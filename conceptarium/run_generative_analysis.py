"""Reconstruct, generate from and steer a trained generative concept model.

Post-hoc counterpart to ``run_experiment.py`` for the generative branch. It
rebuilds a finished run from its Hydra config and checkpoint, then produces the
things a concept bottleneck generative model is *for*:

``overview.png``
    Three rows: real images, their reconstructions through ``q(z | x)``, and
    images decoded from ``z`` drawn from the **prior**. The third row is
    unconditional — nothing lines up with the columns above it — and that is the
    point: sharp reconstructions beside incoherent samples is the signature of a
    posterior that has drifted off the prior.

``steering_<concept>.png``
    One figure per concept. A generated sample on the first row, then that same
    sample decoded once per state of the concept — its own ``z`` and its own
    drawn values for every *other* concept replayed as evidence, so the states
    are the only thing that differs. This is the model's headline claim, that
    intervening on a concept steers the generated output, and nothing else in
    the repository exercises it.

``generative_metrics.csv``
    Per-concept test accuracy and the reconstruction NLL.

Both engines are used, for different jobs: :class:`VariationalInference` is the
only one that consults the guide, so it does the encoding; the ancestral engine
does every decode. They share the same PGM by reference, so no weights are
copied between them.
"""

import csv
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

from conceptarium.evaluate import load_job
from conceptarium.registry import load_registry
from conceptarium.resolvers import register_custom_resolvers
from conceptarium.utils import (
    attach_latent_encoder,
    resolve_graph,
    seed_everything,
    setup_run_env,
    update_config_from_data,
)
from torch_concepts.nn import AncestralSamplingInference, ReconstructionLoss
from torch_concepts.nn.modules.mid.distributions import spec_for

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rebuilding a finished run
# ---------------------------------------------------------------------------
def resolve_job_dir(cfg: DictConfig) -> Path:
    """The Hydra job directory to analyse: explicit, or the newest match."""
    if cfg.get("job_dir"):
        return Path(cfg.job_dir)

    csv_path = os.path.join(get_original_cwd(), "conceptarium", cfg.csv_path)
    filters = dict(cfg.filters) if cfg.get("filters") else None
    rows = load_registry(csv_path, filters=filters)
    if not rows:
        raise SystemExit(
            f"No runs in {csv_path} matching {filters}. Train one first:\n"
            "  python run_experiment.py --config-name cbgm dataset=colormnist\n"
            "(note that `debug: true` suppresses registration), or pass "
            "job_dir=<hydra job dir> explicitly."
        )
    return Path(rows[-1]["run_dir"])


def rebuild(job_cfg: DictConfig):
    """Datamodule and model for a finished job, wired as ``run_experiment`` did.

    Deliberately not ``evaluate.evaluate_job``: that helper still reads the
    pre-``model_cls`` config schema and never builds the latent encoder, so it
    cannot reconstruct a current model.
    """
    # Hydra writes .hydra/config.yaml *before* main() runs, so the saved config
    # has not been through setup_run_env and carries no data root — without this
    # the datamodule falls back to a cwd-relative path and re-downloads.
    job_cfg = setup_run_env(job_cfg)

    datamodule = instantiate(job_cfg.dataset.datamodule, _convert_="all")
    backbone = instantiate(job_cfg.dataset.backbone, _convert_="all")
    datamodule.setup("fit")
    job_cfg = update_config_from_data(job_cfg, datamodule)

    loss = instantiate(job_cfg.loss, _convert_="all")
    metrics = instantiate(job_cfg.metrics, _convert_="all", _partial_=True)(
        annotations=datamodule.annotations
    )
    model = instantiate(job_cfg.model.model_cls, _convert_="all", _partial_=True)(
        annotations=datamodule.annotations,
        graph=resolve_graph(
            datamodule.graph,
            datamodule.annotations,
            job_cfg.dataset.default_task_names,
        ),
        backbone=attach_latent_encoder(job_cfg, backbone),
        loss=loss,
        metrics=metrics,
    )
    return datamodule, model


# ---------------------------------------------------------------------------
# Concept addressing
# ---------------------------------------------------------------------------
def concept_variables(model) -> List:
    """The model's concept variables, in graph order."""
    return [v for v in model.pgm.variables.values() if v.variable_type == "concept"]


def concept_states(variable) -> List[Tuple[str, torch.Tensor]]:
    """Every state of ``variable`` as ``(label, value)``, in the PGM's layout.

    A binary concept is one column that is 0 or 1; a categorical one is a
    one-hot row of width ``size``. Both are what the CPD's value looks like, so
    either can be forced straight into a query.
    """
    size = variable.size
    if size == 1:  # binary
        return [("0", torch.zeros(1, 1)), ("1", torch.ones(1, 1))]
    return [
        (str(s), torch.eye(size)[s].unsqueeze(0)) for s in range(size)
    ]


def observation_of(model, out, name: str = "input") -> torch.Tensor:
    """The reconstructed observation, whichever quantity its family reports.

    ``probs`` for a Bernoulli, ``loc`` for a Normal, ``value`` for a Delta — read
    off the variable so the figures below survive a change of observation family.
    """
    quantity = spec_for(model.pgm.variables[name].distribution).primary_param
    return getattr(out, quantity)[name]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_grid(
    rows: List[torch.Tensor], shape, path: Path, row_labels=None, col_labels=None
) -> None:
    """Write a grid of images, one supplied tensor batch per row.

    Generalises the helper in ``examples/utilization/2_model/15_*.py``: the image
    shape is passed in rather than hard-coded, and the backend is forced to Agg
    so this runs on a headless GPU box. Rows may be of different lengths — the
    grid is as wide as the longest and short rows leave their tail blank.

    ``col_labels`` is one list per row (``None`` for an unlabelled row), titling
    the cells of that row — the state names above a steering sweep, say.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping %s", path.name)
        return

    n_rows = len(rows)
    n_cols = max(r.shape[0] for r in rows)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols, n_rows + 0.4), squeeze=False
    )
    for r, batch in enumerate(rows):
        images = batch.detach().reshape(-1, *shape).cpu()
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c >= images.shape[0]:
                continue
            img = images[c]
            # (C, H, W) -> (H, W, C); a single channel drops to greyscale.
            ax.imshow(img.permute(1, 2, 0).squeeze().numpy().clip(0, 1))
            titles = col_labels[r] if col_labels is not None else None
            if titles is not None and c < len(titles):
                ax.set_title(titles[c], fontsize=6)
        if row_labels is not None:
            # Re-enable the axis only to carry the label, without drawing a box
            # around the first image of the row.
            ax = axes[r][0]
            ax.axis("on")
            ax.set_ylabel(row_labels[r], fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)


def generate(model, engine, query, n_samples):
    """Draw ``n_samples`` from the prior ancestrally.

    Returns ``(images, samples)`` where ``samples`` maps a variable name to its
    *drawn* value — notably ``z`` and each concept — which the steering figures
    replay so that only the intervened concept changes. Unwrapped to plain
    tensors, because that is what an engine accepts back as evidence.
    """
    with torch.no_grad():
        out = engine.query(query=query, evidence={}, n_samples=n_samples)
    samples = {
        name: getattr(out.samples[name], "tensor", out.samples[name])
        for name in out.samples.annotation.labels
    }
    return observation_of(model, out), samples


def figure_overview(model, engine, query, shape, images, generated, out_dir) -> None:
    """Originals, their reconstructions, and unconditional prior samples.

    The generations are *not* reconstructions of the originals above them — they
    are independent draws from ``p(z)``. The three rows sit together because a
    generative model is judged on both at once: sharp reconstructions with
    incoherent samples means the posterior has drifted off the prior.
    """
    with torch.no_grad():
        # The guide is the only route from an image to a posterior z, so the
        # encode step needs the variational engine (the model's own eval one),
        # which in turn requires every variable in its query. The posterior
        # *mean*, not a draw, so the row shows the model's best reconstruction.
        encoded = model(query=list(model.pgm.variables), input=images)
        z = encoded.guide_params["loc"]["z"].tensor
        recon = observation_of(model, engine.query(query=query, evidence={"z": z}))

    save_grid(
        [images, recon, generated],
        shape,
        out_dir / "overview.png",
        row_labels=["original", "reconstruction", "generation"],
    )


def figure_steering(
    model, engine, shape, generated, samples, concepts, variable, out_dir
) -> None:
    """One figure per concept: a generated image, then that image at every state.

    Replays that sample's own ``z`` and its own drawn values for *every other*
    concept as evidence, so the only thing differing across the second row is the
    intervened concept. Clamping the others matters: left free they would be
    redrawn per state, confounding the intervention with fresh sampling noise.
    """
    states = concept_states(variable)
    n = len(states)
    # Column 0 of the generation is "the originally sampled image"; every
    # variation below is that same draw with one concept moved.
    evidence = {"z": samples["z"][:1].expand(n, -1)}
    for other in concepts:
        if other.name != variable.name:
            evidence[other.name] = samples[other.name][:1].expand(n, -1)
    evidence[variable.name] = torch.cat([value for _, value in states], dim=0)

    with torch.no_grad():
        out = engine.query(query=["input"], evidence=evidence)

    save_grid(
        [generated[:1], observation_of(model, out)],
        shape,
        out_dir / f"steering_{variable.name}.png",
        row_labels=["generated", f"do({variable.name})"],
        col_labels=[None, [label for label, _ in states]],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(model, datamodule, query) -> List[Dict[str, object]]:
    """Test metrics plus the reconstruction NLL.

    The concept metrics are the model's own :class:`ConceptMetrics` — the same
    collection ``conf/metrics`` configured and training logged — so binary and
    categorical concepts are routed correctly without this script re-deriving
    the rule, and changing the metrics config changes the report. The NLL comes
    from the same ``ReconstructionLoss`` the objective used.
    """
    loader = datamodule.test_dataloader() or datamodule.val_dataloader()
    reconstruction = ReconstructionLoss(variable="input")
    metrics = model.test_metrics
    metrics.reset()
    total, nll = 0, 0.0

    with torch.no_grad():
        for batch in loader:
            x, c = batch["inputs"]["x"], batch["concepts"]["c"]
            out = model(query=query, input=x)
            out.extra = {"evidence": {"input": x}}
            metrics.update(out, model.prepare_target(c))
            nll += float(reconstruction(out)) * x.shape[0]
            total += x.shape[0]

    rows = [{"metric": k, "value": float(v)} for k, v in metrics.compute().items()]
    rows.append({"metric": "reconstruction_nll", "value": nll / max(total, 1)})
    rows.append({"metric": "n_test_samples", "value": total})
    return rows


# ---------------------------------------------------------------------------
@hydra.main(config_path="conf", config_name="generative_analysis", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.get("seed", 42))

    job_dir = resolve_job_dir(cfg)
    job_cfg, ckpt_path = load_job(job_dir)
    if ckpt_path is None:
        raise SystemExit(f"No checkpoint under {job_dir / 'checkpoints'}.")
    logger.info("analysing %s", job_dir)

    datamodule, model = rebuild(job_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()

    shape = tuple(datamodule.n_features)
    concepts = concept_variables(model)
    # A decoding pass reports only the image, `z` and the concepts: the embedding
    # and bottleneck variables have multi-dimensional events of *differing*
    # widths, which cannot be concatenated into one annotated tensor. They are
    # ancestors of `input`, so they are still computed — just not reported.
    query = ["input", "z", *(v.name for v in concepts)]
    # VariationalInference's contract wants every variable in the query.
    vi_query = list(model.pgm.variables)

    # Decoding engine: ancestral sampling resolves every root through its own
    # prior, so an unconditioned query really does draw z ~ p(z).
    #
    # `hard` must match how the model was TRAINED: the bottleneck mixes state
    # embeddings by the concept score, so decoding a hard assignment when
    # training only ever saw soft blends (or vice versa) is out of distribution.
    # Under `soft_mixing` the training engine samples soft, so this does too.
    hard = cfg.get("hard_sampling")
    if hard is None:
        hard = not getattr(model, "soft_mixing", False)
    # ...and so must the temperature: the relaxation is annealed during
    # training, so decoding at the default 1.0 would sample far softer codes
    # than the trained decoder ever saw. The checkpoint restores the training
    # engine's temperature buffer, so this reads the value training ended on.
    temperature = float(model.train_inference.temperature)
    engine = AncestralSamplingInference(
        model.pgm, p_int=1.0, hard=bool(hard),
        initial_temperature=temperature, annealing="constant",
    )
    logger.info(
        "decoding with %s discrete samples at temperature %.4f "
        "(model.soft_mixing=%s, train engine hard=%s)",
        "hard" if hard else "soft", temperature,
        getattr(model, "soft_mixing", None),
        getattr(model.train_inference, "hard", None),
    )

    out_dir = Path(job_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(cfg.get("n_samples", 10))

    batch = next(iter(datamodule.test_dataloader() or datamodule.val_dataloader()))
    images = batch["inputs"]["x"][:n]

    # One generation pass feeds both figures: the third row of the overview, and
    # the `z`/concept draws the steering figures replay.
    generated, samples = generate(model, engine, query, n)
    figure_overview(model, engine, query, shape, images, generated, out_dir)

    wanted = cfg.get("steer_concepts")
    to_steer = (
        [v for v in concepts if v.name in set(wanted)] if wanted else concepts
    )
    for variable in to_steer:
        figure_steering(
            model, engine, shape, generated, samples, concepts, variable, out_dir
        )

    rows = evaluate(model, datamodule, vi_query)
    csv_out = out_dir / "generative_metrics.csv"
    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", csv_out)
    for row in rows:
        print(f"{row['metric']:>28}: {row['value']}")


if __name__ == "__main__":
    register_custom_resolvers()
    main()
