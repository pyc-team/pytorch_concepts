"""Generate, steer and counterfactually edit a trained generative concept model.

Post-hoc counterpart to ``run_experiment.py`` for the generative branch. It
rebuilds a finished run from its Hydra config and checkpoint, then produces the
three things a concept bottleneck generative model is *for*:

``generation.png``
    Images decoded from ``z`` drawn from the **prior**. Nothing is conditioned
    on: the ancestral engine resolves the root through its own ``FixedPrior``,
    and the guide is not consulted at all.

``steering.png``
    One row per concept: a single ``z``, held fixed, decoded once per state of
    that concept. This is the model's headline claim — that intervening on a
    concept steers the generated output — which nothing else in the repository
    exercises.

``counterfactual.png``
    A real image, its reconstruction through ``q(z | x)``, and the
    reconstruction of the *same* ``z`` with one concept forced to a different
    state. Reading the third row against the second isolates the edit from the
    reconstruction error.

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
def save_grid(rows: List[torch.Tensor], shape, path: Path, row_labels=None) -> None:
    """Write a grid of images, one supplied tensor batch per row.

    Generalises the helper in ``examples/utilization/2_model/15_*.py``: the image
    shape is passed in rather than hard-coded, and the backend is forced to Agg
    so this runs on a headless GPU box.
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


def figure_generation(model, engine, query, shape, n_samples, out_dir) -> None:
    """Decode ``n_samples`` draws from the prior — no evidence, no guide."""
    with torch.no_grad():
        out = engine.query(query=query, evidence={}, n_samples=n_samples)
    save_grid([observation_of(model, out)], shape, out_dir / "generation.png")


def figure_steering(model, engine, query, shape, concepts, out_dir) -> None:
    """One row per concept: the same ``z``, swept across that concept's states."""
    rows, labels = [], []
    for variable in concepts:
        states = concept_states(variable)
        # One z for the whole row: the only thing that varies is the concept.
        z = torch.randn(1, model.latent_size).expand(len(states), -1)
        forced = torch.cat([value for _, value in states], dim=0)
        forced_query = {**{name: None for name in query}, variable.name: forced}
        with torch.no_grad():
            out = engine.query(
                query=forced_query, evidence={"z": z}
            )
        rows.append(observation_of(model, out))
        labels.append(variable.name)
    if rows:
        save_grid(rows, shape, out_dir / "steering.png", row_labels=labels)


def figure_counterfactual(
    model, engine, query, vi_query, shape, images, variable, state, out_dir
) -> None:
    """Original, its reconstruction, and the same ``z`` with one concept forced."""
    with torch.no_grad():
        # The guide is the only route from an image to a posterior z, so the
        # encode step needs the variational engine (the model's own eval one),
        # which in turn requires every variable in its query.
        encoded = model(query=vi_query, input=images)
        # Unwrapped: evidence must be a plain Tensor, and every quantity comes
        # back annotated. The posterior mean, not a sample, so the two decodes
        # differ only by the intervention.
        z = encoded.guide_params["loc"]["z"].tensor

        recon = observation_of(model, engine.query(query=query, evidence={"z": z}))

        forced = state.expand(images.shape[0], -1)
        edited = observation_of(model, engine.query(
            query={**{name: None for name in query}, variable.name: forced},
            evidence={"z": z},
        ))

    save_grid(
        [images, recon, edited],
        shape,
        out_dir / "counterfactual.png",
        row_labels=["original", "reconstruction", f"do({variable.name})"],
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
    # A decoding pass reports only the image and the concepts: the embedding and
    # bottleneck variables have multi-dimensional events of *differing* widths,
    # which cannot be concatenated into one annotated tensor. They are ancestors
    # of `input`, so they are still computed — just not reported.
    query = ["input", *(v.name for v in concepts)]
    # VariationalInference's contract wants every variable in the query.
    vi_query = list(model.pgm.variables)

    # Decoding engine: ancestral sampling resolves every root through its own
    # prior, so an unconditioned query really does draw z ~ p(z). p_int=1.0
    # makes a query value a hard do-intervention, which is what steers.
    # hard=True: an un-intervened concept must still resolve to a genuine state
    # assignment, not a soft blend of every state's embedding — the bottleneck
    # mixes state embeddings by the concept score, so a soft draw would decode
    # a code no training sample (teacher-forced, hence hard) ever produced.
    engine = AncestralSamplingInference(model.pgm, p_int=1.0, hard=True)

    out_dir = Path(job_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(cfg.get("n_samples", 8))

    figure_generation(model, engine, query, shape, n, out_dir)

    wanted = cfg.get("steer_concepts")
    to_steer = (
        [v for v in concepts if v.name in set(wanted)] if wanted else concepts
    )
    figure_steering(model, engine, query, shape, to_steer, out_dir)

    batch = next(iter(datamodule.test_dataloader() or datamodule.val_dataloader()))
    images = batch["inputs"]["x"][:n]
    cf_name = cfg.get("counterfactual_concept") or concepts[0].name
    cf_var = next(v for v in concepts if v.name == cf_name)
    states = concept_states(cf_var)
    which = cfg.get("counterfactual_state")
    cf_state = dict(states).get(str(which), states[-1][1])
    figure_counterfactual(
        model, engine, query, vi_query, shape, images, cf_var, cf_state, out_dir
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
