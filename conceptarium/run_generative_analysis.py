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

By default this runs on the newest matching run. Set ``all_jobs=true`` to
produce the same three artefacts for **every** run in the registry that matches
``filters`` — one set of figures per model, which is what comparing a sweep
needs — plus a ``generative_metrics_all.csv`` collecting them into one table. A
run whose checkpoint will not load (an architecture change since it was trained,
usually) is reported and skipped rather than aborting the rest.

Both engines are used, for different jobs: :class:`VariationalInference` is the
only one that consults the guide, so it does the encoding; the ancestral engine
does every decode. They share the same PGM by reference, so no weights are
copied between them.

On speed: the guide runs the image backbone live and torchvision preprocessing
resizes to 224x224, so scoring the test set is the dominant cost — on
Color-MNIST that is 64x the pixels of the original 28x28, minutes of ResNet on a
CPU. ``accelerator`` picks the device; ``max_eval_batches`` caps the metric pass
when an approximate number will do.
"""

import csv
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
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
def resolve_job_dirs(cfg: DictConfig) -> List[Path]:
    """Every Hydra job directory to analyse, oldest first.

    An explicit ``job_dir`` wins and may be a single path or a list. Otherwise
    the registry is filtered by ``filters``: with ``all_jobs`` **every** match is
    returned, which is what a sweep wants — one set of figures per trained model
    — and without it only the newest.
    """
    if cfg.get("job_dir"):
        given = cfg.job_dir
        if isinstance(given, str):
            return [Path(given)]
        return [Path(d) for d in given]

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
    selected = rows if cfg.get("all_jobs") else rows[-1:]
    return [Path(row["run_dir"]) for row in selected]


def resolve_job_dir(cfg: DictConfig) -> Path:
    """The single newest job directory matching ``cfg``.

    Thin wrapper over :func:`resolve_job_dirs` for the callers that analyse one
    model at a time (``run_generative_diagnostics.py``).
    """
    return resolve_job_dirs(cfg)[-1]


def resolve_device(accelerator: Optional[str]) -> torch.device:
    """Device to run the analysis on.

    ``'auto'`` (or unset) picks CUDA when it is there and CPU otherwise —
    deliberately not MPS, which :mod:`torch_concepts.backbone` refuses for the
    same reason (torchvision preprocessing and HuggingFace models misbehave on
    it).

    This matters more than it looks. The guide runs the image backbone live, and
    torchvision's preprocessing resizes to 224x224 — on Color-MNIST that is 64x
    the pixels of the original 28x28 — so a full test-set pass is minutes of
    ResNet on a CPU and seconds on a GPU.
    """
    if accelerator in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(accelerator)


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


def concept_states(variable, device=None) -> List[Tuple[str, torch.Tensor]]:
    """Every state of ``variable`` as ``(label, value)``, in the PGM's layout.

    A binary concept is one column that is 0 or 1; a categorical one is a
    one-hot row of width ``size``. Both are what the CPD's value looks like, so
    either can be forced straight into a query.
    """
    size = variable.size
    if size == 1:  # binary
        values = [("0", torch.zeros(1, 1)), ("1", torch.ones(1, 1))]
    else:
        values = [(str(s), torch.eye(size)[s].unsqueeze(0)) for s in range(size)]
    if device is None:
        return values
    return [(label, value.to(device)) for label, value in values]


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
    states = concept_states(variable, device=samples["z"].device)
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
def evaluate(
    model,
    datamodule,
    query,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Test metrics plus the reconstruction NLL.

    The concept metrics are the model's own :class:`ConceptMetrics` — the same
    collection ``conf/metrics`` configured and training logged — so binary and
    categorical concepts are routed correctly without this script re-deriving
    the rule, and changing the metrics config changes the report. The NLL comes
    from the same ``ReconstructionLoss`` the objective used.

    This is the expensive part of the script by a wide margin: every batch runs
    the guide, and the guide runs the image backbone on upsampled inputs. Hence
    ``device`` — pass a GPU when there is one — and ``max_batches``, which caps
    the pass when an approximate number is enough. ``n_test_samples`` in the
    output always reports how many samples were actually scored.
    """
    loader = datamodule.test_dataloader() or datamodule.val_dataloader()
    reconstruction = ReconstructionLoss(variable="input")
    metrics = model.test_metrics
    metrics.reset()
    total, nll = 0, 0.0

    # inference_mode over no_grad: same effect on autograd, and it also skips
    # view/version tracking, which is pure overhead for a pass that never
    # backprops.
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                logger.info("stopping evaluation after %d batches (max_batches)", index)
                break
            x, c = batch["inputs"]["x"], batch["concepts"]["c"]
            if device is not None:
                x, c = x.to(device), c.to(device)
            out = model(query=query, input=x)
            out.extra = {"evidence": {"input": x}}
            metrics.update(out, model.prepare_target(c))
            # Accumulate on-device and read once at the end: float() on a CUDA
            # tensor synchronises, so doing it per batch serialises the loop
            # against the GPU it was just handed to.
            nll += reconstruction(out) * x.shape[0]
            total += x.shape[0]

    rows = [{"metric": k, "value": float(v)} for k, v in metrics.compute().items()]
    rows.append({"metric": "reconstruction_nll", "value": float(nll) / max(total, 1)})
    rows.append({"metric": "n_test_samples", "value": total})
    return rows


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------
def analyse_job(cfg: DictConfig, job_dir: Path, device: torch.device) -> List[Dict]:
    """Figures and metrics for a single trained model, written beside it."""
    seed_everything(cfg.get("seed", 42))

    job_cfg, ckpt_path = load_job(job_dir)
    if ckpt_path is None:
        # A plain exception, not SystemExit: the caller analyses a *list* of runs
        # and catches Exception to skip the broken ones, and SystemExit is a
        # BaseException that would sail past that and kill the whole sweep.
        raise FileNotFoundError(f"No checkpoint under {job_dir / 'checkpoints'}.")
    logger.info("analysing %s", job_dir)

    datamodule, model = rebuild(job_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    # Moves the backbone with it: `Backbone` is an nn.Module holding its
    # torchvision model as a submodule, and reads `.device` off its parameters.
    model.to(device)

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
    images = batch["inputs"]["x"][:n].to(device)

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

    rows = evaluate(
        model, datamodule, vi_query,
        device=device, max_batches=cfg.get("max_eval_batches"),
    )
    csv_out = out_dir / "generative_metrics.csv"
    with open(csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", csv_out)
    for row in rows:
        print(f"{row['metric']:>28}: {row['value']}")
    return rows


# ---------------------------------------------------------------------------
@hydra.main(config_path="conf", config_name="generative_analysis", version_base="1.3")
def main(cfg: DictConfig) -> None:
    job_dirs = resolve_job_dirs(cfg)
    device = resolve_device(cfg.get("accelerator"))
    logger.info("analysing %d run(s) on %s", len(job_dirs), device)

    summary: List[Dict] = []
    failures: List[Tuple[Path, Exception]] = []
    for job_dir in job_dirs:
        print(f"\n=== {job_dir} ===")
        try:
            rows = analyse_job(cfg, job_dir, device)
        except Exception as error:
            # One unreadable run must not cost the rest of a sweep its figures.
            # The common cause is a checkpoint predating an architecture change,
            # which surfaces here as a state_dict shape mismatch.
            logger.exception("skipping %s: %s", job_dir, error)
            failures.append((job_dir, error))
            continue
        for row in rows:
            summary.append({"run_dir": str(job_dir), **row})

    # A sweep is only comparable side by side, so collect every run's metrics
    # into one table next to this analysis job rather than only beside each run.
    if len(job_dirs) > 1 and summary:
        # `chdir: false`, so cwd is the original working directory rather than
        # this job's; ask Hydra where its output actually went.
        combined = Path(HydraConfig.get().runtime.output_dir) / "generative_metrics_all.csv"
        with open(combined, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["run_dir", "metric", "value"])
            writer.writeheader()
            writer.writerows(summary)
        logger.info("wrote %s", combined)
        print(f"\ncombined metrics: {combined}")

    if failures:
        print(f"\n{len(failures)} of {len(job_dirs)} run(s) failed:")
        for job_dir, error in failures:
            print(f"  {job_dir}: {type(error).__name__}: {error}")
        # Nothing at all came out: fail the process rather than exiting 0 on an
        # empty result, which a caller would read as success.
        if len(failures) == len(job_dirs):
            raise SystemExit(1)


if __name__ == "__main__":
    register_custom_resolvers()
    main()
