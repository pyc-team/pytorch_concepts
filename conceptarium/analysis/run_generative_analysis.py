"""Qualitative and quantitative analysis of trained generative concept models.

Finds every finished CBGM / CVAE run under ``--outputs``, rebuilds it from its saved
config and checkpoint, and writes per run:

    results/generative/fid.csv                              one row per run
    results/generative/<model>/<run>/generation.png         5x2 prior samples + concepts
    results/generative/<model>/<run>/steering.png           CBGM only, see below

Generation is a single unconditioned ancestral query. Each model's own graph supplies the
order: CBGM draws ``z -> embeddings -> c -> mixing -> x``, CVAE ``c -> z -> x`` (its prior
is conditional), so nothing here branches on the model.

Steering replays a generation with one concept clamped. There is no ``do()`` API for the
PGM path; whole-variable ``evidence`` *is* the intervention -- it clamps the value, skips
that variable's CPD and recomputes every descendant. Clamping ``z`` and the concepts as
drawn therefore reproduces the sample exactly (asserted below), and moving one concept off
that baseline isolates its effect.

Usage:
    cd conceptarium
    python analysis/run_generative_analysis.py --n-fid 256 --device cpu
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conceptarium.resolvers import register_custom_resolvers  # noqa: E402
from conceptarium.utils import (  # noqa: E402
    attach_latent_encoder,
    resolve_graph,
    seed_everything,
    setup_run_env,
    update_config_from_data,
)
from torch_concepts.nn import AncestralSamplingInference  # noqa: E402

logger = logging.getLogger(__name__)

MODELS = {  # _target_ suffix -> short name used in paths and the table
    "ConceptBottleneckGenerativeModel": "cbgm",
    "ConditionalVariationalAutoencoder": "cvae",
}


def find_runs(root: Path):
    """Every finished generative job directory under ``root``, oldest first.

    A job needs both halves to be rebuildable: the config Hydra wrote and a checkpoint
    (which also stands in for "it did not crash"). Filtering on the model target is what
    keeps the discriminative runs sharing this output tree out of the table.
    """
    found = {}
    for config in sorted(root.rglob(".hydra/config.yaml")):
        job = config.parent.parent
        target = OmegaConf.select(OmegaConf.load(config), "model.model_cls._target_") or ""
        if target.split(".")[-1] in MODELS and any((job / "checkpoints").glob("*.ckpt")):
            found.setdefault(job.resolve(), None)
    return sorted(found, key=lambda p: p.stat().st_mtime)


def load(job_dir: Path, device):
    """Rebuild a run's datamodule and model, weights loaded, ready to sample from."""
    cfg = setup_run_env(OmegaConf.load(job_dir / ".hydra" / "config.yaml"))
    datamodule = instantiate(cfg.dataset.datamodule, _convert_="all")
    datamodule.setup("fit")
    cfg = update_config_from_data(cfg, datamodule)

    model = instantiate(cfg.model.model_cls, _convert_="all", _partial_=True)(
        annotations=datamodule.annotations,
        graph=resolve_graph(datamodule.graph, datamodule.annotations,
                            cfg.dataset.default_task_names),
        backbone=attach_latent_encoder(cfg, instantiate(cfg.dataset.backbone, _convert_="all")),
        loss=instantiate(cfg.loss, _convert_="all"),
    )
    ckpt = sorted((job_dir / "checkpoints").glob("*.ckpt"))[0]
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    model.eval().to(device)

    # The relaxation is annealed during training and the checkpoint restores that buffer,
    # so decoding at the default 1.0 would sample far softer concept codes than the trained
    # decoder ever saw.
    engine = AncestralSamplingInference(
        model.pgm, initial_temperature=float(model.train_inference.temperature),
        annealing="constant",
    )
    return cfg, datamodule, model, engine


def grid(rows, shape, path, titles=None, row_labels=None):
    """Write a grid of images, one tensor batch per row; short rows leave a blank tail."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    n_rows, n_cols = len(rows), max(r.shape[0] for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.1 * n_cols, 1.35 * n_rows),
                             squeeze=False)
    for r, batch in enumerate(rows):
        images = batch.detach().reshape(-1, *shape).cpu().clamp(0, 1)
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c < images.shape[0]:
                ax.imshow(images[c].permute(1, 2, 0).squeeze().numpy())
            else:
                ax.set_visible(False)
            if titles is not None and titles[r] is not None and c < len(titles[r]):
                ax.set_title(titles[r][c], fontsize=5)
        if row_labels is not None:
            axes[r][0].set_ylabel(row_labels[r], fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)


def label(annotations, name, value):
    """``"digit=7"``. Falls back to the state index when a dataset names no states."""
    concept = annotations.concept(name)
    state = int(value.argmax()) if concept.cardinality > 1 else int(value.round())
    return f"{name}={concept.states[state] if concept.states else state}"


def states_of(variable, device):
    """Every state of a concept as a one-row value: binary has ``size == 1`` but 2 states."""
    if variable.size == 1:
        return [("0", torch.zeros(1, 1, device=device)), ("1", torch.ones(1, 1, device=device))]
    return [(str(s), torch.eye(variable.size, device=device)[s: s + 1])
            for s in range(variable.size)]


def draw(engine, latent, concepts, n, batch_size):
    """``(images, drawn)`` from ``n`` prior draws. ``drawn`` holds ``z`` and each concept."""
    names = ["input", latent, *(v.name for v in concepts)]
    images, drawn = [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            out = engine.query(query=names, evidence={},
                               n_samples=min(batch_size, n - start))
            images.append(out.value["input"].tensor)
            drawn.append({k: out.samples[k].tensor for k in [latent, *(v.name for v in concepts)]})
    return (torch.cat(images),
            {k: torch.cat([d[k] for d in drawn]) for k in drawn[0]})


def decode(engine, evidence, batch_size):
    """The observation under ``evidence`` -- i.e. an intervention, chunked."""
    n = next(iter(evidence.values())).shape[0]
    with torch.no_grad():
        return torch.cat([
            engine.query(query=["input"],
                         evidence={k: v[i: i + batch_size] for k, v in evidence.items()}
                         ).value["input"].tensor
            for i in range(0, n, batch_size)
        ])


def figure_generation(images, drawn, annotations, concepts, shape, path, rows=2, cols=5):
    """``rows x cols`` prior samples, each captioned with its own concepts."""
    n = rows * cols
    captions = ["\n".join(label(annotations, v.name, drawn[v.name][i]) for v in concepts)
                for i in range(n)]
    grid([images[r * cols:(r + 1) * cols] for r in range(rows)], shape, path,
         titles=[captions[r * cols:(r + 1) * cols] for r in range(rows)])


def figure_steering(engine, images, drawn, model, concepts, shape, path, batch_size, n_rows=5):
    """Each row: one generated sample, then that sample at every state of every concept.

    Every column is decoded from the *same* ``z`` as the original beside it, so a
    difference down a row is the intervention and nothing else.
    """
    base = {name: value[:n_rows] for name, value in drawn.items()}
    replay = decode(engine, base, batch_size)
    # The intervention is only meaningful if the no-op case is exact: clamping the drawn
    # values must reproduce the sample. A silent regression here would still draw a
    # plausible-looking figure, so fail loudly instead.
    assert torch.allclose(replay, images[:n_rows], atol=1e-5), \
        "steering replay does not reproduce the generation; interventions are not isolated"

    columns, titles = [images[:n_rows]], ["original"]
    for variable in concepts:
        for state, value in states_of(model.pgm.variables[variable.name], images.device):
            columns.append(decode(engine, {**base, variable.name: value.expand(n_rows, -1)},
                                  batch_size))
            titles.append(f"{variable.name}={state}")
    grid([torch.stack([col[r] for col in columns]) for r in range(n_rows)], shape, path,
         titles=[titles] + [None] * (n_rows - 1))


class Inception(nn.Module):
    """Inception-v3 pool3 features, the FID convention.

    Hand-rolled because ``FrechetInceptionDistance(feature=2048)`` needs ``torch-fidelity``,
    which is not installed; a custom ``feature=`` module is the documented alternative.
    ``num_features`` is declared so torchmetrics skips its probe, which would call this with
    a dummy *image* while it expects a flat sample.
    """

    num_features = 2048

    def __init__(self, shape):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        self.net, self.shape = net.eval(), tuple(shape)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.inference_mode()
    def forward(self, flat):
        x = flat.reshape(-1, *self.shape).clamp(0, 1)
        x = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)


def fid(generated, datamodule, shape, device, batch_size):
    """FID between the generated samples and an equal number of real test images."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    loader = datamodule.test_dataloader() or datamodule.val_dataloader()
    real, taken = [], 0
    for batch in loader:
        x = batch["inputs"]["x"]
        real.append(x.reshape(x.shape[0], -1)[: len(generated) - taken])
        taken += real[-1].shape[0]
        if taken >= len(generated):
            break
    real = torch.cat(real).to(device)
    n = min(len(real), len(generated))
    if n < 2:
        return float("nan"), n

    metric = FrechetInceptionDistance(feature=Inception(shape).to(device),
                                      normalize=False).to(device)
    for source, is_real in ((real[:n], True), (generated[:n], False)):
        for i in range(0, n, batch_size):
            metric.update(source[i: i + batch_size], real=is_real)
    return float(metric.compute()), n


def analyse(job_dir, args, device, out_root):
    """Figures and the FID row for one trained run."""
    cfg, datamodule, model, engine = load(job_dir, device)
    name = MODELS[cfg.model.model_cls._target_.split(".")[-1]]
    shape = tuple(datamodule.n_features)
    concepts = [v for v in model.pgm.variables.values() if v.variable_type == "concept"]
    if args.steer_concepts:
        concepts = [v for v in concepts if v.name in set(args.steer_concepts)]

    # `<date>_<sweep>_<job>`: a job dir is named for its number alone, so every job of a
    # sweep would otherwise share one output directory and overwrite the last.
    run_id = f"{job_dir.parent.parent.name}_{job_dir.parent.name}_{job_dir.name}"
    out_dir = out_root / name / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    images, drawn = draw(engine, "z", concepts, max(args.n_fid, 10), args.batch_size)

    figure_generation(images, drawn, datamodule.annotations, concepts, shape,
                      out_dir / "generation.png")
    if name == "cbgm":
        figure_steering(engine, images, drawn, model, concepts, shape,
                        out_dir / "steering.png", args.batch_size)

    score, n = fid(images[: args.n_fid], datamodule, shape, device, args.batch_size)
    return {"model": name, "run_dir": str(job_dir), "fid": score, "n_fid_samples": n}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--results", type=Path, default=Path("results/generative"))
    parser.add_argument("--n-fid", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steer-concepts", nargs="*", default=None,
                        help="restrict the steering figure's columns (CelebA has 40 concepts)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    jobs = find_runs(args.outputs)
    if not jobs:
        raise SystemExit(f"No CBGM/CVAE runs with checkpoints under {args.outputs}.")
    logger.info("analysing %d run(s) on %s", len(jobs), device)

    rows, failures = [], []
    for job_dir in jobs:
        print(f"\n=== {job_dir} ===")
        try:
            seed_everything(args.seed)
            rows.append(analyse(job_dir, args, device, args.results))
            print(rows[-1])
        except Exception as error:  # a checkpoint predating an architecture change, say
            logger.exception("skipping %s", job_dir)
            failures.append((job_dir, error))

    if rows:
        args.results.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(rows).sort_values("fid")
        table.to_csv(args.results / "fid.csv", index=False)
        print(f"\n{table.to_string(index=False)}\n\nwrote {args.results / 'fid.csv'}")
    for job_dir, error in failures:
        print(f"skipped {job_dir}: {type(error).__name__}: {error}")
    if failures and not rows:
        raise SystemExit(1)


if __name__ == "__main__":
    register_custom_resolvers()
    main()
