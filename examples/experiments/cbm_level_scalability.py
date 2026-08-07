"""Forward+backward scalability of a CBM across PyC's abstraction levels.

Compares FIVE implementations of the exact same linear Concept Bottleneck
Model (Koh et al., ICML 2020) — ``input -> latent -> concepts -> task`` — as
the number of (binary) concepts ``N`` grows from 20 to 200,000:

- **baseline**  : plain ``torch.nn`` (``nn.Linear`` only, no PyC at all).
- **low-level** : PyC layers (:class:`LinearEmbeddingToConcept`,
  :class:`LinearConceptToConcept`) called directly, pure-PyTorch control flow.
- **mid-level** : PyC's PGM machinery (:class:`EmbeddingVariable`,
  :class:`ConceptVariable`, :class:`ParametricCPD`, :class:`BayesianNetwork`,
  :class:`DeterministicInference`), built by hand. The concepts and the task
  are each a single **plate** :class:`ConceptVariable` (one node holding all
  members) — the same layout the high-level model derives automatically —
  so both wire ONE encoder / ONE predictor rather than ``N`` per-concept CPDs.
- **high-level**: :class:`ConceptBottleneckModel` with ``lightning=True``
  (so the model class carries the Lightning ``BaseLearner`` mixin) and
  ``plate=True`` (the default), which auto-derives the same plate layout.

All five are given IDENTICAL weights (copied from the baseline's
``nn.Linear`` layers after construction — see :func:`sync_weights`) so a
forward pass produces numerically identical concept/task logits; this is
checked with ``torch.allclose`` at every sweep point before timing. Because
all five share the exact same weights, differences in average
forward+backward time isolate each level's framework overhead.

Four of the five (baseline / low-level / mid-level / high-level) are timed
with a plain manual forward/backward loop, so the comparison measures each
model's own compute graph rather than any training-loop overhead.

- **high-level (Trainer)**: the SAME Lightning-enabled model, but trained
  through an actual :class:`pytorch_lightning.Trainer.fit` call — the
  realistic way one would train it — rather than a hand-rolled loop. Timing a
  ``Trainer.fit`` consistently is not as simple as wrapping it in
  ``time.perf_counter()``: a single ``fit`` call carries a one-time setup cost
  (accelerator/strategy resolution, sanity checks, hook dispatch) that has
  nothing to do with per-step compute and would dominate at small ``N``. This
  script instead does ONE ``trainer.fit`` call over a loader that repeats the
  same batch ``N_WARMUP + N_ITERS`` times, uses a :class:`pytorch_lightning.Callback`
  (:class:`_StepTimer`) to timestamp every ``on_train_batch_start``/``_end``,
  and averages only the LAST ``N_ITERS`` step durations — by then Trainer/
  dataloader/sanity-check setup has already happened once, so it does not leak
  into the measurement (the same warmup/timed split every other
  implementation uses).

Model scheme (identical across all five)::

    input (batch, INPUT_DIM)
      -> latent = LeakyReLU(Linear(INPUT_DIM -> LATENT_DIM))
      -> concept_logits = Linear(LATENT_DIM -> N)
      -> concept_probs  = sigmoid(concept_logits)          # soft CBM bottleneck
      -> task_logits    = Linear(N -> N_TASKS)

Data is random (``torch.randn`` / random binary targets) — values are
irrelevant, only compute is measured. Edit the CONFIG block below to change
the sweep; there are no command-line arguments.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import pytorch_lightning as pl

from torch_concepts import seed_everything, Annotations, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.tensor import AnnotatedTensor
from torch_concepts.nn import (
    LinearEmbeddingToConcept,
    LinearConceptToConcept,
    ParametricCPD,
    BayesianNetwork,
    DeterministicInference,
    LearnablePrior,
    ConceptBottleneckModel,
    ConceptLoss,
)

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# CONFIG  (edit these — no command-line arguments)
# ----------------------------------------------------------------------------
CONCEPT_COUNTS = [20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
N_TASKS = 1
INPUT_DIM = 64
LATENT_DIM = 128
BATCH_SIZE = 32
N_WARMUP = 2
N_ITERS = 5
SEED = 42
DEVICE = torch.device("cpu")
OUTDIR = os.path.dirname(os.path.abspath(__file__))

IMPLEMENTATIONS = ["baseline", "low-level", "mid-level", "high-level", "high-level (Trainer)"]


@dataclass
class Impl:
    """One implementation: its module (for .parameters()/.zero_grad()) and a
    ``forward(x) -> (concept_logits, task_logits)`` callable."""
    name: str
    module: nn.Module
    forward: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


# ----------------------------------------------------------------------------
# Four builders — same architecture, different abstraction level.
# ----------------------------------------------------------------------------
def build_baseline(n: int) -> Impl:
    class Baseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(INPUT_DIM, LATENT_DIM), nn.LeakyReLU())
            self.concept_encoder = nn.Linear(LATENT_DIM, n)
            self.task_predictor = nn.Linear(n, N_TASKS)

        def forward(self, x):
            latent = self.backbone(x)
            concept_logits = self.concept_encoder(latent)
            task_logits = self.task_predictor(torch.sigmoid(concept_logits))
            return concept_logits, task_logits

    module = Baseline()
    return Impl("baseline", module, module.forward)


def build_low_level(n: int) -> Impl:
    class LowLevelCBM(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(INPUT_DIM, LATENT_DIM), nn.LeakyReLU())
            self.concept_encoder = LinearEmbeddingToConcept(in_embeddings=LATENT_DIM, out_concepts=n)
            self.task_predictor = LinearConceptToConcept(in_concepts=n, out_concepts=N_TASKS)

        def forward(self, x):
            latent = self.backbone(x)
            concept_logits = self.concept_encoder(embeddings=latent)
            task_logits = self.task_predictor(concepts=torch.sigmoid(concept_logits))
            return concept_logits, task_logits

    module = LowLevelCBM()
    return Impl("low-level", module, module.forward)


def build_mid_level(n: int, concept_names: List[str], task_names: List[str]) -> Impl:
    input_var = EmbeddingVariable("input", distribution=Delta, size=INPUT_DIM)
    latent_var = EmbeddingVariable("latent", distribution=Delta, size=LATENT_DIM)
    # Plate variables — ONE node per level holding all members, matching the
    # layout the high-level model derives automatically for homogeneous concepts.
    concepts = ConceptVariable("concepts", members=concept_names, distribution=Bernoulli)
    tasks = ConceptVariable("tasks", members=task_names, distribution=Bernoulli)

    input_cpd = ParametricCPD(input_var, parametrization=LearnablePrior(INPUT_DIM), parents=[])
    backbone_cpd = ParametricCPD(
        latent_var, parents=[input_var],
        parametrization=nn.Sequential(nn.Linear(INPUT_DIM, LATENT_DIM), nn.LeakyReLU()),
    )
    encoder_cpd = ParametricCPD(
        concepts, parents=[latent_var],
        parametrization={"logits": LinearEmbeddingToConcept(LATENT_DIM, n)},
    )
    predictor_cpd = ParametricCPD(
        tasks, parents=[concepts],
        parametrization={"logits": LinearConceptToConcept(n, N_TASKS)},
    )

    pgm = BayesianNetwork(
        variables=[input_var, latent_var, concepts, tasks],
        factors=[input_cpd, backbone_cpd, encoder_cpd, predictor_cpd],
    )
    engine = DeterministicInference(pgm, activate_before_propagation=True)

    def forward(x):
        out = engine.query(query=["concepts", "tasks"], evidence={"input": x})
        return out.logits["concepts"], out.logits["tasks"]

    return Impl("mid-level", pgm, forward)


def build_high_level(n: int, concept_names: List[str], task_names: List[str]) -> Impl:
    annotations = Annotations(labels=concept_names + task_names)  # all binary by default
    model = ConceptBottleneckModel(
        input_size=INPUT_DIM,
        annotations=annotations,
        task_names=task_names,
        backbone=nn.Sequential(nn.Linear(INPUT_DIM, LATENT_DIM), nn.LeakyReLU()),
        latent_size=LATENT_DIM,
        lightning=True,          # carries the Lightning BaseLearner mixin
        optim_class=torch.optim.AdamW,
        optim_kwargs={"lr": 0.01},
        plate=True,              # auto-derives the same plate layout as mid-level
    )

    def forward(x):
        out = model(query=["concepts", "tasks"], input=x)
        return out.logits["concepts"], out.logits["tasks"]

    return Impl("high-level", model, forward)


def build_high_level_trainer(n: int, concept_names: List[str], task_names: List[str]) -> Tuple[ConceptBottleneckModel, Annotations]:
    """Same architecture as :func:`build_high_level`, but with a ``loss`` and
    optimizer configured so it can be driven by a real ``Trainer.fit`` (see
    :func:`time_trainer_fit`). Returns the model and its ``Annotations`` (the
    latter is needed to build the ``AnnotatedTensor`` training target)."""
    annotations = Annotations(labels=concept_names + task_names)  # all binary by default
    model = ConceptBottleneckModel(
        input_size=INPUT_DIM,
        annotations=annotations,
        task_names=task_names,
        backbone=nn.Sequential(nn.Linear(INPUT_DIM, LATENT_DIM), nn.LeakyReLU()),
        latent_size=LATENT_DIM,
        lightning=True,
        loss=ConceptLoss(binary=nn.BCEWithLogitsLoss()),
        optim_class=torch.optim.AdamW,
        optim_kwargs={"lr": 0.01},
        plate=True,
    )
    return model, annotations


# ----------------------------------------------------------------------------
# Weight sync (identical init -> identical output) and timing.
# ----------------------------------------------------------------------------
def find_linear(module: nn.Module, in_features: int, out_features: int) -> nn.Linear:
    """Locate the unique ``nn.Linear(in_features, out_features)`` inside ``module``.

    Shape-based lookup: robust to the different internal wiring/naming of each
    implementation (plain attribute vs. wrapped-in-CPD vs. LazyConstructor).
    """
    matches = [
        m for m in module.modules()
        if isinstance(m, nn.Linear) and m.in_features == in_features and m.out_features == out_features
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one Linear({in_features}->{out_features}), found {len(matches)}"
        )
    return matches[0]


def sync_weights(baseline: nn.Module, other: nn.Module, n: int) -> None:
    """Copy baseline's backbone/encoder/predictor weights into ``other`` in place."""
    for in_f, out_f in [(INPUT_DIM, LATENT_DIM), (LATENT_DIM, n), (n, N_TASKS)]:
        src = find_linear(baseline, in_f, out_f)
        dst = find_linear(other, in_f, out_f)
        dst.weight.data.copy_(src.weight.data)
        dst.bias.data.copy_(src.bias.data)


loss_fn = nn.BCEWithLogitsLoss()


def _sync_device(device: torch.device) -> None:
    """Block until all queued device work completes.

    MPS/CUDA dispatch kernels asynchronously, so ``time.perf_counter()``
    around unsynchronized code only measures how fast Python can enqueue
    work, not how long it actually takes to run -- silently underestimating
    wall-clock time (and doing so unevenly across implementations, since
    whichever one happens to trigger an incidental host-device read pays for
    the accumulated backlog while the others don't). Always sync immediately
    before starting and immediately after stopping a timed region.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def time_forward_backward(impl: Impl, x, c_true, y_true) -> float:
    """Average wall-clock time (s) of one forward+backward pass, warm-up excluded."""
    impl.module.to(DEVICE).train()
    for _ in range(N_WARMUP):
        impl.module.zero_grad(set_to_none=True)
        c_logits, y_logits = impl.forward(x)
        (loss_fn(c_logits, c_true) + loss_fn(y_logits, y_true)).backward()
    _sync_device(DEVICE)

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        impl.module.zero_grad(set_to_none=True)
        c_logits, y_logits = impl.forward(x)
        (loss_fn(c_logits, c_true) + loss_fn(y_logits, y_true)).backward()
    _sync_device(DEVICE)
    return (time.perf_counter() - t0) / N_ITERS


class _RepeatedBatch:
    """Minimal iterable "dataloader" that yields the SAME pre-built batch dict
    ``n_steps`` times (so one epoch == ``n_steps`` training steps), with no
    Dataset/collate machinery involved — the batch's ``AnnotatedTensor`` is
    passed straight through."""

    def __init__(self, batch: dict, n_steps: int):
        self.batch = batch
        self.n_steps = n_steps

    def __iter__(self):
        for _ in range(self.n_steps):
            yield self.batch

    def __len__(self):
        return self.n_steps


class _StepTimer(pl.Callback):
    """Timestamps every training step's start/end so per-step duration can be
    recovered after ``trainer.fit`` returns, without polluting the measurement
    with one-time Trainer setup cost (see :func:`time_trainer_fit`).

    Syncs the device immediately before each timestamp (see :func:`_sync_device`)
    so a step's recorded duration reflects completed device work rather than
    just async dispatch time, and doesn't leak backlog into the next step.
    """

    def __init__(self):
        self.starts: List[float] = []
        self.ends: List[float] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        _sync_device(DEVICE)
        self.starts.append(time.perf_counter())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        _sync_device(DEVICE)
        self.ends.append(time.perf_counter())


def _lightning_accelerator(device: torch.device) -> str:
    """Map a ``torch.device`` to the Lightning ``accelerator`` string that
    keeps ``Trainer.fit`` on the same device (Lightning otherwise defaults to
    ``"cpu"``, silently moving the model off ``DEVICE`` while the batch stays
    there, causing a device-mismatch error)."""
    return {"cuda": "gpu", "mps": "mps", "cpu": "cpu"}[device.type]


def time_trainer_fit(model: ConceptBottleneckModel, batch: dict) -> float:
    """Average wall-clock time (s) of one real Lightning training step.

    A single ``trainer.fit`` runs ``N_WARMUP + N_ITERS`` steps over a loader
    that repeats ``batch``; a callback timestamps every step and only the LAST
    ``N_ITERS`` are averaged, so the one-time Trainer/dataloader/sanity-check
    setup cost (already paid during the first ``N_WARMUP`` steps) does not
    leak into the measurement — the same warmup/timed split every other
    implementation uses.
    """
    loader = _RepeatedBatch(batch, N_WARMUP + N_ITERS)
    timer = _StepTimer()
    trainer = pl.Trainer(
        max_epochs=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        accelerator=_lightning_accelerator(DEVICE),
        devices=1,
        callbacks=[timer],
    )
    model.to(DEVICE).train()
    trainer.fit(model, train_dataloaders=loader)
    durations = [end - start for start, end in zip(timer.starts, timer.ends)]
    return sum(durations[N_WARMUP:]) / N_ITERS


def main():
    results: Dict[str, List] = {s: [] for s in IMPLEMENTATIONS}

    print(f"device={DEVICE} | batch={BATCH_SIZE} | input_dim={INPUT_DIM} | "
          f"latent={LATENT_DIM} | warmup={N_WARMUP} | iters={N_ITERS}")
    print(f"{'N':>10} | " + " | ".join(f"{s:>20}" for s in IMPLEMENTATIONS) + " | check")
    print("-" * 130)

    for n in CONCEPT_COUNTS:
        concept_names = [f"c{i}" for i in range(n)]
        task_names = [f"t{i}" for i in range(N_TASKS)]

        x = torch.randn(BATCH_SIZE, INPUT_DIM, device=DEVICE)
        c_true = torch.randint(0, 2, (BATCH_SIZE, n), device=DEVICE).float()
        y_true = torch.randint(0, 2, (BATCH_SIZE, N_TASKS), device=DEVICE).float()

        seed_everything(SEED); baseline = build_baseline(n)
        seed_everything(SEED); low = build_low_level(n)
        seed_everything(SEED); mid = build_mid_level(n, concept_names, task_names)
        seed_everything(SEED); high = build_high_level(n, concept_names, task_names)
        seed_everything(SEED); trainer_model, trainer_annotations = build_high_level_trainer(n, concept_names, task_names)

        impls = {"baseline": baseline, "low-level": low, "mid-level": mid, "high-level": high}
        for impl in impls.values():
            impl.module.to(DEVICE)
        trainer_model.to(DEVICE)

        for name, impl in impls.items():
            if name != "baseline":
                sync_weights(baseline.module, impl.module, n)
        sync_weights(baseline.module, trainer_model, n)

        # Correctness check: identical weights -> identical forward output.
        with torch.no_grad():
            ref_c, ref_y = baseline.forward(x)
            ok = True
            max_diff = 0.0
            for name, impl in impls.items():
                if name == "baseline":
                    continue
                c_logits, y_logits = impl.forward(x)
                diff = max((c_logits - ref_c).abs().max().item(), (y_logits - ref_y).abs().max().item())
                max_diff = max(max_diff, diff)
                if not torch.allclose(c_logits, ref_c, atol=1e-4, rtol=1e-4) or \
                   not torch.allclose(y_logits, ref_y, atol=1e-4, rtol=1e-4):
                    ok = False

            trainer_model.eval()
            out = trainer_model(query=["concepts", "tasks"], input=x)
            tc_logits, ty_logits = out.logits["concepts"], out.logits["tasks"]
            diff = max((tc_logits - ref_c).abs().max().item(), (ty_logits - ref_y).abs().max().item())
            max_diff = max(max_diff, diff)
            if not torch.allclose(tc_logits, ref_c, atol=1e-4, rtol=1e-4) or \
               not torch.allclose(ty_logits, ref_y, atol=1e-4, rtol=1e-4):
                ok = False
            trainer_model.train()

        row = {}
        for name in IMPLEMENTATIONS:
            if name == "high-level (Trainer)":
                continue
            impl = impls[name]
            try:
                row[name] = time_forward_backward(impl, x, c_true, y_true)
            except (RuntimeError, MemoryError) as exc:
                row[name] = None
                print(f"  {name} N={n:,}: {type(exc).__name__}: {str(exc)[:60]}")
            results[name].append(row[name])

        # high-level (Trainer): real Trainer.fit loop (see time_trainer_fit).
        c_annotated = AnnotatedTensor(torch.cat([c_true, y_true], dim=1), trainer_annotations, axis=1)
        batch = {"inputs": {"x": x}, "concepts": {"c": c_annotated}}
        try:
            row["high-level (Trainer)"] = time_trainer_fit(trainer_model, batch)
        except (RuntimeError, MemoryError) as exc:
            row["high-level (Trainer)"] = None
            print(f"  high-level (Trainer) N={n:,}: {type(exc).__name__}: {str(exc)[:60]}")
        results["high-level (Trainer)"].append(row["high-level (Trainer)"])

        cells = " | ".join(
            (f"{row[s]:>20.5f}" if row[s] is not None else f"{'-':>20}") for s in IMPLEMENTATIONS
        )
        check = "OK" if ok else f"MISMATCH(max={max_diff:.2e})"
        print(f"{n:>10,} | {cells} | {check}")

    _plot(results)


def _plot(results: Dict[str, List]) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    markers = {"baseline": "o-", "low-level": "s-", "mid-level": "^-",
               "high-level": "d-", "high-level (Trainer)": "x--"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for s in IMPLEMENTATIONS:
        xs = [c for c, t in zip(CONCEPT_COUNTS, results[s]) if t is not None]
        ys = [t for t in results[s] if t is not None]
        if xs:
            ax.plot(xs, ys, markers[s], label=s)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of concepts (N)")
    ax.set_ylabel(f"avg time for 1 forward+backward pass (batch={BATCH_SIZE}) (s)")
    ax.set_title("CBM scalability across PyC abstraction levels")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_png = os.path.join(OUTDIR, "cbm_level_scalability.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved plot to {out_png}")


if __name__ == "__main__":
    main()
