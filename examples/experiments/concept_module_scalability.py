"""Scalability of the concept module: shared vs individual CPDs (inference only).

Measures the wall-clock time to run ONE forward pass (a :class:`ForwardInference`
``query``) over an entire random dataset — one epoch, all batches, **no
backward** — as the number of concepts ``N`` grows, comparing two mid-level
:class:`BayesianNetwork` formulations of the same CBM concept module:

- **individual**: ``N`` independent :class:`ParametricCPD`s, each mapping the
  shared latent to one concept probability (``N`` separate
  ``Linear(latent_dims -> 1)`` encoders).
- **shared**: ONE shared :class:`ParametricCPD` whose encoder maps the latent to
  all ``N`` concept probabilities at once (``Linear(latent_dims -> N)``). Each
  concept stays an individually addressable graph node whose params are a view
  into the stacked output. There are no per-concept embeddings — the shared
  CPD's output size is ``N``.

Model scheme follows ``examples/utilization/1_pgm/0.1_concept_bottleneck_model.py``::

    input (root, supplied as evidence) -> latent (backbone) -> concepts

The data is random (``torch.randn``) — values are irrelevant, we only measure
compute. Edit the CONFIG block below to change the sweep; there are no
command-line arguments.

Scope note: both formulations materialize one graph node per concept, so the
sweep is capped at ``MAX_BUILD`` (sizes above it are skipped and logged rather
than hanging). The plot is log-log over the feasible regime.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ParametricCPD,
    BayesianNetwork,
    ForwardInference,
    LinearEmbeddingToConcept,
    LearnablePrior,
    Sequential,
)

# ----------------------------------------------------------------------------
# CONFIG  (edit these — no command-line arguments)
# ----------------------------------------------------------------------------
CONCEPT_COUNTS = [2, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
N_SAMPLES = 2_048      # rows in the random dataset
BATCH_SIZE = 256       # one epoch == ceil(N_SAMPLES / BATCH_SIZE) batches
INPUT_DIM = 32         # input feature dimension
LATENT_DIMS = 16       # backbone latent dimension
SEED = 42
DEVICE = torch.device("cpu")
MAX_BUILD = 200_000    # skip (and log) any N above this — both setups
                       # materialize one graph node per concept.
OUTDIR = os.path.dirname(os.path.abspath(__file__))


PLATE_NAME = "concepts"   # the plate variable's name; querying it returns all members


def _root_and_backbone():
    """Shared scaffold input (root) -> latent (backbone); returns its vars + CPDs."""
    input_var = EmbeddingVariable("input", distribution=Delta, size=INPUT_DIM)
    latent_var = EmbeddingVariable("latent", distribution=Delta, size=LATENT_DIMS)
    input_cpd = ParametricCPD(input_var, parametrization=LearnablePrior(INPUT_DIM), parents=[])
    backbone = ParametricCPD(
        latent_var, parents=[input_var],
        parametrization=nn.Sequential(nn.Linear(INPUT_DIM, LATENT_DIMS), nn.LeakyReLU()),
    )
    return input_var, latent_var, input_cpd, backbone


def build_individual(names: List[str]) -> BayesianNetwork:
    """N independent concept variables + N independent CPDs (the list path)."""
    input_var, latent_var, input_cpd, backbone = _root_and_backbone()
    concepts = ConceptVariable(names, distribution=Bernoulli)            # N variables
    cpds = ParametricCPD(                                                 # N CPDs
        concepts, parents=[latent_var],
        parametrization=Sequential(LinearEmbeddingToConcept(LATENT_DIMS, 1), nn.Sigmoid()),
    )
    return BayesianNetwork(
        variables=[input_var, latent_var, *concepts],
        factors=[input_cpd, backbone, *cpds],
    )


def build_shared(names: List[str]) -> BayesianNetwork:
    """ONE plate variable + ONE CPD emitting all members at once."""
    input_var, latent_var, input_cpd, backbone = _root_and_backbone()
    concepts = ConceptVariable(PLATE_NAME, members=names, distribution=Bernoulli)
    cpd = ParametricCPD(
        concepts, parents=[latent_var],
        parametrization=Sequential(
            LinearEmbeddingToConcept(LATENT_DIMS, concepts.size), nn.Sigmoid()),
    )
    return BayesianNetwork(variables=[input_var, latent_var, concepts],
                           factors=[input_cpd, backbone, cpd])


def time_epoch(model: BayesianNetwork, query, make_evidence, x: torch.Tensor) -> float:
    """Time one epoch (all batches) of ``query`` under per-batch ``make_evidence``.

    Inference only (no backward). A warm-up batch builds the engine caches and is
    excluded from timing.
    """
    model.to(DEVICE).eval()
    engine = ForwardInference(model, mode="deterministic")
    with torch.no_grad():
        engine.query(query, evidence=make_evidence(x[:BATCH_SIZE]))  # warm-up (not timed)
        t0 = time.perf_counter()
        for i in range(0, x.shape[0], BATCH_SIZE):
            engine.query(query, evidence=make_evidence(x[i:i + BATCH_SIZE]))
        return time.perf_counter() - t0


# Four scenarios. The three "shared" ones use the SAME plate model; only HOW it
# is addressed differs, which is exactly what isolates the addressing/slicing cost:
#   shared:plate       -> query the plate name (one stacked output)        ~ O(K)
#   shared:members     -> query the N member names                         ~ O(N) names+outputs
#   shared:members-ev  -> observe all N members (partial-plate evidence)   ~ O(N) overlay
SETUPS = ["individual", "shared:plate", "shared:members", "shared:members-ev"]


def main():
    seed_everything(SEED)
    x = torch.randn(N_SAMPLES, INPUT_DIM, device=DEVICE)
    results: Dict[str, List] = {s: [] for s in SETUPS}

    print(f"device={DEVICE} | n_samples={N_SAMPLES} | batch={BATCH_SIZE} | "
          f"input_dim={INPUT_DIM} | latent={LATENT_DIMS} | max_build={MAX_BUILD:,}")
    print(f"{'N':>10} | " + " | ".join(f"{s:>17}" for s in SETUPS))
    print("-" * 86)

    for n in CONCEPT_COUNTS:
        if n > MAX_BUILD:
            for s in SETUPS:
                results[s].append(None)
            print(f"{n:>10,} | " + " | ".join(f"{'skip(>MAX_BUILD)':>17}" for _ in SETUPS))
            continue

        names = [f"c{i}" for i in range(n)]
        member_ev = {m: torch.ones(BATCH_SIZE, 1, device=DEVICE) for m in names}
        seed_everything(SEED); individual = build_individual(names)
        seed_everything(SEED); shared = build_shared(names)
        plan = {
            "individual":        (individual, names,         lambda xb: {"input": xb}),
            "shared:plate":      (shared,     [PLATE_NAME],  lambda xb: {"input": xb}),
            "shared:members":    (shared,     names,         lambda xb: {"input": xb}),
            "shared:members-ev": (shared,     [PLATE_NAME],  lambda xb: {"input": xb, **member_ev}),
        }

        row = {}
        for s in SETUPS:
            model, q, make_ev = plan[s]
            try:
                row[s] = time_epoch(model, q, make_ev, x)
            except (RuntimeError, MemoryError) as exc:
                row[s] = None
                print(f"  {s} N={n:,}: {type(exc).__name__}: {str(exc)[:60]}")
            results[s].append(row[s])

        cells = " | ".join(
            (f"{row[s]:>17.3f}" if row[s] is not None else f"{'-':>17}") for s in SETUPS
        )
        print(f"{n:>10,} | {cells}")

    _plot(results)


def _plot(results: Dict[str, List]) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    markers = {"individual": "o-", "shared:plate": "s-",
               "shared:members": "^-", "shared:members-ev": "d-"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for s in SETUPS:
        xs = [c for c, t in zip(CONCEPT_COUNTS, results[s]) if t is not None]
        ys = [t for t in results[s] if t is not None]
        if xs:
            ax.plot(xs, ys, markers[s], label=s)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of concepts (N)")
    ax.set_ylabel(f"time for 1 epoch ({N_SAMPLES} samples, query only) (s)")
    ax.set_title("Concept-module scalability: plate addressing modes")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_png = os.path.join(OUTDIR, "concept_module_scalability.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved plot to {out_png}")


if __name__ == "__main__":
    main()
