"""Verification of the vectorised BeliefPropagation refactor.

For each of three bnlearn networks (``asia``, ``sachs``, ``insurance``) this
script builds the *same* :class:`BayesianNetwork` twice — one CPD per node,
wired along the true DAG — and trains the two copies on the same data with the
same optimiser and the same seed, differing only in the inference engine:

* ``legacy``     — the previous per-edge-dict belief propagation
  (:mod:`legacy_belief_propagation`, a verbatim snapshot);
* ``vectorised`` — the current :class:`~torch_concepts.nn.BeliefPropagation`,
  with flat ``[E, K]`` message tensors, signature-bucketed factors and
  bias-absorbed unary factors.

The training query is ``p(all concepts)``: BP is run with **no evidence**, so
``out`` holds one marginal per node, and the loss is the cross-entropy of those
marginals against every row of the dataset. Its minimiser is exactly the
*empirical* marginal of each node, which is what the accuracy table checks.

Two things are being verified:

1. **Equivalence** — the two engines must learn the same probabilities (they
   are the same algorithm; damping is off, so the arithmetic differs only in
   the LOG0 clamp on padded states).
2. **Speed** — the point of the refactor. ``bp_training_time.png`` plots the
   wall-clock training time of the two engines side by side.

Note that the fitted marginals need not match the empirical ones *exactly*:
the moralised DAGs here all contain cycles, so BP returns approximate
marginals and the optimum of the loss is only as good as that approximation.

Run with::

    python examples/experiments/bp_benchmark/bp_refactor_check.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, OneHotCategorical

from torch_concepts import seed_everything, ConceptVariable
from torch_concepts.data import BnLearnDataset
from torch_concepts.nn import (
    BayesianNetwork,
    BeliefPropagation,
    LearnablePrior,
    ParametricCPD,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from legacy_belief_propagation import BeliefPropagation as LegacyBeliefPropagation  # noqa: E402

DATASETS = ("asia", "sachs", "insurance")
ENGINES = {"legacy": LegacyBeliefPropagation, "vectorised": BeliefPropagation}

N_SAMPLES = 2000
N_EPOCHS = 200
LR = 0.1
BP_ITERS = 15
SEED = 42
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bp_training_time.png")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_network(name: str) -> Tuple[torch.Tensor, List[str], List[int], Dict[str, List[str]]]:
    """``(labels, names, n_states, parents)`` for one bnlearn network.

    ``labels`` is ``(N, n_nodes)`` of integer states (one column per node, in
    ``names`` order) and ``n_states[i]`` is node ``i``'s cardinality. The
    dataset stores a binary node's cardinality as ``1`` (a Bernoulli width),
    which is widened back to its ``2`` states here.
    """
    dataset = BnLearnDataset(name=name, seed=SEED, n_gen=N_SAMPLES)
    names = list(dataset.graph.node_names)
    assert list(dataset.annotations.labels) == names, "column order must follow the DAG"
    labels = dataset.concepts.as_subclass(torch.Tensor).long()
    n_states = [2 if c == 1 else int(c) for c in dataset.annotations.cardinalities]
    parents = {n: list(dataset.graph.get_predecessors(n)) for n in names}
    return labels, names, n_states, parents


def empirical_marginals(
    labels: torch.Tensor, names: List[str], n_states: List[int]
) -> Dict[str, torch.Tensor]:
    """Per-node state frequencies in the data — the target the loss is minimised at."""
    return {
        name: torch.bincount(labels[:, i], minlength=k).float() / labels.shape[0]
        for i, (name, k) in enumerate(zip(names, n_states))
    }


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_network(
    names: List[str], n_states: List[int], parents: Dict[str, List[str]]
) -> BayesianNetwork:
    """One :class:`ParametricCPD` per node, wired along the true DAG.

    A 2-state node is a width-1 ``Bernoulli``; a ``k``-state node is a width-``k``
    ``OneHotCategorical``. Roots get a :class:`LearnablePrior`; every other node
    maps its parents' (concatenated) values to logits with a linear layer, which
    is enough to represent any CPD of binary parents and is a smooth
    approximation otherwise.
    """
    variables = {
        name: ConceptVariable(
            name,
            distribution=Bernoulli if k == 2 else OneHotCategorical,
            size=1 if k == 2 else k,
        )
        for name, k in zip(names, n_states)
    }
    factors = []
    for name in names:
        pa = [variables[p] for p in parents[name]]
        width = variables[name].size
        parametrization = (
            {"logits": LearnablePrior(width)} if not pa
            else {"logits": nn.Linear(sum(p.size for p in pa), width)}
        )
        factors.append(ParametricCPD(variables[name], parametrization, parents=pa))
    return BayesianNetwork(variables=list(variables.values()), factors=factors)


def marginal_loss(
    out, labels: torch.Tensor, names: List[str], n_states: List[int]
) -> torch.Tensor:
    """Cross-entropy of the queried marginals against every row of the dataset.

    With no evidence the query returns one marginal per node, shaped
    ``(1, width)`` — the same distribution for every row — so each term is the
    dataset's mean per-sample cross-entropy for that node, minimised exactly at
    the node's empirical marginal.
    """
    n = labels.shape[0]
    total = labels.new_zeros((), dtype=torch.get_default_dtype())
    for i, (name, k) in enumerate(zip(names, n_states)):
        probs = out.probs[name]
        if k == 2:
            total = total + F.binary_cross_entropy(
                probs.expand(n, 1), labels[:, i : i + 1].float()
            )
        else:
            total = total + F.nll_loss(probs.expand(n, k).log(), labels[:, i])
    return total


def learned_marginals(out, names: List[str], n_states: List[int]) -> Dict[str, torch.Tensor]:
    """``out.probs`` widened back to a per-node state distribution.

    A binary node reports the Bernoulli ``P(x=1)`` of width 1 (the uniform
    engine contract), so it is expanded to ``[P(0), P(1)]`` before it can be
    compared with an empirical frequency vector.
    """
    marginals = {}
    for name, k in zip(names, n_states):
        p = out.probs[name].detach().reshape(-1)
        marginals[name] = torch.cat([1.0 - p, p]) if k == 2 else p
    return marginals


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(
    engine_name: str,
    labels: torch.Tensor,
    names: List[str],
    n_states: List[int],
    parents: Dict[str, List[str]],
) -> Tuple[Dict[str, torch.Tensor], float, float]:
    """Train one model end to end; return ``(marginals, seconds, final loss)``.

    The seed is reset first, so both engines start from bit-identical weights
    and any difference in the learned marginals is the engines', not the
    initialisation's.
    """
    seed_everything(SEED)
    pgm = build_network(names, n_states, parents)
    engine = ENGINES[engine_name](pgm, iters=BP_ITERS)
    optimizer = torch.optim.Adam(pgm.parameters(), lr=LR)

    start = time.perf_counter()
    loss = torch.zeros(())
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        out = engine.query(query=names, evidence={})
        loss = marginal_loss(out, labels, names, n_states)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"    [{engine_name:>10}] epoch {epoch:>4}  loss {loss.item():.4f}")
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        out = engine.query(query=names, evidence={})
    return learned_marginals(out, names, n_states), elapsed, loss.item()


def max_abs_error(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    return max((a[k] - b[k]).abs().max().item() for k in a)


def mean_abs_error(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    diffs = torch.cat([(a[k] - b[k]).abs().reshape(-1) for k in a])
    return diffs.mean().item()


# --------------------------------------------------------------------------
def main() -> None:
    logging.disable(logging.INFO)
    torch.set_num_threads(1)  # so the timings compare arithmetic, not thread luck

    results: Dict[str, Dict[str, float]] = {}
    for name in DATASETS:
        print(f"\n=== {name} ===")
        labels, names, n_states, parents = load_network(name)
        empirical = empirical_marginals(labels, names, n_states)
        print(f"  {len(names)} nodes, cardinalities {sorted(set(n_states))}, "
              f"{labels.shape[0]} samples")

        trained = {}
        for engine_name in ENGINES:
            marginals, seconds, final_loss = train(
                engine_name, labels, names, n_states, parents
            )
            trained[engine_name] = marginals
            results.setdefault(name, {})[f"{engine_name}_time"] = seconds
            results[name][f"{engine_name}_max_err"] = max_abs_error(marginals, empirical)
            results[name][f"{engine_name}_mean_err"] = mean_abs_error(marginals, empirical)
            results[name][f"{engine_name}_loss"] = final_loss
        results[name]["engine_gap"] = max_abs_error(
            trained["legacy"], trained["vectorised"]
        )

    print_report(results)
    plot_times(results)


def print_report(results: Dict[str, Dict[str, float]]) -> None:
    print("\n\n== learned vs empirical marginals (max / mean abs error) ==")
    print(f"{'dataset':>11} | {'legacy':>17} | {'vectorised':>17} | {'engine gap':>10}")
    for name, r in results.items():
        print(
            f"{name:>11} | "
            f"{r['legacy_max_err']:.4f} / {r['legacy_mean_err']:.4f} | "
            f"{r['vectorised_max_err']:.4f} / {r['vectorised_mean_err']:.4f} | "
            f"{r['engine_gap']:>10.2e}"
        )

    print("\n== training wall-clock ==")
    print(f"{'dataset':>11} | {'legacy':>9} | {'vectorised':>11} | {'speedup':>8}")
    for name, r in results.items():
        speedup = r["legacy_time"] / r["vectorised_time"]
        print(
            f"{name:>11} | {r['legacy_time']:>8.2f}s | "
            f"{r['vectorised_time']:>10.2f}s | {speedup:>7.1f}x"
        )


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
# Light-mode tokens from the data-viz reference palette: categorical slots 1
# and 2 (the first three slots are validated all-pairs, worst CVD dE 9.2) on the
# light chart surface, with chrome/ink in text tokens rather than series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"legacy": "#2a78d6", "vectorised": "#eb6834"}

BAR_PX = 22.0      # <= 24px: never fill the band, leave the rest as air
GAP_PX = 2.0       # surface gap between the two touching bars of a group
RADIUS_PX = 4.0    # rounded data-end, square at the baseline


def _px_to_data(ax, dx: float, dy: float) -> Tuple[float, float]:
    """Convert a pixel offset to data units (axes limits must already be set)."""
    inv = ax.transData.inverted().transform
    x0, y0 = inv((0.0, 0.0))
    return abs(inv((dx, 0.0))[0] - x0), abs(inv((0.0, dy))[1] - y0)


def _rounded_bar(ax, value: float, y: float, height: float, color: str) -> None:
    """A horizontal bar whose *data* end is rounded and whose baseline end is square."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    rx, ry = _px_to_data(ax, RADIUS_PX, RADIUS_PX)
    rx, ry = min(rx, value / 2), min(ry, height / 2)
    lo, hi = y - height / 2, y + height / 2
    verts = [
        (0, lo), (value - rx, lo), (value, lo), (value, lo + ry),
        (value, hi - ry), (value, hi), (value - rx, hi), (0, hi), (0, lo),
    ]
    codes = [
        Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
        Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none"))


def plot_times(results: Dict[str, Dict[str, float]]) -> None:
    """Grouped bars of training wall-clock, one group per network.

    Magnitude comparison across two engines, so: bars from a shared zero
    baseline, one fixed hue per engine, seconds labelled at each tip, and the
    speed-up as its own right-hand column (the actual headline).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    slowest = max(r["legacy_time"] for r in results.values())
    ax.set_xlim(0, slowest * 1.16)          # headroom for the tip labels
    ax.set_ylim(len(names) - 0.5, -0.5)     # first network on top
    # The right margin is figure space, not axis space: the speed-up column sits
    # outside the plot so no gridline runs through it.
    fig.subplots_adjust(left=0.12, right=0.86, top=0.76, bottom=0.18)

    # Limits are final, so pixel specs can now be converted exactly.
    bar_h = _px_to_data(ax, 0.0, BAR_PX)[1]
    gap = _px_to_data(ax, 0.0, GAP_PX)[1]

    for row, name in enumerate(names):
        for sign, engine in ((-1, "legacy"), (1, "vectorised")):
            seconds = results[name][f"{engine}_time"]
            y = row + sign * (bar_h + gap) / 2
            _rounded_bar(ax, seconds, y, bar_h, SERIES[engine])
            ax.text(
                seconds + slowest * 0.012, y, f"{seconds:.1f}s",
                va="center", ha="left", fontsize=8, color=INK_SECONDARY,
            )
        ax.text(
            1.15, row, f"{results[name]['legacy_time'] / results[name]['vectorised_time']:.1f}×",
            transform=ax.get_yaxis_transform(), va="center", ha="right",
            fontsize=11, color=INK,
        )

    ax.text(
        1.15, -0.62, "speed-up", transform=ax.get_yaxis_transform(),
        va="center", ha="right", fontsize=8, color=INK_MUTED,
    )

    ax.set_yticks(range(len(names)), names, fontsize=10, color=INK)
    ax.tick_params(axis="x", length=0, colors=INK_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.set_xlabel("training wall-clock (seconds, lower is better)",
                  fontsize=8, color=INK_MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["left"].set_linewidth(0.8)

    fig.text(0.013, 0.94, "Vectorised belief propagation cuts training time",
             fontsize=12, color=INK, weight="semibold")
    fig.text(0.013, 0.875,
             f"{N_EPOCHS} epochs of p(all concepts) per network, {BP_ITERS} BP "
             "iterations, identical models and seed",
             fontsize=8, color=INK_SECONDARY)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=6,
                   markerfacecolor=SERIES[e], markeredgecolor="none", label=e)
        for e in ENGINES
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              ncol=2, frameon=False, fontsize=8, labelcolor=INK_SECONDARY,
              handletextpad=0.4, columnspacing=1.4)

    fig.savefig(OUT_PNG, facecolor=SURFACE)
    print(f"\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
