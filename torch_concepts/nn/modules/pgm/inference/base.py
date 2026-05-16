"""Shared scaffolding for inference engines."""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.nn import PyroModule

from ..models.bayesian_network import BayesianNetwork
from ..models.variable import ExogenousVariable
from .result import InferenceOutput, ParamDict


# Canonical parameter names emitted in InferenceOutput.model_params /
# InferenceOutput.guide_params, keyed by distribution family.
_PARAM_NAMES: Dict[type, Tuple[str, ...]] = {
    dist.Bernoulli: ("probs",),
    dist.Categorical: ("probs",),
    dist.OneHotCategorical: ("probs",),
    dist.Normal: ("loc", "scale"),
    dist.MultivariateNormal: ("loc", "scale_tril"),
    dist.Delta: ("v",),
    dist.RelaxedBernoulliStraightThrough: ("probs", "temperature"),
    dist.RelaxedOneHotCategoricalStraightThrough: ("probs", "temperature"),
}


def _peel(d: dist.Distribution) -> dist.Distribution:
    """Strip Independent/Masked/Expanded wrappers."""
    while True:
        if isinstance(d, dist.Independent):
            d = d.base_dist
            continue
        base = getattr(d, "base_dist", None)
        if base is not None and type(d).__name__ in (
            "MaskedDistribution",
            "ExpandedDistribution",
        ):
            d = base
            continue
        return d


def dist_to_params(d: dist.Distribution) -> ParamDict:
    """Return the canonical named-parameter dict of a distribution
    (e.g. ``{'probs': ...}`` or ``{'loc': ..., 'scale': ...}``), peeling
    ``Independent`` / masked / expanded wrappers first."""
    base = _peel(d)
    names: Optional[Tuple[str, ...]] = None
    for k, v in _PARAM_NAMES.items():
        if isinstance(base, k):
            names = v
            break
    if names is None:
        return {}
    out: ParamDict = {}
    for n in names:
        out[n] = getattr(base, n)
    return out


def trace_to_params(trace: poutine.Trace) -> Dict[str, ParamDict]:
    """Walk a Pyro trace and collect ``dist_to_params`` for every stochastic
    (non-deterministic) sample site, keyed by site name."""
    out: Dict[str, ParamDict] = {}
    for name, node in trace.nodes.items():
        if node["type"] != "sample":
            continue
        if node.get("infer", {}).get("_deterministic", False):
            continue
        pd_ = dist_to_params(node["fn"])
        if pd_:
            out[name] = pd_
    return out


def make_temperature_schedule(
    initial_temperature: float,
    annealing: Union[str, Callable[[int], float]],
    annealing_rate: float,
) -> Callable[[int], float]:
    """Build a ``step -> temperature`` schedule.

    ``annealing`` may be ``'constant'``, ``'exponential'`` (decays as
    ``T0 * exp(-rate * step)``), ``'linear'`` (decays as
    ``max(eps, T0 - rate * step)``), or a user-supplied callable.
    """
    if callable(annealing):
        return annealing
    if annealing == "constant":
        return lambda step: float(initial_temperature)
    if annealing == "exponential":
        return lambda step: float(initial_temperature) * math.exp(-annealing_rate * step)
    if annealing == "linear":
        return lambda step: max(
            1e-6, float(initial_temperature) - annealing_rate * step
        )
    raise ValueError(
        f"Unknown annealing schedule {annealing!r}. Use "
        "'constant', 'exponential', 'linear', or pass a callable."
    )


# -----------------------------------------------------------------------------
class InferenceEngine(PyroModule):
    """Base class for all inference engines.

    Calling ``engine(query, evidence, data)`` validates that ``query`` and
    ``evidence`` reference known variables, that ``evidence`` is a subset of
    ``data.keys()``, and that every ``ExogenousVariable`` of the PGM is
    listed in ``evidence``; it then dispatches to the subclass-specific
    ``_run`` and returns an ``InferenceOutput``.
    """

    name: str = "InferenceEngine"

    def __init__(self, pgm: BayesianNetwork):
        super().__init__()
        self.pgm = pgm

    # ------------------------------------------------------------------
    def _validate(
        self,
        query: List[str],
        evidence: List[str],
        data: Dict[str, torch.Tensor],
    ) -> None:
        all_names = set(self.pgm.name_to_variable.keys())
        unknown_q = set(query) - all_names
        if unknown_q:
            raise ValueError(f"{self.name}: unknown query names {sorted(unknown_q)}.")
        unknown_e = set(evidence) - all_names
        if unknown_e:
            raise ValueError(f"{self.name}: unknown evidence names {sorted(unknown_e)}.")
        unknown_d = set(data.keys()) - all_names
        if unknown_d:
            raise ValueError(f"{self.name}: unknown data keys {sorted(unknown_d)}.")

        missing = set(evidence) - set(data.keys())
        if missing:
            raise ValueError(
                f"{self.name}: every evidence variable must appear in `data`. "
                f"Missing: {sorted(missing)}."
            )

        for v in self.pgm.variables:
            if isinstance(v, ExogenousVariable) and v.name not in evidence:
                raise ValueError(
                    f"{self.name}: ExogenousVariable {v.name!r} must appear "
                    "in `evidence` (and therefore in `data`)."
                )

    # ------------------------------------------------------------------
    def __call__(
        self,
        query: List[str],
        evidence: List[str],
        data: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        self._validate(list(query), list(evidence), data)
        return self._run(list(query), list(evidence), data)

    def _run(
        self,
        query: List[str],
        evidence: List[str],
        data: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        raise NotImplementedError
