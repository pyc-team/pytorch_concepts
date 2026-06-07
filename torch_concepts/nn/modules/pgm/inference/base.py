"""Shared scaffolding for inference engines."""
from __future__ import annotations

import inspect
import math
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import pyro.distributions as dist
import pyro.poutine as poutine
import torch
from pyro.nn import PyroModule

from ..models.bayesian_network import BayesianNetwork
from .outputs import InferenceOutput, ParamDict


# Canonical parameter names emitted in InferenceOutput.params /
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
    """Strip Independent/Masked/Expanded wrappers.
    Pyro wraps distributions in the following way:
        - Independent(base, reinterpreted_batch_ndims): declares batch dimension as independent events.
        - MaskedDistribution(base, mask): masks out some batch dims (e.g., to avoid log prob computation)
        - ExpandedDistribution(base, batch_shape): adds batch dim.
    """
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
    """Use the Pyro trace to collect ``dist_to_params`` for every stochastic
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
class BaseInference(PyroModule):
    """Base class for all inference engines.

    ``query`` and ``evidence`` are attribute-style containers keyed by PGM
    variable name. ``__call__`` delegates to ``query``.
    """

    name: str = "BaseInference"

    def __init__(self, pgm: BayesianNetwork):
        super().__init__()
        self.pgm = pgm

        roots_needing_input: List[str] = [
            v.name
            for v in pgm.variables
            if pgm.factors[v.name].is_root
            and len(
                inspect.signature(
                    pgm.factors[v.name].parametrization.forward
                ).parameters
            ) > 0
        ]
        if roots_needing_input:
            warnings.warn(
                "\033[33m"
                f"{self.name}: the following root variables have a parametrization "
                f"that requires input arguments: {roots_needing_input}. "
                "These must be supplied as constant evidence on every query call."
                "\033[0m",
                UserWarning,
                stacklevel=2,
            )

    def _validate_containers(
        self,
        query: Dict[str, Optional[torch.Tensor]],
        evidence: Dict[str, torch.Tensor],
    ) -> None:
        all_names = set(self.pgm.name_to_variable.keys())
        unknown_q = set(query.keys()) - all_names
        if unknown_q:
            raise ValueError(f"{self.name}: unknown query names {sorted(unknown_q)}.")
        unknown_e = set(evidence.keys()) - all_names
        if unknown_e:
            raise ValueError(f"{self.name}: unknown evidence names {sorted(unknown_e)}.")

        for name, val in evidence.items():
            if not isinstance(val, torch.Tensor):
                raise ValueError(
                    f"{self.name}: evidence[{name!r}] must be a Tensor, "
                    f"got {type(val).__name__}."
                )

        if not query and not evidence:
            raise ValueError("nothing to do")

        all_tensors = {name: val for name, val in query.items() if val is not None}
        all_tensors.update(evidence)
        batch_sizes = {name: t.shape[0] for name, t in all_tensors.items()}
        if len(set(batch_sizes.values())) > 1:
            shapes = {name: tuple(t.shape) for name, t in all_tensors.items()}
            raise ValueError(f"{self.name}: mismatched batch sizes {shapes}.")

    @staticmethod
    def _normalize_query(
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        if isinstance(query, list):
            return {name: None for name in query}
        return query

    def __call__(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        return self.query(query=query, evidence=evidence)
