"""Pyro-specific distribution utilities for the Pyro inference backend.

Provides helpers to extract named parameter dicts from Pyro distributions and
traces, for use by :class:`PyroBaseInference` and related engines.

Entry points:
- :data:`_PARAM_NAMES` — canonical param names per distribution family.
- :func:`_peel` — strip ``Independent``/masked/expanded wrappers.
- :func:`dist_to_params` — convert a Pyro distribution to a param dict.
- :func:`trace_to_params` — harvest param dicts from all sites in a trace.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pyro.distributions as pyro_dist
import torch.distributions as td

from ..outputs import ParamDict


# Canonical parameter names emitted in InferenceOutput.params /
# InferenceOutput.guide_params, keyed by distribution family.
#
# We key on ``torch.distributions`` classes (not pyro's), because the Pyro
# subclasses inherit from them — so ``isinstance(d, td.Normal)`` matches both
# a plain ``torch`` distribution and a ``pyro`` one. ``build_distribution``
# returns plain ``torch.distributions`` instances; the relaxed STE classes
# only exist in Pyro and are added as explicit extra entries below.
#
# Delta is deliberately omitted: it is deterministic and carries no
# learnable params worth surfacing through ``InferenceOutput.params``.
_PARAM_NAMES: Dict[type, Tuple[str, ...]] = {
    td.Bernoulli: ("probs",),
    td.Categorical: ("probs",),
    td.OneHotCategorical: ("probs",),
    td.Normal: ("loc", "scale"),
    td.MultivariateNormal: ("loc", "scale_tril"),
    pyro_dist.RelaxedBernoulliStraightThrough: ("probs", "temperature"),
    pyro_dist.RelaxedOneHotCategoricalStraightThrough: ("probs", "temperature"),
}


def _peel(d: pyro_dist.Distribution) -> pyro_dist.Distribution:
    """Strip ``Independent`` / Masked / Expanded wrappers off a distribution.

    Pyro wraps distributions in the following way:
        - ``Independent(base, reinterpreted_batch_ndims)``: declares batch
          dimension as independent events.
        - ``MaskedDistribution(base, mask)``: masks out some batch dims
          (e.g. to avoid log-prob computation).
        - ``ExpandedDistribution(base, batch_shape)``: adds batch dim.

    Tested with ``torch.distributions.Independent`` (the Pyro subclass
    inherits from it, so plain torch instances match too).
    """
    while True:
        if isinstance(d, td.Independent):
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


def dist_to_params(d: pyro_dist.Distribution) -> ParamDict:
    """Return the canonical named-parameter dict of a Pyro distribution
    (e.g. ``{'probs': ...}`` or ``{'loc': ..., 'scale': ...}``), peeling
    ``Independent`` / masked / expanded wrappers first.
    """
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


def trace_to_params(trace) -> Dict[str, ParamDict]:
    """Use a Pyro trace to collect ``dist_to_params`` for every stochastic
    (non-deterministic) sample site, keyed by site name.

    The ``trace`` argument accepts a ``poutine.Trace`` node dict
    (``trace.nodes``) or anything with the same structure.
    """
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
