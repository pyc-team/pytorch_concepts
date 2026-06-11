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
# Discrete families are handled separately by ``_discrete_prob_key`` because
# they accept either ``probs`` or ``logits`` and we want to preserve whichever
# key was actually used at construction time.
_PARAM_NAMES: Dict[type, Tuple[str, ...]] = {
    td.Normal: ("loc", "scale"),
    td.MultivariateNormal: ("loc", "scale_tril"),
}

# Families whose primary parameter is either ``probs`` or ``logits``.
_DISCRETE_FAMILIES: Tuple[type, ...] = (
    td.Bernoulli, td.Categorical, td.OneHotCategorical,
)

# Relaxed surrogates that also carry a ``temperature`` parameter.
_RELAXED_DISCRETE_FAMILIES: Tuple[type, ...] = (
    pyro_dist.RelaxedBernoulliStraightThrough,
    pyro_dist.RelaxedOneHotCategoricalStraightThrough,
)


def _discrete_prob_key(d) -> str:
    """Return ``'probs'`` or ``'logits'`` reflecting how *d* was constructed.

    Checks ``_param`` to determine the original parametrization of plain
    discrete distributions (``td.Bernoulli``, ``td.Categorical``,
    ``td.OneHotCategorical``). Works because ``torch.distributions`` stores
    the directly-passed tensor in ``_param``.
    """
    source = d if hasattr(d, "_param") else getattr(d, "base_dist", d)
    param = getattr(source, "_param", None)
    if param is None:
        return "probs"  # safe fallback
    probs_attr = getattr(source, "probs", None)
    return "probs" if (probs_attr is not None and param is probs_attr) else "logits"


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
    (e.g. ``{'probs': ...}`` or ``{'logits': ...}`` or ``{'loc': ..., 'scale': ...}``),
    peeling ``Independent`` / masked / expanded wrappers first.

    For **plain discrete** families (observed sites, created by
    ``build_distribution``) the returned key (``'probs'`` or ``'logits'``)
    reflects whichever parametrization was used at construction time.

    For **relaxed discrete** families (latent/guide sites, created by
    ``_pyro_relaxed_distribution``) the key is always ``'probs'`` because
    Pyro's ``LogitRelaxedBernoulli`` always stores logits internally and
    Pyro reconstructs distribution objects during tracing (losing any
    construction-time tag). Callers that need the user's original key should
    post-process using the CPD's ``parametrization.keys()``; see
    :meth:`VariationalInference._align_param_keys`.
    """
    base = _peel(d)

    # Relaxed discrete (STE): always extract probs; temperature too.
    # The internal representation is always logits (LogitRelaxedBernoulli),
    # but .probs is available as a property (sigmoid of stored logits).
    if isinstance(base, _RELAXED_DISCRETE_FAMILIES):
        return {"probs": base.probs, "temperature": base.temperature}

    # Plain discrete: probs or logits, detected via _param.
    if isinstance(base, _DISCRETE_FAMILIES):
        key = _discrete_prob_key(base)
        return {key: getattr(base, key)}

    # All other families: fixed param names.
    names: Optional[Tuple[str, ...]] = None
    for k, v in _PARAM_NAMES.items():
        if isinstance(base, k):
            names = v
            break
    if names is None:
        return {}
    return {n: getattr(base, n) for n in names}


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
