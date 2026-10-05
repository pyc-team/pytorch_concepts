"""Pyro-specific distribution utilities for the Pyro inference backend.

Provides the Pyro-compatible distribution builder for ``pyro.sample`` sites,
and helpers to extract named parameter dicts from Pyro distributions and
traces, for use by :class:`PyroBaseInference` and related engines.

Entry points:
- :func:`build_relaxed_pyro_distribution` — relaxed distribution for an
  unobserved ``pyro.sample`` site.
- :data:`_PARAM_NAMES` — canonical param names per distribution family.
- :func:`_peel` — strip ``Independent``/masked/expanded wrappers.
- :func:`dist_to_params` — convert a Pyro distribution to a param dict.
- :func:`trace_to_params` — harvest param dicts from all sites in a trace.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.distributions as td

from ...distributions import spec_for
from ...variable import Variable
from ....outputs import ParamDict


def build_relaxed_pyro_distribution(
    variable: Variable,
    params: Dict[str, torch.Tensor],
    temperature: torch.Tensor,
) -> pyro_dist.Distribution:
    """Build a Pyro-compatible relaxed distribution for ``pyro.sample`` sites.

    The Pyro counterpart of
    :func:`~torch_concepts.nn.modules.mid.inference.torch.utils.build_relaxed_distribution`.
    It cannot simply reuse that one: an **unobserved** ``pyro.sample`` site
    needs a ``pyro.distributions`` instance (subclass of ``TorchDistribution``),
    while the registry's relaxed factories return plain ``torch.distributions``
    objects for the soft families — and those are not callable, so
    ``pyro.sample`` raises ``TypeError: 'X' object is not callable``. (An
    *observed* site accepts a plain torch distribution, which is why
    ``model_fn`` builds those with ``build_distribution``.)

    Soft or hard is decided by the **declared family**. A variable declared
    ``Bernoulli`` / ``RelaxedBernoulli`` gets the plain relaxed (Concrete)
    distribution, so the sampled value stays soft — what a descendant that
    *mixes* by that value needs, since a hard draw zeroes the gradient to
    every state it did not select. Declaring
    ``RelaxedBernoulliStraightThrough`` instead selects Pyro's own
    straight-through estimator, which yields an exact bit / one-hot row and
    registers correctly with Pyro's effect-handler stack.

    Raises
    ------
    ValueError
        If the family has no relaxed counterpart (a plain ``Categorical``), with
        the registry's reason — the same error the torch backend raises. Such a
        variable can only ever be an observed site.
    """
    # Reached only during inference, after a Pyro engine was constructed, so
    # Pyro is guaranteed importable here.
    import pyro.distributions as pyro_dist

    # Parameters arrive in the member layout (*batch, n_members,
    # *member_shape). Reinterpreting the member axis and the member's own
    # event as the event leaves batch_shape == (*batch,), which is what the
    # ``pyro.plate("batch", ...)`` dim lines up with. ``event_ndims`` is
    # subtracted because a family like OneHotCategorical already claims its
    # trailing class axis.
    D = variable.distribution
    spec = spec_for(D, f"Variable {variable.name!r}")
    n_event = 1 + len(variable.member_shape) - spec.event_ndims
    params = {
        key: variable.to_member(value, key) for key, value in params.items()
    }
    # A straight-through class is a *subclass* of its plain relaxed base, so
    # it must be tested first or it would fall through to the soft branch.
    if issubclass(D, pyro_dist.RelaxedBernoulliStraightThrough):
        d = pyro_dist.RelaxedBernoulliStraightThrough(
            temperature=temperature, **params)
    elif issubclass(D, pyro_dist.RelaxedOneHotCategoricalStraightThrough):
        d = pyro_dist.RelaxedOneHotCategoricalStraightThrough(
            temperature=temperature, **params)
    elif issubclass(D, (td.Bernoulli, td.RelaxedBernoulli)):
        d = pyro_dist.RelaxedBernoulli(temperature=temperature, **params)
    elif issubclass(D, (td.OneHotCategorical, td.RelaxedOneHotCategorical)):
        d = pyro_dist.RelaxedOneHotCategorical(temperature=temperature, **params)
    elif issubclass(D, td.Normal):
        d = pyro_dist.Normal(**params)
    elif issubclass(D, td.MultivariateNormal):
        d = pyro_dist.MultivariateNormal(**params)
    elif D.__name__ == "Delta":
        # Map ``value`` (our Delta convention) to ``v`` (Pyro's).
        return pyro_dist.Delta(params["value"], event_dim=n_event)
    else:
        # No Pyro-samplable relaxation. Returning the exact torch distribution
        # here (as this used to) only moved the failure into ``pyro.sample``,
        # as an opaque "object is not callable".
        reason = spec.no_relaxed_reason or (
            f"{D.__name__} has no relaxed counterpart that Pyro can sample."
        )
        raise ValueError(f"Variable {variable.name!r}: {reason}")
    return d.to_event(n_event)


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

# Relaxed surrogates that also carry a ``temperature`` parameter. Resolved
# lazily so importing this module does not require Pyro to be installed.
def _relaxed_discrete_families() -> Tuple[type, ...]:
    # Reached only during inference, after a Pyro engine was constructed, so
    # Pyro is guaranteed importable here.
    #
    # The base (non-straight-through) classes on purpose: the ``*StraightThrough``
    # variants subclass them, so this covers a variable declared either way.
    # Listing only the straight-through ones would let a soft concept site fall
    # through to the generic branch and report no ``probs`` at all — which
    # surfaces far downstream as a metric or concept loss that cannot find its
    # parameter.
    import pyro.distributions as pyro_dist
    return (
        pyro_dist.RelaxedBernoulli,
        pyro_dist.RelaxedOneHotCategorical,
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
    ``build_relaxed_pyro_distribution``) the key is always ``'probs'`` because
    Pyro's ``LogitRelaxedBernoulli`` always stores logits internally and
    Pyro reconstructs distribution objects during tracing (losing any
    construction-time tag). Callers that need the user's original key should
    post-process using the CPD's ``parametrization.keys()``; see
    :meth:`VariationalInference._align_param_keys`.
    """
    base = _peel(d)

    # Relaxed discrete (STE): extract probs. The internal representation is
    # always logits (LogitRelaxedBernoulli), but .probs is available as a
    # property (sigmoid of stored logits). The relaxation ``temperature`` is
    # deliberately not reported: it is a scalar knob of the engine (readable as
    # ``engine.temperature``), not a per-column distribution parameter, so it has
    # no place on the annotated event axis of the output.
    if isinstance(base, _relaxed_discrete_families()):
        return {"probs": base.probs}

    # Plain discrete: probs or logits, detected via _param.
    if isinstance(base, _DISCRETE_FAMILIES):
        key = _discrete_prob_key(base)
        return {key: getattr(base, key)}

    # Delta: a deterministic node (an embedding, a concept bottleneck context).
    # Latent sites carry Pyro's Delta and observed ones PyC's, which name the
    # point mass differently (``v`` / ``_value``) but agree on ``mean``.
    if type(base).__name__ == "Delta":
        return {"value": base.mean}

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
