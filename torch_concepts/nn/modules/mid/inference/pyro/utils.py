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
- :func:`trace_to_params` — harvest each traced variable's params, under its
  CPD's parameter names.
"""
from __future__ import annotations

from typing import Dict, Tuple

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


# Parameter names reported per family, as ``torch.distributions`` names them.
#
# A discrete family reports both ``probs`` and ``logits``, each read off the
# distribution: a traced distribution no longer says which one it was built
# from, so :func:`trace_to_params` keeps the one its CPD was written with.
# The relaxed entries also match Pyro's relaxed families and their
# straight-through variants, which subclass them. A relaxed site's
# ``temperature`` is not reported: it is an engine knob
# (``engine.temperature``), not a per-column parameter.
_PROBS_AND_LOGITS = ("probs", "logits")
_PARAM_NAMES: Dict[type, Tuple[str, ...]] = {
    td.Normal: ("loc", "scale"),
    td.MultivariateNormal: ("loc", "scale_tril"),
    td.Bernoulli: _PROBS_AND_LOGITS,
    td.Categorical: _PROBS_AND_LOGITS,
    td.OneHotCategorical: _PROBS_AND_LOGITS,
    td.RelaxedBernoulli: _PROBS_AND_LOGITS,
    td.RelaxedOneHotCategorical: _PROBS_AND_LOGITS,
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
    """The named parameters of a Pyro distribution, wrappers peeled off:
    ``{'loc': ..., 'scale': ...}``, ``{'value': ...}``, and for a discrete family
    both ``{'probs': ..., 'logits': ...}`` (see :data:`_PARAM_NAMES`). Empty for
    a family it does not know.
    """
    base = _peel(d)

    # Delta: a deterministic node (an embedding, a concept bottleneck context).
    # Latent sites carry Pyro's Delta and observed ones PyC's, which name the
    # point mass differently (``v`` / ``_value``) but agree on ``mean``.
    if type(base).__name__ == "Delta":
        return {"value": base.mean}

    for family, names in _PARAM_NAMES.items():
        if isinstance(base, family):
            return {name: getattr(base, name) for name in names}
    return {}


def trace_to_params(trace, cpds) -> Dict[str, ParamDict]:
    """Each traced variable's parameters, under the names its CPD uses.

    ``cpds`` maps site names to the CPDs that produced them (``pgm.factors``
    for the model, ``pgm.guides`` for the guide). A site with no CPD there —
    the batch plate, another engine's guide — is skipped. A CPD's
    ``parametrization`` keys are its family's parameter names, so filtering by
    them is what picks ``probs`` or ``logits`` for a discrete site.
    """
    return {
        name: {
            key: value
            for key, value in dist_to_params(node["fn"]).items()
            if key in cpds[name].parametrization
        }
        for name, node in trace.nodes.items()
        if node["type"] == "sample" and name in cpds
    }
