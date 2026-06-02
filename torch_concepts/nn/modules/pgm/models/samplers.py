"""Shared distribution / sampler dispatch.

Centralises the per-family logic used by both ``BayesianNetwork.forward``
(Pyro stochastic function) and ``ForwardInference`` (non-Pyro forward pass).

Four entry points are exposed:

- :func:`build_distribution` — build the *exact* Pyro distribution declared
  by the variable (used when we have a true ``obs`` to score against).
- :func:`build_relaxed_distribution` — build a reparameterisable surrogate
  for the same variable: straight-through relaxations for discrete families,
  the exact distribution for already-reparameterised families. Used when the
  site is unobserved so gradients can flow through the sample.
- :func:`propagated_value` — return the canonical "deterministic" value
  (mean/mode-like) for a parameter dict.
- :func:`sample_from` — reparameterised sample via the same relaxation logic
  but without going through Pyro (used by :class:`ForwardInference`).
"""
from __future__ import annotations

from typing import Dict

import pyro.distributions as dist
import torch

from .variable import Variable


# ---------------------------------------------------------------------------
# Primary parameter name per family — used by ``DeterministicInference`` 
# to propagate the param of the distributions instead of sampling from it
# (mean for Normal/MVN, probs for the discrete families, v for Delta).
# ---------------------------------------------------------------------------
_PRIMARY_PARAM: Dict[type, str] = {
    dist.Bernoulli: "probs",
    dist.OneHotCategorical: "probs",
    dist.Categorical: "probs",
    dist.Normal: "loc",
    dist.MultivariateNormal: "loc",
    dist.Delta: "v",
}


# ---------------------------------------------------------------------------
# Exact distributions (used when ``obs`` is provided).
# ---------------------------------------------------------------------------
def build_distribution(
    variable: Variable, params: Dict[str, torch.Tensor]
) -> dist.Distribution:
    """Build the exact Pyro distribution declared by ``variable``.

    Wraps with ``to_event(1)`` for univariate families when ``size > 1``
    (Bernoulli/Normal/Delta); ``OneHotCategorical`` / ``Categorical`` /
    ``MultivariateNormal`` already carry the right event shape from their
    parameter tensor.
    """
    D = variable.distribution
    if D in (dist.Bernoulli, dist.Normal, dist.Delta):
        d = D(**params, **variable.dist_kwargs)
        if variable.size > 1:
            d = d.to_event(1)
        return d
    return D(**params, **variable.dist_kwargs)


# ---------------------------------------------------------------------------
# Reparameterised surrogate distributions (used when ``obs`` is None and we
# want gradients to flow through the sample).
# ---------------------------------------------------------------------------
def build_relaxed_distribution(
    variable: Variable,
    params: Dict[str, torch.Tensor],
    temperature: torch.Tensor,
) -> dist.Distribution:
    """Build a reparameterised distribution for unobserved sites.

    - ``Bernoulli`` → ``RelaxedBernoulliStraightThrough``.
    - ``OneHotCategorical`` → ``RelaxedOneHotCategoricalStraightThrough``.
    - ``Normal`` / ``MultivariateNormal`` / ``Delta`` → exact distribution
      (already reparameterised).
    - ``Categorical`` → ``ValueError`` (non-differentiable; declare the
      variable as ``OneHotCategorical`` instead).
    """
    D = variable.distribution
    if D is dist.Bernoulli:
        probs = params["probs"]
        logits = torch.log(probs.clamp(min=1e-8)) - torch.log(
            (1.0 - probs).clamp(min=1e-8)
        )
        d = dist.RelaxedBernoulliStraightThrough(temperature=temperature, logits=logits)
        if variable.size > 1:
            d = d.to_event(1)
        return d
    if D is dist.OneHotCategorical:
        logits = torch.log(params["probs"].clamp(min=1e-8))
        return dist.RelaxedOneHotCategoricalStraightThrough(
            temperature=temperature, logits=logits
        )
    if D is dist.Categorical:
        raise ValueError(
            f"Variable {variable.name!r}: plain Categorical cannot be sampled "
            "with gradient flow. Declare it as OneHotCategorical instead, or "
            "always supply this variable as evidence."
        )
    # Normal, MultivariateNormal, Delta: already reparameterised.
    return build_distribution(variable, params)


# ---------------------------------------------------------------------------
# Non-Pyro helpers (used by :class:`ForwardInference`).
# ---------------------------------------------------------------------------
def propagated_value(
    distribution: type, params: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Return the canonical deterministic value (e.g. ``loc`` for Normal,
    ``probs`` for Bernoulli)."""
    try:
        return params[_PRIMARY_PARAM[distribution]]
    except KeyError as exc:
        raise ValueError(f"Unsupported distribution {distribution!r}") from exc


def sample_from(
    variable: Variable,
    params: Dict[str, torch.Tensor],
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Reparameterised sample for the given variable. Mirrors
    :func:`build_relaxed_distribution` but ``rsample()``s directly (no Pyro
    tracing)."""
    return build_relaxed_distribution(variable, params, temperature).rsample()
