"""Pure-PyTorch distribution utilities for the pytorch inference backend.

Provides straight-through estimators (STE) for discrete relaxations, a
deterministic-value dispatcher, and a sampler — all using only
``torch.distributions`` without any Pyro dependency.

Entry points:
- :class:`_StraightThroughBernoulli` — STE relaxed Bernoulli.
- :class:`_StraightThroughOneHotCategorical` — STE relaxed OneHotCategorical.
- :func:`build_relaxed_distribution` — reparameterisable surrogate distribution.
- :func:`propagated_value` — canonical deterministic value from a param dict.
- :func:`sample_from` — reparameterised sample.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.distributions as dist

from ...models.variable import Variable


# ---------------------------------------------------------------------------
# Straight-through estimators for discrete relaxations.
# ---------------------------------------------------------------------------

class _StraightThroughBernoulli(dist.RelaxedBernoulli):
    """Relaxed Bernoulli with straight-through gradient estimator."""

    def rsample(self, sample_shape=()):
        soft = super().rsample(sample_shape)
        hard = (soft > 0.5).to(soft.dtype)
        return hard + (soft - soft.detach())


class _StraightThroughOneHotCategorical(dist.RelaxedOneHotCategorical):
    """Relaxed OneHotCategorical with straight-through gradient estimator."""

    def rsample(self, sample_shape=()):
        soft = super().rsample(sample_shape)
        hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(-1, keepdim=True), 1.0)
        return hard + (soft - soft.detach())


# ---------------------------------------------------------------------------
# Primary parameter name per family — used by propagated_value.
# ---------------------------------------------------------------------------
_PRIMARY_PARAM: Dict[type, str] = {
    dist.Bernoulli: "probs",
    dist.OneHotCategorical: "probs",
    dist.Categorical: "probs",
    dist.Normal: "loc",
    dist.MultivariateNormal: "loc",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_relaxed_distribution(
    variable: Variable,
    params: Dict[str, torch.Tensor],
    temperature: torch.Tensor,
) -> dist.Distribution:
    """Build a reparameterised distribution.

    Discrete families use the straight-through estimator so that gradients
    flow through hard samples. Continuous families fall back to the exact
    distribution (which is already reparameterisable via ``rsample``).
    """
    D = variable.distribution
    if issubclass(D, dist.Bernoulli):
        # Pass whichever key the user provided (probs or logits) directly.
        d = _StraightThroughBernoulli(temperature=temperature, **params)
        return dist.Independent(d, len(variable.shape))
    if issubclass(D, dist.OneHotCategorical):
        return _StraightThroughOneHotCategorical(temperature=temperature, **params)
    if issubclass(D, dist.Categorical):
        raise ValueError(
            f"Variable {variable.name!r}: plain Categorical cannot be sampled "
            "with gradient flow. Declare it as OneHotCategorical instead, or "
            "always supply this variable as evidence."
        )
    from ..utils import build_distribution
    return build_distribution(variable, params)


def propagated_value(
    distribution: type, params: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Return the canonical deterministic value for a parameter dict.

    For discrete families parametrized with ``logits``, converts to
    probabilities (sigmoid / softmax) so chained parent values are always
    in probability space, consistent with the ``probs`` path.
    """
    if distribution.__name__ == "Delta" and "value" in params:
        return params["value"]
    for base_cls, param_name in _PRIMARY_PARAM.items():
        if issubclass(distribution, base_cls):
            if param_name in params:
                return params[param_name]
            # logits path: convert to probs for consistency
            if "logits" in params:
                if issubclass(distribution, dist.Bernoulli):
                    return torch.sigmoid(params["logits"])
                return torch.softmax(params["logits"], dim=-1)  # Categorical / OHC
    raise ValueError(f"Unsupported distribution {distribution!r}")


def sample_from(
    variable: Variable,
    params: Dict[str, torch.Tensor],
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Reparameterised sample for the given variable."""
    return build_relaxed_distribution(variable, params, temperature).rsample()
