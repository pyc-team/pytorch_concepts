"""Pure-PyTorch distribution utilities for the pytorch inference backend.

Provides reparameterisable relaxed surrogates for discrete families, a
deterministic-value dispatcher, and a sampler — all using only
``torch.distributions`` without any Pyro dependency.

Entry points:
- :func:`build_relaxed_distribution` — reparameterisable surrogate distribution.
- :func:`propagated_value` — canonical deterministic value from a param dict.
- :func:`sample_from` — reparameterised sample.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.distributions as dist

from ...models.variable import Variable


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
    validate_args: Optional[bool] = None,
) -> dist.Distribution:
    """Build a reparameterised distribution.

    Discrete families use their relaxed (Concrete / Gumbel-Softmax) counterpart,
    whose ``rsample`` yields differentiable *soft* samples so that gradients flow
    without a straight-through estimator. Continuous families fall back to the
    exact distribution (which is already reparameterisable via ``rsample``).

    ``validate_args`` is forwarded to the distribution constructors. Pass
    ``False`` when the relaxed samples will be scored with ``log_prob`` (e.g.
    importance weighting): at low temperature a relaxed draw lands on the
    boundary of the simplex / unit interval, which torch's argument validation
    rejects even though it is the expected behaviour. The default ``None``
    preserves torch's global setting for callers that only ``rsample``.
    """
    D = variable.distribution
    if issubclass(D, dist.Bernoulli):
        # Pass whichever key the user provided (probs or logits) directly.
        # Params are flat (*batch, size); reinterpret the single size axis as
        # the event so batch_shape stays (*batch,). The variable's declared
        # shape is restored on the sampled realization, not here.
        d = dist.RelaxedBernoulli(temperature=temperature, **params, validate_args=validate_args)
        return dist.Independent(d, 1, validate_args=validate_args)
    if issubclass(D, dist.OneHotCategorical):
        return dist.RelaxedOneHotCategorical(temperature=temperature, **params, validate_args=validate_args)
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
            # primary path, the user provided the primary parameter (e.g. "probs" for Bernoulli)
            if param_name in params:
                return params[param_name]
            # fallback path, the user provided "logits" instead of the primary parameter (e.g. "logits" for Bernoulli)
            if "logits" in params:
                return params["logits"]
    raise ValueError(f"Unsupported distribution {distribution!r}")


def sample_from(
    variable: Variable,
    params: Dict[str, torch.Tensor],
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Reparameterised sample for the given variable."""
    return build_relaxed_distribution(variable, params, temperature).rsample()
