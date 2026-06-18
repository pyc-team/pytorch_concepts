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

from ...models.variable import Variable, DEFAULT_ACTIVATIONS


# ---------------------------------------------------------------------------
# Primary parameter name per family — used by propagated_value.
# ---------------------------------------------------------------------------
_PRIMARY_PARAM: Dict[type, str] = {
    dist.Bernoulli: "probs",
    dist.OneHotCategorical: "probs",
    dist.Categorical: "probs",
    dist.Normal: "loc",
    dist.MultivariateNormal: "loc",
    dist.RelaxedOneHotCategorical: "probs",
    dist.RelaxedBernoulli: "probs",
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


def _activate(distribution: type, param_name: str, value: torch.Tensor) -> torch.Tensor:
    """Apply the default activation for ``(distribution, param_name)``.

    Looks the activation up in
    :data:`~torch_concepts.nn.modules.mid.models.variable.DEFAULT_ACTIVATIONS`,
    matching the distribution family by ``issubclass`` (so relaxed/exact
    variants resolve to the same entry). Falls back to identity when no entry
    exists for the family or parameter.
    """
    for base_cls, activations in DEFAULT_ACTIVATIONS.items():
        if issubclass(distribution, base_cls) and param_name in activations:
            return activations[param_name](value)
    return value


def propagated_value(
    distribution: type, params: Dict[str, torch.Tensor], activate: bool = False,
) -> torch.Tensor:
    """Return the canonical deterministic value for a parameter dict.

    When ``activate`` is ``True`` the selected parameter is mapped through its
    default activation (see :data:`DEFAULT_ACTIVATIONS`) before being returned,
    so that e.g. a CPD producing ``logits`` propagates probabilities to its
    children. When ``False`` the raw parameter is returned unchanged.
    """
    if distribution.__name__ == "Delta" and "value" in params:
        return _activate(distribution, "value", params["value"]) if activate else params["value"]
    for base_cls, param_name in _PRIMARY_PARAM.items():
        if issubclass(distribution, base_cls):
            # primary path, the user provided the primary parameter (e.g. "probs" for Bernoulli)
            if param_name in params:
                return _activate(distribution, param_name, params[param_name]) if activate else params[param_name]
            # fallback path, the user provided "logits" instead of the primary parameter (e.g. "logits" for Bernoulli)
            if "logits" in params:
                return _activate(distribution, "logits", params["logits"]) if activate else params["logits"]
    raise ValueError(f"Unsupported distribution {distribution!r}")


def sample_from(
    variable: Variable,
    params: Dict[str, torch.Tensor],
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Reparameterised sample for the given variable."""
    return build_relaxed_distribution(variable, params, temperature).rsample()
