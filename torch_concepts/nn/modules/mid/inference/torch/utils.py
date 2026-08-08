"""Pure-PyTorch distribution utilities for the pytorch inference backend.

Provides reparameterisable relaxed surrogates for discrete families, a
deterministic-value dispatcher, and a sampler — all using only
``torch.distributions`` without any Pyro dependency.

Entry points:
- :func:`build_relaxed_distribution` — reparameterisable surrogate distribution.
- :func:`propagated_value` — canonical deterministic value from a param dict.
- :func:`mode_value` — hard, most-likely value from a param dict.
- :func:`sample_from` — reparameterised sample.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.distributions as dist

from ...distributions import spec_for
from ...variable import Variable


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
    # A variable may be declared with either the base family (Bernoulli,
    # OneHotCategorical) or its relaxed counterpart — both carry the same
    # ``relaxed`` factory in their spec, with the engine supplying ``temperature``.
    # The factory keeps the params flat (*batch, size) and reinterprets the
    # single size axis as the event, so batch_shape stays (*batch,); the
    # variable's declared shape is restored on the sampled realization, not here.
    spec = spec_for(D, f"Variable {variable.name!r}")
    from ..utils import build_distribution, build_plate

    if spec.relaxed is not None:
        # Same per-member split as the exact builder: a relaxed *categorical*
        # plate is k independent RelaxedOneHotCategoricals, not one over the
        # flattened width. ``build_plate`` is a no-op for the per-element
        # relaxed families (Bernoulli), which already handle a plate column-wise.
        return build_plate(
            variable, spec, params,
            lambda p: spec.relaxed(p, temperature, validate_args),
        )
    if spec.no_relaxed_reason is not None:
        raise ValueError(f"Variable {variable.name!r}: {spec.no_relaxed_reason}")
    # Continuous families are already reparameterisable — use the exact one.
    return build_distribution(variable, params)


def _activate(distribution: type, param_name: str, value: torch.Tensor) -> torch.Tensor:
    """Apply the family's default activation for ``param_name``.

    Reads ``DistributionSpec.activations``, so relaxed and exact variants of a
    family resolve to the same entry. Falls back to identity when the family
    declares no activation for this parameter.
    """
    spec = spec_for(distribution)
    activation = spec.activations.get(param_name)
    return activation(value) if activation is not None else value


def propagated_value(
    distribution: type, params: Dict[str, torch.Tensor], activate: bool = False,
) -> torch.Tensor:
    """Return the canonical deterministic value for a parameter dict.

    Picks the family's ``primary_param`` when present, otherwise falls back to
    ``logits`` (the alternative parametrization of the discrete families).

    When ``activate`` is ``True`` the selected parameter is mapped through its
    default activation (``DistributionSpec.activations``) before being returned,
    so that e.g. a CPD producing ``logits`` propagates probabilities to its
    children. When ``False`` the raw parameter is returned unchanged.
    """
    spec = spec_for(distribution)
    for param_name in (spec.primary_param, "logits"):
        if param_name in params:
            return (
                _activate(distribution, param_name, params[param_name])
                if activate
                else params[param_name]
            )
    raise ValueError(
        f"{distribution.__name__}: cannot propagate a value from parameters "
        f"{sorted(params)}; expected {spec.primary_param!r} or 'logits'."
    )


def _apply_mode(variable: Variable, value: torch.Tensor) -> torch.Tensor:
    """Quantize an already-activated value to the family's hard mode.

    Shared tail of :func:`mode_value` and :func:`sample_from`'s ``hard=True``
    path: both start from a value in the *activated* domain (probs, not logits)
    and need the same per-member rule. Splitting into one row per member first
    makes a ``k``-member categorical plate take ``k`` argmaxes rather than one
    over the flattened ``k * member_size`` columns — every family that declares
    a rule has one scalar per event element, so ``member_size`` is exactly its
    class count and a plate splits like a lone variable. Families whose
    ``primary_param`` already is the mode (Normal's ``loc``, Delta's ``value``)
    declare no rule and come back untouched.
    """
    spec = spec_for(variable.distribution, f"Variable {variable.name!r}")
    if spec.mode is None:
        return value
    per_member = value.reshape(*value.shape[:-1], -1, variable.member_size)
    return spec.mode(per_member).reshape(value.shape)


def mode_value(variable: Variable, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the family's *mode* — its most likely value — for a parameter dict.

    The hard counterpart of :func:`propagated_value`, in the same flat
    ``(*leading, size)`` layout: ``0.``/``1.`` bits for a Bernoulli, a one-hot
    row for a categorical, ``loc`` for a Normal, ``value`` for a Delta.

    The parameter is activated first, which makes each rule
    parametrization-agnostic — ``sigmoid(logits) > 0.5`` is ``logits > 0``, and
    ``argmax`` is invariant under ``softmax``. See :func:`_apply_mode` for the
    quantization itself.
    """
    value = propagated_value(variable.distribution, params, activate=True)
    return _apply_mode(variable, value)


def sample_from(
    variable: Variable,
    params: Dict[str, torch.Tensor],
    temperature: torch.Tensor,
    hard: bool = False,
) -> torch.Tensor:
    """Reparameterised sample for the given variable.

    With ``hard=True`` a discrete draw is quantized to its exact mode by a
    straight-through estimator — hard forward value, soft gradient. Families
    with no mode rule (:func:`_apply_mode`) are unaffected.
    """
    soft = build_relaxed_distribution(variable, params, temperature).rsample()
    if not hard:
        return soft
    hard_value = _apply_mode(variable, soft)
    return soft + (hard_value - soft).detach()
