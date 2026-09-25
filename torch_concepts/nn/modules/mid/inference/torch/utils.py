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
    spec = spec_for(D, f"Variable {variable.name!r}")
    from ..utils import build_distribution, build_in_member_layout

    if spec.relaxed is not None:
        # The same builder as the exact path, so the two cannot disagree about
        # layout: a relaxed categorical plate is k independent
        # RelaxedOneHotCategoricals, one per row of the member axis.
        return build_in_member_layout(
            variable, spec, params,
            lambda p: spec.relaxed(p, temperature, validate_args),
        )
    if spec.no_relaxed_reason is not None:
        raise ValueError(f"Variable {variable.name!r}: {spec.no_relaxed_reason}")
    # Continuous families are already reparameterisable — use the exact one.
    return build_distribution(variable, params)


def _activate(variable: Variable, param_name: str, value: torch.Tensor) -> torch.Tensor:
    """Map ``value`` from ``param_name`` into the primary parameter's domain.

    Identity if ``param_name`` already is the primary parameter. Otherwise
    ``param_name`` is ``"logits"``, converted via
    :class:`~torch_concepts.nn.DefaultActivation` (sigmoid, per-member softmax, ...).
    """
    spec = spec_for(variable.distribution)
    if param_name == spec.primary_param:
        return value
    # ``value`` is in member layout, so each member already owns the last axis:
    # the activation is the lone-variable one (a plain softmax), never the
    # unflatten/flatten sandwich a flat row would need.
    factory = spec.param_activations.get(spec.primary_param)
    if factory is None:
        return value
    width = variable.member_size
    return factory(width, width)(value)


def propagated_value(
    variable: Variable, params: Dict[str, torch.Tensor], activate: bool = False,
) -> torch.Tensor:
    """Return the canonical deterministic value for a parameter dict.

    Picks ``primary_param`` when present, else falls back to ``logits``. If
    ``activate``, the picked parameter is converted to the primary domain
    (:func:`_activate`) before being returned; otherwise it is returned raw.

    ``params`` must be in the member layout every CPD reports. A flat
    ``(*leading, size)`` parameter — an engine's *output*, for instance —
    raises ``ValueError``: activating it would take a categorical plate's
    softmax over all members' classes at once.
    """
    spec = spec_for(variable.distribution, f"Variable {variable.name!r}")
    for param_name in (spec.primary_param, "logits"):
        if param_name in params:
            variable._require_member_layout(params[param_name], param_name)
            return (
                _activate(variable, param_name, params[param_name])
                if activate
                else params[param_name]
            )
    raise ValueError(
        f"{variable.distribution.__name__}: cannot propagate a value from parameters "
        f"{sorted(params)}; expected {spec.primary_param!r} or 'logits'."
    )


def _apply_mode(variable: Variable, value: torch.Tensor) -> torch.Tensor:
    """Quantize an already-activated value to the family's hard mode.

    Starts from a value in the *activated* domain (probs, not logits) and
    applies the family's rule to the last axis. In member layout that axis is
    one member's event, so a ``k``-member categorical plate takes ``k``
    argmaxes for free. Families whose ``primary_param`` already is the mode
    (Normal's ``loc``, Delta's ``value``) declare no rule and come back
    untouched.
    """
    spec = spec_for(variable.distribution, f"Variable {variable.name!r}")
    return value if spec.mode is None else spec.mode(value)


def mode_value(variable: Variable, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the family's *mode* — its most likely value — for a parameter dict.

    The hard counterpart of :func:`propagated_value`, in the same member layout
    ``(*leading, n_members, *member_shape)``: ``0.``/``1.`` bits for a
    Bernoulli, a one-hot row per member for a categorical, ``loc`` for a
    Normal, ``value`` for a Delta. Flat parameters raise, as in
    :func:`propagated_value`.

    The parameter is activated first, which makes each rule
    parametrization-agnostic — ``sigmoid(logits) > 0.5`` is ``logits > 0``, and
    ``argmax`` is invariant under ``softmax``. See :func:`_apply_mode` for the
    quantization itself.
    """
    value = propagated_value(variable, params, activate=True)
    return _apply_mode(variable, value)


def sample_from(
    variable: Variable,
    params: Dict[str, torch.Tensor],
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Reparameterised sample for the given variable.

    Soft or hard is decided by the **declared family**, not by the engine: a
    variable declared ``Bernoulli`` / ``RelaxedBernoulli`` draws a soft Concrete
    sample, while ``RelaxedBernoulliStraightThrough`` draws an exact bit with a
    soft gradient. Both resolve through the family's ``relaxed`` factory (see
    :func:`build_relaxed_distribution`).
    """
    return build_relaxed_distribution(variable, params, temperature).rsample()
