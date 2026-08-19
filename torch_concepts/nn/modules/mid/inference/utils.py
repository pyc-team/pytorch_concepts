"""Distribution utilities shared across all inference backends.

Backend-agnostic helpers used by both the pure-PyTorch and the Pyro engines:
temperature schedules, event reshaping, exact distribution construction,
teacher forcing, and the discrete-state count an enumeration-based engine needs.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dist

from ..distributions import spec_for
from ..variable import Variable


# ---------------------------------------------------------------------------
# Teacher forcing
# ---------------------------------------------------------------------------
# Shared by both backends: ``ForwardInference`` applies it to the value it
# propagates, ``PyroBaseInference.model_fn`` to the value a sample site caches.
# See ``BaseInference``'s ``p_int`` for what the rate means.

def _align_gt(
    gt: torch.Tensor, ref: torch.Tensor, name: Optional[str] = None
) -> torch.Tensor:
    """Cast and reshape ground-truth tensor to match the dtype and shape of ref.

    Step-by-step:
    1. Cast ``gt`` to ``ref``'s dtype so arithmetic ops don't raise type errors
       (e.g. LongTensor label vs FloatTensor network output).
    2. If shapes already match after the cast, return immediately.
    3. Handle the common "extra trailing 1" mismatches that arise when some
       code paths squeeze scalars and others don't:
       - ``gt`` has one more dim than ``ref`` and its last dim is 1 → squeeze it off.
       - ``gt`` has one fewer dim than ``ref`` and ``ref``'s last dim is 1 → unsqueeze.
    4. Finally, broadcast ``gt`` to exactly ``ref``'s shape so downstream ops
       (e.g. per-element masking) can use ``gt`` in place of ``ref``.

    Step 4 also silently rescues genuinely wrong targets — e.g. a ``(B,)`` label
    stretched across a ``(B, k)`` output — so a broadcast that is not one of the
    trailing-1 cases warns once per (name, shape) pair.
    """
    aligned = gt.to(ref.dtype) if gt.dtype != ref.dtype else gt
    if aligned.shape == ref.shape:
        return aligned

    original_shape = tuple(aligned.shape)
    if aligned.dim() == ref.dim() + 1 and aligned.shape[-1] == 1:
        aligned = aligned.squeeze(-1)
    elif aligned.dim() + 1 == ref.dim() and ref.shape[-1] == 1:
        aligned = aligned.unsqueeze(-1)
    if aligned.shape == ref.shape:
        return aligned

    _warn_broadcast(name, original_shape, tuple(ref.shape))
    return aligned.expand_as(ref)


# (name, target shape, reference shape) triples already warned about, so a
# training loop reports a suspicious target once rather than every step.
_BROADCAST_WARNED: set = set()


def _warn_broadcast(name: Optional[str], gt_shape: tuple, ref_shape: tuple) -> None:
    key = (name, gt_shape, ref_shape)
    if key in _BROADCAST_WARNED:
        return
    _BROADCAST_WARNED.add(key)
    target = f"for {name!r}" if name else "for a query variable"
    warnings.warn(
        f"Teacher forcing {target}: the target of shape {gt_shape} does not match "
        f"the predicted shape {ref_shape} and is being broadcast to fit. This is "
        "usually a mis-shaped label (e.g. class indices where a one-hot or "
        "per-element target is expected); pass a target of shape "
        f"{ref_shape} to silence this.",
        UserWarning,
        stacklevel=3,
    )


def teacher_force(
    nn_value: torch.Tensor,
    gt: torch.Tensor,
    p_int: float,
    n_leading: int,
    name: Optional[str] = None,
) -> torch.Tensor:
    """Stochastically replace nn_value with ground truth at rate p_int.

    The draw is per leading (batch-like) element: ``n_leading`` says how many of
    ``nn_value``'s dimensions are leading, so a variable is forced or not as a
    whole, whatever its event shape and however many batch axes there are.

    A rate strictly between 0 and 1 is CEM's **RandInt**; see
    :class:`~torch_concepts.nn.modules.mid.inference.base.BaseInference`'s
    ``p_int``.
    """
    aligned = _align_gt(gt, nn_value, name)
    if p_int >= 1.0:
        return aligned
    if p_int <= 0.0:
        return nn_value
    mask_shape = nn_value.shape[:n_leading] + (1,) * (nn_value.dim() - n_leading)
    mask = (torch.rand(mask_shape, device=nn_value.device) < p_int).to(nn_value.dtype)
    return mask * aligned + (1.0 - mask) * nn_value


def enumerable_cardinality(variable: Variable) -> int:
    """Number of discrete states of ``variable``.

    Used by the enumeration-based engines (currently
    :class:`~torch_concepts.nn.BeliefPropagation`) to size the state axis of a
    variable's messages and to build a factor's log-potential table.

    - Bernoulli-family with ``size == 1`` -> ``2`` (states ``0`` and ``1``).
    - Categorical/OneHot-family -> ``variable.size`` (one state per class).

    The per-family answer comes from ``DistributionSpec.state_count``.

    Parameters
    ----------
    variable : Variable
        The variable whose discrete states are being counted.

    Returns
    -------
    int
        The number of states.

    Raises
    ------
    ValueError
        If the variable cannot be enumerated — either a ``size > 1``
        Bernoulli (a set of independent bits, not one variable) or a
        non-discrete family such as ``Normal`` or ``Delta``.
    """
    D = variable.distribution
    spec = spec_for(D, f"Variable {variable.name!r}")
    if spec.is_enumerable:
        card = spec.state_count(variable.size)
        if card is not None:
            return card
        raise ValueError(
            f"Variable {variable.name!r}: a size>1 {D.__name__} is a set of "
            "independent bits, not a single enumerable variable. Model each bit "
            "as its own binary variable, or use a Categorical/OneHotCategorical."
        )
    raise ValueError(
        f"Variable {variable.name!r}: distribution {D.__name__} is not discretely "
        "enumerable, so it cannot be a free (queried/latent) variable under belief "
        "propagation. Observe it as evidence, or use a discrete distribution."
    )


def make_temperature_schedule(
    initial_temperature: float,
    annealing: Union[str, Callable[[int], float]],
    annealing_rate: float,
    final_temperature: float = 1e-6,
) -> Callable[[int], float]:
    """Build a ``step -> temperature`` schedule.

    ``annealing`` may be ``'constant'``, ``'exponential'`` (decays as
    ``T0 * exp(-rate * step)``), ``'linear'`` (decays as ``T0 - rate * step``),
    or a user-supplied callable.

    Both decays are clamped below at ``final_temperature``, so the schedule
    reaches a floor and stays there rather than sliding towards zero: a relaxed
    sample is only useful while its gradient is, and a temperature that keeps
    shrinking eventually makes every draw a one-hot with a vanishing gradient.
    Around ``0.1`` a Concrete draw already puts ~94% of its mass on one state.
    A user-supplied callable is returned untouched and is responsible for its
    own floor.
    """
    if callable(annealing):
        return annealing
    floor = float(final_temperature)
    if annealing == "constant":
        return lambda step: float(initial_temperature)
    if annealing == "exponential":
        return lambda step: max(
            floor, float(initial_temperature) * math.exp(-annealing_rate * step)
        )
    if annealing == "linear":
        return lambda step: max(
            floor, float(initial_temperature) - annealing_rate * step
        )
    raise ValueError(
        f"Unknown annealing schedule {annealing!r}. Use "
        "'constant', 'exponential', 'linear', or pass a callable."
    )


def reshape_value_to_event(
    variable: Variable, value: torch.Tensor
) -> torch.Tensor:
    """Reshape a variable's *realization* to ``(*leading, *variable.shape)``.

    ``leading`` may be any number of batch-like dimensions: the trailing block
    is identified by :func:`leading_shape`, so a value arriving flat as
    ``(*leading, size)`` (what a CPD produces) and one already in event layout
    (what a user passes as evidence) both land on the same result. A value
    already shaped exactly ``(*leading, *event)`` — including a variable with
    no event shape — is returned unchanged rather than reshaped.
    """
    event = tuple(variable.shape)
    if not event:
        return value
    leading = leading_shape(event, variable.size, value, variable.name)
    target = (*leading, *event)
    return value if tuple(value.shape) == target else value.reshape(*target)


def flatten_event(variable: Variable, value: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`reshape_value_to_event`.

    Flattens ``(*leading, *variable.shape)`` to ``(*leading, size)`` — the layout
    every distribution parameter and every annotated output tensor uses. A value
    that is already flat is returned unchanged.
    """
    if len(variable.shape) <= 1:
        return value
    leading = leading_shape(variable.shape, variable.size, value, variable.name)
    return value.reshape(*leading, variable.size)


def leading_shape(
    event: Tuple[int, ...],
    size: int,
    value: torch.Tensor,
    context: str = "",
) -> torch.Size:
    """The batch-like dimensions of ``value`` given the event it carries.

    Three layouts are accepted, tried in this order:

    1. **event** — ``(*leading, *event)``, what a caller passes as evidence;
    2. **flat** — ``(*leading, size)``, what a CPD produces;
    3. **squeezed scalar** — ``(*leading,)`` for a width-1 variable, where the
       trailing axis of size 1 is left off (a ``(batch,)`` vector of labels for
       a binary concept).

    The leading dimensions are whatever precedes the matched trailing block.
    This is the one place the mid level decides where "batch" ends and "event"
    begins, so engines never have to assume a single leading dimension.

    ``event``/``size`` are passed rather than a :class:`Variable` because a
    plate *member*'s event is its own per-member block, not the whole plate's.

    When several layouts match — they can only differ for a width-1 variable,
    where e.g. ``(1,)`` reads as either one flat observation or one squeezed
    one — the first that leaves a non-empty leading shape wins, since every
    engine expects at least one batch-like dimension.

    Raises
    ------
    ValueError
        If ``value`` matches no layout.
    """
    event = tuple(event)
    n_event = len(event)
    candidates: list = []
    if value.dim() >= n_event and tuple(value.shape[value.dim() - n_event:]) == event:
        candidates.append(value.shape[: value.dim() - n_event])
    if value.dim() >= 1 and value.shape[-1] == size:
        candidates.append(value.shape[:-1])
    if size == 1:
        candidates.append(value.shape)

    for leading in candidates:
        if len(leading):
            return leading
    if candidates:
        return candidates[0]

    prefix = f"{context}: " if context else ""
    raise ValueError(
        f"{prefix}tensor of shape {tuple(value.shape)} matches none of the accepted "
        f"layouts: event (*leading, {', '.join(map(str, event))}), flat "
        f"(*leading, {size})"
        + (", or squeezed (*leading,)." if size == 1 else ".")
    )


#: Hard counterpart of each relaxed family. The estimators draw *exact*
#: samples so that equality matching works, even from a variable declared with
#: a Concrete/relaxed family for gradient flow.
EXACT_FAMILY: Dict[type, type] = {
    dist.RelaxedBernoulli: dist.Bernoulli,
    dist.RelaxedOneHotCategorical: dist.OneHotCategorical,
}

# The straight-through families are looked up by exact key too, so they need
# their own entries — a subclass of RelaxedBernoulli does not match it here.
try:
    import pyro.distributions as _pyro_dist
except ImportError:  # pragma: no cover - pyro not installed
    pass
else:
    EXACT_FAMILY[_pyro_dist.RelaxedBernoulliStraightThrough] = dist.Bernoulli
    EXACT_FAMILY[_pyro_dist.RelaxedOneHotCategoricalStraightThrough] = dist.OneHotCategorical


def _splits_per_member(spec, params: Dict[str, torch.Tensor], variable: Variable) -> bool:
    """Whether ``variable``'s parameters can be folded into one row per member.

    True when every supplied parameter is one scalar per event element, so
    ``(*batch, k * member_size)`` reshapes cleanly to ``(*batch, k,
    member_size)``. It is False for a parameter whose size is a *function* of
    the width rather than proportional to it — ``MultivariateNormal``'s
    ``scale_tril`` holds ``size * (size + 1) / 2`` Cholesky entries, and a plate
    of those needs one factor per member, not a reshape. Such a plate is left
    building a single joint distribution, as it always has.
    """
    return all(
        name in spec.param_sizes and spec.param_sizes[name](variable.size) == variable.size
        for name in params
    )


def build_plate(
    variable: Variable,
    spec,
    params: Dict[str, torch.Tensor],
    make: Callable[[Dict[str, torch.Tensor]], dist.Distribution],
) -> dist.Distribution:
    """Build ``make(params)``, splitting a plate whose event spans the width.

    ``make`` maps a (possibly folded) parameter dict to a distribution over its
    trailing axis. A family whose event is univariate — Bernoulli, Normal, the
    ``wrap_independent`` families — already gives one scalar per member, so it
    is built as-is (the caller wraps it in ``Independent`` if it needs to). A
    family whose event spans the *whole* width (``OneHotCategorical`` and its
    relaxed twin) is wrong on a plate: ``k`` members of ``member_size`` classes
    are ``k`` independent distributions, not one over ``k * member_size``
    classes. So its params are folded to ``(*batch, k, member_size)``, built in
    one call, wrapped ``Independent`` over the member axis, and the event
    reshaped back to the flat ``(*batch, size)`` that every caller passes values
    in and reads samples out with.

    This is the single place the member split lives, so the exact
    (:func:`build_distribution`) and relaxed (``build_relaxed_distribution``)
    builders cannot disagree about it.
    """
    if (
        len(variable.members) > 1
        and not spec.wrap_independent
        and _splits_per_member(spec, params, variable)
    ):
        k, m = len(variable.members), variable.member_size
        folded = {name: v.reshape(*v.shape[:-1], k, m) for name, v in params.items()}
        return dist.TransformedDistribution(
            dist.Independent(make(folded), 1),
            dist.transforms.ReshapeTransform((k, m), (variable.size,)),
        )
    return make(params)


def build_distribution(
    variable: Variable,
    params: Dict[str, torch.Tensor],
    family: Optional[type] = None,
) -> dist.Distribution:
    """Build the exact distribution declared by ``variable``.

    ``family`` overrides which distribution class is built — how an estimator
    asks for a *hard* draw from a variable declared with a relaxed family (see
    :data:`EXACT_FAMILY`). ``variable.dist_kwargs`` (a Concrete family's
    temperature) belongs to the declared family, so it is dropped when the
    family is overridden. Everything else — the plate split below included —
    is shared, which is the point: the layout rules live in one place.

    Parameters arrive flat as ``(*batch, size)`` (the CPD's untouched output), so
    univariate-event families (Bernoulli, Normal) are wrapped in ``Independent``
    over the single trailing ``size`` axis, giving ``batch_shape == (*batch,)``
    and ``event_shape == (size,)``. This keeps the batch dim intact (required for
    Pyro plates to line up) regardless of the variable's declared ``shape``; the
    variable's event shape is restored on the *realization* by
    :func:`reshape_value_to_event`, not on the distribution parameters.
    """
    D = family if family is not None else variable.distribution
    dist_kwargs = {} if family is not None else variable.dist_kwargs
    spec = spec_for(D, f"Variable {variable.name!r}")

    # ``wrap_independent`` marks the families whose event is univariate (a
    # Delta point mass, by contrast, already has ``batch_shape == ()``). For
    # those, a plate already works: one scalar per column, k of them.
    if spec.wrap_independent:
        return dist.Independent(D(**params, **dist_kwargs), 1)

    return build_plate(
        variable, spec, params, lambda p: D(**p, **dist_kwargs)
    )



