"""Distribution utilities shared across all inference backends.

Backend-agnostic helpers used by both the pure-PyTorch and the Pyro engines:
temperature schedules, event reshaping, exact distribution construction, and
the discrete-state count an enumeration-based engine needs.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Tuple, Union

import torch
import torch.distributions as dist

from ..distributions import spec_for
from ..variable import Variable


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
) -> Callable[[int], float]:
    """Build a ``step -> temperature`` schedule.

    ``annealing`` may be ``'constant'``, ``'exponential'`` (decays as
    ``T0 * exp(-rate * step)``), ``'linear'`` (decays as
    ``max(eps, T0 - rate * step)``), or a user-supplied callable.
    """
    if callable(annealing):
        return annealing
    if annealing == "constant":
        return lambda step: float(initial_temperature)
    if annealing == "exponential":
        return lambda step: float(initial_temperature) * math.exp(-annealing_rate * step)
    if annealing == "linear":
        return lambda step: max(
            1e-6, float(initial_temperature) - annealing_rate * step
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
    (what a user passes as evidence) both land on the same result.
    """
    leading = leading_shape(variable.shape, variable.size, value, variable.name)
    return value.reshape(*leading, *variable.shape)


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


def build_distribution(
    variable: Variable, params: Dict[str, torch.Tensor]
) -> dist.Distribution:
    """Build the exact distribution declared by ``variable``.

    Parameters arrive flat as ``(*batch, size)`` (the CPD's untouched output), so
    univariate-event families (Bernoulli, Normal) are wrapped in ``Independent``
    over the single trailing ``size`` axis, giving ``batch_shape == (*batch,)``
    and ``event_shape == (size,)``. This keeps the batch dim intact (required for
    Pyro plates to line up) regardless of the variable's declared ``shape``; the
    variable's event shape is restored on the *realization* by
    :func:`reshape_value_to_event`, not on the distribution parameters.
    """
    # NOTE: a plate of *categorical* members builds one OneHotCategorical over the
    # plate's whole flattened width (len(members) * member_size classes) rather
    # than one distribution per member. Tracked separately; out of scope here.
    D = variable.distribution
    d = D(**params, **variable.dist_kwargs)
    # ``wrap_independent`` marks the families whose event is univariate (a
    # Delta point mass, by contrast, already has ``batch_shape == ()``).
    spec = spec_for(D, f"Variable {variable.name!r}")
    return dist.Independent(d, 1) if spec.wrap_independent else d



