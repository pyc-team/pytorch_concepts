"""Distribution utilities shared across all inference backends.

Backend-agnostic helpers used by both the pure-PyTorch and the Pyro engines:
temperature schedules, event reshaping, exact distribution construction, and
the discrete-state count an enumeration-based engine needs.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Union

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
    """Reshape a variable's *realization* from flat ``(batch, size)`` to
    ``(batch, *variable.shape)``.
    """
    return value.reshape(value.shape[0], *variable.shape)


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



