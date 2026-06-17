"""Distribution utilities shared across all inference backends.

Only ``build_distribution`` lives here — it is used by both the pure-PyTorch
and Pyro backends. 
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Union

import torch
import torch.distributions as dist

from ..models.variable import Variable


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
    D = variable.distribution
    _univariate = (dist.Bernoulli, dist.Normal)
    _is_delta = D.__name__ == "Delta"
    if _is_delta:
        # Delta is a deterministic point mass; wrapping it in ``Independent``
        # doesn't make sense (no extra batch dims to reinterpret) and the
        # ``torch_concepts.distributions.Delta`` we use here has
        # ``batch_shape == ()`` by construction.
        return D(**params, **variable.dist_kwargs)
    if issubclass(D, _univariate):
        d = D(**params, **variable.dist_kwargs)
        return dist.Independent(d, 1)
    return D(**params, **variable.dist_kwargs)



