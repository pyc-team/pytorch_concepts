"""Prior modules for root CPDs.

Two types are provided:

* :class:`LearnablePrior` — a trainable ``nn.Parameter`` (optimised during
  training);
* :class:`FixedPrior` — values known a priori, held as a non-learnable buffer
  (never updated by the optimizer).

The output of the prior modules is an *unconstrained* parameter. 
If the parameter that has to be learned is a probability, an activation function (e.g. ``torch.sigmoid``) 
must be applied to the output of the prior module to map it to the correct domain.
"""

from __future__ import annotations

from typing import Callable, Tuple, Union

import torch
import torch.nn as nn


class LearnablePrior(nn.Module):
    """Learnable parameter module for root (parent-less) CPDs.

    Wraps a single ``nn.Parameter`` of the requested ``size`` and returns it on
    ``forward()``, making it a drop-in parametrization for a root CPD. The
    parameter is randomly initialised from a standard normal distribution.

    Parameters
    ----------
    size : int or tuple of int
        Shape of the parameter. An int gives a 1-D vector of that length; a
        tuple gives a parameter of that shape. Must match the per-parameter
        size the target distribution expects (e.g. ``1`` for a Bernoulli
        ``logits``, ``k`` for a ``k``-way OneHotCategorical ``logits``).
    broadcast : bool, default True
        Whether ``root_params`` broadcasts this prior over the query's leading
        (batch-like) dims. Set ``False`` for model state rather than a
        per-observation quantity (e.g. a fixed matrix): it stays unbatched.
    """

    def __init__(self, size: Union[int, Tuple[int, ...]], broadcast: bool = True) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))
        self.broadcast = broadcast

    def forward(self) -> torch.Tensor:
        return self.param


class FixedPrior(nn.Module):
    """Non-learnable prior module holding parameter values known a priori.

    Mirrors :class:`LearnablePrior` but the parameter is **fixed**: the supplied
    values are registered as a buffer, so they carry no gradient and are never
    touched by the optimizer, while still moving with ``.to(device)`` and being
    saved in the module ``state_dict``. Use this when a root distribution's
    parameter is known in advance (e.g. a fixed prior probability) rather than
    learned.

    Parameters
    ----------
    values : torch.Tensor
        The fixed parameter values (cloned into a buffer). Their shape must
        match the per-parameter shape the target distribution expects.
    broadcast : bool, default True
        Whether ``root_params`` broadcasts this prior over the query's leading
        (batch-like) dims. Set ``False`` for model state rather than a
        per-observation quantity (e.g. a fixed matrix): it stays unbatched.
    """

    def __init__(self, values: torch.Tensor, broadcast: bool = True) -> None:
        super().__init__()
        tensor = values.detach().clone()
        self.register_buffer("values", tensor)
        self.broadcast = broadcast

    def forward(self) -> torch.Tensor:
        return self.values


class TiedPrior(nn.Module):
    """Non-owning prior that borrows its values from an external source — no copy.

    Mirrors :class:`FixedPrior`, but instead of cloning the values into a buffer
    it holds a **callable** and returns its result on ``forward()``. Use this to
    tie a root CPD to values that live elsewhere (e.g. a head's embedding matrix)
    without duplicating them; calling the source each time keeps the result on the
    model's current device/dtype. Nothing is registered, the source owns that.

    Parameters
    ----------
    get_values : Callable[[], torch.Tensor]
        Returns the prior's values, typically a view of existing weights.
    broadcast : bool, default True
        Whether ``root_params`` broadcasts this prior over the query's leading
        (batch-like) dims. Set ``False`` for model state rather than a
        per-observation quantity (e.g. a fixed matrix): it stays unbatched.
    """

    def __init__(self, get_values: Callable[[], torch.Tensor], broadcast: bool = True) -> None:
        super().__init__()
        self._get_values = get_values
        self.broadcast = broadcast

    def forward(self) -> torch.Tensor:
        return self._get_values()
