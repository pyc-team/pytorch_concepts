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

from typing import Sequence, Union

import torch
import torch.nn as nn


class LearnablePrior(nn.Module):
    """Learnable parameter module for root (parent-less) CPDs.

    Wraps a single ``nn.Parameter`` of the requested ``size`` and returns it on
    ``forward()``, making it a drop-in parametrization for a root CPD. The
    parameter is randomly initialised from a standard normal distribution.

    Parameters
    ----------
    size : int
        Length of the parameter vector. This must match the per-parameter size
        the target distribution expects (e.g. ``1`` for a Bernoulli ``logits``,
        ``k`` for a ``k``-way OneHotCategorical ``logits``).
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.randn(size))

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
    values : torch.Tensor or sequence of float
        The fixed parameter values. Their length must match the per-parameter
        size the target distribution expects. Coerced to a 1-D float tensor.
    """

    def __init__(self, values: Union[torch.Tensor, Sequence[float]]) -> None:
        super().__init__()
        if isinstance(values, torch.Tensor):
            tensor = values.detach().clone().float()
        else:
            tensor = torch.as_tensor(values, dtype=torch.float)
        self.register_buffer("values", tensor)

    def forward(self) -> torch.Tensor:
        return self.values
