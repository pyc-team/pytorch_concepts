"""Raw layers and activations for a continuous variable's ``scale`` parameter.

A :class:`~torch_concepts.nn.modules.mid.factors.cpd.ParametricCPD` applies no
activation: every module must already emit a value in its parameter's natural
domain. A scale head is therefore a raw layer followed by the activation that
makes its output valid — a plain
:class:`~torch_concepts.nn.Sequential` of the two.

For a ``Normal``'s per-element ``scale`` that activation is stock
``torch.nn.Softplus``. ``MultivariateNormal``'s ``scale_tril`` has no stock
equivalent — it is a *matrix* with a positive diagonal — so :class:`TrilActivation`
supplies it.

:class:`GlobalScale` is the raw *layer* side of the same contract: one learnable
value shared by every sample and element, for a homoscedastic Gaussian likelihood.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: could these be implemented as PyC layers?

class TrilActivation(nn.Module):
    """Map a flat raw tensor to a Cholesky factor ``scale_tril``.

    ``MultivariateNormal`` is parametrized by the lower-triangular Cholesky
    factor :math:`L` of its covariance, which the network emits flat: the
    ``size * (size + 1) // 2`` free entries (see
    :func:`~torch_concepts.nn.modules.mid.distributions._lower_triangular`).
    This scatters them into a ``(*leading, size, size)`` lower-triangular matrix
    and forces the diagonal positive, which is what makes :math:`L L^\\top` a
    valid positive-definite covariance.

    Entries are read in the row-major order of :func:`torch.tril_indices`; the
    mapping is an arbitrary but fixed bijection, so the layer before it simply
    learns it.

    Parameters
    ----------
    size : int
        Event width of the variable, i.e. the side length of :math:`L`.
    transform : callable, default ``F.softplus``
        Applied to the diagonal entries only; the strictly-lower entries stay
        unconstrained, as they must.
    floor : float, default 1e-6
        Added to the diagonal so :math:`L` is never singular — a zero there
        makes the covariance non-invertible and ``log_prob`` infinite.

    Examples
    --------
    >>> import torch
    >>> from torch_concepts.nn import Sequential, TrilActivation
    >>> head = Sequential(torch.nn.Linear(4, 6), TrilActivation(size=3))
    >>> tril = head(torch.randn(8, 4))
    >>> tril.shape
    torch.Size([8, 3, 3])
    >>> bool((tril.diagonal(dim1=-2, dim2=-1) > 0).all())
    True
    >>> bool((tril.triu(diagonal=1) == 0).all())
    True
    """

    def __init__(
        self,
        size: int,
        transform: Callable[[torch.Tensor], torch.Tensor] = F.softplus,
        floor: float = 1e-6,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.transform = transform
        self.floor = floor
        rows, cols = torch.tril_indices(self.size, self.size)
        # Buffers: they must follow the module across device and dtype casts.
        self.register_buffer("_rows", rows, persistent=False)
        self.register_buffer("_cols", cols, persistent=False)
        self.register_buffer("_is_diag", rows == cols, persistent=False)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        expected = self.size * (self.size + 1) // 2
        if raw.shape[-1] != expected:
            raise ValueError(
                f"{type(self).__name__}: got {raw.shape[-1]} values but a "
                f"{self.size}x{self.size} Cholesky factor needs {expected}."
            )
        # Transform the diagonal entries *before* scattering: writing into the
        # assembled matrix's diagonal view afterwards would be an in-place op on
        # a tensor autograd already tracks.
        values = torch.where(self._is_diag, self.transform(raw) + self.floor, raw)
        tril = raw.new_zeros(*raw.shape[:-1], self.size, self.size)
        tril[..., self._rows, self._cols] = values
        return tril

    def extra_repr(self) -> str:
        name = getattr(self.transform, "__name__", repr(self.transform))
        return f"size={self.size}, transform={name}, floor={self.floor}"


class GlobalScale(nn.Module):
    """One learnable scalar scale, shared by every element and every sample.

    A **raw** head for a continuous variable's ``scale``: it ignores its input
    and broadcasts a single learnable value to ``(batch, size)``, the flat
    layout every distribution parameter uses. ``nn.Softplus`` is composed on top
    like any other raw ``scale`` head, so ``softplus(raw)`` is the standard
    deviation. Costs one parameter where a per-input head (e.g. a second copy of
    a decoder) costs a whole network — the right trade for a homoscedastic
    likelihood. Reads a single leading dimension (``x.shape[0]``).

    Parameters
    ----------
    size : int
        Event width of the variable (``variable.size``).
    init : float, default 1.0
        Standard deviation at initialisation; stored as ``log(expm1(init))`` so
        ``softplus`` returns it exactly.

    Examples
    --------
    >>> import torch
    >>> from torch_concepts.nn import GlobalScale
    >>> scale = GlobalScale(size=784, init=0.1)
    >>> sum(p.numel() for p in scale.parameters())
    1
    >>> raw = scale(torch.randn(5, 32))
    >>> raw.shape
    torch.Size([5, 784])
    >>> torch.allclose(torch.nn.functional.softplus(raw), torch.full_like(raw, 0.1), atol=1e-6)
    True
    """

    def __init__(self, size: int, init: float = 1.0) -> None:
        super().__init__()
        self.size = int(size)
        self.raw = nn.Parameter(torch.tensor(math.log(math.expm1(float(init)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.raw.expand(x.shape[0], self.size)

    def extra_repr(self) -> str:
        return f"size={self.size}"
