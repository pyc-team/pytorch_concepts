import torch
from torch import nn

class SumOp(nn.Module):
    r"""Sum ``n_terms`` equal-size contributions concatenated along the last dim.

    Aggregation layer that splits its input into ``n_terms`` chunks of size
    ``input_size`` along the last dimension and returns their elementwise sum.
    Useful as a fusion node in PGMs or wherever a fixed number of equally
    shaped tensors need to be combined additively.

    Args:
        input_size (int): Feature size of each contribution.
        n_terms (int, optional): Number of contributions to sum. Defaults to 2.

    Shape:
        - Input: ``(*, n_terms * input_size)``
        - Output: ``(*, input_size)``

    Example:
        >>> layer = SumOp(input_size=4, n_terms=3)
        >>> x = torch.ones(2, 12)   # three (2, 4) tensors stacked along the last dim
        >>> out = layer(x)
        >>> out.shape
        torch.Size([2, 4])
        >>> out
        tensor([[3., 3., 3., 3.],
                [3., 3., 3., 3.]])
    """

    def __init__(self, input_size: int, n_terms: int = 2):
        super().__init__()
        if n_terms < 1:
            raise ValueError(f"n_terms must be >= 1, got {n_terms}.")
        self.input_size = input_size
        self.n_terms = n_terms

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        expected = self.n_terms * self.input_size
        if input.shape[-1] != expected:
            raise ValueError(
                f"Expected {self.n_terms} contributions of size {self.input_size} "
                f"(total {expected}), got last dim {input.shape[-1]}."
            )
        if self.n_terms == 1:
            return input
        return input.unflatten(-1, (self.n_terms, self.input_size)).sum(-2)


class ResidualCorrectionOp(nn.Module):
    r"""Correction term :math:`\varepsilon` for an additive reconstruction.

    Given a ``target`` tensor and ``n_terms`` reconstruction tensors whose
    sum approximates ``target``, computes a correction :math:`\varepsilon`
    such that ``target ≈ sum(parts) + epsilon`` with explicit control over
    the autograd flow.

    Two orthogonal mechanisms are combined:

    1.  A *target-residual* contribution selected by ``residual_mode``.
        The mode names describe what happens to the parts' gradient at
        the output of the downstream sum ``h_bar = sum(parts) + epsilon``:

        - ``"block_parts"``:
          :math:`\varepsilon \mathrel{+}= \text{target} - \sum_i p_i`.
          Each part appears with coefficient :math:`-1` inside
          :math:`\varepsilon` and :math:`+1` outside (in the downstream
          sum).  The two cancel, so the gradient w.r.t. each part is
          blocked — gradient flows only through ``target``.
        - ``"keep_parts"``:
          :math:`\varepsilon \mathrel{+}= \text{target} - \sum_i p_i.\mathrm{detach}()`.
          Detaching the parts inside the residual keeps their gradient
          alive outside, so the downstream sum's gradient w.r.t. each
          part is :math:`+1` (the natural reconstruction grad).
        - ``"off"``: no target-residual contribution.  ``target`` is
          still consumed by :meth:`forward` (used for shape/dtype) so
          the layer can plug into a fixed-parents PGM without changes.

    2.  Optional *stop-gradient* contributions for selected parts.  For
        each index ``i`` in ``stop_grad_parts`` the term
        :math:`p_i.\mathrm{detach}() - p_i` is added to :math:`\varepsilon`.
        This adds zero to the value but cancels the gradient through
        :math:`p_i` in the downstream sum.

    Args:
        input_size (int): Feature size of ``target`` and each part.
        n_terms (int): Number of reconstruction parts.
        residual_mode (str): ``"block_parts"``, ``"keep_parts"``, or
            ``"off"``.  Defaults to ``"block_parts"``.
        stop_grad_parts (sequence of int, optional): Indices into the
            parts list that should receive a stop-gradient contribution.
            Defaults to no stop-grad parts.

    Shape:
        - Input ``input``: ``(*, (n_terms + 1) * input_size)`` — the
          concatenation of ``[target, *parts]`` along the last dim.
        - Output: ``(*, input_size)``.

    Behavior table (``n_terms=2``).  ``h_bar = sum(parts) + epsilon``:

    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+
    | ``residual_mode`` | ``stop_grad_parts``| grad ``part[0]``| grad ``part[1]``| value of ``h_bar``                    |
    +===================+====================+=================+=================+=======================================+
    | ``"block_parts"`` | ``()``             | 0               | 0               | ``target``                            |
    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+
    | ``"keep_parts"``  | ``()``             | 1               | 1               | ``target``                            |
    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+
    | ``"keep_parts"``  | ``(1,)``           | 1               | 0               | ``target``                            |
    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+
    | ``"keep_parts"``  | ``(0,)``           | 0               | 1               | ``target``                            |
    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+
    | ``"off"``         | ``()``             | 1               | 1               | ``part[0] + part[1]`` (target unused) |
    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+
    | ``"off"``         | ``(1,)``           | 1               | 0               | ``part[0] + part[1].detach()``        |
    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+

    For ``n_terms=1`` the table collapses to the rows whose
    ``stop_grad_parts`` is ``()`` (``"block_parts"`` → grad=0,
    ``"keep_parts"`` → grad=1, ``"off"`` → grad=1 with ``h_bar = part``).

    Example:
        >>> # ResNet-style residual: y = part + (target - part) == target
        >>> layer = ResidualCorrectionOp(input_size=4, n_terms=1, residual_mode="block_parts")
        >>> target = torch.randn(2, 4)
        >>> part = torch.randn(2, 4, requires_grad=True)
        >>> epsilon = layer(torch.cat([target, part], dim=-1))
        >>> y = part + epsilon
        >>> torch.allclose(y, target)
        True
    """

    _RESIDUAL_MODES = ("block_parts", "keep_parts", "off")

    def __init__(
        self,
        input_size: int,
        n_terms: int,
        residual_mode: str = "block_parts",
        stop_grad_parts=None,
    ):
        super().__init__()
        if n_terms < 1:
            raise ValueError(f"n_terms must be >= 1, got {n_terms}.")
        if residual_mode not in self._RESIDUAL_MODES:
            raise ValueError(
                f"residual_mode must be one of {self._RESIDUAL_MODES}, got {residual_mode!r}."
            )
        self.input_size = input_size
        self.n_terms = n_terms
        self.residual_mode = residual_mode
        self.stop_grad_parts = tuple(stop_grad_parts) if stop_grad_parts else ()
        for idx in self.stop_grad_parts:
            if not 0 <= idx < n_terms:
                raise ValueError(
                    f"stop_grad_parts index {idx} out of range [0, {n_terms})."
                )

    def compute(self, target: torch.Tensor, *parts: torch.Tensor) -> torch.Tensor:
        """Compute :math:`\\varepsilon` from ``target`` and positional ``parts``."""
        if len(parts) != self.n_terms:
            raise ValueError(
                f"Expected {self.n_terms} parts, got {len(parts)}."
            )
        if self.residual_mode == "block_parts":
            epsilon = target - sum(parts)
        elif self.residual_mode == "keep_parts":
            epsilon = target - sum(p.detach() for p in parts)
        else:  # "off"
            epsilon = torch.zeros_like(target)
        for idx in self.stop_grad_parts:
            epsilon = epsilon + (parts[idx].detach() - parts[idx])
        return epsilon

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        expected = (self.n_terms + 1) * self.input_size
        if input.shape[-1] != expected:
            raise ValueError(
                f"Expected target + {self.n_terms} parts of size {self.input_size} "
                f"(total {expected}), got last dim {input.shape[-1]}."
            )
        chunks = input.split(self.input_size, dim=-1)
        return self.compute(chunks[0], *chunks[1:])
