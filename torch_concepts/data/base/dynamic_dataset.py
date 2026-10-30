"""
Concept dataset built from time series.

This module provides the DynamicConceptDataset class, which cuts time series
into windows and uses them as the samples of a ConceptDataset.
"""
from typing import Optional

import torch
from torch import Tensor

from ...annotations import Annotations
from .dataset import ConceptDataset


class DynamicConceptDataset(ConceptDataset):
    """A ConceptDataset whose samples are windows of time series.

    The inputs are ``N`` series of ``T`` steps, ``(T, N, F)``, and the
    optional concepts are series over the same steps, ``(T, N, C)``. Each
    sample is ``window`` consecutive steps of one series, ``(window, F)`` and
    ``(window, C)``; the model picks which steps it observes and which it
    predicts. Steps left over at the end are dropped.

    Attributes:
        n_series (int): Number of series, ``N``.
        start (Tensor): First step of each sample, for splits in time.
        series (Tensor): Series of each sample, for splits by series.

    Args:
        input_data: Input series, ``(T, N, F)`` or ``(T, F)``.
        concepts: Optional concept series, ``(T, N, C)`` or ``(T, C)``.
        annotations: Concept metadata, as in :class:`ConceptDataset`.
        window: Steps per sample (default: ``T``).
        stride: Steps from the start of one window to the start of the next
            (default: ``window``, so windows do not overlap).
        **kwargs: Passed to :class:`ConceptDataset`.

    Example:
        >>> import torch
        >>> from torch_concepts import Annotations
        >>> from torch_concepts.data.base import DynamicConceptDataset
        >>> x, c = torch.randn(100, 3, 4), torch.randn(100, 3, 2)  # 3 series of 100 steps
        >>> annotations = Annotations(labels=["a", "b"], cardinalities=[1, 1], types=["continuous"] * 2)
        >>> dataset = DynamicConceptDataset(x, c, annotations, window=10, stride=5)
        >>> dataset.input_data.shape  # 19 windows of each series
        torch.Size([57, 10, 4])
        >>> dataset.concepts.shape
        torch.Size([57, 10, 2])
        >>> dataset.n_series
        3
    """

    def __init__(
        self,
        input_data: Tensor,
        concepts: Optional[Tensor] = None,
        annotations: Optional[Annotations] = None,
        window: Optional[int] = None,
        stride: Optional[int] = None,
        **kwargs,
    ):
        if input_data.dim() == 2:  # a single series
            input_data = input_data.unsqueeze(1)
            concepts = concepts.unsqueeze(1) if concepts is not None else None
        n_steps, self.n_series = input_data.shape[:2]
        window = window or n_steps
        stride = stride or window

        def samples(series):  # (T, N, ...) -> (n_windows * N, window, ...)
            return series.unfold(0, window, stride).movedim(-1, 2).flatten(0, 1)

        inputs = samples(input_data)  # unfold raises if window > T
        self.start, self.series = torch.cartesian_prod(
            torch.arange(0, n_steps - window + 1, stride), torch.arange(self.n_series)
        ).T
        super().__init__(
            inputs, samples(concepts) if concepts is not None else None, annotations, **kwargs
        )

    def _subset_rows(self, indices) -> None:
        """Subset the samples and keep ``start`` and ``series`` aligned with them."""
        super()._subset_rows(indices)
        self.start, self.series = self.start[indices], self.series[indices]
