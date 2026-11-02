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

    The inputs are ``N`` series of ``T`` steps and ``F`` channels,
    ``(T, N, F)``, and the optional concepts are ``C`` concepts over the same
    steps and series, ``(T, N, C)``. Each series is cut into windows of
    ``window`` consecutive steps, one every ``stride`` steps; steps left over
    at the end are dropped. The windows are then flattened with the series
    into the sample axis: each sample is one window of one series,
    ``(window, F)`` and ``(window, C)``, so there are ``n_windows * N``
    samples. With ``flatten_series=False``, each sample is instead one window
    of all the series, ``(window, N, F)`` and ``(window, N, C)``. The model
    picks which steps of a sample it observes and which it predicts.

    Attributes:
        n_series (int): Number of series, ``N``.
        start (Tensor): First step of each sample, for splits in time.
        series (Tensor): Series of each sample, for splits by series; None
            with ``flatten_series=False``.

    Args:
        input_data: Input series, ``(T, N, F)``, or ``(T, F)`` for one series.
        concepts: Optional concept series, ``(T, N, C)`` or ``(T, C)``.
        annotations: Concept metadata, as in :class:`ConceptDataset`.
        window: Steps per sample (default: ``T``).
        stride: Steps from the start of one window to the start of the next
            (default: ``window``, so windows do not overlap).
        flatten_series: One series per sample (default: True), or all the
            series in each sample (False).
        **kwargs: Passed to :class:`ConceptDataset`.

    Example:
        >>> import torch
        >>> from torch_concepts import Annotations
        >>> from torch_concepts.data.base import DynamicConceptDataset
        >>> x = torch.randn(10, 3, 5)  # T = 10 steps, N = 3 series, F = 5 channels
        >>> c = torch.randn(10, 3, 2)  # C = 2 concepts
        >>> annotations = Annotations(labels=["a", "b"], cardinalities=[1, 1], types=["continuous"] * 2)
        >>> dataset = DynamicConceptDataset(x, c, annotations, window=4, stride=1)
        >>> dataset.input_data.shape  # 7 windows of each of the 3 series
        torch.Size([21, 4, 5])
        >>> dataset.concepts.shape
        torch.Size([21, 4, 2])
        >>> windows = DynamicConceptDataset(
        ...     x, c, annotations, window=4, stride=1, flatten_series=False)
        >>> windows.input_data.shape  # 7 windows, each with the 3 series
        torch.Size([7, 4, 3, 5])
    """

    def __init__(
        self,
        input_data: Tensor,
        concepts: Optional[Tensor] = None,
        annotations: Optional[Annotations] = None,
        window: Optional[int] = None,
        stride: Optional[int] = None,
        flatten_series: bool = True,
        **kwargs,
    ):
        if input_data.dim() == 2:  # one series, (T, F): a series axis of size 1
            input_data = input_data.unsqueeze(1)
            concepts = concepts.unsqueeze(1) if concepts is not None else None
        n_steps, self.n_series = input_data.shape[:2]
        window = window or n_steps
        stride = stride or window

        def samples(series: Tensor) -> Tensor:
            """The windows of ``series`` (T, N, ...) as samples. unfold() raises if window > T."""
            windows = series.unfold(0, window, stride)  # (n_windows, N, ..., window)
            if flatten_series:
                return windows.movedim(-1, 2).flatten(0, 1)  # (n_windows * N, window, ...)
            return windows.movedim(-1, 1)  # (n_windows, window, N, ...)

        # The first step of each sample, and with flatten_series its series, in the order
        # of the samples: window by window, and within a window series by series.
        self.start, self.series = torch.arange(0, n_steps - window + 1, stride), None
        if flatten_series:
            self.start, self.series = torch.cartesian_prod(self.start, torch.arange(self.n_series)).T
        super().__init__(
            samples(input_data), samples(concepts) if concepts is not None else None,
            annotations, **kwargs,
        )

    def _subset_rows(self, indices) -> None:
        """Subset the samples and keep ``start`` and ``series`` aligned with them."""
        super()._subset_rows(indices)
        self.start = self.start[indices]
        self.series = self.series[indices] if self.series is not None else None
