"""Inductive splitting of time series: whole series go to train, validation or test.

This module provides InductiveSplitter, which draws the series of a
DynamicConceptDataset at random and gives each split all the windows of its series.
"""

import torch

from ..base.dataset import ConceptDataset
from .random import RandomSplitter


class InductiveSplitter(RandomSplitter):
    """Random split by series, for the windows of a :class:`DynamicConceptDataset`.

    Draws the series at random and gives each split all the windows of its
    series, so the test set holds series that the model has not seen.

    Args:
        val_size (Union[int, float], optional): Validation set size, in series.
            Defaults to 0.1.
        test_size (Union[int, float], optional): Test set size, in series.
            Defaults to 0.2.
        seed (int, optional): Seed of the draw. Defaults to None.

    Example:
        >>> import torch
        >>> from torch_concepts.data.base import DynamicConceptDataset
        >>> from torch_concepts.data.splitters import InductiveSplitter
        >>> dataset = DynamicConceptDataset(torch.randn(10, 5, 2), window=4, stride=1)
        >>> splitter = InductiveSplitter(val_size=1, test_size=1, seed=0)
        >>> splitter.fit(dataset)  # 5 series of 7 windows each
        >>> splitter.train_len, splitter.val_len, splitter.test_len
        (21, 7, 7)
        >>> [dataset.series[splitter.indices[s]].unique().tolist() for s in ("train", "val", "test")]
        [[0, 1, 3], [4], [2]]

    Breakdown of the splits series by series, for the example above::

        test_series: asked (1), kept (1)
        val_series: asked (1), kept (1)
        train_series: rest (3), kept (3)

        step      0  1  2  3  4  5  6  7  8  9
        series 0  T  T  T  T  T  T  T  T  T  T    train, 7 windows
        series 1  T  T  T  T  T  T  T  T  T  T    train, 7 windows
        series 2  E  E  E  E  E  E  E  E  E  E    test, 7 windows
        series 3  T  T  T  T  T  T  T  T  T  T    train, 7 windows
        series 4  V  V  V  V  V  V  V  V  V  V    val, 7 windows
    """

    def fit(self, dataset: ConceptDataset) -> None:
        # --- Checks ---
        # One series per sample: with flatten_series=False a sample holds every series.
        if getattr(dataset, "series", None) is None:
            raise ValueError(
                "InductiveSplitter splits by series: it needs a DynamicConceptDataset "
                "with flatten_series=True, whose samples record their series."
            )

        # --- Split ---
        series = dataset.series.unique()
        super().fit(series)  # draws the series, with the sizes and the seed of RandomSplitter
        self.set_indices(**{
            split: torch.where(torch.isin(dataset.series, series[idxs]))[0].tolist()
            for split, idxs in self.indices.items()
        })

        # --- Checks ---
        # A split asked for got at least one series: a fraction of few series rounds to 0.
        empty = [s for s in ("val", "test") if getattr(self, f"{s}_size") and not self.indices[s]]
        if empty:
            raise ValueError(
                f"The {' and '.join(empty)} split holds no series: {len(series)} series "
                "are too few for the sizes asked, so give the sizes as numbers of series."
            )
