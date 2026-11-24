"""Transductive splitting of time series: train on the first steps, test on the last.

This module provides TransductiveSplitter, which cuts the windows of a
DynamicConceptDataset in time, so that every series is in every split.
"""

from typing import Union

import torch

from ..base.dataset import ConceptDataset
from ..base.splitter import Splitter
from ..utils import resolve_size


class TransductiveSplitter(Splitter):
    """Split in time, for the windows of a :class:`DynamicConceptDataset`.

    Orders the windows by their first step and gives the first ones to train, the
    next ones to validation and the last ones to test, so every series is in every
    split and the test set measures the forecast of later steps. Each split holds
    the steps from its first window to the first window of the next split. A window
    goes to the split that holds all its steps, and a window across a cut goes to
    no split, so no step is in two splits. A split shorter than a window holds
    none, which raises an error.

    Args:
        val_size (Union[int, float], optional): Validation set size, in window
            starts. Defaults to 0.1.
        test_size (Union[int, float], optional): Test set size, in window starts.
            Defaults to 0.2.

    Example:
        >>> import torch
        >>> from torch_concepts.data.base import DynamicConceptDataset
        >>> from torch_concepts.data.splitters import TransductiveSplitter
        >>> dataset = DynamicConceptDataset(torch.randn(20, 2, 1), window=4, stride=1)
        >>> splitter = TransductiveSplitter(val_size=4, test_size=2)
        >>> splitter.fit(dataset)  # windows start at steps 0, 1, ..., 16
        >>> [dataset.start[splitter.indices[s]].unique().tolist() for s in ("train", "val", "test")]
        [[0, 1, 2, 3, 4, 5, 6, 7], [11], [15, 16]]
    """

    def __init__(
        self,
        val_size: Union[int, float] = 0.1,
        test_size: Union[int, float] = 0.2,
    ):
        super().__init__()
        self.val_size = val_size
        self.test_size = test_size

    def fit(self, dataset: ConceptDataset) -> None:
        start, window = dataset.start, dataset.window
        starts = start.unique()  # sorted
        n_test = resolve_size(self.test_size, len(starts))
        n_val = resolve_size(self.val_size, len(starts))
        n_train = len(starts) - n_val - n_test

        # --- Checks ---
        # Some window starts are left for training.
        if n_train < 1:
            raise ValueError(
                f"val_size and test_size take all the {len(starts)} window starts, "
                "so none is left for training."
            )

        # --- Split ---
        edges = torch.cat([starts, starts[-1:] + window])  # and the end of the last window
        bounds = edges[[0, n_train, n_train + n_val, len(starts)]].tolist()
        splits = {
            split: torch.where((start >= lo) & (start + window <= hi))[0].tolist()
            for split, lo, hi in zip(("train", "val", "test"), bounds, bounds[1:])
        }

        # --- Checks ---
        # Each split asked for holds a window: in one shorter than a window, all cross its cut.
        empty = [s for s, n in zip(splits, (n_train, n_val, n_test)) if n and not splits[s]]
        if empty:
            raise ValueError(
                f"No window of {window} steps fits in the {' and '.join(empty)} split, which is "
                "shorter than a window: use shorter windows, longer series or a larger split."
            )

        self.set_indices(**splits)
        self._fitted = True

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"train_size={self.train_len}, "
            f"val_size={self.val_len}, "
            f"test_size={self.test_len})"
        )
