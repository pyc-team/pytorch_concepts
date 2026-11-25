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
    none, which raises an error: each split before the last needs at least
    ``window / stride`` window starts. The cut drops the last ``window - stride``
    steps of train and validation, but not of test, so test holds a little more
    than its share of the steps.

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

    Breakdown of the splits step by step, for the example above::

        test_windows: asked (2), kept (2)
        val_windows: asked (4), kept (1)
        train_windows: rest (11), kept (8)

        step   0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
        split  T  T  T  T  T  T  T  T  T  T  T |V  V  V  V |E  E  E  E  E

            0  T  T  T  T                                                   train
            1     T  T  T  T                                                train
            2        T  T  T  T                                             train
            3           T  T  T  T                                          train
            4              T  T  T  T                                       train
            5                 T  T  T  T                                    train
            6                    T  T  T  T                                 train
            7                       T  T  T  T                              train
            8                          x  x  x  x                           dropped
            9                             x  x  x  x                        dropped
            10                               x  x  x  x                    dropped
            11                                  V  V  V  V                 val
            12                                     x  x  x  x              dropped
            13                                        x  x  x  x           dropped
            14                                           x  x  x  x        dropped
            15                                              E  E  E  E     test
            16                                                 E  E  E  E  test
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
        # A split asked for gets a window start: a fraction of few starts rounds to 0.
        empty = [s for s, size, n in (("val", self.val_size, n_val), ("test", self.test_size, n_test)) if size and not n]
        if empty:
            raise ValueError(
                f"The {' and '.join(empty)} split gets none of the {len(starts)} window "
                "starts: use shorter windows, longer series or a larger split."
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
