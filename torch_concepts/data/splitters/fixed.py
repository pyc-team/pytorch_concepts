"""Splitter with fixed, pre-computed train/val/test indices.

This module provides FixedIndicesSplitter, which uses caller-supplied index
sequences directly instead of computing a split from the dataset.
"""

from typing import Optional, Sequence

from ..base.dataset import ConceptDataset
from ..base.splitter import Splitter


class FixedIndicesSplitter(Splitter):
    """Splitter that uses fixed, pre-computed indices.

    The train/validation/test indices are provided at construction time and
    used as-is; :meth:`fit` is a no-op. Useful when the split is already known
    (e.g. loaded from a file, or shared across runs/methods for a paired
    comparison).

    Index sequences are copied into plain lists at construction, so mutating the
    caller's sequences afterwards does not affect the splitter. Any argument left
    as None stays unset.

    Args:
        train_idxs (sequence of int, optional): Training indices. Defaults to None.
        val_idxs (sequence of int, optional): Validation indices. Defaults to None.
        test_idxs (sequence of int, optional): Test indices. Defaults to None.

    Example:
        >>> splitter = FixedIndicesSplitter(
        ...     train_idxs=range(0, 70),
        ...     val_idxs=range(70, 90),
        ...     test_idxs=range(90, 100),
        ... )
        >>> splitter.fit(dataset)  # no-op; indices already set
        >>> print(splitter.train_len, splitter.val_len, splitter.test_len)
        70 20 10
    """

    def __init__(
        self,
        train_idxs: Optional[Sequence[int]] = None,
        val_idxs: Optional[Sequence[int]] = None,
        test_idxs: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.set_indices(
            train=list(train_idxs) if train_idxs is not None else None,
            val=list(val_idxs) if val_idxs is not None else None,
            test=list(test_idxs) if test_idxs is not None else None,
        )
        self._fitted = True

    def fit(self, dataset: ConceptDataset) -> None:
        """No-op: the indices are fixed at construction time."""
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"train_size={self.train_len}, "
            f"val_size={self.val_len}, "
            f"test_size={self.test_len})"
        )
