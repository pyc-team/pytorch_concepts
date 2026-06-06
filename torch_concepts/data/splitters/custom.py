"""Custom data splitting via user-provided split functions.

This module provides CustomSplitter, which builds train/val/test splits from
caller-supplied validation and test splitting functions, optionally excluding
the test indices from the validation split.
"""

from typing import Callable, Mapping, Optional

from ..base.dataset import ConceptDataset
from ..base.splitter import Splitter


class CustomSplitter(Splitter):
    """Splitter built from custom validation and test splitting functions.

    The test split is computed first, then the validation split (optionally
    masking out the test indices so they cannot leak into validation). The
    remaining samples returned by ``val_split_fn`` form the training set.

    Each split function is expected to take the dataset as its first positional
    argument and return a ``(train_idxs, split_idxs)`` tuple of index lists:

    - ``test_split_fn(dataset, **test_kwargs) -> (_, test_idxs)``
      (the first element is ignored)
    - ``val_split_fn(dataset, **val_kwargs) -> (train_idxs, val_idxs)``

    When ``mask_test_indices_in_val`` is True and a test split exists, the test
    indices are passed to ``val_split_fn`` as a ``mask`` keyword argument so the
    function can exclude them.

    Args:
        val_split_fn (Callable, optional): Function returning
            ``(train_idxs, val_idxs)``. If None, the validation set is empty and
            every index not assigned to the test set becomes training.
            Defaults to None.
        test_split_fn (Callable, optional): Function returning
            ``(_, test_idxs)``. If None, the test set is empty. Defaults to None.
        val_kwargs (Mapping, optional): Extra keyword arguments forwarded to
            ``val_split_fn``. Defaults to None (empty).
        test_kwargs (Mapping, optional): Extra keyword arguments forwarded to
            ``test_split_fn``. Defaults to None (empty).
        mask_test_indices_in_val (bool, optional): If True, pass the test indices
            to ``val_split_fn`` as ``mask=test_idxs`` so they are excluded from
            the validation split. Defaults to True.

    Example:
        >>> def by_threshold(dataset, frac=0.2, mask=None):
        ...     n = len(dataset)
        ...     pool = [i for i in range(n) if mask is None or i not in set(mask)]
        ...     cut = int((1 - frac) * len(pool))
        ...     return pool[:cut], pool[cut:]
        >>>
        >>> splitter = CustomSplitter(
        ...     val_split_fn=by_threshold, val_kwargs={'frac': 0.1},
        ...     test_split_fn=by_threshold, test_kwargs={'frac': 0.2},
        ... )
        >>> splitter.fit(dataset)
        >>> print(splitter.train_len, splitter.val_len, splitter.test_len)
    """

    def __init__(
        self,
        val_split_fn: Optional[Callable] = None,
        test_split_fn: Optional[Callable] = None,
        val_kwargs: Optional[Mapping] = None,
        test_kwargs: Optional[Mapping] = None,
        mask_test_indices_in_val: bool = True,
    ):
        super().__init__()
        self.val_split_fn = val_split_fn
        self.test_split_fn = test_split_fn
        self.val_kwargs = dict(val_kwargs) if val_kwargs else dict()
        self.test_kwargs = dict(test_kwargs) if test_kwargs else dict()
        self.mask_test_indices_in_val = mask_test_indices_in_val

    @property
    def val_policy(self) -> Optional[str]:
        """Name of the validation split function, if callable."""
        return self.val_split_fn.__name__ if callable(self.val_split_fn) else None

    @property
    def test_policy(self) -> Optional[str]:
        """Name of the test split function, if callable."""
        return self.test_split_fn.__name__ if callable(self.test_split_fn) else None

    def fit(self, dataset: ConceptDataset) -> None:
        """Split the dataset using the custom split functions.

        Args:
            dataset: The ConceptDataset to split.
        """
        # Test split first.
        if self.test_split_fn is not None:
            _, test_idxs = self.test_split_fn(dataset, **self.test_kwargs)
        else:
            test_idxs = []

        # Validation split, optionally masking out the test indices.
        val_kwargs = self.val_kwargs
        if self.mask_test_indices_in_val and len(test_idxs):
            val_kwargs = dict(**self.val_kwargs, mask=test_idxs)

        if self.val_split_fn is not None:
            train_idxs, val_idxs = self.val_split_fn(dataset, **val_kwargs)
        else:
            # No validation function: everything not in test becomes training.
            test_set = {int(i) for i in test_idxs}
            train_idxs = [i for i in range(len(dataset)) if i not in test_set]
            val_idxs = []

        self.set_indices(train=train_idxs, val=val_idxs, test=test_idxs)
        self._fitted = True

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"val_policy={self.val_policy}, "
            f"test_policy={self.test_policy}, "
            f"train_size={self.train_len}, "
            f"val_size={self.val_len}, "
            f"test_size={self.test_len})"
        )
