"""Random data splitting for train/validation/test splits.

This module provides RandomSplitter for randomly dividing datasets into
standard train/val/test splits.
"""

from typing import Union
import numpy as np

from ..utils import resolve_size
from ..base.dataset import ConceptDataset
from ..base.splitter import Splitter

class RandomSplitter(Splitter):
    """Random splitting strategy for datasets.
    
    Randomly divides a dataset into train, validation, and test splits.
    Ensures reproducibility when numpy's random seed is set externally
    before calling fit().
    
    The splitting is done in the following order:
    1. Test (if test_size > 0)
    2. Validation (if val_size > 0)
    3. Training (remaining samples)
    
    Args:
        val_size (Union[int, float], optional): Size of validation set.
            If float, represents fraction of dataset. If int, represents
            absolute number of samples. Defaults to 0.1.
        test_size (Union[int, float], optional): Size of test set.
            If float, represents fraction of dataset. If int, represents
            absolute number of samples. Defaults to 0.2.
            
    Example:
        >>> # 70% train, 10% val, 20% test
        >>> splitter = RandomSplitter(val_size=0.1, test_size=0.2)
        >>> splitter.fit(dataset)
        >>> print(f"Train: {splitter.train_len}, Val: {splitter.val_len}, Test: {splitter.test_len}")
        Train: 700, Val: 100, Test: 200
    """

    def __init__(
        self,
        val_size: Union[int, float] = 0.1,
        test_size: Union[int, float] = 0.2,
    ):
        """Initialize the RandomSplitter.
        
        Args:
            val_size: Size of validation set. If float, represents fraction
                of dataset. If int, represents absolute number of samples.
                Defaults to 0.1.
            test_size: Size of test set. If float, represents fraction
                of dataset. If int, represents absolute number of samples.
                Defaults to 0.2.
        """
        super().__init__()
        self.val_size = val_size
        self.test_size = test_size

    def fit(self, dataset: ConceptDataset) -> None:
        """Randomly split the dataset into train/val/test sets.
        
        Creates a random permutation of dataset indices and divides them
        according to specified split sizes. Sets the _fitted flag to True
        upon completion.
        
        Args:
            dataset: The ConceptDataset to split.
            
        Raises:
            ValueError: If split sizes exceed dataset size.
        """
        n_samples = len(dataset)
        
        # Resolve all sizes to absolute numbers
        n_val = resolve_size(self.val_size, n_samples)
        n_test = resolve_size(self.test_size, n_samples)
        
        # Validate that splits don't exceed dataset size
        total_split = n_val + n_test
        if total_split > n_samples:
            raise ValueError(
                f"Split sizes sum to {total_split} but dataset has only "
                f"{n_samples} samples. "
                f"(val={n_val}, test={n_test})"
            )
        
        n_train = n_samples - total_split
        
        # Create random permutation of indices
        indices = np.random.permutation(n_samples)
        
        # Split indices in order: test, val, train
        test_idxs = indices[:n_test]
        val_idxs = indices[n_test:n_test + n_val]
        train_idxs = indices[n_test + n_val:]
        
        # Store indices
        self.set_indices(
            train=train_idxs.tolist(),
            val=val_idxs.tolist(),
            test=test_idxs.tolist()
        )

        self._fitted = True
        
        # Sanity check
        assert len(self.train_idxs) == n_train, \
            f"Expected {n_train} training samples, got {len(self.train_idxs)}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"train_size={self.train_len}, "
            f"val_size={self.val_len}, "
            f"test_size={self.test_len})"
        )
