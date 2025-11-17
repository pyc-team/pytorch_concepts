"""Random data splitting for train/validation/test splits.

This module provides RandomSplitter for randomly dividing datasets with
support for standard splits plus optional fine-tuning subsets.
"""

from typing import Union
import numpy as np

from torch_concepts.data.base import ConceptDataset

from ..base.splitter import Splitter

class RandomSplitter(Splitter):
    """Random splitting strategy for datasets.
    
    Randomly divides a dataset into train, validation, test, and optionally
    fine-tuning splits. Ensures reproducibility when numpy's random seed is set
    externally before calling fit().
    
    The splitting is done in the following order:
    1. Fine-tuning validation (if ftune_val_size > 0)
    2. Fine-tuning train (if ftune_size > 0)
    3. Test (if test_size > 0)
    4. Validation (if val_size > 0)
    5. Training (remaining samples)
    
    Args:
        val_size (Union[int, float], optional): Size of validation set.
            If float, represents fraction of dataset. If int, represents
            absolute number of samples. Defaults to 0.1.
        test_size (Union[int, float], optional): Size of test set.
            If float, represents fraction of dataset. If int, represents
            absolute number of samples. Defaults to 0.2.
        ftune_size (Union[int, float], optional): Size of fine-tuning set.
            If float, represents fraction of dataset. If int, represents
            absolute number of samples. Defaults to 0.0.
        ftune_val_size (Union[int, float], optional): Size of fine-tuning
            validation set. If float, represents fraction of dataset. If int,
            represents absolute number of samples. Defaults to 0.0.
            
    Example:
        >>> # 70% train, 10% val, 20% test
        >>> splitter = RandomSplitter(val_size=0.1, test_size=0.2)
        >>> splitter.fit(dataset)
        >>> print(f"Train: {splitter.train_len}, Val: {splitter.val_len}, Test: {splitter.test_len}")
        Train: 700, Val: 100, Test: 200
        
        >>> # With fine-tuning splits
        >>> splitter = RandomSplitter(
        ...     val_size=0.1,
        ...     test_size=0.2,
        ...     ftune_size=0.05,
        ...     ftune_val_size=0.05
        ... )
        >>> splitter.fit(dataset)
    """

    def __init__(
        self,
        val_size: Union[int, float] = 0.1,
        test_size: Union[int, float] = 0.2,
        ftune_size: Union[int, float] = 0.0,
        ftune_val_size: Union[int, float] = 0.0,
    ):
        """Initialize the RandomSplitter.
        
        Args:
            val_size: Size of validation set. If float, represents fraction
                of dataset. If int, represents absolute number of samples.
                Defaults to 0.1.
            test_size: Size of test set. If float, represents fraction
                of dataset. If int, represents absolute number of samples.
                Defaults to 0.2.
            ftune_size: Size of fine-tuning set. If float, represents fraction
                of dataset. If int, represents absolute number of samples.
                Defaults to 0.0.
            ftune_val_size: Size of fine-tuning validation set. If float,
                represents fraction of dataset. If int, represents absolute
                number of samples. Defaults to 0.0.
        """
        super().__init__()
        self.val_size = val_size
        self.test_size = test_size
        self.ftune_size = ftune_size
        self.ftune_val_size = ftune_val_size

    def _resolve_size(self, size: Union[int, float], n_samples: int) -> int:
        """Convert size specification to absolute number of samples.
        
        Args:
            size: Either an integer (absolute count) or float (fraction in [0, 1]).
            n_samples: Total number of samples in dataset.
            
        Returns:
            int: Absolute number of samples.
            
        Raises:
            ValueError: If fractional size is not in [0, 1] or absolute size is negative.
            TypeError: If size is neither int nor float.
        """
        if isinstance(size, float):
            if not 0.0 <= size <= 1.0:
                raise ValueError(f"Fractional size must be in [0, 1], got {size}")
            return int(size * n_samples)
        
        elif isinstance(size, int):
            if size < 0:
                raise ValueError(f"Absolute size must be non-negative, got {size}")
            return size
        
        else:
            raise TypeError(f"Size must be int or float, got {type(size).__name__}")

    def fit(self, dataset: ConceptDataset) -> None:
        """Randomly split the dataset into train/val/test/ftune sets.
        
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
        n_val = self._resolve_size(self.val_size, n_samples)
        n_test = self._resolve_size(self.test_size, n_samples)
        n_ftune = self._resolve_size(self.ftune_size, n_samples)
        n_ftune_val = self._resolve_size(self.ftune_val_size, n_samples)
        
        # Validate that splits don't exceed dataset size
        total_split = n_val + n_test + n_ftune + n_ftune_val
        if total_split > n_samples:
            raise ValueError(
                f"Split sizes sum to {total_split} but dataset has only "
                f"{n_samples} samples. "
                f"(val={n_val}, test={n_test}, ftune={n_ftune}, "
                f"ftune_val={n_ftune_val})"
            )
        
        n_train = n_samples - total_split
        
        # Create random permutation of indices
        indices = np.random.permutation(n_samples)
        
        # Split indices in order: ftune_val, ftune, test, val, train
        ftune_val_idxs = indices[:n_ftune_val]
        ftune_idxs = indices[n_ftune_val:n_ftune_val + n_ftune]
        test_idxs = indices[n_ftune_val + n_ftune:n_ftune_val + n_ftune + n_test]
        val_idxs = indices[n_ftune_val + n_ftune + n_test:n_ftune_val + n_ftune + n_test + n_val]
        train_idxs = indices[n_ftune_val + n_ftune + n_test + n_val:]
        
        # Store indices
        self.set_indices(
            train=train_idxs.tolist(),
            val=val_idxs.tolist(),
            test=test_idxs.tolist(),
            ftune=ftune_idxs.tolist(),
            ftune_val=ftune_val_idxs.tolist()
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
            f"test_size={self.test_len}, "
            f"ftune_size={self.ftune_len}, "
            f"ftune_val_size={self.ftune_val_len})"
        )
    