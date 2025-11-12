from typing import Union
import numpy as np

from torch_concepts.data.base import ConceptDataset

from ..base.splitter import Splitter

class RandomSplitter(Splitter):
    """Random splitting strategy for datasets.
    
    Randomly divides a dataset into train, validation, test, and optionally
    fine-tuning splits. Ensures reproducibility when a seed is provided.
    
    The splitting is done in the following order:
    1. Fine-tuning validation (if ftune_val_size > 0)
    2. Fine-tuning train (if ftune_size > 0)
    3. Test (if test_size > 0)
    4. Validation (if val_size > 0)
    5. Training (remaining samples)
    
    Example:
        >>> splitter = RandomSplitter(
        ...     val_size=0.1,
        ...     test_size=0.2,
        ...     ftune_size=0.05,
        ...     ftune_val_size=0.05
        ... )
        >>> splitter.split(dataset)
        >>> print(f"Train: {splitter.n_train}, Val: {splitter.n_val}")
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
                (default: 0.1)
            test_size: Size of test set. If float, represents fraction
                of dataset. If int, represents absolute number of samples.
                (default: 0.2)
            ftune_size: Size of fine-tuning set. If float, represents fraction
                of dataset. If int, represents absolute number of samples.
                (default: 0.0)
            ftune_val_size: Size of fine-tuning validation set. If float,
                represents fraction of dataset. If int, represents absolute
                number of samples. (default: 0.0)
            seed: Random seed for reproducibility. If None, splits will be
                non-deterministic. (default: None)
        """
        super().__init__()
        self.val_size = val_size
        self.test_size = test_size
        self.ftune_size = ftune_size
        self.ftune_val_size = ftune_val_size

    def _resolve_size(self, size: Union[int, float], n_samples: int) -> int:
        """Convert size specification to absolute number of samples.
        Args:
            size: Either an integer (absolute count) or float (fraction).
            n_samples: Total number of samples in dataset.
        Returns:
            Absolute number of samples.
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
        Args:
            dataset: The dataset to split.
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
    