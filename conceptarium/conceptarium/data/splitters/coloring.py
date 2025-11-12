"""Data splitting utilities for train/validation/test splits."""
import json
from abc import ABC, abstractmethod
import os
from typing import Union
import numpy as np

from torch_concepts.data.base import ConceptDataset

from ..base.splitter import Splitter

class ColoringSplitter(Splitter):
    """ Coloring-based splitting strategy for datasets.
    
    It divides a dataset into train, validation, test, and optionally
    fine-tuning splits considering the coloring scheme used in the dataset.
    Specifically, it ensures that the training set and the validation set contains samples
    colored with the 'training_mode', while the test set and the fine_tune sets contains samples
    colored with the 'test_mode'.
    NOTE: it assumes the dataset is already shuffled.
    
    Example:
        >>> splitter = ColoringSplitter(
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
        root: str,
        seed: int = None,
        val_size: Union[int, float] = 0.1,
        test_size: Union[int, float] = 0.2,
        ftune_size: Union[int, float] = 0.0,
        ftune_val_size: Union[int, float] = 0.0
    ):
        """Initialize the ColoringSplitter.
        
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
            coloring_mode_path: Path to the JSON file containing the coloring mode
                for each sample in the dataset. (default: None)
            seed: Random seed for reproducibility. If None, splits will be
                non-deterministic. (default: None)
        """
        super().__init__()
        self.root = root
        self.seed = seed
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
        """Split the dataset into train/val/test/ftune sets based on percentages.
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


        # load coloring mode
        # search for the file f"coloring_mode_seed_{self.seed}.json"
        coloring_mode_path = os.path.join(self.root, f"coloring_mode_seed_{self.seed}.json")
        if not os.path.exists(coloring_mode_path):
            raise ValueError(f"No coloring mode file found for the seed {self.seed}.")
        with open(coloring_mode_path, "r") as f:
            coloring_mode = json.load(f)


        indices = np.arange(len(coloring_mode))
        # get indices for training_mode and test_mode
        train_indices = [int(i) for i in indices if coloring_mode[i] == 'training']
        test_indices  = [int(i) for i in indices if coloring_mode[i] == 'test']

        try:
            val_idxs = np.array(train_indices[:n_val])
            train_idxs = np.array(train_indices[n_val:])
        except ValueError:
            raise ValueError(f"Not enough samples colored with training mode for requested train+val size ({n_train + n_val}).")

        try:
            ftune_val_idxs = np.array(test_indices[:n_ftune_val])
            ftune_idxs = np.array(test_indices[n_ftune_val:n_ftune_val + n_ftune])
            test_idxs = np.array(test_indices[n_ftune_val + n_ftune:])
        except ValueError:
            raise ValueError(f"Not enough samples colored with test mode for requested test size ({n_test}).")

        
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