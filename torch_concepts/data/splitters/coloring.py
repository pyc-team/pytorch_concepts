"""Coloring-based data splitting for distribution shift experiments.

This module provides ColoringSplitter which divides datasets based on
pre-computed coloring schemes. Useful for controlled distribution shift
experiments where training and test sets should have different characteristics.
"""

import json
import os
from typing import Union
import numpy as np

from ..utils import resolve_size
from ..base.dataset import ConceptDataset
from ..base.splitter import Splitter

class ColoringSplitter(Splitter):
    """Coloring-based splitting strategy for distribution shift experiments.
    
    Divides a dataset into train/val/test splits based on a pre-computed
    coloring scheme stored in a JSON file. This ensures that training and
    validation sets contain samples with 'training' coloring, while test
    sets contain samples with 'test' coloring.
    
    This is useful for:
    - Out-of-distribution (OOD) evaluation
    - Domain adaptation experiments
    - Controlled distribution shift scenarios
    
    Note: Assumes the dataset is already shuffled and that a coloring file
    exists at {root}/coloring_mode_seed_{seed}.json
    
    Args:
        root (str): Root directory containing the coloring mode JSON file.
        seed (int, optional): Random seed used to identify the coloring file.
            Defaults to None.
        val_size (Union[int, float], optional): Validation set size (from 'training'
            colored samples). Defaults to 0.1.
        test_size (Union[int, float], optional): Test set size (from 'test'
            colored samples). Defaults to 0.2.
            
    Example:
        >>> # Create a coloring file first: coloring_mode_seed_42.json
        >>> # Format: {"0": "training", "1": "training", "2": "test", ...}
        >>> 
        >>> splitter = ColoringSplitter(
        ...     root='data/my_dataset',
        ...     seed=42,
        ...     val_size=0.1,
        ...     test_size=0.2
        ... )
        >>> splitter.fit(dataset)
        >>> # Train/val from 'training' samples, test from 'test' samples
    """

    def __init__(
        self,
        root: str,
        seed: int = None,
        val_size: Union[int, float] = 0.1,
        test_size: Union[int, float] = 0.2
    ):
        """Initialize the ColoringSplitter.
        
        Args:
            root (str): Root directory containing coloring mode JSON file.
            seed (int, optional): Random seed to identify coloring file.
                File expected at {root}/coloring_mode_seed_{seed}.json.
                Defaults to None.
            val_size: Validation set size (from 'training' samples). 
                If float, represents fraction. If int, absolute count. Defaults to 0.1.
            test_size: Test set size (from 'test' samples).
                If float, represents fraction. If int, absolute count. Defaults to 0.2.
        """
        super().__init__()
        self.root = root
        self.seed = seed
        self.val_size = val_size
        self.test_size = test_size

    def fit(self, dataset: ConceptDataset) -> None:
        """Split dataset based on coloring scheme from JSON file.
        
        Loads the coloring mode file and divides indices into 'training' and
        'test' groups. Then allocates samples from each group to the appropriate
        splits (train/val from 'training', test from 'test').
        
        Args:
            dataset: The ConceptDataset to split.
            
        Raises:
            ValueError: If coloring file doesn't exist, or if there aren't enough
                samples of a particular coloring mode to satisfy the requested splits.
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
            test_idxs = np.array(test_indices[:n_test])
        except ValueError:
            raise ValueError(f"Not enough samples colored with test mode for requested test size ({n_test}).")

        
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