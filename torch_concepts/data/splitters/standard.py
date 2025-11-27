"""Standard data splitting for train/validation/test splits.

This module provides StandardSplitter for dividing datasets into
standard train/val/test splits provided by the dataset authors.
"""

from typing import Union
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

from ..utils import resolve_size
from ..base.dataset import ConceptDataset
from ..base.splitter import Splitter

class StandardSplitter(Splitter):
    """Standard splitting strategy for datasets.
    
    Divides a dataset into train, validation, and test splits based on
    standard splits provided by the dataset authors.
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
        >>> splitter = StandardSplitter(val_size=0.1, test_size=0.2)
        >>> splitter.fit(dataset)
        >>> print(f"Train: {splitter.train_len}, Val: {splitter.val_len}, Test: {splitter.test_len}")
        Train: 700, Val: 100, Test: 200
    """

    def __init__(
        self
    ):
        """Initialize the StandardSplitter.
        
        Args:
            val_size: Size of validation set. If float, represents fraction
                of dataset. If int, represents absolute number of samples.
                Defaults to 0.1.
            test_size: Size of test set. If float, represents fraction
                of dataset. If int, represents absolute number of samples.
                Defaults to 0.2.
        """
        super().__init__()

    def fit(self, dataset: ConceptDataset) -> None:
        """Split the dataset into train/val/test sets based on standard splits.
        
        Args:
            dataset: The ConceptDataset to split.
            
        Raises:
            ValueError: If the dataset does not provide standard splits.
        """

        # Load standard splits from dataset if available
        if any("split_mapping" in path for path in dataset.processed_paths):
            split_series = pd.read_hdf(
                next(path for path in dataset.processed_paths if "split_mapping" in path), key="split_mapping"
            )
            train_idxs = split_series[split_series == "train"].index.tolist()
            val_idxs = split_series[split_series == "val"].index.tolist()
            test_idxs = split_series[split_series == "test"].index.tolist()
            
            # Store indices
            self.set_indices(
                train=train_idxs,
                val=val_idxs,
                test=test_idxs
            )

            self._fitted = True

            logger.info(f"Attention StandardSplitter uses predefined splits provided by the dataset authors."
                        f"Train size: {self.train_len}, "
                        f"Val size: {self.val_len}, "
                        f"Test size: {self.test_len}")
        else:
            raise ValueError("Dataset does not provide standard splits.")


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"train_size={self.train_len}, "
            f"val_size={self.val_len}, "
            f"test_size={self.test_len})"
        )
