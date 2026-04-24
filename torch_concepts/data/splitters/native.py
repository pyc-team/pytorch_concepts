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

class NativeSplitter(Splitter):
    """Native splitting strategy for datasets.
    
    Divides a dataset into train, validation, and test splits based on
    native splits provided by the dataset authors.
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
        >>> splitter = NativeSplitter(val_size=0.1, test_size=0.2)
        >>> splitter.fit(dataset)
        >>> print(f"Train: {splitter.train_len}, Val: {splitter.val_len}, Test: {splitter.test_len}")
        Train: 700, Val: 100, Test: 200
    """

    def __init__(
        self
    ):
        """Initialize the NativeSplitter.
        
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
        """Split the dataset into train/val/test sets based on native splits.
        
        Args:
            dataset: The ConceptDataset to split.
            
        Raises:
            ValueError: If the dataset does not provide native splits.
        """

        # Load native splits from dataset if available
        if any("split_mapping" in path for path in dataset.processed_paths):
            split_series = pd.read_hdf(
                next(path for path in dataset.processed_paths if "split_mapping" in path), key="split_mapping"
            )
            
            # Normalize split labels to handle variations like "valid", "validation", "val", etc.
            split_lower = split_series.str.lower()
            train_idxs = split_series[split_lower.str.startswith("train")].index.tolist()
            val_idxs = split_series[split_lower.str.startswith("val")].index.tolist()
            test_idxs = split_series[split_lower.str.startswith("test")].index.tolist()
            
            # Store indices
            self.set_indices(
                train=train_idxs,
                val=val_idxs,
                test=test_idxs
            )

            self._fitted = True

            logger.info(f"NativeSplitter uses predefined splits native to the dataset."
                        f"Train size: {self.train_len}, "
                        f"Val size: {self.val_len}, "
                        f"Test size: {self.test_len}")
        else:
            raise ValueError("Dataset does not provide native splits.")


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"train_size={self.train_len}, "
            f"val_size={self.val_len}, "
            f"test_size={self.test_len})"
        )
