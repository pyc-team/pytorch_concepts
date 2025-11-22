"""Abstract base class for dataset splitting strategies.

This module defines the Splitter interface for dividing datasets into
train/val/test splits. Splitters manage indices and ensure reproducible
splits through random seeds.
"""

from abc import ABC, abstractmethod

from .dataset import ConceptDataset

class Splitter(ABC):
    """Abstract base class for dataset splitting strategies.
    
    Splitters divide a ConceptDataset into train, validation, and test splits.
    They store indices for each split and provide properties to access split
    sizes and indices. All concrete splitter implementations should inherit
    from this class and implement the fit() method.
    
    Attributes:
        train_idxs (list): Training set indices.
        val_idxs (list): Validation set indices.
        test_idxs (list): Test set indices.
        
    Example:
        >>> class CustomSplitter(Splitter):
        ...     def fit(self, dataset):
        ...         n = len(dataset)
        ...         self.set_indices(
        ...             train=list(range(int(0.7*n))),
        ...             val=list(range(int(0.7*n), int(0.9*n))),
        ...             test=list(range(int(0.9*n), n))
        ...         )
        ...         self._fitted = True
        >>> 
        >>> splitter = CustomSplitter()
        >>> splitter.fit(my_dataset)
        >>> print(f"Train: {splitter.train_len}, Val: {splitter.val_len}")
    """

    def __init__(self):
        self.__indices = dict()
        self._fitted = False
        self.reset()

    @property
    def indices(self):
        return self.__indices

    @property
    def fitted(self):
        return self._fitted

    @property
    def train_idxs(self):
        return self.__indices.get('train')

    @property
    def val_idxs(self):
        return self.__indices.get('val')

    @property
    def test_idxs(self):
        return self.__indices.get('test')

    @property
    def train_len(self):
        return len(self.train_idxs) if self.train_idxs is not None else None

    @property
    def val_len(self):
        return len(self.val_idxs) if self.val_idxs is not None else None

    @property
    def test_len(self):
        return len(self.test_idxs) if self.test_idxs is not None else None

    def set_indices(self, train=None, val=None, test=None):
        if train is not None:
            self.__indices['train'] = train
        if val is not None:
            self.__indices['val'] = val
        if test is not None:
            self.__indices['test'] = test

    def reset(self):
        self.__indices = dict(train=None, val=None, test=None)
        self._fitted = False

    @abstractmethod
    def fit(self, dataset: ConceptDataset):
        """Split the dataset into train/val/test sets.
        
        This method should set the following attributes:
        - self.train_idxs: List of training indices
        - self.val_idxs: List of validation indices
        - self.test_idxs: List of test indices
        
        Args:
            dataset: The dataset to split.
        """
        raise NotImplementedError
        
    def split(self, dataset: ConceptDataset) -> None:
        if self.fitted:
            return self.indices
        else:
            return self.fit(dataset)
