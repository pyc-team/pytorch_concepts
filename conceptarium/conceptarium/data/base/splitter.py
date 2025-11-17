"""Abstract base class for dataset splitting strategies.

This module defines the Splitter interface for dividing datasets into
train/val/test splits with optional fine-tuning subsets. Splitters manage
indices and ensure reproducible splits through random seeds.
"""

from abc import ABC, abstractmethod

from torch_concepts.data.base import ConceptDataset

class Splitter(ABC):
    """Abstract base class for dataset splitting strategies.
    
    Splitters divide a ConceptDataset into train, validation, test, and optionally
    fine-tuning splits. They store indices for each split and provide properties
    to access split sizes and indices. All concrete splitter implementations
    should inherit from this class and implement the fit() method.
    
    Attributes:
        train_idxs (list): Training set indices.
        val_idxs (list): Validation set indices.
        test_idxs (list): Test set indices.
        ftune_idxs (list): Fine-tuning set indices (optional).
        ftune_val_idxs (list): Fine-tuning validation set indices (optional).
        
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
    def ftune_idxs(self):
        return self.__indices.get('ftune')

    @property
    def ftune_val_idxs(self):
        return self.__indices.get('ftune_val')

    @property
    def train_len(self):
        return len(self.train_idxs) if self.train_idxs is not None else None

    @property
    def val_len(self):
        return len(self.val_idxs) if self.val_idxs is not None else None

    @property
    def test_len(self):
        return len(self.test_idxs) if self.test_idxs is not None else None
    
    @property
    def ftune_len(self):
        return len(self.ftune_idxs) if self.ftune_idxs is not None else None

    @property
    def ftune_val_len(self):
        return len(self.ftune_val_idxs) if self.ftune_val_idxs is not None else None

    def set_indices(self, train=None, val=None, test=None, ftune=None, ftune_val=None):
        if train is not None:
            self.__indices['train'] = train
        if val is not None:
            self.__indices['val'] = val
        if test is not None:
            self.__indices['test'] = test
        if ftune is not None:
            self.__indices['ftune'] = ftune
        if ftune_val is not None:
            self.__indices['ftune_val'] = ftune_val

    def reset(self):
        self.__indices = dict(train=None, val=None, test=None, ftune=None, ftune_val=None)
        self._fitted = False

    @abstractmethod
    def fit(self, dataset: ConceptDataset):
        """Split the dataset into train/val/test sets.
        
        This method should set the following attributes:
        - self.train_idxs: List of training indices
        - self.val_idxs: List of validation indices
        - self.test_idxs: List of test indices
        - self.ftune_idxs: (Optional) List of fine-tuning indices
        - self.ftune_val_idxs: (Optional) List of fine-tuning validation indices
        
        Args:
            dataset: The dataset to split.
        """
        raise NotImplementedError
        
    def split(self, dataset: ConceptDataset) -> None:
        if self.fitted:
            return self.indices
        else:
            return self.fit(dataset)
