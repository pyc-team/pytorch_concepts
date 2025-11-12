from abc import ABC, abstractmethod

from torch_concepts.data.base import ConceptDataset

class Splitter(ABC):
    """Abstract base class for dataset splitting strategies.
    
    Splitters divide a dataset into train, validation, test, and optionally
    fine-tuning splits. They maintain reproducibility through random seeds
    and can handle both absolute (int) and relative (float) split sizes.
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
