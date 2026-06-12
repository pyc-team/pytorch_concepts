from abc import ABC, abstractmethod
from typing import Any, List

from torch import Tensor
from torch.utils.data import Dataset

from torch_concepts.data.base.dataset import ConceptDataset


Concepts = List[Any] | Tensor


class Annotator(ABC):
    """Base class for assigning concepts to dataset samples.

    An annotator maps:
        dataset, concepts -> ConceptDataset
    """

    @abstractmethod
    def annotate(
        self,
        dataset: Dataset,
        concepts: Concepts,
        **kwargs: Any,
    ) -> ConceptDataset:
        pass