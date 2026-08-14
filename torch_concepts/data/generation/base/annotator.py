from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset

from torch_concepts import Annotations
from torch_concepts.tensor import AnnotatedTensor


class Annotator(ABC):
    """Base class for assigning concepts to dataset samples.

    An annotator maps:
        dataset + Annotations -> AnnotatedTensor
    """

    @abstractmethod
    def annotate(
        self,
        dataset: Dataset,
        concepts: Annotations,
        **kwargs: Any,
    ) -> AnnotatedTensor:
        pass
