from abc import ABC, abstractmethod
from typing import Any

# TODO: Use AnnotatedTensor instead of Tensor.
from torch.utils.data import Dataset

from torch_concepts import Annotations


class Annotator(ABC):
    """Base class for assigning concepts to dataset samples.

    An annotator maps:
        dataset + AxisAnnotation -> Tensor
    """

    @abstractmethod
    def annotate(
        self,
        dataset: Dataset,
        concepts: Annotations,
        **kwargs: Any,
    ) -> AnnotatedTensor:
        pass
