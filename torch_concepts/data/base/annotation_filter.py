from abc import ABC, abstractmethod

from torch import Tensor

from torch_concepts import Annotations


class AnnotationFilter(ABC):
    """Filter out concepts from individual samples.

    Implementations must preserve the input tensor shape and represent filtered
    sample-concept entries with ``NaN``, meaning that the concept is absent
    from that sample.
    """

    @abstractmethod
    def filter(self, scores: Tensor, concepts: Annotations) -> Tensor:
        """Return scores with selected sample-concept entries filtered out."""
