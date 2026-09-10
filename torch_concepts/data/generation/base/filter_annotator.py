from abc import ABC, abstractmethod

from torch_concepts.tensor import AnnotatedTensor


class FilterAnnotator(ABC):
    """Filter out concepts from individual samples.

    Implementations must preserve the input tensor shape and annotation
    metadata. Concrete filters define how selected entries are represented.
    """

    @abstractmethod
    def filter(self, scores: AnnotatedTensor) -> AnnotatedTensor:
        """Return scores with selected entries filtered out, preserving metadata."""
