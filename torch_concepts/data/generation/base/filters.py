from abc import ABC, abstractmethod
from collections import Counter
from torch import Tensor

from torch_concepts import Annotations


class FilterGenerator(ABC):
    """Filter generated concept names before they are annotated."""

    @abstractmethod
    def filter(self, concepts: list[str]) -> list[str]:
        """Return the generated concept names that should be retained."""

    def filter_annotations(self, concepts: Annotations) -> Annotations:
        """Apply the string filter while preserving concept-axis information."""
        filtered_labels = self.filter(list(concepts.labels))
        if not isinstance(filtered_labels, list) or not all(
            isinstance(label, str) for label in filtered_labels
        ):
            raise TypeError(
                "GeneratorFilter.filter must return a list of strings."
            )

        if Counter(filtered_labels) - Counter(concepts.labels):
            raise ValueError(
                "Generator filters may only remove or reorder generated "
                "concept names."
            )
        return concepts.subset(filtered_labels)


class FilterAnnotator(ABC):
    """Filter out concepts from individual samples.

    Implementations must preserve the input tensor shape and represent filtered
    sample-concept entries with ``NaN``, meaning that the concept is absent
    from that sample.
    """

    @abstractmethod
    def filter(self, scores: Tensor, concepts: Annotations) -> Tensor:
        """Return scores with selected sample-concept entries filtered out."""
