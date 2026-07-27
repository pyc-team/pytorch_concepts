from abc import ABC, abstractmethod
from collections import Counter

from torch_concepts import Annotations


class GeneratorFilter(ABC):
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
