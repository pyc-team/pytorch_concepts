from abc import ABC, abstractmethod

from torch_concepts import Annotations


class FilterGenerator(ABC):
    """Filter generated concept definitions stored in an Annotations object."""

    @abstractmethod
    def filter(self, concepts: Annotations) -> Annotations:
        """Return a removed/reordered subset preserving concept definitions."""
