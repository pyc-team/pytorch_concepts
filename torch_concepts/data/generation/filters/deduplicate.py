from torch_concepts import Annotations
from torch_concepts.data.generation.base.filter_generator import FilterGenerator


class DeduplicateConcepts(FilterGenerator):
    """Deduplicate a generated concept axis, preserving order and metadata."""

    def filter(self, concepts: Annotations) -> Annotations:
        """Keep one occurrence per label, rejecting incompatible definitions."""
        definitions: dict[str, tuple[list[str], int, str]] = {}
        for index, label in enumerate(concepts.labels):
            definition = (
                list(concepts.states[index]),
                concepts.cardinalities[index],
                concepts.types[index],
            )
            if label in definitions and definitions[label] != definition:
                raise ValueError(
                    f"Concept {label!r} has incompatible definitions: "
                    f"{definition} does not match {definitions[label]}."
                )
            definitions[label] = definition
        return concepts.subset(list(definitions))
