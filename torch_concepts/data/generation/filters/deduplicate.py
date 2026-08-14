from torch_concepts import Annotations
from torch_concepts.data.generation.base.filter_generator import FilterGenerator


class DeduplicateConcepts(FilterGenerator):
    """Remove duplicate generated concept names while preserving order."""

    def filter(self, concepts: list[str]) -> list[str]:
        return list(dict.fromkeys(concepts))

    def filter_annotations(self, concepts: Annotations) -> Annotations:
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
        return super().filter_annotations(concepts)
