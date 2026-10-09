from torch_concepts import Annotations
from torch_concepts.data.generation.base.filter_generator import FilterGenerator


class DeduplicateConcepts(FilterGenerator):
    """Deduplicate generated concept definitions, preserving order and metadata.

    Examples
    --------
    >>> from torch_concepts import Annotations
    >>> concepts = Annotations(
    ...     labels=["color", "shape", "color"],
    ...     states=[["red", "blue"], ["circle", "square"], ["red", "blue"]],
    ...     types=["categorical", "categorical", "categorical"],
    ... )
    >>> filtered = DeduplicateConcepts().filter(concepts)
    >>> filtered.labels
    ['color', 'shape']
    >>> filtered.states
    [['red', 'blue'], ['circle', 'square']]
    """

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
