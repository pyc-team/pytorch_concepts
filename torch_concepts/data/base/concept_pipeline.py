from __future__ import annotations

from typing import Any, Callable, Literal, Sequence

from torch import Tensor
from torch.utils.data import Dataset

from torch_concepts import AxisAnnotation
from torch_concepts.data.base.annotator import Annotator
from torch_concepts.data.base.concept_generator import ConceptGenerator


# merged: merges all generated concepts, then sends them to all annotators
# cartesian: sends each generated concept to all annotators
# zip: sends each generated concept to the corresponding annotator, producing a one-to-one mapping of results.
# The mapping is determined by the order of generators and annotators in the pipeline configuration.
# Ex. routing='zip' with generators=[G1, G2] and annotators=[A1, A2] produces A1(G1) and A2(G2), while routing='cartesian' produces A1(G1), A1(G2), A2(G1), and A2(G2).
RoutingMode = Literal["merged", "cartesian", "zip"]


class UnionConceptFilter:
    """Merge concept axes in order while preserving categorical structure."""

    def __call__(self, concepts: dict[str, AxisAnnotation]) -> AxisAnnotation:
        labels: list[str] = []
        states: list[list[str]] = []
        cardinalities: list[int] = []
        metadata: dict[str, dict[str, Any]] | None = None
        definitions: dict[str, tuple[list[str], int]] = {}

        if any(axis.metadata is not None for axis in concepts.values()):
            metadata = {}

        for source_name, axis in concepts.items():
            for index, label in enumerate(axis.labels):
                state_names = list(axis.states[index])
                cardinality = axis.cardinalities[index]
                definition = (state_names, cardinality)

                if label in definitions:
                    if definitions[label] != definition:
                        previous_states, previous_cardinality = definitions[label]
                        raise ValueError(
                            f"Concept {label!r} has incompatible definitions while "
                            f"merging {source_name!r}: states/cardinality "
                            f"{state_names}/{cardinality} do not match "
                            f"{previous_states}/{previous_cardinality}."
                        )
                    if metadata is not None and axis.metadata is not None:
                        existing = metadata.setdefault(label, {})
                        for key, value in axis.metadata.get(label, {}).items():
                            existing.setdefault(key, value)
                    continue

                definitions[label] = definition
                labels.append(label)
                states.append(state_names)
                cardinalities.append(cardinality)
                if metadata is not None:
                    metadata[label] = dict(
                        axis.metadata.get(label, {}) if axis.metadata else {}
                    )

        return AxisAnnotation(
            labels=labels,
            states=states,
            cardinalities=cardinalities,
            metadata=metadata,
        )


class ConceptSupervisionPipeline:
    """Compose concept generation, annotation, filtering, and aggregation.

    Parameters
    ----------
    generators : ConceptGenerator or sequence of ConceptGenerator
        Concept generators to produce concept annotations from the dataset.
    annotators : Annotator or sequence of Annotator
        Annotators to produce concept values from the dataset and concept annotations.
    concept_filter : callable, optional
        Function to filter or merge concept annotations from multiple generators.
        If None, no filtering is applied. If routing='merged', a default filter is used to merge concept axes while preserving categorical structure.
    aggregator : callable, optional
        Function to aggregate the generated concept values into a single tensor.
        If None, no aggregation is performed.
    routing : {'merged', 'cartesian', 'zip'}, default='merged'
        Routing mode for combining generators and annotators:
        - 'merged': merges all generated concepts, then sends them to all annotators.
        - 'cartesian': sends each generated concept to all annotators.
        - 'zip': sends each generated concept to the corresponding annotator, producing a one-to-one mapping of results. The mapping is determined by the order of generators and annotators in the pipeline configuration.
    name : str, optional
        Name of the pipeline. If None, the class name is used.
    """

    def __init__(
        self,
        generators: ConceptGenerator | Sequence[ConceptGenerator],
        annotators: Annotator | Sequence[Annotator],
        concept_filter: Callable[[dict[str, AxisAnnotation]], AxisAnnotation] | None = None,
        aggregator: Callable[[dict[str, Tensor]], Tensor] | None = None,
        routing: RoutingMode = "merged",
        name: str | None = None,
    ):
        if routing not in {"merged", "cartesian", "zip"}:
            raise ValueError(
                "routing must be one of: 'merged', 'cartesian', or 'zip'."
            )

        self.generators = self._as_list(generators, ConceptGenerator, "generators")
        self.annotators = self._as_list(annotators, Annotator, "annotators")
        if not self.generators:
            raise ValueError("At least one concept generator is required.")
        if not self.annotators:
            raise ValueError("At least one annotator is required.")
        if routing == "zip" and len(self.generators) != len(self.annotators):
            raise ValueError(
                "routing='zip' requires the same number of generators and annotators."
            )

        self.concept_filter = (
            concept_filter
            if concept_filter is not None
            else UnionConceptFilter() if routing == "merged" else None
        )
        self.aggregator = aggregator
        self.routing = routing
        self.name = name or self.__class__.__name__

    def __call__(
        self,
        dataset: Dataset,
        class_names: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Tensor], dict[str, AxisAnnotation]]:
        generator_names = self._component_names(self.generators)
        annotator_names = self._component_names(self.annotators)
        concepts = {
            generator_name: generator.generate(
                dataset=dataset,
                class_names=class_names,
                **kwargs,
            )
            for generator_name, generator in zip(generator_names, self.generators)
        }

        values: dict[str, Tensor] = {}
        annotations: dict[str, AxisAnnotation] = {}
        if self.routing == "merged":
            if self.concept_filter is None:
                raise RuntimeError("Merged routing requires a concept filter.")
            merged = self.concept_filter(concepts)
            for annotator_name, annotator in zip(annotator_names, self.annotators):
                concept_values = annotator.annotate(dataset, merged, **kwargs)
                self._insert_result(
                    values,
                    annotations,
                    annotator_name,
                    concept_values,
                    merged,
                    dataset,
                )
        elif self.routing == "cartesian":
            for generator_name, annotation in concepts.items():
                for annotator_name, annotator in zip(
                    annotator_names, self.annotators
                ):
                    route_name = f"{generator_name}_{annotator_name}"
                    concept_values = annotator.annotate(
                        dataset, annotation, **kwargs
                    )
                    self._insert_result(
                        values,
                        annotations,
                        route_name,
                        concept_values,
                        annotation,
                        dataset,
                    )
        else:
            for generator_name, annotation, annotator_name, annotator in zip(
                generator_names,
                concepts.values(),
                annotator_names,
                self.annotators,
            ):
                route_name = f"{generator_name}_{annotator_name}"
                concept_values = annotator.annotate(
                    dataset, annotation, **kwargs
                )
                self._insert_result(
                    values,
                    annotations,
                    route_name,
                    concept_values,
                    annotation,
                    dataset,
                )

        if self.aggregator is not None:
            aggregate_annotation = self._common_annotation(annotations)
            aggregate_values = self.aggregator(values)
            aggregate_name = self._unique_name("aggregated", values)
            self._insert_result(
                values,
                annotations,
                aggregate_name,
                aggregate_values,
                aggregate_annotation,
                dataset,
                unique=False,
            )

        return values, annotations

    @staticmethod
    def _as_list(value: Any, expected_type: type, name: str) -> list[Any]:
        if isinstance(value, expected_type):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result = list(value)
            if not all(isinstance(item, expected_type) for item in result):
                raise TypeError(
                    f"All {name} must be {expected_type.__name__} instances."
                )
            return result
        raise TypeError(
            f"{name} must be a {expected_type.__name__} or a sequence of them."
        )

    @staticmethod
    def _component_names(components: Sequence[Any]) -> list[str]:
        names: list[str] = []
        used: dict[str, int] = {}
        for component in components:
            base = getattr(component, "name", None) or component.__class__.__name__
            count = used.get(base, 0)
            used[base] = count + 1
            names.append(base if count == 0 else f"{base}_{count}")
        return names

    @staticmethod
    def _unique_name(name: str, values: dict[str, Tensor]) -> str:
        if name not in values:
            return name
        index = 1
        while f"{name}_{index}" in values:
            index += 1
        return f"{name}_{index}"

    @classmethod
    def _insert_result(
        cls,
        values: dict[str, Tensor],
        annotations: dict[str, AxisAnnotation],
        requested_name: str,
        concept_values: Tensor,
        annotation: AxisAnnotation,
        dataset: Dataset,
        unique: bool = True,
    ) -> None:
        name = (
            cls._unique_name(requested_name, values)
            if unique else requested_name
        )
        cls._validate_value(name, concept_values, annotation, dataset)
        values[name] = concept_values
        annotations[name] = annotation

    @staticmethod
    def _validate_value(
        name: str,
        values: Tensor,
        annotation: AxisAnnotation,
        dataset: Dataset,
    ) -> None:
        if not isinstance(values, Tensor):
            raise TypeError(
                f"Generated concept values {name!r} must be a Tensor."
            )
        if values.ndim != 2:
            raise ValueError(
                f"Generated concept values {name!r} must be two-dimensional; "
                f"got shape {tuple(values.shape)}."
            )
        if values.shape[0] != len(dataset):
            raise ValueError(
                f"Generated concept values {name!r} have {values.shape[0]} "
                f"samples, but the dataset has {len(dataset)}."
            )
        if values.shape[1] != annotation.shape:
            raise ValueError(
                f"Generated concept values {name!r} have {values.shape[1]} "
                f"outputs, but their annotation defines {annotation.shape}."
            )

    @staticmethod
    def _common_annotation(
        annotations: dict[str, AxisAnnotation],
    ) -> AxisAnnotation:
        if not annotations:
            raise ValueError("Cannot aggregate an empty set of concept values.")
        iterator = iter(annotations.values())
        first = next(iterator)
        first_definition = first.to_dict()
        if any(axis.to_dict() != first_definition for axis in iterator):
            raise ValueError(
                "Aggregation requires all generated concept tensors to share "
                "the same AxisAnnotation."
            )
        return first
