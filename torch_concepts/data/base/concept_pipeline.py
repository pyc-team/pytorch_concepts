from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Literal, Sequence

from torch import Tensor
from torch.utils.data import Dataset

from torch_concepts import Annotations
from torch_concepts.data.base.annotator import Annotator
from torch_concepts.data.base.concept_generator import ConceptGenerator


# merged: merge all generated concepts, then send them to all annotators.
# cartesian: send each generated concept axis to all annotators.
# zip: send each generated concept axis to the corresponding annotator.
# Annotation can run over the generation dataset or over named datasets such as
# {"train": train_dataset, "val": val_dataset}.
RoutingMode = Literal["merged", "cartesian", "zip"]


class DeduplicateConcepts:
    """Remove duplicate concept labels from one concept axis.

    The first occurrence is kept. Later occurrences must have the same states,
    cardinality, and type so that filtering cannot silently collapse
    incompatible concept definitions.
    """

    def __call__(self, concepts: Annotations) -> Annotations:
        labels: list[str] = []
        states: list[list[str]] = []
        cardinalities: list[int] = []
        types: list[str] = []
        metadata: dict[str, dict[str, Any]] | None = None
        definitions: dict[str, tuple[list[str], int, str]] = {}

        if concepts.metadata is not None:
            metadata = {}

        for index, label in enumerate(concepts.labels):
            state_names = list(concepts.states[index])
            cardinality = concepts.cardinalities[index]
            concept_type = concepts.types[index]
            definition = (state_names, cardinality, concept_type)

            if label in definitions:
                if definitions[label] != definition:
                    previous_states, previous_cardinality, previous_type = (
                        definitions[label]
                    )
                    raise ValueError(
                        f"Concept {label!r} has incompatible definitions: "
                        f"states/cardinality/type "
                        f"{state_names}/{cardinality}/{concept_type} do not "
                        f"match {previous_states}/{previous_cardinality}/"
                        f"{previous_type}."
                    )
                if metadata is not None and concepts.metadata is not None:
                    existing = metadata.setdefault(label, {})
                    for key, value in concepts.metadata.get(label, {}).items():
                        existing.setdefault(key, value)
                continue

            definitions[label] = definition
            labels.append(label)
            states.append(state_names)
            cardinalities.append(cardinality)
            types.append(concept_type)
            if metadata is not None:
                metadata[label] = dict(
                    concepts.metadata.get(label, {}) if concepts.metadata else {}
                )

        return Annotations(
            labels=labels,
            states=states,
            cardinalities=cardinalities,
            types=types,
            metadata=metadata,
            concept_space=concepts.concept_space,
        )


class ConceptSupervisionPipeline:
    """Compose concept generation, annotation, filtering, and aggregation.

    Calling the pipeline uses ``dataset`` as the concept-generation dataset.
    By default, that same dataset is annotated. Pass ``annotation_datasets`` to
    annotate one or more named datasets with the generated concept axis while
    still generating concepts from ``dataset``.

    Parameters
    ----------
    generators : ConceptGenerator or sequence of ConceptGenerator
        Concept generators to produce concept annotations from the dataset.
    annotators : Annotator or sequence of Annotator
        Annotators to produce concept values from the dataset and concept annotations.
    concept_filter : callable, optional
        Transformation applied to each concept axis before annotation. Defaults
        to :class:`DeduplicateConcepts`, so duplicate labels are removed even
        when they come from a single generator. For ``routing='merged'``,
        generator axes are first merged by routing and then the filter is
        applied to the merged axis. For ``cartesian`` and ``zip``, the filter is
        applied to each routed generator axis.
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
        concept_filter: Callable[[Annotations], Annotations] | None = DeduplicateConcepts(),
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

        self.concept_filter = concept_filter
        self.aggregator = aggregator
        self.routing = routing
        self.name = name or self.__class__.__name__

    def __call__(
        self,
        dataset: Dataset,
        class_names: list[str] | None = None,
        annotation_datasets: Mapping[str, Dataset] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Tensor], dict[str, Annotations]]:
        """Generate concepts from ``dataset`` and annotate datasets.

        Parameters
        ----------
        dataset : Dataset
            Dataset used by concept generators. If ``annotation_datasets`` is
            omitted, this dataset is also annotated.
        class_names : list[str], optional
            Class names forwarded to concept generators.
        annotation_datasets : mapping of str to Dataset, optional
            Named datasets to annotate with the concepts generated from
            ``dataset``. Output keys are prefixed with each mapping key, e.g.
            ``"train_CLIPAnnotator"`` and ``"val_CLIPAnnotator"``.
        **kwargs
            Additional keyword arguments forwarded to generators and annotators.

        Returns
        -------
        values, annotations : tuple[dict[str, Tensor], dict[str, Annotations]]
            Sample-level concept values and their concept axes. With named
            annotation datasets, both dictionaries use split-prefixed keys.
        """
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
        annotations: dict[str, Annotations] = {}
        datasets_to_annotate, prefix_outputs = self._annotation_dataset_map(
            dataset,
            annotation_datasets,
        )
        for dataset_name, annotation_dataset in datasets_to_annotate.items():
            dataset_values, dataset_annotations = self._annotate_dataset(
                concepts=concepts,
                generator_names=generator_names,
                annotator_names=annotator_names,
                dataset=annotation_dataset,
                kwargs=kwargs,
            )
            for name, concept_values in dataset_values.items():
                output_name = (
                    f"{dataset_name}_{name}" if prefix_outputs else name
                )
                self._insert_result(
                    values,
                    annotations,
                    output_name,
                    concept_values,
                    dataset_annotations[name],
                    annotation_dataset,
                )

        return values, annotations

    def _annotate_dataset(
        self,
        concepts: dict[str, Annotations],
        generator_names: list[str],
        annotator_names: list[str],
        dataset: Dataset,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Tensor], dict[str, Annotations]]:
        values: dict[str, Tensor] = {}
        annotations: dict[str, Annotations] = {}
        if self.routing == "merged":
            merged = self._merge_concept_axes(concepts)
            merged = self._filter_concepts(merged)
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
                annotation = self._filter_concepts(annotation)
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
                annotation = self._filter_concepts(annotation)
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
    def _merge_concept_axes(concepts: dict[str, Annotations]) -> Annotations:
        """Concatenate generator concept axes in generator order."""
        labels: list[str] = []
        states: list[list[str]] = []
        cardinalities: list[int] = []
        types: list[str] = []
        metadata: dict[str, dict[str, Any]] | None = None
        concept_space = False

        if any(axis.metadata is not None for axis in concepts.values()):
            metadata = {}

        for axis in concepts.values():
            labels.extend(axis.labels)
            states.extend([list(state_names) for state_names in axis.states])
            cardinalities.extend(axis.cardinalities)
            types.extend(axis.types)
            concept_space = concept_space or axis.concept_space
            if metadata is not None and axis.metadata is not None:
                for label in axis.labels:
                    metadata.setdefault(label, dict(axis.metadata.get(label, {})))

        return Annotations(
            labels=labels,
            states=states,
            cardinalities=cardinalities,
            types=types,
            metadata=metadata,
            concept_space=concept_space,
        )

    def _filter_concepts(self, concepts: Annotations) -> Annotations:
        if self.concept_filter is None:
            return concepts
        return self.concept_filter(concepts)

    @staticmethod
    def _annotation_dataset_map(
        default_dataset: Dataset,
        annotation_datasets: Mapping[str, Dataset] | None,
    ) -> tuple[dict[str, Dataset], bool]:
        if annotation_datasets is None:
            return {"": default_dataset}, False
        if not isinstance(annotation_datasets, Mapping):
            raise TypeError(
                "annotation_datasets must be a mapping of names to datasets."
            )
        if not annotation_datasets:
            raise ValueError("annotation_datasets must not be empty.")

        result: dict[str, Dataset] = {}
        for name, dataset in annotation_datasets.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "annotation_datasets keys must be non-empty strings."
                )
            if not isinstance(dataset, Dataset):
                raise TypeError(
                    f"annotation_datasets[{name!r}] must be a Dataset."
                )
            result[name] = dataset
        return result, True

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
        annotations: dict[str, Annotations],
        requested_name: str,
        concept_values: Tensor,
        annotation: Annotations,
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
        annotation: Annotations,
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
        if values.shape[1] != annotation.size:
            raise ValueError(
                f"Generated concept values {name!r} have {values.shape[1]} "
                f"outputs, but their annotation defines {annotation.size}."
            )

    @staticmethod
    def _common_annotation(
        annotations: dict[str, Annotations],
    ) -> Annotations:
        if not annotations:
            raise ValueError("Cannot aggregate an empty set of concept values.")
        iterator = iter(annotations.values())
        first = next(iterator)
        first_definition = first.to_dict()
        if any(axis.to_dict() != first_definition for axis in iterator):
            raise ValueError(
                "Aggregation requires all generated concept tensors to share "
                "the same Annotations."
            )
        return first
