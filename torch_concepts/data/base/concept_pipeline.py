from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Literal, Sequence

from torch import Tensor
from torch.utils.data import Dataset

from torch_concepts import Annotations
from torch_concepts.data.base.annotation_filter import AnnotationFilter
from torch_concepts.data.base.annotator import Annotator
from torch_concepts.data.base.calibrator import Calibrator
from torch_concepts.data.base.concept_generator import ConceptGenerator
from torch_concepts.data.base.generator_filter import GeneratorFilter
from torch_concepts.data.lf_postprocessing import DeduplicateConcepts


# merged: merge all generated concepts, then send them to all annotators.
# cartesian: send each generated concept axis to all annotators.
# zip: send each generated concept axis to the corresponding annotator.
# Annotation can run over the generation dataset or over named datasets such as
# {"train": train_dataset, "val": val_dataset}.
RoutingMode = Literal["merged", "cartesian", "zip"]


DEFAULT_GENERATOR_FILTER = DeduplicateConcepts()


class ConceptSupervisionPipeline:
    """Compose concept generation, annotation, calibration, and filtering.

    Calling the pipeline uses ``dataset`` as the concept-generation dataset.
    By default, that same dataset is annotated. Pass ``annotation_datasets`` to
    annotate one or more named datasets with the generated concept axis while
    still generating concepts from ``dataset``.

    Routing controls which generated concept axes are sent to each annotator:

    - ``"merged"`` concatenates all generator outputs, applies the generator
      filter once to the merged axis, and sends that shared axis to every
      annotator. One result is produced per annotator.
    - ``"cartesian"`` applies the generator filter independently to every
      generator output and sends every filtered axis to every annotator. One
      result is produced for each generator-annotator pair.
    - ``"zip"`` applies the generator filter independently to every generator
      output and pairs generators and annotators by their configuration order.
      It requires equal numbers of generators and annotators.

    Every routed annotation tensor then passes through the same optional
    processing stages in this order: ``raw_annotation_filter``, ``calibrator``,
    and ``calibrated_annotation_filter``. Aggregation, when configured, runs
    after those stages.

    Parameters
    ----------
    generators : ConceptGenerator or sequence of ConceptGenerator
        Concept generators to produce concept annotations from the dataset.
    annotators : Annotator or sequence of Annotator
        Annotators that produce raw concept scores.
    generator_filter : GeneratorFilter, optional
        Filter applied to generated concept names before annotation. Defaults
        to :class:`DeduplicateConcepts`. For merged routing it is applied after
        merging; for cartesian and zip routing it is applied independently to
        each generator output. Pass ``None`` to disable generator filtering.
    raw_annotation_filter : AnnotationFilter, optional
        Sample-level filter applied to each raw annotation tensor. Filtered
        entries are represented by ``NaN``.
    calibrator : Calibrator, optional
        Transformation applied after raw annotation filtering.
    calibrated_annotation_filter : AnnotationFilter, optional
        Sample-level filter applied after optional calibration. If no
        calibrator is configured, it receives the raw-filtered scores.
    aggregator : callable, optional
        Function that aggregates the final generated concept tensors.
        Aggregation is supported only with merged routing. Aggregators are
        responsible for deciding how to handle any ``NaN`` entries introduced
        by annotation filters.
    routing : {'merged', 'cartesian', 'zip'}, default='merged'
        Routing mode used to combine generators and annotators, as described
        above.
    name : str, optional
        Name of the pipeline. If None, the class name is used.
    """

    def __init__(
        self,
        generators: ConceptGenerator | Sequence[ConceptGenerator],
        annotators: Annotator | Sequence[Annotator],
        generator_filter: GeneratorFilter | None = DEFAULT_GENERATOR_FILTER,
        raw_annotation_filter: AnnotationFilter | None = None,
        calibrator: Calibrator | None = None,
        calibrated_annotation_filter: AnnotationFilter | None = None,
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
        if aggregator is not None and routing != "merged":
            raise ValueError(
                "aggregator is only supported with routing='merged'."
            )
        if generator_filter is not None and not isinstance(
            generator_filter,
            GeneratorFilter,
        ):
            raise TypeError(
                "generator_filter must be a GeneratorFilter or None."
            )
        if raw_annotation_filter is not None and not isinstance(
            raw_annotation_filter,
            AnnotationFilter,
        ):
            raise TypeError(
                "raw_annotation_filter must be an AnnotationFilter or None."
            )
        if calibrator is not None and not isinstance(calibrator, Calibrator):
            raise TypeError("calibrator must be a Calibrator or None.")
        if calibrated_annotation_filter is not None and not isinstance(
            calibrated_annotation_filter,
            AnnotationFilter,
        ):
            raise TypeError(
                "calibrated_annotation_filter must be an AnnotationFilter "
                "or None."
            )

        self.generator_filter = generator_filter
        self.raw_annotation_filter = raw_annotation_filter
        self.calibrator = calibrator
        self.calibrated_annotation_filter = calibrated_annotation_filter
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
        """Annotate one dataset using the configured routing strategy.

        The method constructs each route selected by ``self.routing``, invokes
        the corresponding annotator, runs the annotation post-processing
        stages, and records each tensor together with its concept axis. With
        merged routing, the optional aggregator is invoked after all annotator
        routes have been processed.

        Parameters
        ----------
        concepts : dict[str, Annotations]
            Generated concept axes keyed by their unique generator names.
        generator_names : list[str]
            Unique names in the same order as ``self.generators``.
        annotator_names : list[str]
            Unique names in the same order as ``self.annotators``.
        dataset : Dataset
            Dataset whose samples are passed to annotators.
        kwargs : dict[str, Any]
            Additional keyword arguments forwarded to annotators.

        Returns
        -------
        values, annotations : tuple[dict[str, Tensor], dict[str, Annotations]]
            Processed annotation tensors and their corresponding concept axes,
            keyed by route name.
        """
        values: dict[str, Tensor] = {}
        annotations: dict[str, Annotations] = {}
        if self.routing == "merged":
            merged = self._merge_concept_axes(concepts)
            merged = self._filter_concepts(merged)
            for annotator_name, annotator in zip(annotator_names, self.annotators):
                concept_values = annotator.annotate(dataset, merged, **kwargs)
                concept_values = self._process_annotation_values(
                    concept_values,
                    merged,
                    dataset,
                    annotator_name,
                )
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
                    concept_values = self._process_annotation_values(
                        concept_values,
                        annotation,
                        dataset,
                        route_name,
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
                concept_values = self._process_annotation_values(
                    concept_values,
                    annotation,
                    dataset,
                    route_name,
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
        """Concatenate generated concept axes into one annotation definition.

        Labels, state names, cardinalities, and types retain dictionary
        insertion order. The returned axis is marked as a concept space when
        at least one input axis is a concept space.

        Parameters
        ----------
        concepts : dict[str, Annotations]
            Concept axes keyed in generator order.

        Returns
        -------
        Annotations
            A new annotation definition containing every input concept axis.
        """
        labels: list[str] = []
        states: list[list[str]] = []
        cardinalities: list[int] = []
        types: list[str] = []
        concept_space = False

        for axis in concepts.values():
            labels.extend(axis.labels)
            states.extend([list(state_names) for state_names in axis.states])
            cardinalities.extend(axis.cardinalities)
            types.extend(axis.types)
            concept_space = concept_space or axis.concept_space

        return Annotations(
            labels=labels,
            states=states,
            cardinalities=cardinalities,
            types=types,
            concept_space=concept_space,
        )

    def _filter_concepts(self, concepts: Annotations) -> Annotations:
        """Apply the configured generator filter to a concept axis.

        Parameters
        ----------
        concepts : Annotations
            Generated concept axis to filter.

        Returns
        -------
        Annotations
            The filtered concept axis, or the original object when generator
            filtering is disabled.
        """
        if self.generator_filter is None:
            return concepts
        return self.generator_filter.filter_annotations(concepts)

    def _process_annotation_values(
        self,
        values: Tensor,
        concepts: Annotations,
        dataset: Dataset,
        route_name: str,
    ) -> Tensor:
        """Validate and post-process one annotator output tensor.

        The raw tensor is first checked against the dataset and concept axis.
        Enabled stages then run in this order: raw annotation filter,
        calibrator, and calibrated annotation filter. Every stage receives the
        current tensor and the unchanged concept metadata, and must return a
        tensor with the same shape.

        Parameters
        ----------
        values : Tensor
            Raw annotation scores with shape ``(samples, concept_outputs)``.
        concepts : Annotations
            Concept axis describing the tensor's second dimension.
        dataset : Dataset
            Annotated dataset, used to validate the sample dimension.
        route_name : str
            Route identifier included in validation error messages.

        Returns
        -------
        Tensor
            Scores after all configured post-processing stages.

        Raises
        ------
        TypeError
            If the raw values or a stage result is not a tensor.
        ValueError
            If the raw values are incompatible with the dataset or concepts,
            or a processing stage changes the tensor shape.
        """
        self._validate_value(route_name, values, concepts, dataset)
        stages = (
            (
                "raw_annotation_filter",
                self.raw_annotation_filter,
                "filter",
            ),
            ("calibrator", self.calibrator, "calibrate"),
            (
                "calibrated_annotation_filter",
                self.calibrated_annotation_filter,
                "filter",
            ),
        )
        for stage_name, stage, method_name in stages:
            if stage is None:
                continue
            processed = getattr(stage, method_name)(values, concepts)
            if not isinstance(processed, Tensor):
                raise TypeError(
                    f"{stage_name} must return a Tensor; "
                    f"got {type(processed).__name__}."
                )
            if processed.shape != values.shape:
                raise ValueError(
                    f"{stage_name} must preserve annotation tensor shape; "
                    f"got {tuple(processed.shape)} instead of "
                    f"{tuple(values.shape)}."
                )
            values = processed
        return values

    @staticmethod
    def _annotation_dataset_map(
        default_dataset: Dataset,
        annotation_datasets: Mapping[str, Dataset] | None,
    ) -> tuple[dict[str, Dataset], bool]:
        """Resolve and validate the datasets that should be annotated.

        When no explicit mapping is supplied, the generation dataset is
        returned under an empty name and output prefixing is disabled.
        Otherwise, mapping names and dataset values are validated and retained
        in their original order.

        Parameters
        ----------
        default_dataset : Dataset
            Generation dataset used as the annotation fallback.
        annotation_datasets : mapping of str to Dataset, optional
            Explicitly named datasets to annotate.

        Returns
        -------
        datasets, prefix_outputs : tuple[dict[str, Dataset], bool]
            The resolved dataset mapping and whether its names should prefix
            output keys.

        Raises
        ------
        TypeError
            If the supplied value is not a mapping or a mapping value is not a
            dataset.
        ValueError
            If the mapping is empty or contains an empty or non-string name.
        """
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
        """Normalize one component or a component sequence to a list.

        Strings and bytes are rejected even though they are sequences. Every
        returned item is guaranteed to be an instance of ``expected_type``.

        Parameters
        ----------
        value : Any
            Single component or sequence of components to normalize.
        expected_type : type
            Required base type for every component.
        name : str
            Component category used in error messages.

        Returns
        -------
        list[Any]
            Components in their supplied order.

        Raises
        ------
        TypeError
            If ``value`` is neither a matching component nor a valid sequence,
            or if any sequence item has the wrong type.
        """
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
        """Create deterministic, unique names for configured components.

        A component's non-empty ``name`` attribute is preferred; otherwise its
        class name is used. Repeated base names receive ``_1``, ``_2``, and so
        on in configuration order.

        Parameters
        ----------
        components : sequence
            Components for which to derive route-safe names.

        Returns
        -------
        list[str]
            Unique names in the same order as ``components``.
        """
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
        """Return an output name that is absent from an existing result map.

        The requested name is returned unchanged when available. On collision,
        the first free numeric suffix (``_1``, ``_2``, and so on) is appended.

        Parameters
        ----------
        name : str
            Preferred output name.
        values : dict[str, Tensor]
            Existing results whose keys are already occupied.

        Returns
        -------
        str
            An unused output name.
        """
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
        """Validate and insert a tensor and its annotation metadata together.

        Keeping both mappings in one helper ensures their keys remain aligned.
        By default, collisions are resolved with :meth:`_unique_name`; callers
        may disable this when they have already chosen a unique name.

        Parameters
        ----------
        values : dict[str, Tensor]
            Result mapping updated with ``concept_values``.
        annotations : dict[str, Annotations]
            Metadata mapping updated with ``annotation`` under the same key.
        requested_name : str
            Preferred key for both output mappings.
        concept_values : Tensor
            Sample-level concept tensor to validate and store.
        annotation : Annotations
            Concept axis describing ``concept_values``.
        dataset : Dataset
            Dataset used to validate the tensor's sample dimension.
        unique : bool, default=True
            Whether to resolve a name collision automatically.

        Raises
        ------
        TypeError
            If ``concept_values`` is not a tensor.
        ValueError
            If its dimensions do not match the dataset and annotation.
        """
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
        """Check that a result tensor matches its dataset and concept axis.

        A valid result is a two-dimensional tensor whose first dimension equals
        the number of dataset samples and whose second dimension equals
        ``annotation.size``.

        Parameters
        ----------
        name : str
            Result name included in validation errors.
        values : Tensor
            Tensor to validate.
        annotation : Annotations
            Definition of the expected output dimension.
        dataset : Dataset
            Definition of the expected sample dimension.

        Raises
        ------
        TypeError
            If ``values`` is not a tensor.
        ValueError
            If ``values`` is not two-dimensional or either dimension has the
            wrong size.
        """
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
        """Ensure aggregate inputs share one annotation definition.

        Parameters
        ----------
        annotations : dict[str, Annotations]
            Concept axes associated with tensors passed to the aggregator.

        Returns
        -------
        Annotations
            The first annotation object when every definition is equivalent.

        Raises
        ------
        ValueError
            If there are no annotations or their serialized definitions differ.
        """
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
