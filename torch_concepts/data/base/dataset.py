"""
Base dataset class for concept-annotated datasets.

This module provides the ConceptDataset class, which serves as the foundation
for all concept-based datasets in the torch_concepts package.
"""
from abc import abstractmethod
import os
import logging
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm
from typing import Dict, List, Optional, Union
import warnings

from ...concept_graph import ConceptGraph
from ...annotations import Annotations
from ...tensor import AnnotatedTensor
from ..utils import files_exist, parse_tensor, convert_precision
from ..generation.base.pipeline import ConceptGenerationPipeline

# TODO: implement masks for missing values
# TODO: add exogenous
# TODO: range for continuous concepts
# TODO: add possibility to annotate multiple axis (e.g., for relational concepts)

logger = logging.getLogger(__name__)

class ConceptDataset(Dataset):
    """
    Base class for concept-annotated datasets.

    This class extends PyTorch's Dataset to support concept annotations,
    concept graphs, and various metadata. It provides a unified interface
    for working with datasets that have both input features and concept labels.

    Attributes:
        name (str): Name of the dataset.
        precision (int or str): Numerical precision for tensors (16, 32, or 64).
        input_data (Tensor): Input features/images.
        native_concepts (AnnotatedTensor, optional): Persistent native concept
            values and their metadata.
        concepts (AnnotatedTensor, optional): Concept supervision currently
            selected for model training.
        generated_concepts (dict[str, AnnotatedTensor]): Generated sample-level
            concept values and metadata keyed by pipeline output name.

    Args:
        input_data: Input features as numpy array, pandas DataFrame, or Tensor.
        concepts: Optional native concept values as a numpy array, pandas
            DataFrame, Tensor, or AnnotatedTensor, with shape
            (n_samples, n_concepts). Categorical values are class indices.
        annotations: Optional metadata for plain concept values. Must be omitted when
            concepts is an AnnotatedTensor, whose attached metadata is used.
            When no metadata is provided the dataset assumes every concept is binary,
            with a warning. Omit both arguments for a dataset without native concepts;
            annotations without concept values are not supported.
        graph: Optional concept graph as pandas DataFrame or tensor.
        concept_names_subset: Optional list to select subset of concepts.
        reorder_by_type: Group same-type concepts contiguously -- binary, then
            categorical (ascending cardinality), then continuous (default:
            True), so type-based slicing on the resulting AnnotatedTensor is a
            view instead of a copy. Ties keep their relative order.
        precision: Numerical precision (16, 32, or 64, default: 32).
        exogenous: Optional exogenous variables (not yet implemented).

    Raises:
        TypeError: If concepts uses an unsupported input type.
        ValueError: If both an AnnotatedTensor and separate annotations are
            supplied; annotations are supplied without concepts; annotated
            values do not describe the concept columns of a two-dimensional
            tensor, or an invalid concept shape or subset is requested.
        RuntimeError: If concept values and inputs have different sample counts.

    Example:
        >>> import torch
        >>> from torch_concepts import Annotations, AnnotatedTensor
        >>> from torch_concepts.data.base.dataset import ConceptDataset
        >>> X = torch.randn(100, 28, 28)  # 100 images
        >>> C = torch.randint(0, 2, (100, 5))  # 5 binary concepts
        >>> annotations = Annotations(labels=['c1', 'c2', 'c3', 'c4', 'c5'])
        >>> dataset = ConceptDataset(X, C, annotations=annotations)
        >>> len(dataset)
        100

        An AnnotatedTensor can supply both values and metadata:

        >>> annotated_concepts = AnnotatedTensor(C, annotations, axis=1)
        >>> dataset = ConceptDataset(X, concepts=annotated_concepts)
        >>> dataset.concept_names
        ['c1', 'c2', 'c3', 'c4', 'c5']
    """

    # Set by ``ConceptDataModule(max_samples=...)``: the rows are a random draw,
    # and which draw (None = from the global RNG, so not reproducible).
    is_subset: bool = False
    subset_seed: Optional[int] = None

    def __init__(
        self,
        input_data: Union[np.ndarray, pd.DataFrame, Tensor],
        concepts: Optional[Union[np.ndarray, pd.DataFrame, Tensor, AnnotatedTensor]] = None,
        annotations: Optional[Annotations] = None,
        graph: Optional[pd.DataFrame] = None,
        concept_names_subset: Optional[List[str]] = None,
        reorder_by_type: bool = True,
        precision: Union[int, str] = 32,
        name: Optional[str] = None,
        # TODO: implement handling of exogenous inputs
    ):
        super(ConceptDataset, self).__init__()

        # Set info
        self.name = name if name is not None else self.__class__.__name__
        self.precision = precision
        self.embs_precomputed = False  # whether input_data 
                                       # contains precomputed embeddings
        self.native_concepts: Optional[AnnotatedTensor] = None
        self.concepts: Optional[AnnotatedTensor] = None
        self.use_as_gt = False
        self.generated_gt_name: Optional[str] = None
        self.generated_concepts: Dict[str, AnnotatedTensor] = {}
        self._ground_truth_annotation: Optional[Annotations] = None
        self._ground_truth_source: Optional[str] = None

        self.input_data: Tensor = parse_tensor(input_data, 'input', self.precision)
        native_values, annotations = self._normalize_native_concepts(
            concepts, annotations
        )

        # sanity check
        axis_annotation = annotations

        if axis_annotation is not None and axis_annotation.cardinalities is not None:
            concept_names_with_cardinality = [name for name, card in zip(axis_annotation.labels, axis_annotation.cardinalities) if card is not None]
            concept_names_without_cardinality = [name for name in axis_annotation.labels if name not in concept_names_with_cardinality]
            if concept_names_without_cardinality:
                raise ValueError(f"Cardinalities list provided but missing cardinality for concepts: {concept_names_without_cardinality}")

        # set concept annotations
        self._annotations = annotations
        self._all_concept_annotation: Optional[Annotations] = None
        if annotations is None:
            if concept_names_subset is not None:
                raise ValueError(
                    "concept_names_subset requires native concept annotations."
                )
        else:
            # maybe reduce annotations based on subset of concept names
            self._maybe_reduce_annotations(annotations,
                                           concept_names_subset)
            # group same-type concepts contiguously (stable within each type) so
            # AnnotatedTensor.binary()/.categorical()/.continuous() resolve to a
            # view instead of a per-batch advanced-index copy
            if reorder_by_type:
                self._annotations = self._maybe_reorder_by_type(self._annotations)

        # Store native concept data C
        if native_values is not None:
            self.set_concepts(native_values)
        else:
            self._resolve_ground_truth()

        # Store graph
        self._graph = None
        if graph is not None:
            self.set_graph(graph)  # graph among all concepts

        self.scalers = {}  # dict of fitted scalers for input and concepts

    def _normalize_native_concepts(
        self,
        concepts: Optional[Union[np.ndarray, pd.DataFrame, Tensor, AnnotatedTensor]],
        annotations: Optional[Annotations],
    ) -> tuple[Optional[AnnotatedTensor], Optional[Annotations]]:
        """Return labeled native values and the complete declared native schema."""
        if concepts is None:
            if annotations is not None:
                raise ValueError("annotations requires native concept values.")
            return None, None

        if isinstance(concepts, AnnotatedTensor):
            if annotations is not None:
                raise ValueError(
                    "Do not provide annotations when concepts is an "
                    "AnnotatedTensor; use its attached annotation."
                )
            self._validate_native_concepts(concepts)
            annotations = concepts.annotation
            values = concepts.tensor
        elif isinstance(concepts, (Tensor, np.ndarray, pd.DataFrame)):
            if concepts.ndim != 2:
                raise ValueError(
                    "Native concepts must be two-dimensional with shape "
                    "(n_samples, n_concepts)."
                )
            if annotations is None:
                warnings.warn("No concept annotations provided. These will be set to default numbered "
                             "concepts 'concept_{i}'. All concepts will be treated as binary.")
                n = concepts.shape[1]
                annotations = Annotations(
                    labels=[f"concept_{i}" for i in range(n)],
                    cardinalities=[1] * n,
                    types=['binary'] * n,
                )
            elif isinstance(concepts, pd.DataFrame):
                missing = [label for label in annotations.labels if label not in concepts.columns]
                if missing:
                    raise ValueError(f"Native concepts are missing required labels: {missing}.")
                concepts = concepts[annotations.labels]
            values = concepts
        else:
            raise TypeError(
                "concepts must be a Tensor, np.ndarray, pd.DataFrame, "
                "AnnotatedTensor, or None."
            )

        values = parse_tensor(values, 'concepts', self.precision)
        concept_annotation = annotations.to_concept_space()
        if values.shape[1] != concept_annotation.size:
            raise ValueError(
                "Native concepts must have one column per concept; "
                f"got {values.shape[1]} columns for {concept_annotation.size} concepts."
            )
        native_values = AnnotatedTensor(values, concept_annotation, axis=1)
        self._validate_native_concepts(native_values)
        return native_values, annotations

    def _validate_native_concepts(self, concepts: AnnotatedTensor) -> None:
        """Validate native tensor layout and alignment with dataset rows."""
        if not isinstance(concepts, AnnotatedTensor):
            raise TypeError("Native concepts must be an AnnotatedTensor.")
        if concepts.dim() != 2 or concepts.axis not in (1, -1):
            raise ValueError(
                "Annotated concepts must be two-dimensional with "
                "metadata on the concept columns (axis 1 or -1)."
            )
        if concepts.shape[0] != self.n_samples:
            raise RuntimeError(
                f"Concepts has {concepts.shape[0]} samples but "
                f"input_data has {self.n_samples}."
            )
        if concepts.shape[1] != len(concepts.annotation.labels):
            raise ValueError(
                "Native concepts must have one column per concept; "
                "categorical values must be class indices, not per-state scores."
            )

    def __repr__(self):
        """
        Return string representation of the dataset.

        Returns:
            str: String showing dataset name and dimensions.
        """
        return f"{self.name}(n_samples={self.n_samples}, n_features={self.n_features}, n_concepts={self.n_concepts})"

    def __len__(self) -> int:
        """
        Return number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.n_samples
    
    def __getitem__(self, item):
        """Return a sample using the common concept-dataset dictionary shape.

        ``concepts['c']`` is the sole learner-facing supervision key and is
        indexed directly from the dataset's currently selected concepts.
        """
        x = self.input_data[item]
        selected = (
            self.concepts[item]
            if self.concepts is not None
            else None
        )
        native = (
            self.native_concepts[item]
            if self.native_concepts is not None
            else None
        )
        generated = {
            name: values[item]
            for name, values in self.generated_concepts.items()
        }

        return {
            "inputs": {"x": x},
            "concepts": {
                "c": selected,
                "native": native,
                "generated": generated,
            },
        }

    def collate(self, samples):
        """Collate samples into a batch, re-annotating the ground-truth concepts.

        Per-sample (plain, 1-D) concept rows are stacked into
        ``(batch, n_concepts)`` tensors and re-wrapped as
        :class:`~torch_concepts.tensor.AnnotatedTensor` carrying the same
        metadata as their selected, native, or generated source. Unavailable
        selected/native views remain ``None``. Any fitted scalers are attached
        under ``'scalers'`` (a reference to the dataset-level dict, so the
        learner can transform in scaled space and report metrics in the
        original scale). Used as the DataLoader ``collate_fn`` by
        :class:`ConceptDataModule`.
        """
        def collate_optional(values, annotation, name):
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise ValueError(
                    f"Cannot collate {name}: only some samples contain values."
                )
            collated = default_collate(values)
            if annotation is not None:
                collated = AnnotatedTensor(collated, annotation, axis=1)
            return collated

        generated_keys = tuple(self.generated_concepts)
        expected_keys = set(generated_keys)
        for index, sample in enumerate(samples):
            sample_keys = set(sample["concepts"]["generated"])
            if sample_keys != expected_keys:
                raise ValueError(
                    "Generated concept sources must be consistent across "
                    f"samples; sample {index} has {sorted(sample_keys)}, "
                    f"expected {sorted(expected_keys)}."
                )

        concept_samples = [sample["concepts"] for sample in samples]
        batch = {
            "inputs": default_collate([sample["inputs"] for sample in samples]),
            "concepts": {
                "c": collate_optional(
                    [concepts["c"] for concepts in concept_samples],
                    self._ground_truth_annotation,
                    "selected concepts",
                ),
                "native": collate_optional(
                    [concepts["native"] for concepts in concept_samples],
                    (
                        self.native_concepts.annotation
                        if self.native_concepts is not None
                        else None
                    ),
                    "native concepts",
                ),
                "generated": {
                    name: collate_optional(
                        [concepts["generated"][name] for concepts in concept_samples],
                        self.generated_concepts[name].annotation,
                        f"generated concepts {name!r}",
                    )
                    for name in generated_keys
                },
            },
        }
        if self.scalers:
            batch['scalers'] = self.scalers
        return batch


    # Dataset properties #####################################################

    @property
    def n_samples(self) -> int:
        """
        Number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.input_data.size(0)

    @property
    def n_features(self) -> tuple:
        """
        Shape of features in dataset's input (excluding number of samples).

        Returns:
            tuple: Shape of input features.
        """
        return tuple(self.input_data.size()[1:])

    @property
    def n_concepts(self) -> int:
        """
        Number of concepts in the dataset.

        Returns:
            int: Number of concepts, or 0 if no concepts.
        """
        return len(self.concept_names) if self.has_concepts else 0

    @property
    def concept_names(self) -> List[str]:
        """
        List of concept names in the dataset.

        Returns:
            List[str]: Names of all concepts.
        """
        if self.concepts is None:
            return []
        return self.concepts.annotation.labels

    @property
    def annotations(self) -> Optional[Annotations]:
        """Annotations for the concepts in the dataset."""
        return (
            self.concepts.annotation
            if self.concepts is not None
            else None
        )

    @property
    def shape(self) -> tuple:
        """Shape of the input tensor."""
        return tuple(self.input_data.size())

    @property
    def exogenous(self) -> Dict[str, Tensor]:
        """Mapping of dataset's exogenous variables."""
        # return {name: attr['value'] for name, attr in self._exogenous.items()}
        raise NotImplementedError("Exogenous variables are not supported for now.")

    @property
    def n_exogenous(self) -> int:
        """Number of exogenous variables in the dataset."""
        # return len(self._exogenous)
        raise NotImplementedError("Exogenous variables are not supported for now.")

    @property
    def graph(self) -> Optional[ConceptGraph]:
        """Adjacency matrix of the causal graph between concepts."""
        return self._graph

    # Dataset flags #####################################################

    @property
    def has_exogenous(self) -> bool:
        """Whether the dataset has exogenous information."""
        # return self.n_exogenous > 0
        raise NotImplementedError("Exogenous variables are not supported for now.")

    @property
    def has_native_concepts(self) -> bool:
        """Whether the dataset provides native concept annotations."""
        return self.native_concepts is not None

    @property
    def has_generated_concepts(self) -> bool:
        """Whether generated concept vocabularies are available."""
        return bool(self.generated_concepts)

    @property
    def has_concepts(self) -> bool:
        """Whether concept supervision is available for training."""
        return self.concepts is not None

    @property
    def root_dir(self) -> str:
        if isinstance(self.root, str):
            root = os.path.expanduser(os.path.normpath(self.root))
        else:
            raise ValueError("Invalid root directory")
        return root
        
    @property
    @abstractmethod
    def raw_filenames(self) -> List[str]:
        """The list of raw filenames in the :obj:`self.root_dir` folder that must be
        present in order to skip `download()`. Should be implemented by subclasses."""
        pass

    @property
    @abstractmethod
    def processed_filenames(self) -> List[str]:
        """The list of processed filenames in the :obj:`self.root_dir` folder that must be
        present in order to skip `build()`. Should be implemented by subclasses."""
        pass

    @property
    def raw_paths(self) -> List[str]:
        """The absolute paths of the raw files that must be present in order to skip downloading."""
        return [os.path.join(self.root_dir, f) for f in self.raw_filenames]

    @property
    def processed_paths(self) -> List[str]:
        """The absolute paths of the processed files that must be present in order to skip building."""
        return [os.path.join(self.root_dir, f) for f in self.processed_filenames]

    # Directory utilities ###########################################################

    # Loading pipeline: load() → load_raw() → build() → download()

    def maybe_download(self):
        if not files_exist(self.raw_paths):
            os.makedirs(self.root_dir, exist_ok=True)
            self.download()

    def maybe_build(self):
        if not files_exist(self.processed_paths):
            os.makedirs(self.root_dir, exist_ok=True)
            self.build()

    def download(self) -> None:
        """Downloads dataset's files to the :obj:`self.root_dir` folder."""
        raise NotImplementedError

    def build(self) -> None:
        """Eventually build the dataset from raw data to :obj:`self.root_dir`
        folder."""
        pass

    def load_raw(self, *args, **kwargs):
        """Loads raw dataset without any data preprocessing."""
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """Loads raw dataset and preprocess data.
        Default to :obj:`load_raw`."""
        return self.load_raw(*args, **kwargs)

    # Embedding precomputation #############################################

    def precompute_embeddings(
        self,
        backbone,
        batch_size: int = 64,
        workers: int = 0,
        cache: bool = True,
        cache_dir: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """Precompute backbone embeddings and swap them in as ``input_data``.

        Runs the (frozen) ``backbone`` over the whole dataset once. Afterwards
        ``input_data`` holds the ``(n_samples, backbone.out_features)``
        embeddings and ``embs_precomputed`` is True, so ``__getitem__`` serves
        embeddings.

        With ``cache=True`` (default) the embeddings are persisted under
        ``cache_dir`` (by default ``root_dir``) and reloaded on later calls;
        ``force=True`` recomputes anyway. The file name identifies the rows it
        covers -- e.g. ``bkb_embs_resnet18_n4000_seed7.pt`` -- so the full
        dataset and each ``max_samples`` subset keep separate caches. An
        unseeded subset redraws its rows every run, and is never cached.

        Parameters
        ----------
        backbone : Backbone
            Feature extractor (needs ``filename``, ``source`` and
            ``__call__``).
        batch_size : int, default 64
            Batch size for the extraction pass.
        workers : int, default 0
            DataLoader workers for the extraction pass.
        cache : bool, default True
            Persist the embeddings to disk and reuse them across calls. Pass
            False to compute in memory only.
        cache_dir : str, optional
            Directory for the cache file. Defaults to the dataset's
            ``root_dir``; set it when the data lives on read-only/shared
            storage and the cache should go elsewhere (e.g. local scratch).
        force : bool, default False
            Recompute even if a cache file exists.
        """
        embs = None
        if cache and self.is_subset and self.subset_seed is None:
            warnings.warn(
                "Embeddings of an unseeded subset are not cached: its rows are "
                "redrawn every run. Pass `seed` to the datamodule to make the "
                "subset -- and its cache -- reproducible."
            )
            cache = False
        if cache:
            if cache_dir is None:
                cache_dir = self.root_dir
            os.makedirs(cache_dir, exist_ok=True)
            # Key the cache by the rows it holds, so no two sets share a file.
            stem, ext = os.path.splitext(backbone.filename)
            stem += f"_n{self.n_samples}"
            if self.is_subset:
                stem += f"_seed{self.subset_seed}"
            cache_path = os.path.join(cache_dir, stem + ext)
            if os.path.exists(cache_path) and not force:
                logger.info(f"Loading cached embeddings from {cache_path}")
                embs = torch.load(cache_path)
                if embs.shape[0] != self.n_samples:  # stale cache (e.g. written from a subset)
                    embs = None
        if embs is None:
            embs = self._compute_embeddings(backbone, batch_size, workers)
            if cache:
                logger.info(f"Saving embeddings to {cache_path}")
                torch.save(embs, cache_path)
        self.input_data = embs
        self.embs_precomputed = True

    def _compute_embeddings(self, backbone, batch_size: int, workers: int):
        """Run ``backbone`` over the whole dataset (original order) and return
        the stacked ``(n_samples, emb_dim)`` embeddings on CPU."""
        def collate_fn(batch):
            images = [sample['inputs']['x'] for sample in batch]
            if backbone.source != "huggingface" and isinstance(images[0], Tensor):
                return torch.stack(images)
            return images

        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_fn,
        )

        # Force eval so BatchNorm/Dropout stay deterministic (embeddings are
        # cached); restore the caller's mode afterwards.
        was_training = backbone.training
        backbone.eval()
        embeddings_list = []
        try:
            with torch.no_grad():
                for batch_data in tqdm(dataloader, desc="Extracting embeddings"):
                    embeddings_list.append(backbone(batch_data).cpu())
        finally:
            backbone.train(was_training)
        return torch.cat(embeddings_list, dim=0)

    def _subset_rows(self, indices) -> None:
        """Subset every row-aligned source and rebuild selected supervision."""
        row_indices = (
            indices.tolist()
            if hasattr(indices, "tolist")
            else list(indices)
        )
        n_samples = len(self)

        sources = []
        if self.native_concepts is not None:
            sources.append(("native", self.native_concepts))
        sources.extend(
            (f"generated:{name}", values)
            for name, values in self.generated_concepts.items()
        )
        for name, values in sources:
            if values.shape[0] != n_samples:
                raise RuntimeError(
                    f"Concept source {name!r} has {values.shape[0]} rows, "
                    f"but input_data has {n_samples}."
                )

        if isinstance(self.input_data, list):
            subset_input_data = [self.input_data[index] for index in row_indices]
        else:
            subset_input_data = self.input_data[row_indices]

        def subset_concepts(values: AnnotatedTensor) -> AnnotatedTensor:
            return AnnotatedTensor(
                values.tensor[row_indices],
                values.annotation,
                axis=1,
            )

        subset_native = (
            subset_concepts(self.native_concepts)
            if self.native_concepts is not None
            else None
        )
        subset_generated = {
            name: subset_concepts(values)
            for name, values in self.generated_concepts.items()
        }

        self.input_data = subset_input_data
        self.native_concepts = subset_native
        self.generated_concepts = subset_generated
        self._resolve_ground_truth()
    
    def generate_concepts(
        self,
        concept_pipeline: ConceptGenerationPipeline,
        class_names: Optional[List[str]] = None,
        use_as_gt: bool = False,
        generated_gt_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, AnnotatedTensor]:
        """Generate concepts aligned one-to-one with and attach them to this dataset.

        Call :class:`ConceptGenerationPipeline` directly for split-specific
        tensors or annotation of multiple target datasets.

        Args:
            concept_pipeline: Pipeline that generates and annotates concepts.
            class_names: Optional task labels forwarded to the concept
                generator prompt. For example, an LLM generator can use
                ``["cat", "dog"]`` to discover properties that distinguish
                cats from dogs. Annotators do not use these labels.
            use_as_gt: Select generated concepts as the learner-facing
                ``dataset.concepts``. If ``True`` and the pipeline returns
                multiple named sources, ``generated_gt_name`` is required.
            generated_gt_name: Name of the generated source to use as
                ``dataset.concepts`` when ``use_as_gt=True``. A pipeline can
                return several sources, for example one output per annotator
                plus an aggregated output; this argument chooses exactly one.
                It must exactly match one key in the dictionary returned by
                the pipeline. Inspect ``generated.keys()`` to see the valid
                names (for example, ``"aggregated"`` or
                ``"train_aggregated"``). Omit it when there is only one
                generated source.
            **kwargs: Additional keyword arguments forwarded to
                ``concept_pipeline``.

        Returns:
            Generated annotated concept tensors keyed by pipeline output name.

        Example:
            This pipeline asks an LLM for properties that distinguish cats
            from dogs, then scores those properties with two CLIP annotators.
            It returns one source per annotator and a third, averaged source.

            .. code-block:: python

                generator = LLMConceptGenerator(
                    llm=llm_backend,
                    prompt=(
                        "List visual properties that distinguish {class_names}."
                    ),
                )
                pipeline = ConceptGenerationPipeline(
                    generators=generator,
                    annotators=[clip_annotator_a, clip_annotator_b],
                    aggregator=average_scores,
                )
                generated = dataset.generate_concepts(
                    concept_pipeline=pipeline,
                    class_names=["cat", "dog"],
                    use_as_gt=True,
                    generated_gt_name="aggregated",
                )
                print(generated.keys())
                # CLIPAnnotator, CLIPAnnotator_1, aggregated
                # dataset.concepts is now generated["aggregated"]
        """
        if not callable(concept_pipeline):
            raise TypeError("concept_pipeline must be callable.")

        generated_concepts = concept_pipeline(
            self,
            class_names=class_names,
            **kwargs,
        )
        self.set_generated_concepts(
            generated_concepts,
            use_as_gt=use_as_gt,
            generated_gt_name=generated_gt_name,
        )
        return generated_concepts

    def set_generated_concepts(
        self,
        concepts: Dict[str, AnnotatedTensor],
        use_as_gt: bool = False,
        generated_gt_name: Optional[str] = None,
    ) -> None:
        """Store generated concept values together with their metadata."""
        normalized = {}
        for name, values in concepts.items():
            if not isinstance(values, AnnotatedTensor):
                raise TypeError(
                    f"Generated concept source {name!r} must be an "
                    f"AnnotatedTensor, got {type(values).__name__}."
                )
            if values.dim() != 2:
                raise ValueError(
                    f"Generated concept source {name!r} must be "
                    f"2-dimensional, got shape {tuple(values.shape)}."
                )
            if values.shape[0] != len(self):
                raise ValueError(
                    f"Generated concept source {name!r} has "
                    f"{values.shape[0]} samples, but dataset has {len(self)}."
                )
            normalized[name] = (
                values
                if values.axis == 1
                else AnnotatedTensor(values.tensor, values.annotation, axis=1)
            )

        selects_generated = bool(normalized) and (
            use_as_gt or self.native_concepts is None
        )
        if (
            selects_generated
            and generated_gt_name is None
            and len(normalized) > 1
        ):
            available = ", ".join(normalized)
            raise ValueError(
                "generated_gt_name must be specified when selecting from "
                f"multiple generated concept sources. Available sources: "
                f"{available}."
            )
        if (
            selects_generated
            and generated_gt_name is not None
            and generated_gt_name not in normalized
        ):
            available = ", ".join(normalized)
            raise ValueError(
                f"generated_gt_name={generated_gt_name!r} is not a generated "
                f"concept source. Available sources: {available}."
            )

        self.use_as_gt = use_as_gt
        self.generated_gt_name = generated_gt_name
        self.generated_concepts = normalized
        self._resolve_ground_truth()

    def _resolve_ground_truth(self) -> None:
        """Resolve the tensor and annotation used as training supervision."""
        if self.use_as_gt and self.generated_concepts:
            name = self._resolve_generated_gt_name()
            selected = self.generated_concepts[name]
            self._ground_truth_source = name
        elif self.native_concepts is not None:
            selected = self.native_concepts
            self._ground_truth_source = "native"
        elif self.generated_concepts:
            name = self._resolve_generated_gt_name()
            selected = self.generated_concepts[name]
            self._ground_truth_source = name
        else:
            selected = None
            self._ground_truth_source = None
        self.concepts = selected
        self._ground_truth_annotation = (
            selected.annotation if selected is not None else None
        )

    def _resolve_generated_gt_name(self) -> str:
        """Return the generated source selected for ground-truth supervision."""
        if not self.generated_concepts:
            raise ValueError("No generated concepts are available.")
        if self.generated_gt_name is None:
            if len(self.generated_concepts) > 1:
                available = ", ".join(self.generated_concepts)
                raise ValueError(
                    "generated_gt_name must be specified when selecting from "
                    "multiple generated concept sources. Available sources: "
                    f"{available}."
                )
            return next(iter(self.generated_concepts))
        if self.generated_gt_name not in self.generated_concepts:
            available = ", ".join(self.generated_concepts)
            raise ValueError(
                f"generated_gt_name={self.generated_gt_name!r} is not a "
                f"generated concept source. Available sources: {available}."
            )
        return self.generated_gt_name
    # Setters ##############################################################

    def _maybe_reduce_annotations(self,
                                annotations: Annotations,
                                concept_names_subset: Optional[List[str]] = None):
        """If ``concept_names_subset`` is provided, the annotations are reduced
        to include only the specified concepts.

        Args:
            annotations: Annotations object for all concepts.
            concept_names_subset: List of strings naming the subset of concepts to use.
                                    If :obj:`None`, will use all concepts.
        """
        self._all_concept_annotation = annotations
        if concept_names_subset is not None:
            self._annotations = annotations.subset(concept_names_subset)

    def _maybe_reorder_by_type(self, annotations: Annotations) -> Annotations:
        """Reorder ``annotations`` so same-type concepts sit contiguously
        (binary, then categorical, then continuous), categorical concepts
        further sorted by ascending cardinality. Ties keep their relative
        order. A no-op if already in this order.
        """
        sorted_labels = [
            label
            for labels in annotations.labels_by_type.values()
            for label in sorted(labels, key=lambda l: annotations.concept(l).cardinality)
        ]
        if sorted_labels == list(annotations.labels):
            return annotations
        return annotations.subset(sorted_labels)

    def set_graph(self, graph: pd.DataFrame):
        """Set the adjacency matrix of the causal graph between concepts 
        as a pandas DataFrame.
        
        If a concept subset was selected via ``concept_names_subset``,
        the graph is automatically subsetted to match the current concepts.

        Args:
            graph: A pandas DataFrame representing the adjacency matrix of the 
                   causal graph. Rows and columns should be named after the 
                   variables in the dataset.
        """
        if not isinstance(graph, pd.DataFrame):
            raise TypeError(f"Graph must be a pandas DataFrame, got {type(graph).__name__}.")
        if self._annotations is None:
            raise ValueError(
                "A native concept graph requires native concept annotations."
            )
        # Subset the native graph to match the selected native annotation axis.
        native_concept_names = list(self._annotations.labels)
        subgraph = graph.loc[native_concept_names, native_concept_names]
        self._graph = ConceptGraph(
            data=parse_tensor(subgraph, 'graph', self.precision),
            node_names=native_concept_names
        )
        
    def set_concepts(self, concepts: AnnotatedTensor):
        """Replace native values using the metadata established at construction.

        This does not define a new native schema; datasets constructed without
        native concepts must use generated-concept APIs to attach generated data.

        Args:
            concepts: Annotated native values of shape (n_samples, n_concepts),
                with one column per concept and categorical values stored as
                class indices. Must contain all selected native labels; columns
                are selected/reordered by name. Plain values are accepted only
                by the constructor, which attaches their metadata first.

        Raises:
            TypeError: If concepts is not an AnnotatedTensor.
            ValueError: If the native schema is unavailable, the tensor layout
                is invalid, or required native labels are missing.
            RuntimeError: If the number of samples differs from the dataset.
        """
        self._validate_native_concepts(concepts)
        if self._annotations is None:
            raise ValueError(
                "Native concepts cannot be set without native concept annotations."
            )

        selected_labels = list(self._annotations.labels)
        missing = [label for label in selected_labels if label not in concepts.annotation.labels]
        if missing:
            raise ValueError(f"Native concepts are missing required labels: {missing}.")
        concepts = concepts[selected_labels]
        # Axis 1 makes individual rows plain tensors for collation.
        self.native_concepts = AnnotatedTensor(
            convert_precision(concepts.tensor, self.precision),
            self._annotations.to_concept_space(),
            axis=1,
        )
        self._resolve_ground_truth()

    def add_exogenous(self,
                      name: str,
                      value: Union[np.ndarray, pd.DataFrame, Tensor],
                      convert_precision: bool = True):
        raise NotImplementedError("Exogenous variables are not supported for now.")

    def remove_exogenous(self, name: str):
        raise NotImplementedError("Exogenous variables are not supported for now.")

    def add_scaler(self, key: str, scaler):
        """Add a scaler for preprocessing a specific tensor.

        Args:
            key (str): The name of the tensor to scale ('input', 'concepts').
            scaler (Scaler): The fitted scaler to use.
        """
        if key not in ['input', 'concepts']:
            raise KeyError(f"{key} not in dataset. Valid keys: 'input', 'concepts'")
        self.scalers[key] = scaler

    # Utilities ###########################################################
