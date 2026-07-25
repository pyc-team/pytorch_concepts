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
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Union
import warnings

from ...concept_graph import ConceptGraph
from ...annotations import Annotations
from ...tensor import AnnotatedTensor
from ..utils import files_exist, parse_tensor, convert_precision
from .concept_pipeline import ConceptSupervisionPipeline

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
        concepts (Tensor, optional): Native concept annotations originally
            provided by the dataset.
        generated_concepts (dict[str, Annotations]): Generated concept
            vocabularies keyed by pipeline output name.
        generated_annotations (dict[str, Tensor]): Sample-level annotations
            binding dataset samples to generated concept values.
        ground_truth (Tensor, optional): Concept supervision selected for model
            training.

    Args:
        input_data: Input features as numpy array, pandas DataFrame, or Tensor.
        concepts: Optional native concept annotations as a numpy array, pandas
            DataFrame, or Tensor.
        annotations: Optional annotations for the native concepts.
        graph: Optional concept graph as pandas DataFrame or tensor.
        concept_names_subset: Optional list to select subset of concepts.
        reorder_by_type: Group same-type concepts contiguously -- binary, then
            categorical (ascending cardinality), then continuous (default:
            True), so type-based slicing on the resulting AnnotatedTensor is a
            view instead of a copy. Ties keep their relative order.
        precision: Numerical precision (16, 32, or 64, default: 32).
        exogenous: Optional exogenous variables (not yet implemented).

    Raises:
        ValueError: If native concepts are provided without an axis-1
            annotation, or if an invalid concept subset is requested.
        NotImplementedError: If continuous concepts or exogenous variables are used.

    Example:
        >>> X = torch.randn(100, 28, 28)  # 100 images
        >>> C = torch.randint(0, 2, (100, 5))  # 5 binary concepts
        >>> annotations = Annotations(labels=['c1', 'c2', 'c3', 'c4', 'c5'])
        >>> dataset = ConceptDataset(X, C, annotations=annotations)
        >>> len(dataset)
        100
    """
    def __init__(
        self,
        input_data: Union[np.ndarray, pd.DataFrame, Tensor],
        concepts: Optional[Union[np.ndarray, pd.DataFrame, Tensor]] = None,
        annotations: Optional[Annotations] = None,
        graph: Optional[pd.DataFrame] = None,
        concept_names_subset: Optional[List[str]] = None,
        reorder_by_type: bool = True,
        precision: Union[int, str] = 32,
        # TODO: implement handling of exogenous inputs
    ):
        Dataset.__init__(self)

        # Set info
        self.name = name if name is not None else self.__class__.__name__
        self.precision = precision
        self.embs_precomputed = False  # whether input_data 
                                       # contains precomputed embeddings
        self.concepts: Optional[Tensor] = None
        self.use_as_gt = False
        self.generated_gt_name: Optional[str] = None
        self.generated_concepts: Dict[str, Annotations] = {}
        self.generated_annotations: Dict[str, Tensor] = {}
        self.ground_truth: Optional[Tensor] = None
        self._ground_truth_annotation: Optional[Annotations] = None
        self._ground_truth_source: Optional[str] = None

        # sanity check on concept annotations and metadata
        if annotations is None and concepts is not None:
            warnings.warn("No concept annotations provided. These will be set to default numbered "
                         "concepts 'concept_{i}'. All concepts will be treated as binary.")
            n = concepts.shape[1]
            annotations = Annotations(labels=[f"concept_{i}" for i in range(n)],
                                      cardinalities=[1] * n, # assume binary
                                      types=['binary'] * n)

        # sanity check
        axis_annotation = annotations

        if axis_annotation.cardinalities is not None:
            concept_names_with_cardinality = [name for name, card in zip(axis_annotation.labels, axis_annotation.cardinalities) if card is not None]
            concept_names_without_cardinality = [name for name in axis_annotation.labels if name not in concept_names_with_cardinality]
            if concept_names_without_cardinality:
                raise ValueError(f"Cardinalities list provided but missing cardinality for concepts: {concept_names_without_cardinality}")

        # set concept annotations
        self._annotations = annotations
        # maybe reduce annotations based on subset of concept names
        self._maybe_reduce_annotations(annotations,
                                       concept_names_subset)
        # group same-type concepts contiguously (stable within each type) so
        # AnnotatedTensor.binary()/.categorical()/.continuous() resolve to a
        # view instead of a per-batch advanced-index copy
        if reorder_by_type:
            self._annotations = self._maybe_reorder_by_type(self._annotations)

        # Set dataset's input data X
        # TODO: input is assumed to be a one of "np.ndarray, pd.DataFrame, Tensor" for now
        # allow more complex data structures in the future with a custom parser
        self.input_data: Tensor = parse_tensor(input_data, 'input', self.precision)

        # Store native concept data C
        if concepts is not None:
            self.set_concepts(concepts)
        else:
            self._resolve_ground_truth()

        # Store graph
        self._graph = None
        if graph is not None:
            self.set_graph(graph)  # graph among all concepts

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

        ``concepts['c']`` is the sole learner-facing supervision key. It is the
        exact same Python object as either ``concepts['native']`` or the selected
        entry of ``concepts['generated']`` for this sample.
        """
        x = self.input_data[item]
        native = self.concepts[item] if self.concepts is not None else None
        generated = {
            name: values[item]
            for name, values in getattr(
                self,
                "generated_annotations",
                {},
            ).items()
        }
        ground_truth_source = getattr(self, "_ground_truth_source", "native")
        if ground_truth_source == "native":
            selected = native
        elif ground_truth_source is not None:
            selected = generated[ground_truth_source]
        else:
            selected = native

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

        The default collate stacks the per-sample (plain, 1-D) concept rows into a
        ``(batch, n_concepts)`` tensor; this re-wraps that tensor as an
        :class:`~torch_concepts.tensor.AnnotatedTensor` carrying the same
        concept-space annotation as :attr:`concepts`, so every batch's concepts
        are label/type aware. Inputs and any other keys are collated unchanged.
        Used as the DataLoader ``collate_fn`` by :class:`ConceptDataModule`.
        """
        batch = default_collate(samples)
        annotation = self._ground_truth_annotation
        if annotation is not None and isinstance(batch, dict):
            concepts = batch.get('concepts')
            if isinstance(concepts, dict):
                c = concepts.get('c')
                if isinstance(c, Tensor) and c.dim() >= 2 and c.shape[1] == annotation.size:
                    # axis=1 to match how the concepts are stored (see the
                    # explanatory comment in ``_set_concepts``); for this 2-D
                    # batch it is the same axis as the default -1, but pinning it
                    # keeps the stored and collated representations consistent.
                    concepts['c'] = AnnotatedTensor(c, annotation, axis=1)
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
        if self._ground_truth_annotation is None:
            return []
        return self._ground_truth_annotation.labels

    @property
    def annotations(self) -> Optional[Annotations]:
        """Annotations for the concepts in the dataset."""
        return self._ground_truth_annotation

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
        return self.concepts is not None

    @property
    def has_generated_concepts(self) -> bool:
        """Whether generated concept vocabularies are available."""
        return bool(self.generated_concepts)

    @property
    def has_concepts(self) -> bool:
        """Whether concept supervision is available for training."""
        return self.ground_truth is not None

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

        With ``cache=True`` (default) the embeddings are persisted to
        ``{root_dir}/{backbone.filename}`` and loaded from there on subsequent
        calls instead of recomputing. Use ``force=True`` to force recomputing 
        embeddings even if a cache file exists.

        
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
            False to compute in memory only (e.g. on a dataset subset, to
            avoid writing a subset-sized cache into a shared ``root_dir``).
        cache_dir : str, optional
            Directory for the cache file. Defaults to the dataset's
            ``root_dir``; set it when the data lives on read-only/shared
            storage and the cache should go elsewhere (e.g. local scratch).
        force : bool, default False
            Recompute even if a cache file exists.
        """
        embs = None
        if cache:
            if cache_dir is None:
                cache_dir = self.root_dir
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, backbone.filename)
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
    
    def generate_concepts(
        self,
        concept_pipeline: ConceptSupervisionPipeline,
        class_names: Optional[List[str]] = None,
        datasets_to_annotate: Optional[Mapping[str, Any]] = None,
        self_annotation_name: Optional[str] = None,
        use_as_gt: bool = False,
        generated_gt_name: Optional[str] = None,
        **kwargs,
    ) -> tuple[Dict[str, Tensor], Dict[str, Annotations]]:
        """Run a concept-supervision pipeline explicitly on this dataset.

        Concepts are generated from ``self``. ``datasets_to_annotate`` only
        controls which datasets are annotated with those generated concepts.
        If no extra datasets are provided, only ``self`` is annotated.

        ``self_annotation_name`` lets the current dataset be included in named
        annotation outputs alongside additional datasets. For example, a
        training dataset can pass ``self_annotation_name="train"`` and
        ``datasets_to_annotate={"val": val_dataset}``.

        Args:
            concept_pipeline: Pipeline that generates and annotates concepts.
            class_names: Optional task or class names forwarded to the concept
                generator prompt.
            datasets_to_annotate: Optional mapping from output names to datasets
                that should be annotated with the concepts generated from this
                dataset, in addition to ``self`` when
                ``self_annotation_name`` is provided.
            self_annotation_name: Optional output name used to include ``self``
                in the annotation outputs when annotating multiple partitions.
            use_as_gt: Select generated annotations as learner supervision.
            generated_gt_name: Generated output name selected when
                ``use_as_gt=True``.
            **kwargs: Additional keyword arguments forwarded to
                ``concept_pipeline``.

        Returns:
            A pair ``(annotation_values, concepts)`` with dictionaries keyed by
            pipeline output name.
        """
        if not callable(concept_pipeline):
            raise TypeError("concept_pipeline must be callable.")
        if self_annotation_name is not None or datasets_to_annotate is not None:
            datasets = {}
            if self_annotation_name is not None:
                datasets[self_annotation_name] = self
            datasets.update(dict(datasets_to_annotate or {}))
            kwargs["annotation_datasets"] = datasets

        annotation_values, concepts = concept_pipeline(
            self,
            class_names=class_names,
            **kwargs,
        )
        self.set_generated_concepts(
            concepts,
            annotation_values,
            use_as_gt=use_as_gt,
            generated_gt_name=generated_gt_name,
        )
        return annotation_values, concepts

    def set_generated_concepts(
        self,
        concepts: Dict[str, Annotations],
        annotations: Dict[str, Tensor],
        use_as_gt: bool = False,
        generated_gt_name: Optional[str] = None,
    ) -> None:
        """Store generated vocabularies and their sample annotations."""
        self.use_as_gt = use_as_gt
        self.generated_gt_name = generated_gt_name
        self.generated_concepts = dict(concepts)
        self.generated_annotations = dict(annotations)
        if set(self.generated_concepts) != set(self.generated_annotations):
            raise ValueError(
                "Generated concepts and annotations must use the same keys."
            )
        self._resolve_ground_truth()

    def _resolve_ground_truth(self) -> None:
        """Resolve the tensor and annotation used as training supervision."""
        if self.use_as_gt and self.generated_concepts:
            name = self._resolve_generated_gt_name()
            self.ground_truth = self.generated_annotations[name]
            self._ground_truth_annotation = self.generated_concepts[name]
            self._ground_truth_source = name
        elif getattr(self, "concepts", None) is not None:
            self.ground_truth = self.concepts
            self._ground_truth_annotation = self._annotations
            self._ground_truth_source = "native"
        elif self.generated_concepts:
            name = self._resolve_generated_gt_name()
            self.ground_truth = self.generated_annotations[name]
            self._ground_truth_annotation = self.generated_concepts[name]
            self._ground_truth_source = name
        else:
            self.ground_truth = None
            self._ground_truth_annotation = None
            self._ground_truth_source = None

    def _resolve_generated_gt_name(self) -> str:
        """Return the generated source selected for ground-truth supervision."""
        if not self.generated_concepts:
            raise ValueError("No generated concepts are available.")
        if self.generated_gt_name is None:
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
        self.concept_names_all = annotations.labels
        self._all_concept_annotation = annotations
        if concept_names_subset is not None:
            # sanity check, all subset concepts must be in all concepts
            missing_concepts = set(concept_names_subset) - set(self.concept_names_all)
            assert not missing_concepts, f"Concepts not found in dataset: {missing_concepts}"
            to_select = deepcopy(concept_names_subset)
            
            # Get indices of selected concepts
            indices = [self.concept_names_all.index(name) for name in to_select]
            
            # Reduce annotations by extracting only the selected concepts
            axis_annotation = annotations
            reduced_labels = tuple(axis_annotation.labels[i] for i in indices)
            
            # Reduce cardinalities
            reduced_cardinalities = tuple(axis_annotation.cardinalities[i] for i in indices)
        
            # Reduce states
            reduced_states = tuple(axis_annotation.states[i] for i in indices)

            # Reduce types
            reduced_types = tuple(axis_annotation.types[i] for i in indices)

            # Create reduced annotations
            self._annotations = Annotations(
                labels=reduced_labels,
                cardinalities=reduced_cardinalities,
                states=reduced_states,
                types=reduced_types,
            )

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
        # Subset graph to match current concept_names
        subgraph = graph.loc[self.concept_names, self.concept_names]
        self._graph = ConceptGraph(
            data=parse_tensor(subgraph, 'graph', self.precision),
            node_names=self.concept_names
        )
        
    def set_concepts(self, concepts: Union[np.ndarray, pd.DataFrame, Tensor]):
        """Set concept annotations for the dataset.
        
        Args:
            concepts: Tensor of shape (n_samples, n_concepts) containing concept values
            concept_names: List of strings naming each concept. If None, will use
                         numbered concepts like "concept_0", "concept_1", etc.
        """
        # Validate shape
        # concepts' length must match dataset's length
        if concepts.shape[0] != self.n_samples:
            raise RuntimeError(f"Concepts has {concepts.shape[0]} samples but "
                f"input_data has {self.n_samples}.")
        
        if not isinstance(concepts, (pd.DataFrame, np.ndarray, Tensor)):
            raise TypeError(f"Concepts must be a np.ndarray, pd.DataFrame, "
                f"or Tensor, got {type(concepts).__name__}.")

        #########################################################################
        ###### modify this to change convention for how to store concepts  ######
        #########################################################################
        # convert pd.Dataframe to tensor
        values = parse_tensor(concepts, 'concepts', self.precision)
        columns = self._all_concept_annotation.get_slice(self._annotations.labels)
        values = values[:, columns]
        #########################################################################

        # Wrap the full concept tensor with a *concept-space* annotation (one
        # integer-coded column per concept, so categorical labels are class
        # indices) so it carries the concept labels/types.
        #
        # ``axis=1`` is passed explicitly rather than taking the default: it is
        # what makes per-sample ``__getitem__`` indexing return a *plain* 1-D
        # row. An annotation on axis 1 needs 2+ dims, so indexing a row drops
        # it, and ``default_collate`` can then stack the rows as ordinary
        # tensors; :meth:`collate` re-annotates the assembled batch. Under the
        # default ``axis=-1`` the row would keep its annotation (its last axis
        # is still the concept axis) and collation would fail on a list of
        # AnnotatedTensors.
        concept_ann = self.annotations.to_concept_space()
        if concepts.dim() >= 2 and concepts.shape[1] == concept_ann.size:
            self.concepts = AnnotatedTensor(concepts, concept_ann, axis=1)
        else:
            self.concepts = concepts
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
