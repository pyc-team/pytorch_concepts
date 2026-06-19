"""
Base dataset class for concept-annotated datasets.

This module provides the ConceptDataset class, which serves as the foundation
for all concept-based datasets in the torch_concepts package.
"""
from abc import abstractmethod
import os
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from copy import deepcopy
from typing import Dict, List, Optional, Union
import warnings

from ...nn.modules.mid.constructors.concept_graph import ConceptGraph
from ...annotations import Annotations, AxisAnnotation
from ..utils import files_exist, parse_tensor, convert_precision
from .concept_pipeline import ConceptSupervisionPipeline

# TODO: implement masks for missing values
# TODO: add exogenous
# TODO: range for continuous concepts
# TODO: add possibility to annotate multiple axis (e.g., for relational concepts)


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
        generated_concepts (dict[str, Tensor]): Generated concept supervision
            keyed by pipeline output name.
        generated_annotations (dict[str, AxisAnnotation]): Axis annotations for
            generated concept tensors.
        ground_truth (Tensor, optional): Concept supervision selected for model
            training.

    Args:
        input_data: Input features as numpy array, pandas DataFrame, or Tensor.
        concepts: Optional native concept annotations as a numpy array, pandas
            DataFrame, or Tensor.
        annotations: Optional annotations for the native concepts.
        graph: Optional concept graph as pandas DataFrame or tensor.
        concept_names_subset: Optional list to select subset of concepts.
        concept_pipeline: Optional pipeline used to generate concept
            supervision.
        use_as_gt: Whether the selected generated concepts should replace native
            concepts as training supervision.
        name: Optional dataset name.
        precision: Numerical precision (16, 32, or 64, default: 32).
        exogenous: Optional exogenous variables (not yet implemented).

    Raises:
        ValueError: If native concepts are provided without an axis-1
            annotation, or if an invalid concept subset is requested.
        NotImplementedError: If continuous concepts or exogenous variables are used.

    Example:
        >>> X = torch.randn(100, 28, 28)  # 100 images
        >>> C = torch.randint(0, 2, (100, 5))  # 5 binary concepts
        >>> annotations = Annotations({1: AxisAnnotation(labels=['c1', 'c2', 'c3', 'c4', 'c5'])})
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
        concept_pipeline: Optional[ConceptSupervisionPipeline] = None,
        use_as_gt: bool = False,
        name: Optional[str] = None,
        precision: Union[int, str] = 32,
        # TODO: implement handling of exogenous inputs
    ):
        super(ConceptDataset, self).__init__()

        # Set info
        self.name = name if name is not None else self.__class__.__name__
        self.precision = precision
        self.embs_precomputed = False  # whether input_data 
                                       # contains precomputed embeddings
        self.concept_pipeline = concept_pipeline
        self.use_as_gt = use_as_gt
        self.concepts: Optional[Tensor] = None
        self.generated_concepts: Dict[str, Tensor] = {}
        self.generated_annotations: Dict[str, AxisAnnotation] = {}
        self.ground_truth: Optional[Tensor] = None
        self._ground_truth_annotation: Optional[AxisAnnotation] = None

        # sanity check on concept annotations and metadata
        if annotations is None and concepts is not None:
            warnings.warn("No concept annotations provided. These will be set to default numbered "
                         "concepts 'concept_{i}'. All concepts will be treated as binary.")
            annotations = Annotations({
                    1: AxisAnnotation(labels=[f"concept_{i}" for i in range(concepts.shape[1])],
                                      cardinalities=None, # assume binary
                                      metadata={f"concept_{i}": {'type': 'discrete', # assume discrete (bernoulli)
                                                                } for i in range(concepts.shape[1])})
                                      })
        # assert first axis is annotated axis for concepts
        if concepts is not None and 1 not in annotations.annotated_axes:
            raise ValueError("Concept annotations must include axis 1 for concepts. " \
            "Axis 0 is always assumed to be the batch dimension")

        # sanity check
        axis_annotation = annotations[1] if concepts is not None else None
        if axis_annotation is not None and axis_annotation.metadata is not None:
            assert all('type' in v  for v in axis_annotation.metadata.values()), \
                "Concept metadata must contain 'type' for each concept."
            assert all(v['type'] in ['discrete', 'continuous'] for v in axis_annotation.metadata.values()), \
                "Concept metadata 'type' must be either 'discrete' or 'continuous'."

        if axis_annotation is not None and axis_annotation.cardinalities is not None:
            concept_names_with_cardinality = [name for name, card in zip(axis_annotation.labels, axis_annotation.cardinalities) if card is not None]
            concept_names_without_cardinality = [name for name in axis_annotation.labels if name not in concept_names_with_cardinality]
            if concept_names_without_cardinality:
                raise ValueError(f"Cardinalities list provided but missing cardinality for concepts: {concept_names_without_cardinality}")
            
            
        # sanity check on unsupported concept types     
        if axis_annotation is not None and axis_annotation.metadata is not None:
            for name, meta in axis_annotation.metadata.items():
                # raise error if type metadata contain 'continuous': this is not supported yet
                # TODO: implement continuous concept types
                if meta['type'] == 'continuous':
                    raise NotImplementedError("Continuous concept types are not supported yet.")


        # set concept annotations
        # this defines self.annotations property
        self._annotations = annotations
        if concepts is not None:
            # maybe reduce annotations based on subset of concept names
            self._maybe_reduce_annotations(annotations,
                                           concept_names_subset)
        else:
            self.concept_names_all = []
            if concept_names_subset is not None:
                raise ValueError(
                    "concept_names_subset requires native concepts."
                )

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
        """
        Get a single sample from the dataset.

        Args:
            item (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing 'inputs' and 'concepts' sub-dictionaries.
        """
        # Get raw input data and concepts
        x = self.input_data[item]

        # TODO: handle missing values with masks

        # Create sample dictionary
        sample = {
            'inputs': {'x': x},    # input data: multiple inputs can be stored in a dict
            'concepts': {
                'ground_truth': (
                    self.ground_truth[item]
                    if self.ground_truth is not None else None
                ),
                'native': (
                    self.concepts[item]
                    if self.concepts is not None else None
                ),
                'generated': {
                    name: values[item]
                    for name, values in self.generated_concepts.items()
                },
            },
            # TODO: add scalers when these are set
            # also check if batch transforms work correctly inside the model training loop
            # 'transforms': {'x': self.scalers.get('input', None),
            #               'c': self.scalers.get('concepts', None)}
        }

        return sample


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
        return list(self._ground_truth_annotation.labels)

    @property
    def annotations(self) -> Optional[Annotations]:
        """Annotations for the concepts in the dataset."""
        if self._ground_truth_annotation is None:
            return None
        return Annotations({1: self._ground_truth_annotation})

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
        """Whether generated concept tensors have been attached."""
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

    def generate_concepts(
        self,
        class_names: Optional[List[str]] = None,
        **kwargs,
    ) -> tuple[Dict[str, Tensor], Dict[str, AxisAnnotation], Optional[str]]:
        """Run the configured concept-supervision pipeline once.

        Dataset subclasses with processed files can call this method from
        :meth:`build` and cache the returned tensors and annotations alongside
        their other processed data.
        """
        if self.concept_pipeline is None:
            raise RuntimeError("No concept_pipeline configured.")
        values, annotations, selected = self.concept_pipeline(
            self,
            class_names=class_names,
            **kwargs,
        )
        self.set_generated_concepts(values, annotations, selected)
        return values, annotations, selected

    def set_generated_concepts(
        self,
        values: Dict[str, Tensor],
        annotations: Dict[str, AxisAnnotation],
        selected: Optional[str] = None,
    ) -> None:
        """Attach generated concept tensors and update training supervision.

        Args:
            values: Generated concept tensors keyed by pipeline output name.
            annotations: Axis annotations keyed like ``values``.
            selected: Name of the generated tensor selected for supervision.
        """
        self.generated_concepts = dict(values)
        self.generated_annotations = dict(annotations)
        if set(self.generated_concepts) != set(self.generated_annotations):
            raise ValueError(
                "Generated concept values and annotations must use the same keys."
            )
        if (
            selected is not None
            and selected not in self.generated_concepts
        ):
            raise ValueError(
                f"Selected generated concepts {selected!r} were not provided."
            )
        self._resolve_ground_truth(selected)

    def _resolve_ground_truth(
        self,
        selected_generated: Optional[str] = None,
    ) -> None:
        """Resolve the tensor and annotation used as training supervision."""
        if self.use_as_gt and selected_generated is not None:
            self.ground_truth = self.generated_concepts[selected_generated]
            self._ground_truth_annotation = self.generated_annotations[
                selected_generated
            ]
        elif self.concepts is not None:
            self.ground_truth = self.concepts
            self._ground_truth_annotation = self._annotations[1]
        elif selected_generated is not None:
            self.ground_truth = self.generated_concepts[selected_generated]
            self._ground_truth_annotation = self.generated_annotations[
                selected_generated
            ]
        elif self.generated_concepts:
            selected_generated = next(iter(self.generated_concepts))
            self.ground_truth = self.generated_concepts[selected_generated]
            self._ground_truth_annotation = self.generated_annotations[
                selected_generated
            ]
        else:
            self.ground_truth = None
            self._ground_truth_annotation = None



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
        self.concept_names_all = annotations.get_axis_labels(1)
        self._all_concept_annotation = annotations[1]
        if concept_names_subset is not None:
            # sanity check, all subset concepts must be in all concepts
            missing_concepts = set(concept_names_subset) - set(self.concept_names_all)
            assert not missing_concepts, f"Concepts not found in dataset: {missing_concepts}"
            to_select = deepcopy(concept_names_subset)

            # Reduce annotations using concept-level slices. AxisAnnotation
            # preserves states, cardinalities, and metadata for categorical
            # concepts.
            self._annotations = Annotations({
                1: annotations[1].subset(to_select)
            })

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
        columns = self._all_concept_annotation.get_slice(self._annotations[1].labels)
        values = values[:, columns]
        #########################################################################

        self.concepts = values
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
