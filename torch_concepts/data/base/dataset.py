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
        concepts (Tensor): Concept annotations.
        annotations (Annotations): Detailed concept annotations with metadata.

    Args:
        input_data: Input features as numpy array, pandas DataFrame, or Tensor.
        concepts: Concept annotations as numpy array, pandas DataFrame, or Tensor.
        annotations: Optional Annotations object with concept metadata.
        graph: Optional concept graph as pandas DataFrame or tensor.
        concept_names_subset: Optional list to select subset of concepts.
        precision: Numerical precision (16, 32, or 64, default: 32).
        name: Optional dataset name.
        exogenous: Optional exogenous variables (not yet implemented).

    Raises:
        ValueError: If concepts is None or annotations don't include axis 1.
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
        concepts: Union[np.ndarray, pd.DataFrame, Tensor],
        annotations: Optional[Annotations] = None,
        graph: Optional[pd.DataFrame] = None,
        concept_names_subset: Optional[List[str]] = None,
        precision: Union[int, str] = 32,
        name: Optional[str] = None,
        # TODO: implement handling of exogenous inputs
    ):
        super(ConceptDataset, self).__init__()

        # Set info
        self.name = name if name is not None else self.__class__.__name__
        self.precision = precision

        if concepts is None:
            raise ValueError("Concepts must be provided for ConceptDataset.")

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
        if 1 not in annotations.annotated_axes:
            raise ValueError("Concept annotations must include axis 1 for concepts. " \
            "Axis 0 is always assumed to be the batch dimension")

        # sanity check
        axis_annotation = annotations[1]
        if axis_annotation.metadata is not None:
            assert all('type' in v  for v in axis_annotation.metadata.values()), \
                "Concept metadata must contain 'type' for each concept."
            assert all(v['type'] in ['discrete', 'continuous'] for v in axis_annotation.metadata.values()), \
                "Concept metadata 'type' must be either 'discrete' or 'continuous'."

        if axis_annotation.cardinalities is not None:
            concept_names_with_cardinality = [name for name, card in zip(axis_annotation.labels, axis_annotation.cardinalities) if card is not None]
            concept_names_without_cardinality = [name for name in axis_annotation.labels if name not in concept_names_with_cardinality]
            if concept_names_without_cardinality:
                raise ValueError(f"Cardinalities list provided but missing cardinality for concepts: {concept_names_without_cardinality}")
            
            
        # sanity check on unsupported concept types     
        if axis_annotation.metadata is not None:
            for name, meta in axis_annotation.metadata.items():
                # raise error if type metadata contain 'continuous': this is not supported yet
                # TODO: implement continuous concept types
                if meta['type'] == 'continuous':
                    raise NotImplementedError("Continuous concept types are not supported yet.")


        # set concept annotations
        # this defines self.annotations property
        self._annotations = annotations
        # maybe reduce annotations based on subset of concept names
        self.maybe_reduce_annotations(annotations,
                                      concept_names_subset)

        # Set dataset's input data X
        # TODO: input is assumed to be a one of "np.ndarray, pd.DataFrame, Tensor" for now
        # allow more complex data structures in the future with a custom parser
        self.input_data: Tensor = parse_tensor(input_data, 'input', self.precision)

        # Store concept data C
        self.concepts = None
        if concepts is not None:
            self.set_concepts(concepts) # Annotat

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
        c = self.concepts[item]

        # TODO: handle missing values with masks

        # Create sample dictionary
        sample = {
            'inputs': {'x': x},    # input data: multiple inputs can be stored in a dict
            'concepts': {'c': c},  # concepts: multiple concepts can be stored in a dict
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
        return self.annotations.get_axis_labels(1)
    
    @property
    def annotations(self) -> Optional[Annotations]:
        """Annotations for the concepts in the dataset."""
        return self._annotations if hasattr(self, '_annotations') else None

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
    def has_concepts(self) -> bool:
        """Whether the dataset has concept annotations."""
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



    # Setters ##############################################################

    def maybe_reduce_annotations(self,
                                annotations: Annotations,
                                concept_names_subset: Optional[List[str]] = None):
        """Set concept and labels for the dataset.
        Args:
            annotations: Annotations object for all concepts.
            concept_names_subset: List of strings naming the subset of concepts to use. 
                                    If :obj:`None`, will use all concepts.
        """
        self.concept_names_all = annotations.get_axis_labels(1)
        if concept_names_subset is not None:
            # sanity check, all subset concepts must be in all concepts
            missing_concepts = set(concept_names_subset) - set(self.concept_names_all)
            assert not missing_concepts, f"Concepts not found in dataset: {missing_concepts}"
            to_select = deepcopy(concept_names_subset)
            
            # Get indices of selected concepts
            indices = [self.concept_names_all.index(name) for name in to_select]
            
            # Reduce annotations by extracting only the selected concepts
            axis_annotation = annotations[1]
            reduced_labels = tuple(axis_annotation.labels[i] for i in indices)
            
            # Reduce cardinalities
            reduced_cardinalities = tuple(axis_annotation.cardinalities[i] for i in indices)
        
            # Reduce states
            reduced_states = tuple(axis_annotation.states[i] for i in indices)

            # Reduce metadata if present
            if axis_annotation.metadata is not None:
                reduced_metadata = {reduced_labels[i]: axis_annotation.metadata[axis_annotation.labels[indices[i]]] 
                                   for i in range(len(indices))}
            else:
                reduced_metadata = None
            
            # Create reduced annotations
            self._annotations = Annotations({
                1: AxisAnnotation(
                    labels=reduced_labels,
                    cardinalities=reduced_cardinalities,
                    states=reduced_states,
                    metadata=reduced_metadata
                )
            })

    def set_graph(self, graph: pd.DataFrame):
        """Set the adjacency matrix of the causal graph between concepts 
        as a pandas DataFrame.
        
        Args:
            graph: A pandas DataFrame representing the adjacency matrix of the 
                   causal graph. Rows and columns should be named after the 
                   variables in the dataset.
        """
        if not isinstance(graph, pd.DataFrame):
            raise TypeError(f"Graph must be a pandas DataFrame, got {type(graph).__name__}.")
        # eventually extract subset
        graph = graph.loc[self.concept_names, self.concept_names]
        self._graph = ConceptGraph(
            data=parse_tensor(graph, 'graph', self.precision),
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
        
        # eventually extract subset
        if isinstance(concepts, pd.DataFrame):
            concepts = concepts.loc[:, self.concept_names]
        elif isinstance(concepts, np.ndarray) or isinstance(concepts, Tensor):
            rows = [self.concept_names_all.index(name) for name in self.concept_names]
            concepts = concepts[:, rows]
        else:
            raise TypeError(f"Concepts must be a np.ndarray, pd.DataFrame, "
                f"or Tensor, got {type(concepts).__name__}.")
        
        #########################################################################
        ###### modify this to change convention for how to store concepts  ######
        #########################################################################
        # convert pd.Dataframe to tensor
        concepts = parse_tensor(concepts, 'concepts', self.precision)
        #########################################################################

        self.concepts = concepts

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
