"""
Toy DAG Dataset Module

This module implements a toy dataset with customizable DAG structure,
conditional probability tables, and autoencoder-based embeddings.
"""
import os
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

from ...annotations import Annotations, AxisAnnotation
from ..base import ConceptDataset
from ..preprocessing.autoencoder import extract_embs_from_autoencoder

logger = logging.getLogger(__name__)


class ToyDAGGenerator:
    """
    Generator for toy datasets based on DAG structure and conditional probability tables.
    
    This class generates synthetic data by sampling from a Bayesian Network defined
    by a DAG structure and conditional probability tables.
    """
    
    def __init__(
        self,
        variables: List[str],
        cardinalities: Dict[str, int],
        dag: List[Tuple[str, str]],
        conditional_probs: Dict[Union[Tuple[str, str], Tuple[str]], np.ndarray],
        root_priors: Optional[Dict[str, np.ndarray]] = None,
        seed: int = 42
    ):
        """
        Initialize the toy DAG generator.
        
        Args:
            variables: List of variable names (e.g., ['v1', 'v2', 'v3'])
            cardinalities: Dictionary mapping variable names to their cardinality
                          (e.g., {'v1': 2, 'v2': 3, 'v3': 2})
            dag: List of edges representing the DAG (e.g., [('v1', 'v2'), ('v2', 'v3')])
            conditional_probs: Dictionary mapping child nodes to conditional probability tables.
                              For a child with single parent, use key (parent, child) with shape
                              (child_cardinality, parent_cardinality).
                              For a child with multiple parents, use key (child,) with shape
                              (child_cardinality, parent1_cardinality, parent2_cardinality, ...).
                              Each CPT should sum to 1.0 along the first (child) dimension.
            root_priors: Optional dictionary mapping root variable names to their prior
                        probability arrays.  Each array has length equal to the
                        variable's cardinality and must sum to 1.
                        E.g. ``{'v1': np.array([0.3, 0.7])}`` for P(v1=0)=0.3, P(v1=1)=0.7.
                        Root variables without an entry are sampled uniformly.
            seed: Random seed for reproducibility
        """
        self.variables = variables
        self.cardinalities = cardinalities
        self.dag = dag
        self.conditional_probs = conditional_probs
        self.root_priors = root_priors if root_priors is not None else {}
        self.seed = seed
        
        # Build adjacency structure
        self.parents = defaultdict(list)
        self.children = defaultdict(list)
        for parent, child in dag:
            self.parents[child].append(parent)
            self.children[parent].append(child)
        
        # Find root nodes (no parents)
        self.roots = [v for v in variables if not self.parents[v]]
        
        # Topological ordering for sampling
        self.topo_order = self._topological_sort()
        
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort on the DAG."""
        in_degree = {v: len(self.parents[v]) for v in self.variables}
        queue = [v for v in self.variables if in_degree[v] == 0]
        topo_order = []
        
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for child in self.children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return topo_order
    
    def generate_sample(self) -> Dict[str, np.ndarray]:
        """
        Generate a single sample from the DAG.
        
        Returns:
            Dictionary mapping variable names to one-hot encoded values
        """
        sample = {}
        
        for var in self.topo_order:
            cardinality = self.cardinalities[var]
            
            if not self.parents[var]:
                # Root node: use prior if provided, otherwise uniform
                if var in self.root_priors:
                    probs = np.asarray(self.root_priors[var], dtype=np.float64)
                    value = np.random.choice(cardinality, p=probs)
                else:
                    value = np.random.randint(0, cardinality)
            else:
                # Non-root: sample based on conditional probability
                parents = self.parents[var]
                
                # Get parent values
                parent_values = tuple(np.argmax(sample[p]) for p in parents)
                
                # Get conditional probability table
                # Try multi-parent format first, then fall back to single-parent format
                if (var,) in self.conditional_probs:
                    # Multi-parent format: key is (child,)
                    cpt = np.asarray(self.conditional_probs[(var,)])
                    # Index: cpt[:, parent1_val, parent2_val, ...]
                    probs = cpt[(slice(None),) + parent_values]
                elif len(parents) == 1:
                    # Single-parent format: key is (parent, child)
                    edge = (parents[0], var)
                    cpt = np.asarray(self.conditional_probs[edge])
                    probs = cpt[:, parent_values[0]]
                else:
                    raise ValueError(
                        f"Variable '{var}' has {len(parents)} parents but no CPT found. "
                        f"Expected key ('{var}',) in conditional_probs."
                    )
                
                # Sample from the conditional distribution
                value = np.random.choice(cardinality, p=probs)
            
            # Store as one-hot encoding
            one_hot = np.zeros(cardinality, dtype=np.float32)
            one_hot[value] = 1.0
            sample[var] = one_hot
        
        return sample
    
    def generate_dataset(self, size: int) -> Dict[str, torch.Tensor]:
        """
        Generate a complete dataset.
        
        Args:
            size: Number of samples to generate
        
        Returns:
            Dictionary mapping variable names to tensors of shape (size, cardinality)
        """
        samples = []
        
        for _ in range(size):
            sample = self.generate_sample()
            samples.append(sample)
        
        # Convert to tensors
        dataset = {}
        for var in self.variables:
            var_data = np.stack([s[var] for s in samples])
            dataset[var] = torch.from_numpy(var_data).float()
        
        return dataset


class ToyDAGDataset(ConceptDataset):
    """
    Dataset class for toy DAG-based synthetic datasets.
    
    This dataset generates synthetic data based on a user-defined Directed Acyclic Graph (DAG)
    and conditional probability tables. It supports:
    - Custom DAG structures
    - Custom conditional probability tables
    - Optional latent variables (used for embedding generation but not exposed as concepts)
    - Autoencoder-based embedding generation
    
    Args:
        variables: List of all variable names in the DAG.
        cardinalities: Dictionary mapping variable names to their cardinality.
        dag: List of edges representing the DAG structure as (parent, child) tuples.
        conditional_probs: Dictionary mapping variables to their conditional probability tables.
                          Format: {(parent, child): array} or {(child,): array for multi-parent}
        root_priors: Optional dictionary mapping root variable names to their prior
                    probability arrays (length = cardinality, must sum to 1).
                    Root variables without an entry are sampled uniformly.
        root: Root directory to store/load the dataset. If None, creates local folder.
        seed: Random seed for data generation and reproducibility.
        n_gen: Total number of samples to generate.
        target_variable: Name of the target variable (optional, for metadata).
        latent_variables: List of latent variable names (used for embeddings but hidden from concepts).
        concept_subset: Optional subset of concept labels to use.
        label_descriptions: Optional dict mapping concept names to descriptions.
        autoencoder_kwargs: Configuration for autoencoder-based feature extraction.
    """
    
    def __init__(
        self,
        variables: List[str],
        cardinalities: Dict[str, int],
        dag: List[Tuple[str, str]],
        conditional_probs: Dict[Union[Tuple[str, str], Tuple[str]], Union[np.ndarray, list]],
        root_priors: Optional[Dict[str, Union[np.ndarray, list]]] = None,
        root: str = None,
        seed: int = 42,
        n_gen: int = 10000,
        target_variable: Optional[str] = None,
        latent_variables: Optional[List[str]] = None,
        concept_subset: Optional[list] = None,
        label_descriptions: Optional[dict] = None,
        autoencoder_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.variables = variables
        self.cardinalities = cardinalities
        self.dag = dag
        self.root_priors = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in root_priors.items()
        } if root_priors is not None else {}
        self.seed = seed
        self.n_gen = n_gen
        self.target_variable = target_variable
        self.latent_variables = latent_variables if latent_variables is not None else []
        self.autoencoder_kwargs = autoencoder_kwargs if autoencoder_kwargs is not None else {}
        self.label_descriptions = label_descriptions
        
        # Validate latent variables
        for lv in self.latent_variables:
            if lv not in variables:
                raise ValueError(f"Latent variable '{lv}' not in variables list")
        
        # Validate target variable
        if target_variable is not None and target_variable in self.latent_variables:
            raise ValueError(f"Target variable '{target_variable}' cannot be a latent variable")
        
        # Parse conditional probabilities (convert lists to numpy arrays, parse string keys)
        self.conditional_probs = self._parse_conditional_probs(conditional_probs, variables, dag)
        
        # If root is not provided, create a local folder automatically
        if root is None:
            root = os.path.join(os.getcwd(), 'data', 'toy_dag')
            
        self.root = root
        
        # Load or generate data
        embeddings, concepts, annotations, graph = self.load()
        
        # Initialize parent class
        super().__init__(
            input_data=embeddings,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
        )
    
    def _parse_conditional_probs(
        self, 
        conditional_probs: Dict, 
        variables: List[str], 
        dag: List[Tuple[str, str]]
    ) -> Dict:
        """Parse and validate conditional probability tables.
        
        Supports multiple formats:
        1. Direct numpy arrays: {(parent, child): array} or {(child,): array}
        2. Explicit parent states (NEW): {child: {"parent1=0,parent2=1": [probs], ...}}
        3. String keys: {"parent_child": array}
        """
        parsed_probs = {}
        
        # Build parent lists for context
        parents_dict = defaultdict(list)
        for parent, child in dag:
            parents_dict[child].append(parent)
        
        if conditional_probs is None or len(conditional_probs) == 0:
            # Generate default random CPTs
            for child, parents in parents_dict.items():
                child_card = self.cardinalities[child]
                
                if len(parents) == 1:
                    parent = parents[0]
                    parent_card = self.cardinalities[parent]
                    cpt = np.random.dirichlet(np.ones(child_card), size=parent_card).T
                    parsed_probs[(parent, child)] = cpt
                else:
                    parent_cards = tuple(self.cardinalities[p] for p in parents)
                    shape = (child_card,) + parent_cards
                    cpt = np.zeros(shape)
                    for idx in np.ndindex(parent_cards):
                        cpt[(slice(None),) + idx] = np.random.dirichlet(np.ones(child_card))
                    parsed_probs[(child,)] = cpt
        else:
            # Parse provided CPTs
            for key, value in conditional_probs.items():
                # Check if this is the new explicit parent states format
                if isinstance(key, str) and key in variables and isinstance(value, dict):
                    # New format: child: {"parent1=0,parent2=1": [probs], ...}
                    child = key
                    parents = parents_dict[child]
                    child_card = self.cardinalities[child]
                    
                    if len(parents) == 1:
                        # Single parent case
                        parent = parents[0]
                        parent_card = self.cardinalities[parent]
                        cpt = np.zeros((child_card, parent_card), dtype=np.float32)
                        
                        for state_str, probs in value.items():
                            # Parse "parent=0" format
                            parent_val = int(state_str.split('=')[1])
                            probs_array = np.array(probs, dtype=np.float32)
                            cpt[:, parent_val] = probs_array
                        
                        parsed_probs[(parent, child)] = cpt
                    else:
                        # Multiple parents case
                        parent_cards = tuple(self.cardinalities[p] for p in parents)
                        shape = (child_card,) + parent_cards
                        cpt = np.zeros(shape, dtype=np.float32)
                        
                        for state_str, probs in value.items():
                            # Parse "parent1=0,parent2=1,..." format
                            parent_vals = []
                            for assignment in state_str.split(','):
                                var_name, var_val = assignment.split('=')
                                var_val = int(var_val.strip())
                                parent_vals.append(var_val)
                            
                            probs_array = np.array(probs, dtype=np.float32)
                            idx = tuple([slice(None)] + parent_vals)
                            cpt[idx] = probs_array
                        
                        parsed_probs[(child,)] = cpt
                
                elif isinstance(key, str):
                    # Old string format "parent_child" or "child"
                    parts = key.split('_')
                    if len(parts) == 2:
                        parent, child = parts[0], parts[1]
                        parent_var = next((v for v in variables if v.endswith(parent)), parent)
                        child_var = next((v for v in variables if v.endswith(child)), child)
                        key = (parent_var, child_var)
                    elif len(parts) == 1:
                        child = parts[0]
                        child_var = next((v for v in variables if v.endswith(child)), child)
                        key = (child_var,)
                    
                    # Convert list to numpy array if necessary
                    if isinstance(value, list):
                        value = np.array(value, dtype=np.float32)
                    
                    parsed_probs[key] = value
                else:
                    # Direct tuple key format
                    if isinstance(value, list):
                        value = np.array(value, dtype=np.float32)
                    parsed_probs[key] = value
        
        return parsed_probs
    
    @property
    def raw_filenames(self) -> List[str]:
        """List of raw filenames that must be present to skip downloading."""
        return []  # Synthetic data, no download needed
    
    @property
    def processed_filenames(self) -> List[str]:
        """List of processed filenames that will be created during build step."""
        return [
            f"embeddings_N_{self.n_gen}_seed_{self.seed}.pt",
            f"concepts_N_{self.n_gen}_seed_{self.seed}.h5",
            f"annotations_N_{self.n_gen}_seed_{self.seed}.pt",
            f"graph_N_{self.n_gen}_seed_{self.seed}.h5"
        ]
    
    def download(self):
        """Download raw data files to root directory."""
        pass  # No external data to download
    
    def build(self):
        """Build processed dataset from raw files."""
        logger.info(f"Generating toy DAG dataset with {self.n_gen} samples...")
        
        # Create generator
        generator = ToyDAGGenerator(
            variables=self.variables,
            cardinalities=self.cardinalities,
            dag=self.dag,
            conditional_probs=self.conditional_probs,
            root_priors=self.root_priors,
            seed=self.seed
        )
        
        # Generate data (includes all variables, including latent)
        data = generator.generate_dataset(self.n_gen)
        
        # Convert to DataFrame for autoencoder
        # For binary variables, convert one-hot [1,0] or [0,1] to single value 0 or 1
        data_for_ae = {}
        ae_column_names = []
        for var in self.variables:
            if self.cardinalities[var] == 2:
                # Binary: extract single value (argmax of one-hot)
                data_for_ae[var] = data[var].argmax(dim=1).float().unsqueeze(1)
                ae_column_names.append(var)
            else:
                # Categorical: keep one-hot
                data_for_ae[var] = data[var]
                for i in range(self.cardinalities[var]):
                    ae_column_names.append(f"{var}_{i}")
        
        df = pd.DataFrame(
            torch.cat([data_for_ae[var] for var in self.variables], dim=1).numpy(),
            columns=ae_column_names
        )
        
        # Extract embeddings using autoencoder
        logger.info("Training autoencoder for embedding extraction...")
        embeddings = extract_embs_from_autoencoder(df, self.autoencoder_kwargs)
        
        # Create concepts tensor (exclude latent variables)
        # Keep original encoding format (one-hot for categorical, single value for binary)
        non_latent_vars = [v for v in self.variables if v not in self.latent_variables]
        concept_data = []
        column_names = []
        
        for var in non_latent_vars:
            if self.cardinalities[var] == 2:
                # Binary: use single value (argmax)
                concept_data.append(data[var].argmax(dim=1).float().unsqueeze(1))
                column_names.append(var)
            else:
                # Categorical: use one-hot with multiple columns
                concept_data.append(data[var])
                # Add column names for each dimension: var_0, var_1, ..., var_K-1
                for i in range(self.cardinalities[var]):
                    column_names.append(f"{var}_{i}")
        
        concepts_tensor = torch.cat(concept_data, dim=1)
        concepts = pd.DataFrame(concepts_tensor.numpy(), columns=column_names)
        
        # Create concept annotations
        concept_names = non_latent_vars
        concept_metadata = {
            name: {'type': 'discrete'} for name in concept_names
        }
        
        # Cardinalities: binary (2) -> 1, categorical (K) -> K
        cardinalities = [
            1 if self.cardinalities[var] == 2 else self.cardinalities[var] 
            for var in non_latent_vars
        ]
        
        annotations = Annotations({
            1: AxisAnnotation(
                labels=concept_names,
                cardinalities=cardinalities,
                metadata=concept_metadata
            )
        })
        
        # Create graph (adjacency matrix) - include all non-latent variables
        graph = pd.DataFrame(
            0, index=non_latent_vars, columns=non_latent_vars
        )
        for parent, child in self.dag:
            # Only include edges where neither parent nor child is latent
            if parent not in self.latent_variables and child not in self.latent_variables:
                graph.loc[parent, child] = 1
        graph = graph.astype(int)
        
        # Save all components
        logger.info(f"Saving dataset to {self.root_dir}")
        torch.save(embeddings, self.processed_paths[0])
        concepts.to_hdf(self.processed_paths[1], key="concepts", mode="w")
        torch.save(annotations, self.processed_paths[2])
        graph.to_hdf(self.processed_paths[3], key="graph", mode="w")
    
    def load_raw(self):
        """Load raw processed files."""
        self.maybe_build()
        
        logger.info(f"Loading dataset from {self.root_dir}")
        embeddings = torch.load(self.processed_paths[0], weights_only=False)
        concepts = pd.read_hdf(self.processed_paths[1], "concepts")
        annotations = torch.load(self.processed_paths[2], weights_only=False)
        graph = pd.read_hdf(self.processed_paths[3], "graph")
        
        # Ensure proper column names (for backward compatibility with cached files)
        # Reconstruct expected column names based on variables and cardinalities
        non_latent_vars = [v for v in self.variables if v not in self.latent_variables]
        expected_columns = []
        for var in non_latent_vars:
            if self.cardinalities[var] == 2:
                expected_columns.append(var)
            else:
                for i in range(self.cardinalities[var]):
                    expected_columns.append(f"{var}_{i}")
        
        # Set column names if not already set
        if list(concepts.columns) != expected_columns:
            concepts.columns = expected_columns
        
        return embeddings, concepts, annotations, graph
    
    def load(self):
        """Load and optionally preprocess dataset."""
        embeddings, concepts, annotations, graph = self.load_raw()
        
        # Add any additional preprocessing here if needed
        
        return embeddings, concepts, annotations, graph
