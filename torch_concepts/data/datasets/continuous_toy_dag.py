"""
Toy Function-based DAG Dataset Module

This module implements a toy dataset where variables are generated according to
user-defined functions (sympy expressions) in a DAG structure. Supports both
binary and continuous variables.
"""
import os
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None

from ...annotations import Annotations, AxisAnnotation
from ..base import ConceptDataset
from ..preprocessing.autoencoder import extract_embs_from_autoencoder

logger = logging.getLogger(__name__)


class ToyFunctionDAGGenerator:
    """
    Generator for datasets based on DAG structure with function-based node generation.
    
    Uses sympy expressions to define how child nodes are computed from parent nodes.
    """
    
    def __init__(
        self,
        variables: List[str],
        dag: List[Tuple[str, str]],
        node_functions: Dict[str, callable],
        variable_type: str = 'continuous',
        cardinalities: Optional[Dict[str, int]] = None,
        source_mean: float = 0.0,
        source_std: float = 1.0,
        gamma: float = 0.0,
        seed: int = 42
    ):
        """
        Initialize the function-based DAG generator.
        
        Args:
            variables: List of variable names
            dag: List of (parent, child) edges
            node_functions: Dict mapping node names to callable functions
            variable_type: 'binary' or 'continuous'
            cardinalities: For binary variables, dict of cardinalities
            source_mean: Mean for sampling root nodes (continuous only)
            source_std: Std for sampling root nodes (continuous only)
            gamma: Noise parameter (continuous: additive, binary: swap probability)
            seed: Random seed
        """
        self.variables = variables
        self.dag = dag
        self.node_functions = node_functions
        self.variable_type = variable_type.lower()
        self.source_mean = source_mean
        self.source_std = source_std
        self.gamma = gamma
        self.seed = seed
        
        if self.variable_type == 'binary':
            self.cardinalities = cardinalities if cardinalities else {v: 2 for v in variables}
        else:
            self.cardinalities = None
        
        # Build adjacency structure
        self.parents = defaultdict(list)
        self.children = defaultdict(list)
        for parent, child in dag:
            self.parents[child].append(parent)
            self.children[parent].append(child)
        
        self.roots = [v for v in variables if not self.parents[v]]
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
    
    def generate_sample(self) -> Dict[str, float]:
        """Generate a single sample using function-based rules."""
        sample = {}
        
        for var in self.topo_order:
            if not self.parents[var]:
                # Root node
                if self.variable_type == 'binary':
                    value = np.random.randint(0, self.cardinalities[var])
                else:
                    value = np.random.normal(self.source_mean, self.source_std)
            else:
                # Non-root: compute using function
                parents = self.parents[var]
                parent_values = [sample[p] for p in parents]
                
                if var in self.node_functions:
                    value = self.node_functions[var](parent_values)
                    
                    if self.variable_type == 'binary':
                        value = int(value) % self.cardinalities[var]
                else:
                    # Default: random sampling
                    if self.variable_type == 'binary':
                        value = np.random.randint(0, self.cardinalities[var])
                    else:
                        value = np.random.normal(0, 1)
            
            sample[var] = value
        
        return sample
    
    def add_noise(self, sample: Dict[str, float]) -> Dict[str, float]:
        """Add noise based on variable type."""
        noisy_sample = sample.copy()
        
        if self.variable_type == 'continuous':
            for var in self.variables:
                noise = np.random.normal(0, 1)
                noisy_sample[var] = sample[var] + self.gamma * noise
        else:  # binary
            for var in self.variables:
                if np.random.random() < self.gamma:
                    current_value = int(noisy_sample[var])
                    possible_values = [v for v in range(self.cardinalities[var]) if v != current_value]
                    if possible_values:
                        noisy_sample[var] = np.random.choice(possible_values)
        
        return noisy_sample
    
    def generate_dataset(self, size: int) -> np.ndarray:
        """Generate dataset as matrix (size, num_variables)."""
        samples = []
        
        for _ in range(size):
            sample = self.generate_sample()
            sample = self.add_noise(sample)
            sample_list = [sample[var] for var in self.variables]
            samples.append(sample_list)
        
        if self.variable_type == 'binary':
            return np.array(samples, dtype=np.int32)
        else:
            return np.array(samples, dtype=np.float32)


class ToyFunctionDAGDataset(ConceptDataset):
    """
    Dataset for function-based DAG generation using sympy expressions.
    
    Supports both binary and continuous variables where child nodes are computed
    from parent nodes using user-defined mathematical functions.
    
    Args:
        variables: List of variable names
        dag: List of (parent, child) edges
        node_functions: Dict mapping node names to sympy expression strings
        variable_type: 'binary' or 'continuous'
        cardinalities: For binary, dict of cardinalities (default: 2 for all)
        source_mean: For continuous, mean for root node sampling
        source_std: For continuous, std for root node sampling
        gamma: Noise parameter (continuous: additive Gaussian, binary: swap prob)
        theta: Embedding noise parameter
        root: Root directory for data storage
        seed: Random seed
        n_gen: Number of samples to generate
        latent_variables: Variables used for embedding but hidden from concepts
        concept_subset: Subset of concepts to use
        label_descriptions: Descriptions for variables
        autoencoder_kwargs: Autoencoder configuration
    """
    
    def __init__(
        self,
        variables: List[str],
        dag: List[Tuple[str, str]],
        node_functions: Dict[str, str],
        variable_type: str = 'continuous',
        cardinalities: Optional[Dict[str, int]] = None,
        source_mean: float = 0.0,
        source_std: float = 1.0,
        gamma: float = 0.0,
        theta: float = 0.0,
        root: str = None,
        seed: int = 42,
        n_gen: int = 10000,
        latent_variables: Optional[List[str]] = None,
        concept_subset: Optional[list] = None,
        label_descriptions: Optional[dict] = None,
        autoencoder_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        if not SYMPY_AVAILABLE:
            raise ImportError("sympy is required for ToyFunctionDAGDataset. Install it with: pip install sympy")
        
        self.variables = variables
        self.dag = dag
        self.node_functions_str = node_functions
        self.variable_type = variable_type.lower()
        self.source_mean = source_mean
        self.source_std = source_std
        self.gamma = gamma
        self.theta = theta
        self.seed = seed
        self.n_gen = n_gen
        self.latent_variables = latent_variables if latent_variables else []
        self.autoencoder_kwargs = autoencoder_kwargs if autoencoder_kwargs else {}
        self.label_descriptions = label_descriptions
        
        if self.variable_type == 'binary':
            self.cardinalities = cardinalities if cardinalities else {v: 2 for v in variables}
        else:
            self.cardinalities = None
        
        # Validate
        for lv in self.latent_variables:
            if lv not in variables:
                raise ValueError(f"Latent variable '{lv}' not in variables list")
        
        # Parse node functions
        self.parsed_functions = self._parse_node_functions(node_functions)
        
        if root is None:
            root = os.path.join(os.getcwd(), 'data', 'toy_function_dag')
        
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
    
    def _parse_node_functions(self, node_functions: Dict[str, str]) -> Dict[str, callable]:
        """Parse sympy expression strings into callable functions."""
        parsed = {}
        
        parents_dict = defaultdict(list)
        for parent, child in self.dag:
            parents_dict[child].append(parent)
        
        for node, func_str in node_functions.items():
            if callable(func_str):
                parsed[node] = func_str
                continue
            
            try:
                parents = parents_dict.get(node, [])
                
                if self.variable_type == 'binary':
                    parent_symbols = {parent: sp.Symbol(parent, bool=True) for parent in parents}
                else:
                    parent_symbols = {parent: sp.Symbol(parent) for parent in parents}
                
                expr = sp.sympify(func_str)
                
                def make_func(expression, parents_list, symbols_dict, node_name=node):
                    def eval_func(parent_values):
                        if not parent_values:
                            return 0.0 if self.variable_type == 'continuous' else 0
                        
                        if self.variable_type == 'binary':
                            int_values = [1 if v != 0 else 0 for v in parent_values]
                            subs_dict = {symbols_dict[parents_list[i]]: int_values[i] 
                                       for i in range(len(parent_values))}
                        else:
                            subs_dict = {symbols_dict[parents_list[i]]: parent_values[i] 
                                       for i in range(len(parent_values))}
                        
                        result = expression.subs(subs_dict)
                        
                        if self.variable_type == 'binary':
                            if hasattr(result, 'is_Boolean') and result.is_Boolean:
                                result = 1 if bool(result) else 0
                            else:
                                result = int(float(result.evalf()))
                            
                            if self.cardinalities:
                                result = result % self.cardinalities.get(node_name, 2)
                        else:
                            result = float(result.evalf())
                        
                        return result
                    return eval_func
                
                parsed[node] = make_func(expr, parents, parent_symbols, node)
                
            except Exception as e:
                raise ValueError(f"Failed to parse function for node '{node}': {func_str}\nError: {e}")
        
        return parsed
    
    @property
    def raw_filenames(self) -> List[str]:
        return []
    
    @property
    def processed_filenames(self) -> List[str]:
        return [
            f"embeddings_{self.variable_type}_N_{self.n_gen}_seed_{self.seed}.pt",
            f"concepts_{self.variable_type}_N_{self.n_gen}_seed_{self.seed}.h5",
            f"annotations_{self.variable_type}_N_{self.n_gen}_seed_{self.seed}.pt",
            f"graph_{self.variable_type}_N_{self.n_gen}_seed_{self.seed}.h5"
        ]
    
    def download(self):
        pass
    
    def build(self):
        logger.info(f"Generating function-based {self.variable_type} DAG dataset with {self.n_gen} samples...")
        
        generator = ToyFunctionDAGGenerator(
            variables=self.variables,
            dag=self.dag,
            node_functions=self.parsed_functions,
            variable_type=self.variable_type,
            cardinalities=self.cardinalities,
            source_mean=self.source_mean,
            source_std=self.source_std,
            gamma=self.gamma,
            seed=self.seed
        )
        
        # Generate data matrix
        data_matrix = generator.generate_dataset(self.n_gen)
        
        # Convert matrix to DataFrame for autoencoder
        df = pd.DataFrame(data_matrix)
        
        # Extract embeddings (with theta noise applied internally if supported)
        logger.info("Training autoencoder for embedding extraction...")
        embeddings = extract_embs_from_autoencoder(df, self.autoencoder_kwargs)
        
        # Add theta noise to embeddings if specified
        if self.theta > 0:
            noise = torch.randn_like(embeddings)
            embeddings = embeddings + self.theta * noise
        
        # Convert matrix to concept tensors
        non_latent_vars = [v for v in self.variables if v not in self.latent_variables]
        non_latent_indices = [i for i, v in enumerate(self.variables) if v not in self.latent_variables]
        
        concept_data = []
        column_names = []
        
        for idx, var in zip(non_latent_indices, non_latent_vars):
            if self.variable_type == 'binary':
                # One-hot encoding for categorical
                card = self.cardinalities[var]
                states = data_matrix[:, idx].astype(np.int32)
                one_hot = np.zeros((len(states), card), dtype=np.float32)
                one_hot[np.arange(len(states)), states] = 1.0
                
                if card == 2:
                    # Binary: use single column
                    concept_data.append(states.reshape(-1, 1).astype(np.float32))
                    column_names.append(var)
                else:
                    concept_data.append(one_hot)
                    for i in range(card):
                        column_names.append(f"{var}_{i}")
            else:
                # Continuous: single column
                values = data_matrix[:, idx].astype(np.float32).reshape(-1, 1)
                concept_data.append(values)
                column_names.append(var)
        
        concepts_array = np.concatenate(concept_data, axis=1)
        concepts = pd.DataFrame(concepts_array, columns=column_names)
        
        # Create annotations
        concept_names = non_latent_vars
        concept_metadata = {
            name: {'type': self.variable_type} for name in concept_names
        }
        
        if self.variable_type == 'binary':
            cardinalities = [
                1 if self.cardinalities[var] == 2 else self.cardinalities[var]
                for var in non_latent_vars
            ]
        else:
            cardinalities = [1 for _ in non_latent_vars]
        
        annotations = Annotations({
            1: AxisAnnotation(
                labels=concept_names,
                cardinalities=cardinalities,
                metadata=concept_metadata
            )
        })
        
        # Create graph
        graph = pd.DataFrame(0, index=non_latent_vars, columns=non_latent_vars)
        for parent, child in self.dag:
            if parent not in self.latent_variables and child not in self.latent_variables:
                graph.loc[parent, child] = 1
        graph = graph.astype(int)
        
        # Save
        logger.info(f"Saving dataset to {self.root_dir}")
        torch.save(embeddings, self.processed_paths[0])
        concepts.to_hdf(self.processed_paths[1], key="concepts", mode="w")
        torch.save(annotations, self.processed_paths[2])
        graph.to_hdf(self.processed_paths[3], key="graph", mode="w")
    
    def load_raw(self):
        self.maybe_build()
        
        logger.info(f"Loading dataset from {self.root_dir}")
        embeddings = torch.load(self.processed_paths[0], weights_only=False)
        concepts = pd.read_hdf(self.processed_paths[1], "concepts")
        annotations = torch.load(self.processed_paths[2], weights_only=False)
        graph = pd.read_hdf(self.processed_paths[3], "graph")
        
        return embeddings, concepts, annotations, graph
    
    def load(self):
        embeddings, concepts, annotations, graph = self.load_raw()
        return embeddings, concepts, annotations, graph
