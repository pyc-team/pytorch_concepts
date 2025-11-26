import numpy as np
import torch
import pandas as pd
import os
import logging
from numpy.random import multivariate_normal, uniform
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_spd_matrix, make_low_rank_matrix
from typing import List, Optional, Union

from ..base.dataset import ConceptDataset
from ...annotations import Annotations, AxisAnnotation

logger = logging.getLogger(__name__)


def _xor(size, random_state=42):
    # sample from uniform distribution
    np.random.seed(random_state)
    x = np.random.uniform(0, 1, (size, 2))
    c = np.stack([
        x[:, 0] > 0.5,
        x[:, 1] > 0.5,
    ]).T
    y = np.logical_xor(c[:, 0], c[:, 1])

    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.FloatTensor(y)

    cy = torch.cat([c, y.unsqueeze(-1)], dim=-1)
    cy_names = ['C1', 'C2', 'xor']
    graph_c_to_y = pd.DataFrame(
        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 0]],
        index=cy_names,
        columns=cy_names,
    )
    return x, cy, cy_names, graph_c_to_y


def _trigonometry(size, random_state=42):
    np.random.seed(random_state)
    h = np.random.normal(0, 2, (size, 3))
    x, y, z = h[:, 0], h[:, 1], h[:, 2]

    # raw features
    input_features = np.stack([
        np.sin(x) + x,
        np.cos(x) + x,
        np.sin(y) + y,
        np.cos(y) + y,
        np.sin(z) + z,
        np.cos(z) + z,
        x ** 2 + y ** 2 + z ** 2,
    ]).T

    # concepts
    concepts = np.stack([
        x > 0,
        y > 0,
        z > 0,
    ]).T

    # task
    downstream_task = (x + y + z) > 1

    input_features = torch.FloatTensor(input_features)
    concepts = torch.FloatTensor(concepts)
    downstream_task = torch.FloatTensor(downstream_task)

    cy = torch.cat([concepts, downstream_task.unsqueeze(-1)], dim=-1)
    cy_names = ['C1', 'C2', 'C3', 'sumGreaterThan1']
    graph_c_to_y = pd.DataFrame(
        [[0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 0]],
        index=cy_names,
        columns=cy_names,
    )

    return input_features, cy, cy_names, graph_c_to_y


def _dot(size, random_state=42):
    # sample from normal distribution
    emb_size = 2
    np.random.seed(random_state)
    v1 = np.random.randn(size, emb_size) * 2
    v2 = np.ones(emb_size)
    np.random.seed(random_state)
    v3 = np.random.randn(size, emb_size) * 2
    v4 = -np.ones(emb_size)
    x = np.hstack([v1+v3, v1-v3])
    c = np.stack([
        np.dot(v1, v2).ravel() > 0,
        np.dot(v3, v4).ravel() > 0,
    ]).T
    y = ((v1*v3).sum(axis=-1) > 0).astype(np.int64)

    x = torch.FloatTensor(x)
    c = torch.FloatTensor(c)
    y = torch.Tensor(y)

    cy = torch.cat([c, y.unsqueeze(-1)], dim=-1)
    cy_names = ['dotV1V2GreaterThan0', 'dotV3V4GreaterThan0', 'dotV1V3GreaterThan0']
    graph_c_to_y = pd.DataFrame(
        [[0, 0, 1],
         [0, 0, 1],
         [0, 0, 0]],
        index=cy_names,
        columns=cy_names,
    )

    return x, cy, cy_names, graph_c_to_y


def _toy_problem(n_samples: int = 10, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    A = torch.randint(0, 2, (n_samples,), dtype=torch.bool)
    torch.manual_seed(seed + 1)
    B = torch.randint(0, 2, (n_samples,), dtype=torch.bool)

    # Column C is true if B is true, randomly true/false if B is false
    C = ~B

    # Column D is true if A or C is true, randomly true/false if both are false
    D = A & C

    # Combine all columns into a matrix
    return torch.stack((A, B, C, D), dim=1).float()


def _checkmark(n_samples: int = 10, seed: int =42, perturb: float = 0.1):
    x = _toy_problem(n_samples, seed)
    c = x.clone()
    torch.manual_seed(seed)
    x = x * 2 - 1 + torch.randn_like(x) * perturb

    # Create DAG as pandas DataFrame with proper column/row names
    concept_names = ['A', 'B', 'C', 'D']
    dag_array = [[0, 0, 0, 1],  # A influences D
                 [0, 0, 1, 0],  # B influences C
                 [0, 0, 0, 1],  # C influences D
                 [0, 0, 0, 0]]  # D doesn't influence others
    dag = pd.DataFrame(dag_array, index=concept_names, columns=concept_names)

    return x, c, concept_names, dag


class ToyDataset(ConceptDataset):
    """
    Synthetic datasets for concept-based learning experiments.

    This class provides several toy datasets with known ground-truth concept
    relationships and causal structures. Each dataset includes input features,
    binary concepts, tasks, and a directed acyclic graph (DAG) representing
    concept-to-task relationships.

    Available Datasets
    ------------------
    - **xor**: Simple XOR dataset with 2 input features, 2 concepts (C1, C2), and
      1 task (xor). The task is the XOR of the two concepts.
    - **trigonometry**: Dataset with 7 trigonometric input features derived from
      3 hidden variables, 3 concepts (C1, C2, C3) representing the signs of the
      hidden variables, and 1 task (sumGreaterThan1).
    - **dot**: Dataset with 4 input features, 2 concepts based on dot products
      (dotV1V2GreaterThan0, dotV3V4GreaterThan0), and 1 task (dotV1V3GreaterThan0).
    - **checkmark**: Dataset with 4 input features and 4 concepts (A, B, C, D),
      where C = NOT B and D = A AND C, demonstrating causal relationships.

    Parameters
    ----------
    dataset : str
        Name of the toy dataset to load. Must be one of: 'xor', 'trigonometry',
        'dot', or 'checkmark'.
    root : str, optional
        Root directory to store/load the dataset files. If None, defaults to
        './data/toy_datasets/{dataset_name}'. Default: None
    seed : int, optional
        Random seed for reproducible data generation. Default: 42
    n_gen : int, optional
        Number of samples to generate. Default: 10000
    concept_subset : list of str, optional
        Subset of concept names to use. If provided, only the specified concepts
        will be included in the dataset. Default: None (use all concepts)

    Attributes
    ----------
    input_data : torch.Tensor
        Input features tensor of shape (n_samples, n_features).
    concepts : torch.Tensor
        Concepts and tasks tensor of shape (n_samples, n_concepts + n_tasks).
        Note: This includes both concepts and tasks concatenated.
    annotations : Annotations
        Metadata about concept names, cardinalities, and types.
    graph : pandas.DataFrame
        Directed acyclic graph representing concept-to-task relationships.
        Stored as an adjacency matrix with concept/task names as indices.
    concept_names : list of str
        Names of all concepts and tasks in the dataset.
    n_concepts : int
        Total number of concepts and tasks (includes both).
    n_features : tuple or int
        Dimensionality of input features.

    Examples
    --------
    Basic usage with XOR dataset:

    >>> from torch_concepts.data.datasets import ToyDataset
    >>>
    >>> # Create XOR dataset with 1000 samples
    >>> dataset = ToyDataset(dataset='xor', seed=42, n_gen=1000)
    >>> print(f"Dataset size: {len(dataset)}")
    >>> print(f"Input features: {dataset.n_features}")
    >>> print(f"Concepts: {dataset.concept_names}")
    >>>
    >>> # Access a single sample
    >>> sample = dataset[0]
    >>> x = sample['inputs']['x']  # input features
    >>> c = sample['concepts']['c']  # concepts and task
    >>>
    >>> # Get concept graph
    >>> print(dataset.graph)

    References
    ----------
    .. [1] Espinosa Zarlenga, M., et al. "Concept Embedding Models:
           Beyond the Accuracy-Explainability Trade-Off",
           NeurIPS 2022. https://arxiv.org/abs/2209.09056
    .. [2] Dominici, G., et al. (2025). "Causal Concept Graph
           Models: Beyond Causal Opacity in Deep Learning."
           ICLR 2025. https://arxiv.org/abs/2405.16507

    See Also
    --------
    CompletenessDataset : Synthetic dataset for concept completeness experiments
    """

    def __init__(
            self,
            dataset: str,  # name of the toy dataset ('xor', 'trigonometry', 'dot', 'checkmark')
            root: str = None,  # root directory to store/load the dataset
            seed: int = 42,  # seed for data generation
            n_gen: int = 10000,  # number of samples to generate
            concept_subset: Optional[list] = None,  # subset of concept labels
    ):
        if dataset.lower() not in TOYDATASETS:
            raise ValueError(f"Dataset {dataset} not found. Available datasets: {TOYDATASETS}")

        self.dataset_name = dataset.lower()
        self.name = dataset.lower()
        self.seed = seed

        # If root is not provided, create a local folder automatically
        if root is None:
            root = os.path.join(os.getcwd(), 'data', 'toy_datasets', self.dataset_name)

        self.root = root
        self.n_gen = n_gen

        # Load data (will generate if not exists)
        input_data, concepts, annotations, graph = self.load()

        # Initialize parent class
        super().__init__(
            input_data=input_data,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
            name=f"ToyDataset_{dataset}"
        )

    @property
    def raw_filenames(self) -> List[str]:
        """No raw files needed - data is generated."""
        return []

    @property
    def processed_filenames(self) -> List[str]:
        """List of processed filenames that will be created during build step."""
        files = [
            f"{self.dataset_name}_input_N_{self.n_gen}_seed_{self.seed}.pt",
            f"{self.dataset_name}_concepts_N_{self.n_gen}_seed_{self.seed}.pt",
            f"{self.dataset_name}_annotations.pt",
            f"{self.dataset_name}_graph.h5",
        ]
        return files

    def download(self):
        """No download needed for toy datasets."""
        pass

    def build(self):
        """Generate synthetic data and save to disk."""
        logger.info(f"Generating {self.dataset_name} dataset with n_gen={self.n_gen}, seed={self.seed}")

        # Select the appropriate data generation function
        if self.dataset_name == 'xor':
            input_data, concepts, concept_names, graph = _xor(self.n_gen, self.seed)
        elif self.dataset_name == 'trigonometry':
            input_data, concepts, concept_names, graph = _trigonometry(self.n_gen, self.seed)
        elif self.dataset_name == 'dot':
            input_data, concepts, concept_names, graph = _dot(self.n_gen, self.seed)
        elif self.dataset_name == 'checkmark':
            input_data, concepts, concept_names, graph = _checkmark(self.n_gen, self.seed)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Create annotations
        concept_metadata = {
            name: {'type': 'discrete'} for name in concept_names
        }
        cardinalities = tuple([1] * len(concept_names))  # All binary concepts

        annotations = Annotations({
            1: AxisAnnotation(
                labels=concept_names,
                cardinalities=cardinalities,
                metadata=concept_metadata
            )
        })

        # Save all data
        logger.info(f"Saving dataset to {self.root_dir}")
        os.makedirs(self.root_dir, exist_ok=True)

        torch.save(input_data, self.processed_paths[0])
        torch.save(concepts, self.processed_paths[1])
        torch.save(annotations, self.processed_paths[2])
        graph.to_hdf(self.processed_paths[3], key="graph", mode="w")

    def load_raw(self):
        """Load the generated dataset from disk."""
        self.maybe_build()
        logger.info(f"Loading dataset from {self.root_dir}")

        input_data = torch.load(self.processed_paths[0], weights_only=False)
        concepts = torch.load(self.processed_paths[1], weights_only=False)
        annotations = torch.load(self.processed_paths[2], weights_only=False)
        graph = pd.read_hdf(self.processed_paths[3], "graph")

        return input_data, concepts, annotations, graph

    def load(self):
        """Load the dataset (wraps load_raw)."""
        return self.load_raw()


def _relu(x):
    return x * (x > 0)


def _random_nonlin_map(n_in, n_out, n_hidden, rank=1000):
    W_0 = make_low_rank_matrix(n_in, n_hidden, effective_rank=rank)
    W_1 = make_low_rank_matrix(n_hidden, n_hidden, effective_rank=rank)
    W_2 = make_low_rank_matrix(n_hidden, n_out, effective_rank=rank)
    # No biases
    b_0 = np.random.uniform(0, 0, (1, n_hidden))
    b_1 = np.random.uniform(0, 0, (1, n_hidden))
    b_2 = np.random.uniform(0, 0, (1, n_out))

    nlin_map = lambda x: np.matmul(
        _relu(
            np.matmul(
                _relu(np.matmul(x, W_0) + np.tile(b_0, (x.shape[0], 1))),
                W_1,
            ) +
            np.tile(b_1, (x.shape[0], 1))
        ),
        W_2,
    ) + np.tile(b_2, (x.shape[0], 1))

    return nlin_map


def _complete(
    n_samples: int = 10,
    p: int = 2,
    n_views: int = 10,
    n_concepts: int = 2,
    n_hidden_concepts: int = 0,
    n_tasks: int = 1,
    seed: int = 42,
):
    total_concepts = n_concepts + n_hidden_concepts

    # Replicability
    np.random.seed(seed)

    # Generate covariates
    mu = uniform(-5, 5, p * n_views)
    sigma = make_spd_matrix(p * n_views, random_state=seed)
    X = multivariate_normal(mean=mu, cov=sigma, size=n_samples)
    ss = StandardScaler()
    X = ss.fit_transform(X)
    # Produce different views
    X_views = np.zeros((n_samples, n_views, p))
    for v in range(n_views):
        X_views[:, v] = X[:, (v * p):(v * p + p)]

    # Nonlinear maps
    g = _random_nonlin_map(
        n_in=p * n_views,
        n_out=total_concepts,
        n_hidden=int((p * n_views + total_concepts) / 2),
    )
    f = _random_nonlin_map(
        n_in=total_concepts,
        n_out=n_tasks,
        n_hidden=int(total_concepts / 2),
    )

    # Generate concepts
    c = g(X)
    c = torch.sigmoid(torch.FloatTensor(c))
    c = (c >= 0.5) * 1.0

    # Generate labels
    y = f(c.detach().numpy())
    y = torch.sigmoid(torch.FloatTensor(y))
    y = (y >= 0.5) * 1.0

    u = c[:, :n_concepts]
    X = torch.FloatTensor(X)
    u = torch.FloatTensor(u)
    y = torch.FloatTensor(y)

    uy = torch.cat([u, y], dim=-1)
    uy_names = [f'C{i}' for i in range(n_concepts)] + [f'y{i}' for i in range(n_tasks)]
    graph_c_to_y = pd.DataFrame(
        np.zeros((n_concepts + n_tasks, n_concepts + n_tasks)),
        index=uy_names,
        columns=uy_names,
    )
    for i in range(n_concepts):
        for j in range(n_tasks):
            graph_c_to_y.iloc[i, n_concepts + j] = 1  # concepts influence tasks

    return X, uy, uy_names, graph_c_to_y


class CompletenessDataset(ConceptDataset):
    """
    Synthetic dataset for concept bottleneck completeness experiments.

    This dataset generates synthetic data to study complete vs. incomplete concept
    bottlenecks. Data is generated using randomly initialized multi-layer perceptrons
    with ReLU activations. Input features are sampled from a multivariate normal
    distribution, and concepts are derived through nonlinear transformations.
    Hidden concepts can be included to simulate incomplete bottlenecks.

    The dataset uses a two-stage generation process:
    1. Map inputs X to concepts C (both observed and hidden) via nonlinear function g
    2. Map concepts C to tasks Y via nonlinear function f

    Parameters
    ----------
    name : str
        Name identifier for the dataset (used for file storage).
    root : str, optional
        Root directory to store/load the dataset files. If None, defaults to
        './data/completeness_datasets/{name}'. Default: None
    seed : int, optional
        Random seed for reproducible data generation. Default: 42
    n_gen : int, optional
        Number of samples to generate. Default: 10000
    p : int, optional
        Dimensionality of each view (feature group). Default: 2
    n_views : int, optional
        Number of views/feature groups. Total input features = p * n_views.
        Default: 10
    n_concepts : int, optional
        Number of observable concepts (not including hidden concepts). Default: 2
    n_hidden_concepts : int, optional
        Number of hidden concepts not observable in the bottleneck. Use this to
        simulate incomplete concept bottlenecks. Default: 0
    n_tasks : int, optional
        Number of downstream tasks to predict. Default: 1
    concept_subset : list of str, optional
        Subset of concept names to use. If provided, only the specified concepts
        will be included. Concept names follow format 'C0', 'C1', etc. Default: None

    Attributes
    ----------
    input_data : torch.Tensor
        Input features tensor of shape (n_samples, p * n_views).
    concepts : torch.Tensor
        Concepts and tasks tensor of shape (n_samples, n_concepts + n_tasks).
        Note: Hidden concepts are NOT included in this tensor.
    annotations : Annotations
        Metadata about concept names, cardinalities, and types.
    graph : pandas.DataFrame
        Directed acyclic graph representing concept-to-task relationships.
        All concepts influence all tasks in this dataset.
    concept_names : list of str
        Names of all concepts and tasks. Format: ['C0', 'C1', ..., 'y0', 'y1', ...]
    n_concepts : int
        Total number of observable concepts and tasks (includes both, excludes hidden).
    n_features : tuple or int
        Dimensionality of input features (p * n_views).

    Examples
    --------
    Basic usage with complete bottleneck:

    >>> from torch_concepts.data.datasets import CompletenessDataset
    >>>
    >>> # Create dataset with complete bottleneck (no hidden concepts)
    >>> dataset = CompletenessDataset(
    ...     name='complete_exp',
    ...     n_gen=5000,
    ...     n_concepts=5,
    ...     n_hidden_concepts=0,
    ...     seed=42
    ... )
    >>> print(f"Dataset size: {len(dataset)}")
    >>> print(f"Input features: {dataset.n_features}")
    >>> print(f"Concepts: {dataset.concept_names}")

    Creating incomplete bottleneck with hidden concepts:

    >>> from torch_concepts.data.datasets import CompletenessDataset
    >>>
    >>> # Create dataset with incomplete bottleneck
    >>> dataset = CompletenessDataset(
    ...     name='incomplete_exp',
    ...     n_gen=5000,
    ...     n_concepts=3,          # 3 observable concepts
    ...     n_hidden_concepts=2,   # 2 hidden concepts (not in bottleneck)
    ...     seed=42
    ... )
    >>> # The hidden concepts affect tasks but are not observable
    >>> print(f"Observable concepts: {dataset.n_concepts}")

    References
    ----------
    .. [1] Laguna, S., et al. "Beyond Concept Bottleneck Models: How to Make Black Boxes
           Intervenable?", NeurIPS 2024. https://arxiv.org/abs/2401.13544
    """

    def __init__(
            self,
            name: str,  # name of the dataset
            root: str = None,  # root directory to store/load the dataset
            seed: int = 42,  # seed for data generation
            n_gen: int = 10000,  # number of samples to generate
            p: int = 2,  # dimensionality of each view
            n_views: int = 10,  # number of views
            n_concepts: int = 2,  # number of concepts
            n_hidden_concepts: int = 0,  # number of hidden concepts
            n_tasks: int = 1,  # number of tasks
            concept_subset: Optional[list] = None,  # subset of concept labels
    ):
        self.name = name
        self.seed = seed

        # If root is not provided, create a local folder automatically
        if root is None:
            root = os.path.join(os.getcwd(), 'data', 'completeness_datasets', name)

        self.root = root
        self.n_gen = n_gen
        self.p = p
        self.n_views = n_views
        self._n_concepts = n_concepts  # Use internal variable to avoid property conflict
        self._n_hidden_concepts = n_hidden_concepts
        self._n_tasks = n_tasks

        # Load data (will generate if not exists)
        input_data, concepts, annotations, graph = self.load()

        # Initialize parent class
        super().__init__(
            input_data=input_data,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,
            name=name
        )

    @property
    def raw_filenames(self) -> List[str]:
        """No raw files needed - data is generated."""
        return []

    @property
    def processed_filenames(self) -> List[str]:
        """List of processed filenames that will be created during build step."""
        return [
            f"input_N_{self.n_gen}_p_{self.p}_views_{self.n_views}_concepts_{self._n_concepts}_hidden_{self._n_hidden_concepts}_seed_{self.seed}.pt",
            f"concepts_N_{self.n_gen}_p_{self.p}_views_{self.n_views}_concepts_{self._n_concepts}_hidden_{self._n_hidden_concepts}_seed_{self.seed}.pt",
            f"annotations_concepts_{self._n_concepts}.pt",
            "graph.h5",
        ]

    def download(self):
        """No download needed for synthetic datasets."""
        pass

    def build(self):
        """Generate synthetic completeness data and save to disk."""
        logger.info(f"Generating completeness dataset with n_gen={self.n_gen}, seed={self.seed}")

        # Generate data using _complete function
        input_data, concepts, concept_names, graph = _complete(
            n_samples=self.n_gen,
            p=self.p,
            n_views=self.n_views,
            n_concepts=self._n_concepts,
            n_hidden_concepts=self._n_hidden_concepts,
            n_tasks=self._n_tasks,
            seed=self.seed,
        )

        # Create annotations
        concept_metadata = {
            name: {'type': 'discrete'} for name in concept_names
        }
        cardinalities = tuple([1] * len(concept_names))  # All binary concepts

        annotations = Annotations({
            1: AxisAnnotation(
                labels=concept_names,
                cardinalities=cardinalities,
                metadata=concept_metadata
            )
        })

        # Save all data
        logger.info(f"Saving dataset to {self.root_dir}")
        os.makedirs(self.root_dir, exist_ok=True)

        torch.save(input_data, self.processed_paths[0])
        torch.save(concepts, self.processed_paths[1])
        torch.save(annotations, self.processed_paths[2])
        graph.to_hdf(os.path.join(self.root_dir, "graph.h5"), key="graph", mode="w")

    def load_raw(self):
        """Load the generated dataset from disk."""
        self.maybe_build()
        logger.info(f"Loading dataset from {self.root_dir}")

        input_data = torch.load(self.processed_paths[0], weights_only=False)
        concepts = torch.load(self.processed_paths[1], weights_only=False)
        annotations = torch.load(self.processed_paths[2], weights_only=False)
        graph = pd.read_hdf(os.path.join(self.root_dir, "graph.h5"), "graph")

        return input_data, concepts, annotations, graph

    def load(self):
        """Load the dataset (wraps load_raw)."""
        return self.load_raw()


TOYDATASETS = ['xor', 'trigonometry', 'dot', 'checkmark']
