import numpy as np
import torch
import pandas as pd
import os
import logging
from scipy.special import ndtr
from typing import List, Optional, Union

from ..base.dataset import ConceptDataset
from ...annotations import Annotations

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
        Random seed for reproducible data *generation* (also determines the
        on-disk cache filename). Default: 42
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

    >>> from torch_concepts.data import ToyDataset
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
        cardinalities = tuple([1] * len(concept_names))  # All binary concepts

        annotations = Annotations(
            labels=concept_names,
            cardinalities=cardinalities,
            types=['binary'] * len(concept_names),
        )

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


def _leaky_relu(h):
    """Leaky ReLU, scaled to keep the variance of a symmetric input."""
    return np.where(h > 0, h, 0.2 * h) * np.sqrt(2 / 1.04)


def _unit(v):
    norm = np.linalg.norm(v)
    return v / norm if norm else v


class CompletenessDataset(ConceptDataset):
    """
    Synthetic dataset with controlled concept (in)completeness.

    ``n_concepts`` binary concepts and a binary task ``y`` are generated from
    Gaussian latents, which an invertible network maps to the input ``x``. So
    ``y`` is a function of ``x`` (the best accuracy from ``x`` is 1), and any
    shortfall of the concepts is due to the concepts alone. The first
    ``observed_fraction * n_concepts`` concepts (rounded half up) are annotated
    (``C0``, ``C1``, ...); the others are hidden, available in
    ``hidden_concepts``.

    Generative process, with ``rho = hidden_observed_corr``::

        u ~ N(0, I)                        one latent per concept
        z_i = u_i                          observed concept i
        z_j = u_j                          hidden concept j; with probability |rho|
            = sign(rho) |u_j| sgn(z_p)     it takes the sign of its observed partner p
        c = 1[z > 0]                       balanced binary concepts
        x = mix(z, e)                      side latents e ~ N(0, I); mix is invertible
        s = concept part + side part
        y = 1[s > t]                       t: median of s, midway between two values (*)

    Given the concepts, their latents carry no information on each other or on
    ``y``: a concept predictor's confidence leaks no hidden information.

    The concept part of ``s`` sums linear terms ``a_i c_i`` and pairwise terms
    ``b_ij c_i c_j`` (concepts coded -1/1) with random weights; the side part
    is linear in ``e``. With ``rho = 0`` all terms are uncorrelated and ``s``
    has unit variance, so the importances below are exact shares of the
    variance of ``s``. Their effect on accuracy is not linear, nor always
    monotone (with ``rho != 0``, a hidden concept can cancel its partner's
    effect): measure it with :meth:`bayes_accuracy`. (*) So ``y`` is balanced,
    within about 0.005 at ``rho = 0``; with ``rho != 0`` and a few concepts,
    ``s`` takes few values with unequal probabilities, and ``P(y = 1)`` can
    deviate by a few points.

    With a fixed ``seed``, changing ``hidden_importance``, ``nonlinearity`` or
    ``side_importance`` changes only ``y``. With ``hidden_importance=None`` and
    ``hidden_observed_corr=0``, changing ``observed_fraction`` changes only
    which concepts are annotated.

    The input has ``n_concepts + n_side`` features. The graph links to ``y``
    the annotated concepts with a nonzero weight in ``s``.

    Parameters
    ----------
    n_gen : int, optional
        Number of samples. Default: 10000
    n_concepts : int, optional
        Number of concepts, observed and hidden. Default: 10
    observed_fraction : float, optional
        Fraction of the concepts that is annotated. Default: 1 (complete)
    hidden_importance : float, optional
        Share of the concept part of ``s`` in terms involving a hidden
        concept: 0 makes the observed concepts complete, 1 makes ``y`` depend
        on hidden concepts only. None does not split the weights: all concepts
        matter alike and ``observed_fraction`` only decides which are
        annotated. Default: None
    nonlinearity : float, optional
        Share of each part of ``s`` in pairwise terms. At 0, ``y`` is a
        threshold of a linear function of the concepts, so a linear
        concept-to-task predictor suffices. Default: 0
    hidden_observed_corr : float, optional
        Correlation between each hidden concept and its observed partner
        (hidden concepts take the observed ones as partners in turn). At 1,
        hidden concepts copy observed ones; at -1, they negate them. Default: 0
    side_importance : float, optional
        Share of ``s`` from the side latents ``e``, which are in ``x`` but in no
        concept. Above 0, even all concepts together are incomplete. Default: 0
    n_side : int, optional
        Number of side latents, part of ``x`` even when ``side_importance=0``
        (they then affect nothing). Default: 4
    mixing_layers : int, optional
        Depth of the network from latents to ``x``, from 0 to 4, i.e. how hard
        the concepts are to predict from ``x``. At 0, each concept is a linear
        threshold of ``x``. Deeper networks are not exactly invertible in
        float32. Default: 2
    seed : int, optional
        Seed of the generation. Default: 42
    concept_subset : list of str, optional
        Annotated concepts to keep (include ``'y'`` to keep the task).
        Default: None

    Attributes
    ----------
    hidden_concepts : torch.Tensor
        Hidden concepts, shape ``(n_gen, n_concepts - n_observed)``.
    side_latents : torch.Tensor
        Side latents ``e``, shape ``(n_gen, n_side)``.

    Examples
    --------
    >>> from torch_concepts.data import CompletenessDataset
    >>> dataset = CompletenessDataset(n_concepts=10, observed_fraction=0.7,
    ...                               hidden_importance=0.3)
    >>> dataset.concept_names
    ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'y']
    >>> round(dataset.bayes_accuracy(), 2)  # best accuracy of y from C0..C6
    0.81
    """

    def __init__(
            self,
            n_gen: int = 10000,
            n_concepts: int = 10,
            observed_fraction: float = 1.0,
            hidden_importance: Optional[float] = None,
            nonlinearity: float = 0.0,
            hidden_observed_corr: float = 0.0,
            side_importance: float = 0.0,
            n_side: int = 4,
            mixing_layers: int = 2,
            seed: int = 42,
            concept_subset: Optional[list] = None,
    ):
        k = int(observed_fraction * n_concepts + 0.5)  # observed concepts, rounded half up
        h = hidden_importance
        shares = [observed_fraction, nonlinearity, side_importance] + ([] if h is None else [h])
        if n_concepts < 1 or not all(0 <= v <= 1 for v in shares) or abs(hidden_observed_corr) > 1:
            raise ValueError("Need n_concepts >= 1, fractions and importances in [0, 1] and "
                             "hidden_observed_corr in [-1, 1].")
        if h is not None and ((h > 0 and k == n_concepts) or (h < 1 and k == 0)):
            raise ValueError("hidden_importance > 0 needs hidden concepts, < 1 needs observed ones.")
        if side_importance > 0 and n_side == 0:
            raise ValueError("side_importance > 0 needs n_side > 0.")
        if not 0 <= mixing_layers <= 4:
            raise ValueError("mixing_layers must be in 0..4: deeper mixing is not exactly invertible in float32.")

        self.seed = seed
        self._k, self._dims, self._corr = k, (n_concepts, n_side), hidden_observed_corr
        self._partner = np.arange(n_concepts - k) % k if k else None
        # one random stream per component: changing a knob leaves the other draws unchanged
        task_seq, mix_seq, data_seq, ref_seq, self._eval_seq = np.random.SeedSequence(seed).spawn(5)

        # task weights over all concepts, then rescaled per part
        rng = np.random.default_rng(task_seq)
        a = rng.choice([-1.0, 1.0], n_concepts) * rng.uniform(0.5, 1.5, n_concepts)  # no negligible concept
        b = np.triu(rng.standard_normal((n_concepts, n_concepts)), 1)
        g = rng.standard_normal(n_side)
        obs = np.arange(n_concepts) < k
        both = np.outer(obs, obs)
        parts = [(1.0, np.ones_like(obs), np.ones_like(both))] if h is None else [(1 - h, obs, both), (h, ~obs, ~both)]
        self._a, self._b = np.zeros(n_concepts), np.zeros((n_concepts, n_concepts))
        for w, lin, pairs in parts:
            lam = nonlinearity if (b * pairs).any() else 0.0  # a lone concept has no pairs: stays linear
            self._a += np.sqrt(w * (1 - lam)) * _unit(a * lin)
            self._b += np.sqrt(w * lam) * _unit(b * pairs)
        self._a *= np.sqrt(1 - side_importance)
        self._b *= np.sqrt(1 - side_importance)
        self._g = np.sqrt(side_importance) * _unit(g)

        rng = np.random.default_rng(mix_seq)
        d = sum(self._dims)
        self._mix_w = [np.linalg.qr(rng.standard_normal((d, d)))[0] for _ in range(mixing_layers + 1)]
        self._mix_b = [0.5 * rng.standard_normal(d) for _ in range(mixing_layers)]

        # threshold halfway between the score values around the median: y is balanced, and when s is
        # discrete (no side latents) no concept combination lies on it. Rounding makes equal scores tie.
        s = np.round(self._score(*self._latents(np.random.default_rng(ref_seq), 100_000)), 9)
        values, counts = np.unique(s, return_counts=True)
        j = np.abs(np.cumsum(counts)[:-1] / len(s) - 0.5).argmin()
        self._tau = (values[j] + values[j + 1]) / 2

        z, e = self._latents(np.random.default_rng(data_seq), n_gen)
        c = (z > 0).astype(np.float32)
        y = self._label(self._score(z, e))
        self.hidden_concepts = torch.tensor(c[:, k:])
        self.side_latents = torch.tensor(e, dtype=torch.float32)

        labels = [f"C{i}" for i in range(k)] + ["y"]
        graph = pd.DataFrame(np.zeros((k + 1, k + 1)), index=labels, columns=labels)
        graph.iloc[:k, k] = ((self._a != 0) | (self._b != 0).any(0) | (self._b != 0).any(1))[:k].astype(float)
        super().__init__(
            input_data=torch.tensor(self._mix(np.concatenate([z, e], 1)), dtype=torch.float32),
            concepts=torch.tensor(np.column_stack([c[:, :k], y]), dtype=torch.float32),
            annotations=Annotations(labels=labels, cardinalities=[1] * (k + 1), types=["binary"] * (k + 1)),
            graph=graph,
            concept_names_subset=concept_subset,
        )

    def bayes_accuracy(self, given: str = "observed", n: int = 4000, n_draws: int = 500) -> float:
        """
        Best possible accuracy of ``y`` from the annotated concepts
        (``given="observed"``) or from all concepts (``given="all"``).

        From ``x`` it is 1 by construction. Estimated on ``n`` fresh samples,
        integrating the hidden concepts out with ``n_draws`` Monte Carlo draws
        (the side latents analytically); the defaults are within about 0.005.
        ``concept_subset`` is ignored.
        """
        rng = np.random.default_rng(self._eval_seq)
        z = self._latents(rng, n)[0]
        if given == "all":
            p = self._p_y(z)
            return float(np.maximum(p, 1 - p).mean())
        if given != "observed":
            raise ValueError(f"given must be 'observed' or 'all', got {given!r}.")
        sign = np.where(z[:, :self._k] > 0, 1.0, -1.0)
        p = np.array([self._p_y(self._latents(rng, n, sign)[0]) for _ in range(n_draws)])
        # half of the draws guess the likelier y, the other half score the guess
        # (scoring on the same draws would bias the estimate up)
        guess, check = p[::2].mean(0) > 0.5, p[1::2].mean(0)
        return float(np.where(guess, check, 1 - check).mean())

    def _latents(self, rng, n, obs_sign=None):
        """Concept latents z and side latents e; ``obs_sign`` fixes the signs of the observed concepts."""
        n_concepts, n_side = self._dims
        k = self._k
        z = rng.standard_normal((n, n_concepts))
        e = rng.standard_normal((n, n_side))
        copy = rng.random((n, n_concepts - k)) < abs(self._corr)  # drawn last: k changes no other draw
        if obs_sign is not None:
            z[:, :k] = obs_sign * np.abs(z[:, :k])
        if self._partner is not None:  # hidden concepts: with probability |corr|, the partner's sign
            partner = np.where(z[:, self._partner] > 0, 1.0, -1.0)
            z[:, k:] = np.where(copy, np.sign(self._corr) * partner * np.abs(z[:, k:]), z[:, k:])
        return z, e

    def _score(self, z, e):
        x = np.where(z > 0, 1.0, -1.0)
        return x @ self._a + ((x @ self._b) * x).sum(1) + e @ self._g

    def _label(self, s):
        return (s > self._tau).astype(np.float32)

    def _p_y(self, z):
        """P(y = 1 | concepts); the side part e @ g is N(0, |g|^2)."""
        s = self._score(z, np.zeros((len(z), len(self._g))))
        sd = np.linalg.norm(self._g)
        return ndtr((s - self._tau) / sd) if sd else self._label(s)

    def _mix(self, v):
        for w, b in zip(self._mix_w, self._mix_b):
            v = _leaky_relu(v @ w + b)
        return v @ self._mix_w[-1]


TOYDATASETS = ['xor', 'trigonometry', 'dot', 'checkmark']
