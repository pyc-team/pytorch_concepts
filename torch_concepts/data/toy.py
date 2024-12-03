import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.random import multivariate_normal, uniform
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_spd_matrix, make_low_rank_matrix


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
    return x, c, y.unsqueeze(-1), None, ['C1', 'C2'], ['xor']


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
    return (
        input_features,
        concepts,
        downstream_task.unsqueeze(-1),
        None,
        ['C1', 'C2', 'C3'],
        ['sumGreaterThan1'],
    )


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
    return (
        x,
        c,
        y.unsqueeze(-1),
        None,
        ['dotV1V2GreaterThan0', 'dotV3V4GreaterThan0'],
        ['dotV1V3GreaterThan0'],
    )


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

    dag = torch.FloatTensor([[0, 0, 0, 1],  # A influences D
                            [0, 0, 1, 0],  # B influences C
                            [0, 0, 0, 1],  # C influences D
                            [0, 0, 0, 0],  # D doesn't influence others
                            ])

    return (
        x,
        c[:, [0, 1, 2]],
        c[:, 3].unsqueeze(1),
        dag,
        ['A', 'B', 'C'],
        ['D'],
    )


class ToyDataset(Dataset):
    """
    This class loads a synthetic dataset.
    Available datasets are:
    - XOR: A simple XOR dataset. The input features are two random variables,
        the concepts are Boolean values of the input features, and the task is
        the XOR of the concepts.
    - Trigonometry: A dataset where the input features are random variables
        sampled from a normal distribution, the concepts are the signs of the
        input features, and the task is the sum of the input features being
        greater than 1.
    - Dot: A dataset where the input features are random variables sampled from
        a normal distribution, the concepts are the signs of the dot product of
        the input features with fixed vectors, and the task is the dot product
        of the input features being greater than 0.
    - Checkmark: A dataset where the concepts A and B are random Boolean
        variables, the concept C is the negation of B, and the task is the
        logical AND of A and C.

    Main references for XOR, Trigonometry, and Dot datasets: `"Concept
    Embedding Models: Beyond the Accuracy-Explainability
    Trade-Off" <https://arxiv.org/abs/2209.09056>`_

    Main reference for Checkmark dataset: `"Causal Concept Embedding Models:
    Beyond Causal Opacity in Deep Learning" <https://arxiv.org/abs/2405.16507>`_

    Attributes:
        dataset: The name of the dataset to load. Available datasets are 'xor',
            'trigonometry', 'dot', and 'checkmark'.
        size: The number of samples in the dataset.
        random_state: The random seed for generating the data. Default is 42.
    """
    def __init__(self, dataset: str, size: int, random_state: int = 42):
        self.size = size
        self.random_state = random_state
        (
            self.data,
            self.concept_labels,
            self.target_labels,
            self.dag,
            self.concept_attr_names,
            self.task_attr_names
        ) = self._load_data(dataset)

    def _load_data(self, dataset):
        if dataset == 'xor':
            return _xor(self.size, self.random_state)
        elif dataset == 'trigonometry':
            return _trigonometry(self.size, self.random_state)
        elif dataset == 'dot':
            return _dot(self.size, self.random_state)
        elif dataset == 'checkmark':
            return _checkmark(self.size, self.random_state)
        else:
            raise ValueError(f"Unknown dataset '{dataset}'")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data = self.data[index]
        concept_label = self.concept_labels[index]
        target_label = self.target_labels[index]
        return data, concept_label, target_label


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
    # tmp = np.tile(np.median(c, 0), (X.shape[0], 1))
    # c = (c >= tmp) * 1.0

    # Generate labels
    y = f(c.detach().numpy())
    y = torch.sigmoid(torch.FloatTensor(y))
    y = (y >= 0.5) * 1.0
    # tmp = np.tile(np.median(y, 0), (X.shape[0], 1))
    # y = (y >= tmp) * 1.0

    u = c[:, :n_concepts]
    X = torch.FloatTensor(X)
    u = torch.FloatTensor(u)
    y = torch.FloatTensor(y)
    return (
        X,
        u,
        y,
        None,
        [f'c{i}' for i in range(n_concepts)],
        [f'y{i}' for i in range(n_tasks)],
    )


class CompletenessDataset:
    """
    This class loads a synthetic dataset where the bottleneck is complete or
    incomplete. The dataset is generated using the activations of randomly
    initialised multilayer perceptrons with ReLU nonlinearities. The input
    features are sampled from a multivariate normal distribution. The concepts
    correspond to the median activations of the hidden layers of the bottleneck.
    The tasks correspond to the median activations of the output layer of the
    bottleneck.

    Main reference: `"Beyond Concept Bottleneck Models: How to Make Black Boxes
    Intervenable?" <https://arxiv.org/abs/2401.13544>`_

    Attributes:
        n_samples: The number of samples in the dataset.
        p: The number of covariates per view.
        n_views: The number of views in the dataset.
        n_concepts: The number of concepts to be learned.
        n_hidden_concepts: The number of hidden concepts to be learned.
        n_tasks: The number of tasks to be learned.
        emb_size: The size of concept embeddings.
        random_state: The random seed for generating the data. Default is 42.
    """
    def __init__(
        self,
        n_samples: int = 10,
        p: int = 2,
        n_views: int = 10,
        n_concepts: int = 2,
        n_hidden_concepts: int = 0,
        n_tasks: int = 1,
        random_state: int = 42,
    ):
        (
            self.data,
            self.concept_labels,
            self.target_labels,
            self.dag,
            self.concept_attr_names,
            self.task_attr_names,
        ) = _complete(
            n_samples,
            p,
            n_views,
            n_concepts,
            n_hidden_concepts,
            n_tasks,
            random_state,
        )
        self.dag = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        concept_label = self.concept_labels[index]
        target_label = self.target_labels[index]
        return data, concept_label, target_label