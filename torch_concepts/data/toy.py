import numpy as np
import torch
from torch.utils.data import Dataset


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

    # concetps
    concetps = np.stack([
        x > 0,
        y > 0,
        z > 0,
    ]).T

    # task
    downstream_task = (x + y + z) > 1

    input_features = torch.FloatTensor(input_features)
    concetps = torch.FloatTensor(concetps)
    downstream_task = torch.FloatTensor(downstream_task)
    return input_features, concetps, downstream_task.unsqueeze(-1), None, ['C1', 'C2', 'C3'], ['sumGreaterThan1']


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
    return x, c, y.unsqueeze(-1), None, ['dotV1V2GreaterThan0', 'dotV3V4GreaterThan0'], ['dotV1V3GreaterThan0']


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


def _checkmark(n_samples: int = 10, seed: int =42, perturb: float = 0.1) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _toy_problem(n_samples, seed)
    c = x.clone()
    torch.manual_seed(seed)
    x = x * 2 - 1 + torch.randn_like(x) * perturb

    dag = torch.FloatTensor([[0, 0, 0, 1],  # A influences D
                            [0, 0, 1, 0],  # B influences C
                            [0, 0, 0, 1],  # C influences D
                            [0, 0, 0, 0],  # D doesn't influence others
                            ])

    return x, c[:, [0, 1, 2]], c[:, 3].unsqueeze(1), dag, ['A', 'B', 'C'], ['D']


class ToyDataset(Dataset):
    def __init__(self, dataset: str, size: int, random_state: int = 42):
        """
        This class loads a synthetic dataset.
        Available datasets are:
        - XOR: A simple XOR dataset. The input features are two random variables, the concepts are Boolean values of the input features, and the task is the XOR of the concepts.
        - Trigonometry: A dataset where the input features are random variables sampled from a normal distribution, the concepts are the signs of the input features, and the task is the sum of the input features being greater than 1.
        - Dot: A dataset where the input features are random variables sampled from a normal distribution, the concepts are the signs of the dot product of the input features with fixed vectors, and the task is the dot product of the input features being greater than 0.
        - Checkmark: A dataset where the concepts A and B are random Boolean variables, the concept C is the negation of B, and the task is the logical AND of A and C.

        Args:
            dataset: The name of the dataset to load. Available datasets are 'xor', 'trigonometry', 'dot', and 'checkmark'.
            size: The number of samples in the dataset.
            random_state: The random seed for generating the data. Default is 42.
        """
        self.size = size
        self.random_state = random_state
        self.data, self.concept_labels, self.target_labels, self.dag, self.concept_attr_names, self.task_attr_names = self._load_data(dataset)

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
