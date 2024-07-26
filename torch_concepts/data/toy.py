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
    """
    This class loads a synthetic dataset.
    Available datasets are:
    - XOR: A simple XOR dataset. The input features are two random variables, the concepts are Boolean values of the input features, and the task is the XOR of the concepts.
    - Trigonometry: A dataset where the input features are random variables sampled from a normal distribution, the concepts are the signs of the input features, and the task is the sum of the input features being greater than 1.
    - Dot: A dataset where the input features are random variables sampled from a normal distribution, the concepts are the signs of the dot product of the input features with fixed vectors, and the task is the dot product of the input features being greater than 0.
    - Checkmark: A dataset where the concepts A and B are random Boolean variables, the concept C is the negation of B, and the task is the logical AND of A and C.
    Main references for XOR, Trigonometry, and Dot datasets: `"Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_
    Main reference for Checkmark dataset: `"Causal Concept Embedding Models: Beyond Causal Opacity in Deep Learning" <https://arxiv.org/abs/2405.16507>`_

    Attributes:
        dataset: The name of the dataset to load. Available datasets are 'xor', 'trigonometry', 'dot', and 'checkmark'.
        size: The number of samples in the dataset.
        random_state: The random seed for generating the data. Default is 42.
    """
    def __init__(self, dataset: str, size: int, random_state: int = 42):
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


def _complete(n_samples: int = 10, n_features: int = 2, n_concepts: int = 2, n_hidden_concepts: int = 0,
              n_tasks: int = 1, emb_size: int = 10, seed: int = 42):
    # Randomly sample μ ∈ Rp s.t.μj ∼ Uniform(−5, 5) for 1 ≤ j ≤ p
    torch.manual_seed(seed)
    mu = torch.rand(n_features) * 10 - 5
    # Generate a random symmetric, positive-definite matrix Σ ∈ Rp×p
    torch.manual_seed(seed + 1)
    A = torch.rand(n_features, n_features)
    Sigma = A @ A.T
    # Randomly sample a design matrix X ∈ RN×p s.t. Xi,: ∼ Np(μ, Σ)
    torch.manual_seed(seed + 2)
    dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=Sigma)
    X = dist.rsample((n_samples,))

    if n_hidden_concepts > 0:
        # the bottleneck is incomplete
        # Let h : Rp → R{K+J} and g : R{K+J} → R be randomly initialised multilayer perceptrons with ReLU nonlinearities.
        torch.manual_seed(seed + 3)
        h = torch.nn.Sequential(torch.nn.Linear(n_features, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, n_concepts + n_hidden_concepts), torch.nn.LeakyReLU())
        torch.manual_seed(seed + 4)
        g = torch.nn.Sequential(torch.nn.Linear(n_concepts + n_hidden_concepts, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, n_tasks), torch.nn.LeakyReLU())
        # Let ui,k+l = 1{[h(Xi,:)]k ≥mk }, where mk = median({[h(X_{l,:})]k}_{l=1}^N), for 1 ≤ i ≤ N and 1 ≤ k ≤ K
        h_X = h(X)
        m = torch.median(h_X, dim=0).values
        u = (h_X >= m).float()
        c = u[:, :n_concepts]
        r = u[:, n_concepts:]
        # Let yi = 1{g(ui)≥my }, where my = median({g(ui)}_{l=1}^N), for 1 ≤ i ≤ N
        g_u = g(u)
        my = torch.median(g_u, dim=0).values
        y = (g_u >= my).float()
        concept_attr_names = [f'C{i}' for i in range(n_concepts)] + [f'R{i}' for i in range(n_hidden_concepts)]

    else:
        # the bottleneck is complete
        # Let h : Rp → RK and g : RK → R be randomly initialised multilayer perceptrons with ReLU nonlinearities.
        torch.manual_seed(seed + 3)
        h = torch.nn.Sequential(torch.nn.Linear(n_features, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, n_concepts), torch.nn.LeakyReLU())
        torch.manual_seed(seed + 4)
        g = torch.nn.Sequential(torch.nn.Linear(n_concepts, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, n_tasks), torch.nn.LeakyReLU())
        # Let ci,k = 1{[h(Xi,:)]k ≥mk }, where mk = median({[h(X_{l,:})]k}_{l=1}^N), for 1 ≤ i ≤ N and 1 ≤ k ≤ K
        h_X = h(X)
        m = torch.median(h_X, dim=0).values
        u = c = (h_X >= m).float()
        # Let yi = 1{g(ci)≥my }, where my = median({g(ci)}_{l=1}^N), for 1 ≤ i ≤ N
        g_c = g(c)
        my = torch.median(g_c, dim=0).values
        y = (g_c >= my).float()
        concept_attr_names = [f'C{i}' for i in range(n_concepts)]

    return X, u, y, None, concept_attr_names, [f'y{i}' for i in range(n_tasks)]


class CompletenessDataset:
    """
    This class loads a synthetic dataset where the bottleneck is complete or incomplete.
    The dataset is generated using the activations of randomly initialised multilayer perceptrons with ReLU nonlinearities.
    The input features are sampled from a multivariate normal distribution.
    The concepts correspond to the median activations of the hidden layers of the bottleneck.
    The tasks correspond to the median activations of the output layer of the bottleneck.
    Main reference: `"Beyond Concept Bottleneck Models: How to Make Black Boxes Intervenable?" <https://arxiv.org/abs/2401.13544>`_

    Attributes:
        n_samples: The number of samples in the dataset.
        n_features: The number of input features.
        n_concepts: The number of concepts to be learned.
        n_hidden_concepts: The number of hidden concepts to be learned.
        n_tasks: The number of tasks to be learned.
        emb_size: The size of concept embeddings.
        random_state: The random seed for generating the data. Default is 42.
    """
    def __init__(self, n_samples: int = 10, n_features: int = 2, n_concepts: int = 2, n_hidden_concepts: int = 0,
                 n_tasks: int = 1, emb_size: int = 10, random_state: int = 42):
        self.data, self.concept_labels, self.target_labels, self.dag, self.concept_attr_names, self.task_attr_names = _complete(n_samples, n_features, n_concepts, n_hidden_concepts, n_tasks, emb_size, random_state)
        self.dag = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        concept_label = self.concept_labels[index]
        target_label = self.target_labels[index]
        return data, concept_label, target_label
