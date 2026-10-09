"""Tensor initializers shared by learnable graph methods.

Each initializer writes a weight tensor under no_grad. Sources choose the target
tensor and handle their own masks and additional parameters.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

def initialize_from_entropy(data: Any) -> Callable:
    """Initialize from 1 - H(i | j), with the final node treated as a task.

    Follow CausalCGM: zero the last row, divide by the mean and clamp to [0, 0.99].
    Copy into the supplied weights, preserving their identity, device and dtype.
    """
    @torch.no_grad()
    def initialize(weights):
        values = data.concepts if hasattr(data, "concepts") else data
        values = values.tensor if hasattr(values, "tensor") else values
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise ValueError("Entropy initialization requires a 2D tensor.")
        if values.shape[0] == 0:
            raise ValueError("Entropy initialization requires at least one row.")
        count = values.shape[1]
        if tuple(weights.shape) != (count, count):
            raise ValueError("Entropy initialization requires one column per graph node.")
        values = values.detach().cpu().numpy()
        adjacency = np.zeros((count, count))
        entropies = [_entropy(values[:, index:index + 1]) for index in range(count)]
        for source in range(count):
            for target in range(count):
                if source != target:
                    joint = _entropy(values[:, [source, target]])
                    adjacency[source, target] = 1 - (joint - entropies[target])
        cov = torch.tensor(adjacency).float()
        cov[-1, :] = 0
        mean = cov.mean()
        if mean != 0:
            cov = cov / mean
        cov = torch.clamp(cov, 0, 0.99)
        weights.copy_(cov.to(weights))
        weights.requires_grad_(True)

    return initialize


def _entropy(values: np.ndarray) -> np.float64:
    """Empirical joint entropy in bits; observations must already be discrete."""
    _, counts = np.unique(values, axis=0, return_counts=True)
    probabilities = counts / len(values)
    return np.sum(-probabilities * np.log2(probabilities))
