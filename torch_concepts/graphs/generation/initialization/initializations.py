"""Tensor initializers shared by learnable graph methods.

Each initializer writes a weight tensor under no_grad. Sources choose the target
tensor and handle their own masks and additional parameters.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

def initialize_from_entropy(
    data: Any, *, apply_mask: bool = False, add_noise: bool = False,
    mask: torch.Tensor | None = None,
) -> Callable:
    """Initialize from 1 - H(i | j), using discrete training observations.

    Apply mask/noise before normalizing by the mean and clamping to [0, 0.99].
    apply_mask zeros forbidden entries; mask must be a boolean weight-shaped tensor.
    add_noise fills allowed off-diagonal zeros with noise in [0.0001, 0.001].
    Copy into the supplied weights, preserving their identity, device and dtype.
    """
    @torch.no_grad()
    def initialize(weights):
        if apply_mask and mask is None:
            raise ValueError("apply_mask requires a mask.")
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != weights.shape:
                raise ValueError("mask must be a boolean tensor with the same shape as weights.")
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
        if apply_mask:
            cov.masked_fill_(~mask.cpu(), 0)
        if add_noise:
            zeros = (cov == 0)
            zeros.fill_diagonal_(False)
            if mask is not None:
                zeros &= mask.cpu()
            cov[zeros] = torch.empty_like(cov[zeros]).uniform_(1e-4, 1e-3)
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
