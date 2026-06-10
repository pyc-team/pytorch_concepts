from typing import List, Union

import torch

from ...base.intervention import BaseConceptInterventionStrategy


class DistributionIntervention(BaseConceptInterventionStrategy):
    """
    Intervention that replaces predicted concepts with ground truth values.

    Implements do(C=c_true) operations by mixing predicted and ground truth
    concept values based on a binary mask.

    Args:
        ground_truth: Ground truth concept values of shape (batch_size, n_concepts).
    """

    def __init__(self, dist: Union[torch.distributions.Distribution, List[torch.distributions.Distribution]]):
        super().__init__()
        self.dist = dist

    def forward(self, x, *args, **kwargs):
        B, F = x.shape
        device, dtype = x.device, x.dtype

        def _sample(d, shape):
            # Try rsample first (for reparameterization), fall back to sample if not supported
            if hasattr(d, "rsample"):
                try:
                    return d.rsample(shape)
                except NotImplementedError:
                    pass
            return d.sample(shape)

        if hasattr(self.dist, "sample"):  # one distribution for all features
            t = _sample(self.dist, (B, F))
        else:  # per-feature list/tuple
            dists = list(self.dist)
            assert len(dists) == F, f"Need {F} per-feature distributions, got {len(dists)}"
            cols = [_sample(d, (B,)) for d in dists]  # each [B]
            t = torch.stack(cols, dim=1)  # [B, F]

        return t.to(device=device, dtype=dtype)
