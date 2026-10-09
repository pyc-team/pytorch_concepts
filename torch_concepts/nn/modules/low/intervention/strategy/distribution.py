from typing import List, Union

import torch

from ...base.intervention import ConceptInterventionStrategy


class DistributionIntervention(ConceptInterventionStrategy):
    """
    Intervention that samples the intervened outputs from a distribution.

    Args:
        dist: One distribution shared by all outputs, or a list with one
            distribution per output (length ``F``).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import DistributionIntervention
        >>>
        >>> strategy = DistributionIntervention(torch.distributions.Normal(0.0, 1.0))
        >>> strategy(torch.randn(3, 2)).shape
        torch.Size([3, 2])
    """

    def __init__(self, dist: Union[torch.distributions.Distribution, List[torch.distributions.Distribution]]):
        super().__init__()
        self.dist = dist

    def forward(self, x, *args, **kwargs):
        *lead, F = x.shape
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
            t = _sample(self.dist, (*lead, F))
        else:  # per-feature list/tuple
            dists = list(self.dist)
            assert len(dists) == F, f"Need {F} per-feature distributions, got {len(dists)}"
            cols = [_sample(d, tuple(lead)) for d in dists]  # each [...lead]
            t = torch.stack(cols, dim=-1)  # [..., F]

        return t.to(device=device, dtype=dtype)
