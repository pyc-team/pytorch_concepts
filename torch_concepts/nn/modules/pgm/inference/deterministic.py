"""Deterministic inference engine — propagates NN-derived parameters (§5.2)."""
from __future__ import annotations

from typing import Dict, List

import pyro.distributions as dist
import torch

from ..models.probabilistic_model import ProbabilisticModel
from .base import InferenceEngine
from .result import InferenceOutput


def _propagated_value(distribution, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Pick the value to propagate to children from a CPD parameter dict."""
    if distribution in (dist.Bernoulli, dist.Categorical, dist.OneHotCategorical):
        return params["probs"]
    if distribution in (dist.Normal, dist.MultivariateNormal):
        return params["loc"]
    if distribution is dist.Delta:
        return params["v"]
    raise ValueError(f"Unsupported distribution {distribution!r}")


def _align_gt(gt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Coerce a ground-truth label to match the dtype/shape of ``ref``."""
    g = gt
    if g.dtype != ref.dtype:
        g = g.to(ref.dtype)
    # If gt has a trailing singleton missing (e.g. Bernoulli (B,) vs (B,)).
    if g.shape != ref.shape:
        # Try to squeeze/unsqueeze trailing dim by 1.
        if g.dim() == ref.dim() + 1 and g.shape[-1] == 1:
            g = g.squeeze(-1)
        elif g.dim() + 1 == ref.dim() and ref.shape[-1] == 1:
            g = g.unsqueeze(-1)
    return g.expand_as(ref) if g.shape != ref.shape else g


def _teacher_force(
    nn_value: torch.Tensor,
    gt: torch.Tensor,
    p_int: float,
) -> torch.Tensor:
    """Per-sample-per-variable Bernoulli mask with probability ``p_int`` (§5.2 step 4)."""
    gt = _align_gt(gt, nn_value)
    if p_int >= 1.0:
        return gt
    if p_int <= 0.0:
        return nn_value
    # Mask shape: batch dim + 1s for event dims. Assume nn_value has at least
    # one batch dim (the leading one).
    batch_dim = nn_value.shape[:1]
    mask_shape = batch_dim + (1,) * (nn_value.dim() - 1)
    mask = (torch.rand(mask_shape, device=nn_value.device) < p_int).to(nn_value.dtype)
    return mask * gt + (1.0 - mask) * nn_value


class DeterministicInference(InferenceEngine):
    """Propagate the NN output (after bounding); optional teacher-forcing (§5.2)."""

    name = "DeterministicInference"

    def __init__(self, pgm: ProbabilisticModel, p_int: float = 1.0):
        super().__init__(pgm)
        if not 0.0 <= float(p_int) <= 1.0:
            raise ValueError(f"p_int must be in [0, 1], got {p_int!r}.")
        self.p_int = float(p_int)

    def _run(
        self,
        query: List[str],
        evidence: List[str],
        data: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        out = InferenceOutput()
        cache: Dict[str, torch.Tensor] = {}
        evidence_set = set(evidence)
        query_set = set(query)

        for var in self.pgm.sorted_variables:
            name = var.concept
            f = self.pgm.factors[name]

            if f.is_root:
                # Roots are always in evidence (validated). Apply parametrisation.
                value = f(evidence_value=data[name])
                cache[name] = value
                if name in query_set:
                    out.model_params[name] = {"v": value}
                continue

            parent_values = [cache[p.concept] for p in f.parents]
            params = f(parent_values=parent_values)
            nn_value = _propagated_value(var.distribution, params)

            if name in evidence_set:
                # Always teacher-force on evidence (regardless of p_int).
                propagated = _align_gt(data[name], nn_value)
            elif name in data:
                propagated = _teacher_force(nn_value, data[name], self.p_int)
            else:
                propagated = nn_value

            cache[name] = propagated
            if name in query_set:
                out.model_params[name] = params

        return out
