"""Deterministic inference engine — sample-free parameter propagation."""
from __future__ import annotations

from typing import Dict, List

import pyro.distributions as dist
import torch

from ..models.bayesian_network import BayesianNetwork
from .base import InferenceEngine
from .result import InferenceOutput


def _propagated_value(distribution, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Pick the parameter of the distribution to propagate to the children from a CPD's
    parameter dict: ``probs`` for discrete families, ``loc`` for Gaussian
    families, ``v`` for ``Delta``."""
    if distribution in (dist.Bernoulli, dist.Categorical, dist.OneHotCategorical):
        return params["probs"]
    if distribution in (dist.Normal, dist.MultivariateNormal):
        return params["loc"]
    if distribution is dist.Delta:
        return params["v"]
    raise ValueError(f"Unsupported distribution {distribution!r}")


def _align_gt(gt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Adjust the ground-truth tensor ``gt`` to match the dtype and shape of
    the CPD point-estimate ``ref``.

    Two adjustments are applied, in order:

    1. **dtype** — casts ``gt`` to ``ref.dtype`` if they differ.

    2. **trailing singleton** — if the two tensors differ by exactly one
    dimension of size 1 on the right, it is squeezed or unsqueezed to
    match:
    - ``(B, 1)`` → ``(B,)`` when ``ref`` has shape ``(B,)``.
    - ``(B,)`` → ``(B, 1)`` when ``ref`` has shape ``(B, 1)``.

    If the shapes still differ after those adjustments, ``expand_as`` is
    attempted for broadcast-compatible cases (e.g. a scalar label broadcast
    over a batch). Incompatible shapes raise a ``RuntimeError`` from PyTorch.
    """
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
    """Mix the model's prediction with the ground truth on a per-sample basis:
    each sample is replaced by ``gt`` with probability ``p_int`` and left as
    ``nn_value`` otherwise."""
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
    """Deterministic inference engine.

    Walks the PGM in topological order and propagates each CPD's parameter
    point estimate (``probs`` / ``loc`` / ``v``) to its children.
    When a non-evidence variable also appears in ``data``, its label is
    teacher-forced with per-sample probability ``p_int`` (default ``1.0``).
    The choice of ``p_int`` determines the effective training mode:

    - ``p_int=1.0`` — sequential training [1].
    - ``p_int=0.0`` — joint training [1].
    - ``0.0 < p_int < 1.0`` — sequential with random interventions [2].

    Returns the CPD parameters of every queried variable in
    ``InferenceOutput.model_params``.
    """

    name = "DeterministicInference"

    def __init__(self, pgm: BayesianNetwork, p_int: float = 1.0):
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
            name = var.name
            f = self.pgm.factors[name]

            if f.is_root:
                # Roots are always in evidence (validated). Apply parametrisation.
                value = f(evidence_value=data[name])
                cache[name] = value
                if name in query_set:
                    out.model_params[name] = {"v": value}
                continue

            parent_values = {p.name: cache[p.name] for p in f.parents}
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
