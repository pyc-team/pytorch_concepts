"""
Base intervention classes for concept-based models.

This module provides abstract base classes for implementing intervention strategies in concept-based models.
"""
import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseConceptInterventionStrategy(nn.Module, ABC):
    """
    Abstract base class for intervention strategies.

    Intervention strategies define how to intervene on layers (either on the parametrization or on the output).
    """
    def __init__(self, *args, **kwargs):
        """Initialize the intervention module."""
        super(BaseConceptInterventionStrategy, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward method to be implemented by subclasses."""
        raise NotImplementedError


class BaseModuleInterventionStrategy(ABC):
    """
    Abstract base class for intervention strategies.

    Intervention strategies define how to intervene on layers (either on the parametrization or on the output).
    """
    def __init__(self, *args, **kwargs):
        """Initialize the intervention module."""
        super(BaseModuleInterventionStrategy, self).__init__()

    @abstractmethod
    def transform(self, module: nn.Module, *args, **kwargs) -> nn.Module:
        """Forward method to be implemented by subclasses."""
        raise NotImplementedError


class BaseInterventionPolicy(nn.Module, ABC):
    def __init__(self):
        super(BaseInterventionPolicy, self).__init__()

    @abstractmethod
    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        """Forward method to compute the intervention scores based on input x."""
        raise NotImplementedError

    def build_mask(
            self,
            policy_scores: torch.tensor,
            sel_idx: Optional[torch.LongTensor] = None,
            quantile: float = 1.0,
            eps: float = 1e-12
    ) -> torch.Tensor:
        B, F = policy_scores.shape
        device = policy_scores.device
        dtype = policy_scores.dtype

        if sel_idx is None:
            sel_idx = torch.arange(F, dtype=torch.long, device=device)
        else:
            sel_idx = sel_idx.to(device=device)

        if len(sel_idx) == 0:
            return torch.ones_like(policy_scores)

        K = sel_idx.numel()
        sel = policy_scores.index_select(dim=1, index=sel_idx)  # [B, K]

        if K == 1:
            # Edge case: single selected column.
            # q < 1 => keep; q == 1 => replace.
            keep_col = torch.ones((B, 1), device=device, dtype=dtype) if quantile < 1.0 \
                else torch.zeros((B, 1), device=device, dtype=dtype)
            mask = torch.ones((B, F), device=device, dtype=dtype)
            mask.scatter_(1, sel_idx.unsqueeze(0).expand(B, -1), keep_col)

            # STE proxy (optional; keeps gradients flowing on the selected col)
            row_max = sel.max(dim=1, keepdim=True).values + eps
            soft_sel = torch.log1p(sel) / torch.log1p(row_max)  # [B,1]
            soft_proxy = torch.ones_like(policy_scores)
            soft_proxy.scatter_(1, sel_idx.unsqueeze(0).expand(B, -1), soft_sel)
            mask = (mask - soft_proxy).detach() + soft_proxy
            return mask

        # K > 1: standard per-row quantile via kthvalue
        k = int(max(1, min(K, 1 + math.floor(quantile * (K - 1)))))
        thr, _ = torch.kthvalue(sel, k, dim=1, keepdim=True)  # [B,1]

        # Use strict '>' so ties at the threshold are replaced (robust near edges)
        sel_mask_hard = (sel > (thr - 0.0)).to(dtype)  # [B,K]

        mask = torch.ones((B, F), device=device, dtype=dtype)
        mask.scatter_(1, sel_idx.unsqueeze(0).expand(B, -1), sel_mask_hard)

        # STE proxy (unchanged)
        row_max = sel.max(dim=1, keepdim=True).values + 1e-12
        soft_sel = torch.log1p(sel) / torch.log1p(row_max)
        soft_proxy = torch.ones_like(policy_scores)
        soft_proxy.scatter_(1, sel_idx.unsqueeze(0).expand(B, -1), soft_sel)
        mask = (mask - soft_proxy).detach() + soft_proxy
        return mask
