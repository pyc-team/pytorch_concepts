"""
Base intervention classes for concept-based models.

This module provides abstract base classes for implementing intervention strategies in concept-based models.
"""
import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class InterventionStrategy(ABC):
    """
    Abstract base class shared by all intervention strategies.

    It carries no behaviour: concept strategies override ``forward`` to rewrite a
    layer's output, module strategies override ``transform`` to rewrite the layer
    itself. It exists so that both kinds can be named and type-checked as one thing.
    """


class ConceptInterventionStrategy(nn.Module, InterventionStrategy):
    """
    Abstract base class for intervention strategies.

    Intervention strategies define how to intervene on layers (either on the parametrization or on the output).
    """
    def __init__(self, *args, **kwargs):
        """Initialize the intervention module."""
        super(ConceptInterventionStrategy, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward method to be implemented by subclasses."""
        raise NotImplementedError


class ModuleInterventionStrategy(InterventionStrategy):
    """
    Abstract base class for intervention strategies.

    Intervention strategies define how to intervene on layers (either on the parametrization or on the output).
    """
    def __init__(self, *args, **kwargs):
        """Initialize the intervention module."""
        super(ModuleInterventionStrategy, self).__init__()

    @abstractmethod
    def transform(self, module: nn.Module, *args, **kwargs) -> nn.Module:
        """Forward method to be implemented by subclasses."""
        raise NotImplementedError


class InterventionPolicy(nn.Module, ABC):
    def __init__(self):
        super(InterventionPolicy, self).__init__()

    @abstractmethod
    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        """Forward method to compute the intervention scores based on input x."""
        raise NotImplementedError

    @staticmethod
    def _ste_soft_sel(sel: torch.Tensor, eps: float, dtype: torch.dtype) -> torch.Tensor:
        """Straight-through soft proxy for the hard mask decision, safe against
        degenerate ties (e.g. under UniformPolicy, where every candidate scores
        exactly 0, or any other policy forced into that regime).

        ``soft_sel = log1p(sel) / log1p(row_max)`` blows up as row_max -> 0:
        the *value* underflows to a 0/0 NaN in low precision (float16), and
        even where the value stays finite (float32), the *gradient* still
        explodes to ~1/row_max (order 1e12), which overflows to inf, and then
        combines with an upstream zero-subgradient (e.g. abs()'s subgradient
        at 0) into NaN once cast back down to float16. Both failure modes are
        avoided by substituting a safe dummy denominator for degenerate rows
        before dividing (so the division itself never explodes, in value or
        gradient), then explicitly zeroing the result for those rows -- there
        is no informative ranking signal among candidates that are all tied.
        """
        sel32 = sel.float()
        row_max = sel32.max(dim=1, keepdim=True).values
        degenerate = row_max < 1e-6
        safe_row_max = torch.where(degenerate, torch.ones_like(row_max), row_max + eps)
        soft_sel = torch.log1p(sel32) / torch.log1p(safe_row_max)
        soft_sel = torch.where(degenerate, torch.zeros_like(soft_sel), soft_sel)
        # Belt-and-suspenders: catches anything unrelated to the tie case above,
        # e.g. a custom policy returning scores <= -1 (log1p domain violation).
        soft_sel = torch.nan_to_num(soft_sel, nan=0.0, posinf=0.0, neginf=0.0)
        return soft_sel.to(dtype)

    def build_mask(
            self,
            policy_scores: torch.tensor,
            sel_idx: Optional[torch.LongTensor] = None,
            quantile: float = 1.0,
            eps: float = 1e-12
    ) -> torch.Tensor:
        # Arbitrary leading dims are supported by flattening them into a single
        # synthetic batch axis; the quantile/threshold logic below then runs
        # independently per leading-dim slice, exactly as it did per row for
        # plain [B, F] input.
        *lead, F = policy_scores.shape
        device = policy_scores.device
        dtype = policy_scores.dtype

        scores = policy_scores.reshape(-1, F)
        B = scores.shape[0]

        # Normalize sel_idx to [B, K] once, up front, so the rest of the
        # method is agnostic to how much of the leading shape was spelled out.
        # sel_idx's leading dims are broadcast against `lead` with standard
        # (right-aligned) rules, e.g. for lead=(Batch, Time):
        #   - None            -> every column, shared     -> [K=F] broadcasts
        #   - [K]              -> shared across everything -> () broadcasts
        #   - [Batch, 1, K]    -> shared across Time only   -> (Batch, 1) broadcasts
        #   - [1, Time, K]     -> shared across Batch only  -> (1, Time) broadcasts
        #   - [Batch, Time, K] -> one set per element        -> exact match
        # A bare [Batch, K] does *not* implicitly mean "broadcast Time": the
        # trailing dim of a shorter leading-shape aligns against `lead`'s
        # trailing dim (Time), not its leading one, under right-aligned
        # broadcasting -- write [Batch, 1, K] to be explicit.
        if sel_idx is None:
            sel_idx = torch.arange(F, dtype=torch.long, device=device)
        else:
            sel_idx = sel_idx.to(device=device)
        try:
            sel_idx = torch.broadcast_to(sel_idx, (*lead, sel_idx.shape[-1]))
        except RuntimeError as e:
            raise ValueError(
                f"members_to_intervene_on leading dims {tuple(sel_idx.shape[:-1])} "
                f"cannot be broadcast against the policy_scores leading dims "
                f"{tuple(lead)}"
            ) from e
        sel_idx = sel_idx.reshape(-1, sel_idx.shape[-1])

        K = sel_idx.shape[1]
        if K == 0:
            return torch.ones_like(scores).reshape(*lead, F)

        sel = torch.gather(scores, dim=1, index=sel_idx)  # [B, K]

        if K == 1:
            # Edge case: single selected column.
            # q < 1 => keep; q == 1 => replace.
            keep_col = torch.ones((B, 1), device=device, dtype=dtype) if quantile < 1.0 \
                else torch.zeros((B, 1), device=device, dtype=dtype)
            mask = torch.ones((B, F), device=device, dtype=dtype)
            mask.scatter_(1, sel_idx, keep_col)

            # STE proxy (optional; keeps gradients flowing on the selected col).
            soft_sel = self._ste_soft_sel(sel, eps, dtype)  # [B,1]
            soft_proxy = torch.ones_like(scores)
            soft_proxy.scatter_(1, sel_idx, soft_sel)
            mask = (mask - soft_proxy).detach() + soft_proxy
            return mask.reshape(*lead, F)

        # K > 1: standard per-row quantile via kthvalue
        k = int(max(1, min(K, 1 + math.floor(quantile * (K - 1)))))
        try:
            thr, _ = torch.kthvalue(sel, k, dim=1, keepdim=True)  # [B,1]
        except NotImplementedError:
            # kthvalue has no MPS kernel; fall back to CPU for this op only.
            thr, _ = torch.kthvalue(sel.cpu(), k, dim=1, keepdim=True)
            thr = thr.to(device)

        # Use strict '>' so ties at the threshold are replaced (robust near edges)
        sel_mask_hard = (sel > (thr - 0.0)).to(dtype)  # [B,K]

        mask = torch.ones((B, F), device=device, dtype=dtype)
        mask.scatter_(1, sel_idx, sel_mask_hard)

        # STE proxy, safe against degenerate ties; see _ste_soft_sel above.
        soft_sel = self._ste_soft_sel(sel, eps, dtype)
        soft_proxy = torch.ones_like(scores)
        soft_proxy.scatter_(1, sel_idx, soft_sel)
        mask = (mask - soft_proxy).detach() + soft_proxy
        return mask.reshape(*lead, F)
