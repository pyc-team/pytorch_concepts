"""
Inference and intervention modules for concept-based models.

This module provides intervention strategies that modify concept values during
inference, enabling causal reasoning and what-if analysis in concept-based models.
"""
import math
import contextlib
from abc import abstractmethod
from typing import List, Sequence, Union, Optional
import torch
import torch.nn as nn

from ...mid.models.cpd import ParametricCPD
from ..base.inference import BaseIntervention

# ---------------- core helpers ----------------

def _get_submodule(model: nn.Module, dotted: str) -> nn.Module:
    cur = model
    for name in dotted.split("."):
        cur = cur.get_submodule(name)
    return cur

def _set_submodule(model: nn.Module, dotted: str, new: nn.Module) -> None:
    parts = dotted.split(".")
    parent = model.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else model
    if len(parts) > 1:
        setattr(parent, parts[-1], new)
    elif len(parts) == 1:
        if isinstance(new, ParametricCPD):
            setattr(parent, parts[0], new)
        else:
            setattr(parent, parts[0], ParametricCPD(concepts=dotted, parametrization=new))
    else:
        raise ValueError("Dotted path must not be empty")

def _as_list(x, n: int):
    # broadcast a singleton to length n; if already a list/tuple, validate length
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(f"Expected list of length {n}, got {len(x)}")
        return list(x)
    return [x for _ in range(n)]

# ---------------- strategy ----------------

class RewiringIntervention(BaseIntervention):
    """
    Base class for rewiring-based interventions.

    Rewiring interventions replace predicted concept values with target values
    based on a binary mask, implementing do-calculus operations.

    Args:
        model: The concept-based model to intervene on.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import RewiringIntervention
        >>>
        >>> # Subclass to create custom intervention
        >>> class MyIntervention(RewiringIntervention):
        ...     def _make_target(self, y, *args, **kwargs):
        ...         return torch.ones_like(y)
        >>>
    """

    def __init__(self, model: nn.Module, *args, **kwargs):
        super().__init__(model)

    @abstractmethod
    def _make_target(self, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Create target tensor for intervention.

        Args:
            y: Predicted concept values.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Target values for intervention.
        """
        raise NotImplementedError

    def query(self, original_module: nn.Module, mask: torch.Tensor, *args, **kwargs) -> nn.Module:
        """
        Create an intervention wrapper module.

        Args:
            original_module: The original module to wrap.
            mask: Binary mask (1=keep prediction, 0=replace with target).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            nn.Module: Wrapped module with intervention applied.
        """
        parent = self

        class _Rewire(nn.Module):
            def __init__(self, orig: nn.Module, mask_: torch.Tensor):
                super().__init__()
                self.orig = orig
                self.register_buffer("mask", mask_.clone())

            def forward(self, **kwargs) -> torch.Tensor:
                y = self.orig(**kwargs)  # [B, F]
                assert y.dim() == 2, "RewiringIntervention expects 2-D tensors [Batch, N_concepts]. Got shape: {}" \
                    .format(y.shape)
                t = parent._make_target(y)  # [B, F]
                m = self.mask.to(dtype=y.dtype)
                return y * m + t * (1.0 - m)

        return _Rewire(original_module, mask)

# -------------------- Concrete strategies --------------------

class GroundTruthIntervention(RewiringIntervention):
    """
    Intervention that replaces predicted concepts with ground truth values.

    Implements do(C=c_true) operations by mixing predicted and ground truth
    concept values based on a binary mask.

    Args:
        model: The concept-based model to intervene on.
        ground_truth: Ground truth concept values of shape (batch_size, n_concepts).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import GroundTruthIntervention
        >>>
        >>> # Create a dummy model
        >>> model = torch.nn.Linear(10, 5)
        >>>
        >>> # Ground truth values
        >>> c_true = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0],
        ...                        [0.0, 1.0, 0.0, 1.0, 0.0]])
        >>>
        >>> # Create intervention
        >>> intervention = GroundTruthIntervention(model, c_true)
        >>>
        >>> # Apply intervention (typically done via context manager)
        >>> # See intervention() context manager for complete usage
    """

    def __init__(self, model: nn.Module, ground_truth: torch.Tensor):
        super().__init__(model)
        self.register_buffer("ground_truth", ground_truth)

    def _make_target(self, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.ground_truth.to(dtype=y.dtype, device=y.device)

class DoIntervention(RewiringIntervention):
    """
    Intervention that sets concepts to constant values (do-calculus).

    Implements do(C=constant) operations, supporting scalar, per-concept,
    or per-sample constant values with automatic broadcasting.

    Args:
        model: The concept-based model to intervene on.
        constants: Constant values (scalar, [F], [1,F], or [B,F]).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import DoIntervention
        >>>
        >>> # Create a dummy model
        >>> model = torch.nn.Linear(10, 3)
        >>>
        >>> # Set all concepts to 1.0
        >>> intervention_scalar = DoIntervention(model, 1.0)
        >>>
        >>> # Set each concept to different values
        >>> intervention_vec = DoIntervention(
        ...     model,
        ...     torch.tensor([0.5, 1.0, 0.0])
        ... )
        >>>
        >>> # Set per-sample values
        >>> intervention_batch = DoIntervention(
        ...     model,
        ...     torch.tensor([[0.0, 1.0, 0.5],
        ...                   [1.0, 0.0, 0.5]])
        ... )
        >>>
        >>> # Use via context manager - see intervention()
    """

    def __init__(self, model: nn.Module, constants: torch.Tensor | float):
        super().__init__(model)
        const = constants if torch.is_tensor(constants) else torch.tensor(constants)
        self.register_buffer("constants", const)

    # unified signature matching base
    def _make_target(self, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        B, F = y.shape
        v = self.constants

        if v.dim() == 0:  # scalar
            v = v.view(1, 1).expand(B, F)
        elif v.dim() == 1:  # [F]
            assert v.numel() == F, f"constants [F] must have F={F}, got {v.numel()}"
            v = v.unsqueeze(0).expand(B, F)
        elif v.dim() == 2:
            b, f = v.shape
            assert f == F, f"constants second dim must be F={F}, got {f}"
            if b == 1:
                v = v.expand(B, F)  # [1, F] -> [B, F]
            else:
                assert b == B, f"constants first dim must be B={B} or 1, got {b}"
        else:
            raise ValueError("constants must be scalar, [F], [1, F], or [B, F]")

        return v.to(dtype=y.dtype, device=y.device)

class DistributionIntervention(RewiringIntervention):
    """
    Intervention that samples concept values from distributions.

    Implements do(C~D) operations where concepts are sampled from specified
    probability distributions, enabling distributional interventions.

    Args:
        model: The concept-based model to intervene on.
        dist: A torch.distributions.Distribution or list of per-concept distributions.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import DistributionIntervention
        >>> from torch.distributions import Bernoulli, Normal
        >>>
        >>> # Create a dummy model
        >>> model = torch.nn.Linear(10, 3)
        >>>
        >>> # Single distribution for all concepts
        >>> intervention_single = DistributionIntervention(
        ...     model,
        ...     Bernoulli(torch.tensor(0.7))
        ... )
        >>>
        >>> # Per-concept distributions
        >>> intervention_multi = DistributionIntervention(
        ...     model,
        ...     [Bernoulli(torch.tensor(0.3)),
        ...      Normal(torch.tensor(0.0), torch.tensor(1.0)),
        ...      Bernoulli(torch.tensor(0.8))]
        ... )
        >>>
        >>> # Use via context manager - see intervention()
    """

    def __init__(self, model: nn.Module, dist):
        super().__init__(model)
        self.dist = dist

    # unified signature matching base
    def _make_target(self, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        B, F = y.shape
        device, dtype = y.device, y.dtype

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

# ---------------- wrapper ----------------

class _InterventionWrapper(nn.Module):
    def __init__(
        self,
        original: nn.Module,
        policy: nn.Module,
        strategy: RewiringIntervention,
        quantile: float,
        subset: Optional[List[int]] = None,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.original = original
        self.policy = policy
        self.strategy = strategy
        self.quantile = float(quantile)
        self.subset = subset
        self.eps = eps
        if hasattr(original, "parametrization"):
            if hasattr(original.parametrization, "forward_to_check"):
                self.forward_to_check = original.parametrization.forward_to_check
            elif hasattr(original.parametrization, "forward"):
                self.forward_to_check = original.parametrization.forward
        else:
            self.forward_to_check = original.forward

    def _build_mask(self, policy_logits: torch.Tensor) -> torch.Tensor:
        B, F = policy_logits.shape
        device = policy_logits.device
        dtype = policy_logits.dtype

        sel_idx = torch.tensor(self.subset, device=device, dtype=torch.long) if self.subset is not None else torch.arange(F, device=device, dtype=torch.long)
        if len(sel_idx) == 0:
            return torch.ones_like(policy_logits)

        K = sel_idx.numel()
        sel = policy_logits.index_select(dim=1, index=sel_idx)  # [B, K]

        if K == 1:
            # Edge case: single selected column.
            # q < 1 => keep; q == 1 => replace.
            keep_col = torch.ones((B, 1), device=device, dtype=dtype) if self.quantile < 1.0 \
                else torch.zeros((B, 1), device=device, dtype=dtype)
            mask = torch.ones((B, F), device=device, dtype=dtype)
            mask.scatter_(1, sel_idx.unsqueeze(0).expand(B, -1), keep_col)

            # STE proxy (optional; keeps gradients flowing on the selected col)
            row_max = sel.max(dim=1, keepdim=True).values + self.eps
            soft_sel = torch.log1p(sel) / torch.log1p(row_max)  # [B,1]
            soft_proxy = torch.ones_like(policy_logits)
            soft_proxy.scatter_(1, sel_idx.unsqueeze(0).expand(B, -1), soft_sel)
            mask = (mask - soft_proxy).detach() + soft_proxy
            return mask

        # K > 1: standard per-row quantile via kthvalue
        k = int(max(1, min(K, 1 + math.floor(self.quantile * (K - 1)))))
        thr, _ = torch.kthvalue(sel, k, dim=1, keepdim=True)  # [B,1]

        # Use strict '>' so ties at the threshold are replaced (robust near edges)
        sel_mask_hard = (sel > (thr - 0.0)).to(dtype)  # [B,K]

        mask = torch.ones((B, F), device=device, dtype=dtype)
        mask.scatter_(1, sel_idx.unsqueeze(0).expand(B, -1), sel_mask_hard)

        # STE proxy (unchanged)
        row_max = sel.max(dim=1, keepdim=True).values + 1e-12
        soft_sel = torch.log1p(sel) / torch.log1p(row_max)
        soft_proxy = torch.ones_like(policy_logits)
        soft_proxy.scatter_(1, sel_idx.unsqueeze(0).expand(B, -1), soft_sel)
        mask = (mask - soft_proxy).detach() + soft_proxy
        return mask

    def forward(self, **kwargs) -> torch.Tensor:
        y = self.original(**kwargs)
        logits = self.policy(y)          # [B,F], 0 = most uncertain, +inf = most certain
        mask = self._build_mask(logits)  # 1 keep, 0 replace

        # 3) proxy that returns the cached y instead of recomputing
        class _CachedOutput(nn.Module):
            def __init__(self, y_cached: torch.Tensor):
                super().__init__()
                self.y_cached = y_cached        # keep graph-connected tensor; do NOT detach
            def forward(self, **kwargs) -> torch.Tensor:
                return self.y_cached

        cached = _CachedOutput(y)

        # 4) use existing strategy API; no changes to GroundTruthIntervention
        replacer = self.strategy.query(cached, mask)
        return replacer(**kwargs)

# ---------------- context manager (now multi-layer) ----------------

@contextlib.contextmanager
def intervention(
    *,
    policies: Union[nn.Module, Sequence[nn.Module]],
    strategies: Union[RewiringIntervention, Sequence[RewiringIntervention]],
    target_concepts: Union[str, int, Sequence[Union[str, int]]],
    quantiles: Optional[Union[float, Sequence[float]]] = 1.,
    model: nn.Module = None,
):
    """
    Context manager for applying interventions to concept-based models.

    Enables interventions on concept modules by temporarily replacing model
    components with intervention wrappers. Supports single or multiple layers.

    Args:
        policies: Policy module(s) that determine which concepts to intervene on.
        strategies: Intervention strategy/strategies (e.g., DoIntervention).
        target_concepts: Concept names/paths or indices to intervene on.
        quantiles: Quantile thresholds for selective intervention (default: 1.0).
        model: Optional model reference (default: strategies[0].model).

    Yields:
        The intervention wrapper (if target_concepts are indices) or None.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import (
        ...     DoIntervention, intervention, RandomPolicy
        ... )
        >>> from torch_concepts import Variable
        >>>
        >>> # Create a simple model
        >>> class SimplePGM(torch.nn.Module):
        ...     def __init__(self, in_features, out_features):
        ...         super().__init__()
        ...         self.encoder = torch.nn.Linear(in_features, 3)
        ...         self.predictor = torch.nn.Linear(3, out_features)
        ...     def forward(self, x):
        ...         c = torch.sigmoid(self.encoder(x))
        ...         y = self.predictor(c)
        ...         return y
        >>>
        >>> model = SimplePGM(10, 3)
        >>>
        >>> # Create intervention strategy (set concepts to 1)
        >>> strategy = DoIntervention(model, torch.FloatTensor([1.0, 0.0, 1.0]))
        >>>
        >>> # Create policy (random selection)
        >>> policy = RandomPolicy(out_features=3)
        >>>
        >>> # Apply intervention on specific concept indices
        >>> x = torch.randn(4, 10)
        >>> with intervention(
        ...     policies=policy,
        ...     strategies=strategy,
        ...     target_concepts=[0, 2],  # Intervene on concepts 0 and 2
        ...     quantiles=0.8
        ... ) as wrapper:
        ...     # Inside context, interventions are active
        ...     output = wrapper(x=x)
        >>>
        >>> print(f"Output shape: {output.shape}")
        Output shape: torch.Size([4, 3])
    """
    # Normalise on_layers to list and compute N
    if isinstance(target_concepts, str):
        target_concepts = [target_concepts]
    N = len(target_concepts)

    # Choose the reference model
    if isinstance(strategies, Sequence):
        ref_model = strategies[0].model
    else:
        ref_model = strategies.model

    originals: List[tuple[str, nn.Module]] = []

    try:
        if isinstance(target_concepts[0], int):
            # in this case we expect a single module to replace
            assert not isinstance(policies, Sequence), "When target_concepts are indices, only a single policy is supported"
            assert not isinstance(strategies, Sequence), "When target_concepts are indices, only a single strategy is supported"
            assert not isinstance(quantiles, Sequence), "When target_concepts are indices, only a single quantile is supported"
            wrap = _InterventionWrapper(
                original=strategies.model,
                policy=policies,
                strategy=strategies,
                quantile=quantiles,
                subset=target_concepts  # type: ignore
            )
            yield wrap

        else:
            # Broadcast/validate others
            policies = _as_list(policies, N)
            strategies = _as_list(strategies, N)
            quantiles = _as_list(quantiles, N)

            for path, pol, strat, q in zip(target_concepts, policies, strategies, quantiles):
                orig = _get_submodule(ref_model, path)
                originals.append((path, orig))
                wrap = _InterventionWrapper(
                    original=orig,
                    policy=pol,
                    strategy=strat,
                    quantile=q,
                )
                _set_submodule(ref_model, path, wrap)
            yield
    finally:
        # restore originals
        for path, orig in originals:
            _set_submodule(ref_model, path, orig)
