"""
Inference and intervention modules for concept-based models.

This module provides intervention strategies that modify concept values during
inference, enabling causal reasoning and what-if analysis in concept-based models.
"""
import math
import contextlib
from abc import abstractmethod
from typing import List, Sequence, Union, Optional, Dict
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
    # validate
    if len(parts) == 0 or (len(parts) == 1 and parts[0] == ""):
        raise ValueError("Dotted path must not be empty")

    parent = model.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else model
    name = parts[-1]

    # If parent supports indexed assignment (e.g., nn.Sequential) and the name is an index, set by index
    if name.isdigit() and hasattr(parent, "__setitem__"):
        idx = int(name)
        parent[idx] = new
        return

    # Otherwise set as attribute on parent.
    # If the new module is already a ParametricCPD, keep it. If not and we're attaching
    # it as a plain attribute on a Module, wrap it into a ParametricCPD so semantics are preserved.
    if isinstance(new, ParametricCPD):
        setattr(parent, name, new)
    else:
        setattr(parent, name, ParametricCPD(concepts=dotted, parametrization=new))

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
                assert y.dim() == 2, (f"RewiringIntervention expects 2-D tensors [Batch, N_concepts]. "
                                      f"Got shape: {y.shape}")
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

    def _build_mask(self, policy_endogenous: torch.Tensor) -> torch.Tensor:
        B, F = policy_endogenous.shape
        device = policy_endogenous.device
        dtype = policy_endogenous.dtype

        sel_idx = torch.tensor(self.subset, device=device, dtype=torch.long) if self.subset is not None else torch.arange(F, device=device, dtype=torch.long)
        if len(sel_idx) == 0:
            return torch.ones_like(policy_endogenous)

        K = sel_idx.numel()
        sel = policy_endogenous.index_select(dim=1, index=sel_idx)  # [B, K]

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
            soft_proxy = torch.ones_like(policy_endogenous)
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
        soft_proxy = torch.ones_like(policy_endogenous)
        soft_proxy.scatter_(1, sel_idx.unsqueeze(0).expand(B, -1), soft_sel)
        mask = (mask - soft_proxy).detach() + soft_proxy
        return mask

    def forward(self, **kwargs) -> torch.Tensor:
        y = self.original(**kwargs)
        endogenous = self.policy(y)          # [B,F], 0 = most uncertain, +inf = most certain
        mask = self._build_mask(endogenous)  # 1 keep, 0 replace

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

# ---------------- global policy wrapper ----------------

class _GlobalPolicyState:
    """
    Shared state for coordinating global policy across multiple wrappers.

    This state object is shared among all wrappers when global_policy=True.
    It collects policy endogenous from all layers, computes a global mask once,
    then distributes slices to each wrapper.

    This implementation works with sequential, threaded, and CUDA stream execution.
    """
    def __init__(self, n_wrappers: int, quantile: float, eps: float = 1e-12):
        self.n_wrappers = n_wrappers
        self.quantile = float(quantile)
        self.eps = eps
        # Store endogenous and outputs indexed by wrapper_id
        self.endogenous_cache: Dict[int, torch.Tensor] = {}
        self.outputs_cache: Dict[int, torch.Tensor] = {}
        self.global_mask: Optional[torch.Tensor] = None
        self.batch_size: Optional[int] = None

    def reset(self):
        """Reset state for a new forward pass."""
        self.endogenous_cache.clear()
        self.outputs_cache.clear()
        self.global_mask = None
        self.batch_size = None

    def register(self, wrapper_id: int, endogenous: torch.Tensor, output: torch.Tensor):
        """Register endogenous and output from a wrapper."""
        # Detect new batch by checking batch size change
        if self.batch_size is not None and endogenous.shape[0] != self.batch_size:
            self.reset()
        self.batch_size = endogenous.shape[0]

        self.endogenous_cache[wrapper_id] = endogenous
        self.outputs_cache[wrapper_id] = output

    def is_ready(self) -> bool:
        """Check if all wrappers have registered their endogenous."""
        return len(self.endogenous_cache) == self.n_wrappers

    def compute_global_mask(self):
        """Compute the global mask once all endogenous are collected."""
        if self.global_mask is not None:
            return  # Already computed

        if not self.is_ready():
            raise RuntimeError(
                f"Cannot compute global mask: only {len(self.endogenous_cache)}/{self.n_wrappers} wrappers registered"
            )

        # Concatenate all endogenous in wrapper_id order
        all_endogenous = torch.cat([self.endogenous_cache[i] for i in range(self.n_wrappers)], dim=1)
        B, F_total = all_endogenous.shape
        device = all_endogenous.device
        dtype = all_endogenous.dtype

        if F_total == 0:
            self.global_mask = torch.ones((B, 0), device=device, dtype=dtype)
            return

        # quantile determines the fraction of concepts to intervene on
        # quantile=0 -> intervene on 0% (mask=1 for all, keep all)
        # quantile=1 -> intervene on 100% (mask=0 for all, replace all)
        num_to_intervene = int(max(0, min(F_total, math.ceil(self.quantile * F_total))))

        if num_to_intervene == 0:
            # Don't intervene on any concepts - keep all predictions
            # mask=1 means keep, so all ones
            self.global_mask = torch.ones((B, F_total), device=device, dtype=dtype)
            return

        if num_to_intervene == F_total:
            # Intervene on all concepts - replace all predictions
            # mask=0 means intervene, so all zeros
            self.global_mask = torch.zeros((B, F_total), device=device, dtype=dtype)
            return

        # Find the threshold: intervene on the top num_to_intervene concepts by policy endogenous
        # kthvalue(k) returns the k-th smallest value, so for top-k we use (F_total - num_to_intervene + 1)
        k = F_total - num_to_intervene + 1
        thr, _ = torch.kthvalue(all_endogenous, k, dim=1, keepdim=True)  # [B,1]

        # mask=1 means keep (don't intervene), mask=0 means replace (do intervene)
        # Intervene on concepts with endogenous >= threshold (top-k by policy score)
        # So those get mask=0, others get mask=1
        mask_hard = (all_endogenous < thr).to(dtype)  # [B, F_total] - 1 where we keep, 0 where we intervene

        # STE proxy
        row_max = all_endogenous.max(dim=1, keepdim=True).values + self.eps
        soft_proxy = torch.log1p(all_endogenous) / torch.log1p(row_max)
        self.global_mask = (mask_hard - soft_proxy).detach() + soft_proxy

    def get_mask_slice(self, wrapper_id: int) -> torch.Tensor:
        """Get the mask slice for a specific wrapper."""
        if self.global_mask is None:
            raise RuntimeError("Global mask not computed yet")

        # Calculate start/end index for this wrapper based on output shapes
        start_idx = sum(self.outputs_cache[i].shape[1] for i in range(wrapper_id))
        end_idx = start_idx + self.outputs_cache[wrapper_id].shape[1]

        return self.global_mask[:, start_idx:end_idx]


class _GlobalPolicyInterventionWrapper(nn.Module):
    """
    Intervention wrapper that uses a shared global state for coordinated masking.

    This wrapper defers intervention application until all wrappers in the level
    have computed their policy endogenous. During forward pass, it only collects
    endogenous and returns the original output. The actual intervention is applied
    via apply_intervention() after all wrappers are ready.
    """
    def __init__(
        self,
        original: nn.Module,
        policy: nn.Module,
        strategy: RewiringIntervention,
        wrapper_id: int,
        shared_state: '_GlobalPolicyState',
    ):
        super().__init__()
        self.original = original
        self.policy = policy
        self.strategy = strategy
        self.wrapper_id = wrapper_id
        self.shared_state = shared_state

        if hasattr(original, "parametrization"):
            if hasattr(original.parametrization, "forward_to_check"):
                self.forward_to_check = original.parametrization.forward_to_check
            elif hasattr(original.parametrization, "forward"):
                self.forward_to_check = original.parametrization.forward
        else:
            self.forward_to_check = original.forward

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass that collects policy endogenous but does NOT apply intervention.

        Returns the original output. Intervention is applied later via apply_intervention().
        """
        # Get output from original module
        y = self.original(**kwargs)

        # Compute policy endogenous
        endogenous = self.policy(y)  # [B, F_i]

        # Register with shared state
        self.shared_state.register(self.wrapper_id, endogenous, y)

        # Always return original output - intervention applied later
        return y

    def apply_intervention(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply intervention to the output after all wrappers are ready.

        This should be called after all wrappers in the level have completed forward().

        Args:
            y: The original output from forward()

        Returns:
            Intervened output
        """
        if not self.shared_state.is_ready():
            raise RuntimeError(
                f"Cannot apply intervention: only {len(self.shared_state.endogenous_cache)}/{self.shared_state.n_wrappers} wrappers registered"
            )

        # Compute global mask if not already computed
        if self.shared_state.global_mask is None:
            self.shared_state.compute_global_mask()

        # Get mask slice for this wrapper
        mask = self.shared_state.get_mask_slice(self.wrapper_id)

        # Create cached output wrapper
        class _CachedOutput(nn.Module):
            def __init__(self, y_cached: torch.Tensor):
                super().__init__()
                self.y_cached = y_cached
            def forward(self, **kwargs) -> torch.Tensor:
                return self.y_cached

        cached = _CachedOutput(y)
        replacer = self.strategy.query(cached, mask)
        result = replacer()

        return result

# ---------------- context manager (now multi-layer) ----------------

@contextlib.contextmanager
def intervention(
    *,
    policies: Union[nn.Module, Sequence[nn.Module]],
    strategies: Union[RewiringIntervention, Sequence[RewiringIntervention]],
    target_concepts: Union[str, int, Sequence[Union[str, int]]],
    quantiles: Optional[Union[float, Sequence[float]]] = 1.,
    model: nn.Module = None,
    global_policy: bool = False,
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
        global_policy: If True, multiple policies are coordinated globally to create
            a unified mask across all layers. If False (default), each policy operates
            independently on its layer. Only applies when target_concepts are strings
            and multiple policies are provided.

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
        >>>
        >>> # Example with global_policy=True for coordinated multi-layer intervention
        >>> # (requires multiple layers and policies)
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
            assert not global_policy, "global_policy not supported for index-based interventions"
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

            if global_policy:
                # Global policy mode: coordinate all policies to create unified global mask

                # Validate: all quantiles must be the same for global policy
                if not all(q == quantiles[0] for q in quantiles):
                    raise ValueError(
                        "When global_policy=True, all quantiles must be the same. "
                        f"Got: {quantiles}"
                    )

                global_quantile = quantiles[0]

                # Create shared state for coordination
                shared_state = _GlobalPolicyState(n_wrappers=N, quantile=global_quantile)

                # Create global wrappers for each layer
                for wrapper_id, (path, pol, strat) in enumerate(zip(target_concepts, policies, strategies)):
                    orig = _get_submodule(ref_model, path)
                    originals.append((path, orig))

                    wrapper = _GlobalPolicyInterventionWrapper(
                        original=orig,
                        policy=pol,
                        strategy=strat,
                        wrapper_id=wrapper_id,
                        shared_state=shared_state,
                    )
                    _set_submodule(ref_model, path, wrapper)

                # Don't yield anything - wrappers coordinate automatically during forward pass
                yield
            else:
                # Independent mode (default/backward compatible): each policy creates its own mask
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
