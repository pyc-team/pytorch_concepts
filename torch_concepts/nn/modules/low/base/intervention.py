"""
Interventions on the outputs of a layer.

A policy scores the outputs to choose which ones to intervene on, and a strategy
gives them their new values. :class:`InterventionModule` wraps a layer with both;
:func:`intervention` applies them to a layer for the duration of a ``with`` block.
"""
import functools
import inspect
import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from torch_concepts import Annotations


class InterventionStrategy(ABC):
    """Common base of :class:`ConceptInterventionStrategy` and :class:`ModuleInterventionStrategy`. 
    Intervention strategies define how to intervene on layers (either on the parametrization or on the output)."""


class ConceptInterventionStrategy(nn.Module, InterventionStrategy):
    """
    Strategy that computes new values for a layer's outputs.

    Subclasses implement ``forward(x, ...)``, returning a tensor shaped like the
    layer output ``x``.
    """
    def __init__(self, *args, **kwargs):
        super(ConceptInterventionStrategy, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """New values for the layer output ``x``, same shape as ``x``."""
        raise NotImplementedError


class ModuleInterventionStrategy(InterventionStrategy):
    """
    Strategy that evaluates a modified version of the layer.

    Subclasses implement ``transform(module, ...)``, returning a callable that is
    called like ``module``.
    """
    def __init__(self, *args, **kwargs):
        super(ModuleInterventionStrategy, self).__init__()

    @abstractmethod
    def transform(self, module: nn.Module, *args, **kwargs):
        """A callable evaluated like ``module`` under the intervention. Do not modify
        ``module`` in place, or the change outlives the intervention."""
        raise NotImplementedError


class InterventionPolicy(nn.Module, ABC):
    """
    Scores a layer's outputs to choose which ones to intervene on.

    Subclasses implement ``forward(x, ...)``, returning one score per output of
    ``x``; the lowest-scoring outputs are intervened on first.
    """
    def __init__(self):
        super(InterventionPolicy, self).__init__()

    @abstractmethod
    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        """Scores for the layer output ``x``, same shape as ``x``."""
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
                f"out_concepts_to_intervene_on leading dims {tuple(sel_idx.shape[:-1])} "
                f"cannot be broadcast against the policy_scores leading dims "
                f"{tuple(lead)}"
            ) from e
        # Use the already-known B rather than -1: when K == 0 (empty subset),
        # the element count is 0 and -1 is ambiguous (any size satisfies it).
        sel_idx = sel_idx.reshape(B, sel_idx.shape[-1])

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


class InterventionModule(nn.Module):
    """
    Wraps a layer so that calling it returns intervened outputs.

    The policy scores the selected outputs, ``quantile`` sets how many of them are
    intervened on, and the strategy gives their new values. All other outputs are
    returned unchanged.

    Args:
        original_module: Layer to intervene on.
        intervention_strategy: Computes the new values of the intervened outputs.
        intervention_policy: Scores the outputs; the lowest are intervened on first.
        out_concepts_to_intervene_on: Outputs that may be intervened on: names
            (if ``original_module.out_concepts`` is an :class:`Annotations`),
            positions, or per-sample positions as a ``[*lead, K]`` nested list or
            LongTensor.
        quantile: Fraction of the selected outputs to intervene on. Defaults to
            1.0 (all of them).
        eps: Numerical stability constant used when building the mask.
        build_context: Optional callable returning extra kwargs for the policy
            and the strategy; see :meth:`build_context`.
        extra_modules: Optional ``{name: module}`` registered on this module and
            passed to ``build_context``.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import InterventionModule, DoIntervention, UniformPolicy
        >>>
        >>> layer = torch.nn.Linear(8, 3)
        >>> intervened = InterventionModule(layer, DoIntervention(0.0), UniformPolicy(), [1])
        >>> out = intervened(torch.randn(4, 8))
        >>> out[:, 1].tolist()
        [0.0, 0.0, 0.0, 0.0]
    """

    def __init__(
            self,
            original_module: nn.Module,
            intervention_strategy: InterventionStrategy,
            intervention_policy: InterventionPolicy,
            out_concepts_to_intervene_on: Union[List[str], List[int], List[List[int]], torch.Tensor],
            quantile: float = 1.0,
            eps: float = 1e-12,
            build_context: Optional[Callable] = None,
            extra_modules: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__()
        if not isinstance(intervention_strategy, InterventionStrategy):
            raise ValueError("Intervention strategy must be an instance of "
                             "ConceptInterventionStrategy or ModuleInterventionStrategy.")
        if out_concepts_to_intervene_on is None or isinstance(out_concepts_to_intervene_on, str):
            raise ValueError("out_concepts_to_intervene_on is required, as a list: the names (if the "
                             "module has annotated out_concepts) or positions of the outputs to intervene on.")
        self.original_module = original_module
        self.intervention_strategy = intervention_strategy
        self.intervention_policy = intervention_policy
        self.out_concepts_to_intervene_on = out_concepts_to_intervene_on
        self.quantile = quantile
        self.eps = eps
        self._build_context_fn = build_context
        if extra_modules:
            for name, module in extra_modules.items():
                self.add_module(name, module)
        self._patch_forward_signature()

    def _patch_forward_signature(self):
        """Give ``self.forward`` the wrapped layer's signature plus a keyword-only
        ``extra_tensors``, so that ``help()`` and IDEs show the real arguments."""
        try:
            orig_sig = inspect.signature(self.original_module.forward)
            params = [p for p in orig_sig.parameters.values() if p.name != 'self']
            extra_param = inspect.Parameter(
                'extra_tensors',
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[Dict[str, torch.Tensor]]
            )
            # insert before **kwargs if present, otherwise append
            var_kw_idx = next(
                (i for i, p in enumerate(params) if p.kind == inspect.Parameter.VAR_KEYWORD),
                None
            )
            if var_kw_idx is not None:
                params.insert(var_kw_idx, extra_param)
            else:
                params.append(extra_param)
            new_sig = orig_sig.replace(parameters=params)

            original_forward = type(self).forward

            @functools.wraps(original_forward)
            def patched_forward(*args, **kwargs):
                return original_forward(self, *args, **kwargs)

            patched_forward.__signature__ = new_sig
            self.forward = patched_forward
        except (ValueError, TypeError):
            pass  # silently skip if signature cannot be determined

    @property
    def sel_idx(self):
        """``out_concepts_to_intervene_on`` as a LongTensor of positions."""
        spec = self.out_concepts_to_intervene_on

        # LongTensor: either [K] (shared across the batch) or [*lead, K]
        # (fixed-size, per-leading-dim-element indices). Passed through as-is;
        # build_mask normalizes both to [B, K].
        if torch.is_tensor(spec):
            return spec.to(dtype=torch.long)

        if len(spec) == 0:
            return torch.empty(0, dtype=torch.long)

        first = spec[0]
        if isinstance(first, str):
            original_annotations = getattr(self.original_module, "out_concepts", None)
            if not isinstance(original_annotations, Annotations):
                raise ValueError("To use string-based concept selection, the original module must have an "
                                 "'out_concepts' attribute of type Annotations.")
            return torch.tensor(original_annotations.get_slice(spec), dtype=torch.long)
        elif isinstance(first, int):
            return torch.tensor(spec, dtype=torch.long)
        elif isinstance(first, (list, tuple)):
            # per-leading-dim-element indices as nested lists, fixed K: [*lead, K]
            return torch.tensor(spec, dtype=torch.long)
        else:
            raise ValueError(
                "out_concepts_to_intervene_on must be a list of integers (shared "
                "indices), a list of strings (shared names), a list of lists of "
                "integers (per-batch-element indices, fixed K), or a LongTensor "
                "of shape [K] or [*lead, K]."
            )

    def build_context(
            self,
            original_module_inputs: Dict[str, torch.Tensor],
            original_module: nn.Module,
            original_module_predictions: torch.Tensor,
            extra_tensors: Dict[str, torch.Tensor] = None,
            extra_modules: Dict[str, nn.Module] = None,
    ) -> dict:
        """
        Extra kwargs for the policy and the strategy; empty by default.

        Override it in a subclass, or pass a ``build_context`` callable at
        construction, which is called as ``build_context(original_module_predictions,
        original_module, original_module_inputs, extra_tensors, extra_modules)``.

        Args:
            original_module_inputs: Arguments of the layer call, by name
                (e.g. ``{"embeddings": x}``).
            original_module: The wrapped layer.
            original_module_predictions: The layer output, shape ``[..., F]``.
            extra_tensors: Tensors passed at call time as ``extra_tensors=...``.
            extra_modules: The ``extra_modules`` given at construction.
        """
        if self._build_context_fn is not None:
            return self._build_context_fn(
                original_module_predictions,
                self.original_module,
                original_module_inputs,
                extra_tensors,
                extra_modules,
            )
        return {}

    def forward(self, *args, **kwargs) -> torch.Tensor:
        extra_tensors = kwargs.pop('extra_tensors', None)
        return self._intervene(self.original_module(*args, **kwargs), args, kwargs, extra_tensors)

    def _intervene(
            self,
            original_module_predictions: torch.Tensor,
            args: tuple,
            kwargs: dict,
            extra_tensors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Intervene on ``original_module_predictions``, the output of
        ``original_module(*args, **kwargs)``, without running the module again."""
        extra_tensors = extra_tensors or {}

        # bind positional and keyword args to parameter names of the wrapped module
        try:
            sig = inspect.signature(self.original_module.forward)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            original_module_inputs = dict(bound.arguments)
        except TypeError:
            original_module_inputs = {}

        assert original_module_predictions.dim() >= 1, (
            f"ConceptInterventionStrategy expects tensors of shape "
            f"[..., N_concepts] (arbitrary leading dims, concepts last). "
            f"Got shape: {original_module_predictions.shape}"
        )

        extra_modules = {
            name: module
            for name, module in self._modules.items()
            if name not in ("original_module", "intervention_strategy", "intervention_policy")
        }

        context = self.build_context(
            original_module_inputs,
            self.original_module,
            original_module_predictions,
            extra_tensors,
            extra_modules,
        )

        policy_scores = self.intervention_policy(original_module_predictions, *args, **kwargs, **context)
        intervention_mask = self.intervention_policy.build_mask(
            policy_scores,
            sel_idx=self.sel_idx,
            quantile=self.quantile,
            eps=self.eps
        ).to(dtype=original_module_predictions.dtype)

        if isinstance(self.intervention_strategy, ConceptInterventionStrategy):
            intervened_predictions = self.intervention_strategy(original_module_predictions, *args, **kwargs, **context)

        elif isinstance(self.intervention_strategy, ModuleInterventionStrategy):
            intervened_module = self.intervention_strategy.transform(self.original_module, *args, **kwargs)
            intervened_predictions = intervened_module(*args, **kwargs)

        else:
            raise ValueError("Intervention strategy must be an instance of "
                             "ConceptInterventionStrategy or ModuleInterventionStrategy.")

        return (original_module_predictions * intervention_mask +
                intervened_predictions * (1.0 - intervention_mask))


def _locate(
        target: nn.Module,
        out_concepts_to_intervene_on: Union[List[str], List[int], List[List[int]], torch.Tensor],
        parameter_to_intervene_on: Optional[str],
):
    """The layer ``target`` reaches, and what to intervene on in its output."""
    from ...mid.graph.probabilistic_model import ProbabilisticModel  # mid imports this module

    pgm = getattr(target, "pgm", target)  # a high-level model intervenes through its PGM
    if not isinstance(pgm, ProbabilisticModel):
        if parameter_to_intervene_on is not None:
            raise TypeError("intervention: parameter_to_intervene_on only applies to a "
                            "ProbabilisticModel or high-level model target.")
        return target, out_concepts_to_intervene_on

    names = out_concepts_to_intervene_on
    if (not isinstance(names, (list, tuple)) or not names
            or not all(isinstance(n, str) for n in names)):
        raise TypeError(
            "intervention: on a ProbabilisticModel or high-level model, "
            "out_concepts_to_intervene_on must be a non-empty list of variable or "
            "member names, which also select the layer. To intervene by position, "
            "pass the layer itself as target."
        )
    unknown = [n for n in names if n not in pgm.queryable_names]
    if unknown:
        raise KeyError(f"intervention: unknown names {unknown}; available names are "
                       f"{sorted(pgm.queryable_names)}.")
    owners = sorted({pgm.resolve(n).name for n in names})
    if len(owners) > 1:
        raise ValueError(f"intervention: {list(names)} are produced by different layers "
                         f"(variables {owners}); use one intervention per layer, e.g. "
                         f"nested `with` blocks.")

    variable = pgm.resolve(names[0])
    if variable.name not in pgm.factors:
        raise KeyError(f"intervention: no factor named {variable.name!r}; available "
                       f"factors are {sorted(pgm.factors.keys())}.")
    parametrization = pgm.factors[variable.name].parametrization
    if parameter_to_intervene_on is None:
        if len(parametrization) > 1:
            raise ValueError(f"intervention: factor {variable.name!r} has parameters "
                             f"{sorted(parametrization.keys())}; pass parameter_to_intervene_on.")
        parameter_to_intervene_on = next(iter(parametrization))
    elif parameter_to_intervene_on not in parametrization:
        raise KeyError(f"intervention: factor {variable.name!r} has no parameter "
                       f"{parameter_to_intervene_on!r}; available parameters are "
                       f"{sorted(parametrization.keys())}.")

    # the variable's own name stands for all of its members
    columns = [col for n in names
               for col in variable.flat_columns(variable.members if n == variable.name else n)]
    return parametrization[parameter_to_intervene_on], columns


@contextmanager
def intervention(
        target: nn.Module,
        intervention_strategy: Optional[InterventionStrategy] = None,
        intervention_policy: Optional[InterventionPolicy] = None,
        out_concepts_to_intervene_on: Union[List[str], List[int], List[List[int]], torch.Tensor] = None,
        parameter_to_intervene_on: Optional[str] = None,
        quantile: float = 1.0,
        eps: float = 1e-12,
        build_context: Optional[Callable] = None,
        extra_modules: Optional[Dict[str, nn.Module]] = None,
):
    """
    Intervene on a layer for the duration of a ``with`` block.

    Inside the block the layer returns intervened outputs to every caller: your
    code, an inference engine, or a high-level model. The layer is reached through
    ``target``:

    - a layer: ``out_concepts_to_intervene_on`` holds names (if the layer has
      annotated ``out_concepts``) or positions of its outputs;
    - a :class:`ProbabilisticModel` or a high-level model: it holds names of one
      variable or of members of one plate, which also select the layer (a
      variable's name covers all its members). To use positions, pass the layer;
    - an :class:`InterventionModule`: then no other argument is taken.

    Args:
        target: Layer, probabilistic model, high-level model, or InterventionModule.
        intervention_strategy: Computes the new values of the intervened outputs.
        intervention_policy: Scores the outputs; the lowest are intervened on first.
        out_concepts_to_intervene_on: Outputs that may be intervened on (see above).
        parameter_to_intervene_on: Head of the variable's factor to intervene on
            (e.g. ``"loc"``), needed only when the factor has more than one.
        quantile, eps, build_context, extra_modules: As in :class:`InterventionModule`.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import intervention, DoIntervention, UniformPolicy
        >>>
        >>> layer = torch.nn.Linear(8, 3)
        >>> with intervention(layer, DoIntervention(0.0), UniformPolicy(), [1]):
        ...     out = layer(torch.randn(4, 8))
        >>> out[:, 1].tolist()
        [0.0, 0.0, 0.0, 0.0]

        On a probabilistic or high-level model, name the concept instead::

            with intervention(model, DoIntervention(1.0), UniformPolicy(), ["c1"]):
                out = model(query=["c1", "y"], input=x)
    """
    if isinstance(target, InterventionModule):
        others = (intervention_strategy, intervention_policy, out_concepts_to_intervene_on,
                  parameter_to_intervene_on, build_context, extra_modules)
        if any(arg is not None for arg in others) or (quantile, eps) != (1.0, 1e-12):
            raise TypeError("intervention takes no other arguments if an InterventionModule "
                            "is passed as target; set them on the InterventionModule instead.")
        intervention_module = target
    else:
        layer, out_concepts_to_intervene_on = _locate(
            target, out_concepts_to_intervene_on, parameter_to_intervene_on
        )
        intervention_module = InterventionModule(
            layer,
            intervention_strategy,
            intervention_policy,
            out_concepts_to_intervene_on,
            quantile,
            eps,
            build_context=build_context,
            extra_modules=extra_modules,
        )

    running = False

    def hook(_, args, kwargs, output):
        nonlocal running
        if running:  # the layer called again while intervening, e.g. by a module strategy
            return None
        running = True
        try:
            # reuse the output the layer just computed instead of running it again
            return intervention_module._intervene(output, args, kwargs)
        finally:
            running = False

    handle = intervention_module.original_module.register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
