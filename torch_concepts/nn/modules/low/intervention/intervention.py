import inspect
import functools
from itertools import chain
from abc import abstractmethod, ABC

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Union

from torch_concepts import Annotations
from ..base.intervention import (
    BaseConceptInterventionStrategy,
    BaseModuleInterventionStrategy,
    BaseInterventionPolicy
)


class BaseInterventionModule(nn.Module, ABC):
    """
    Base class for intervention modules that wrap an original module with a specified
    intervention strategy and policy.

    This module applies the intervention strategy to the outputs of the original module
    according to the intervention policy, allowing for flexible interventions on concept encoders.

    Subclasses should implement the specific logic for applying the intervention strategy
    and policy in the forward method.
    """

    def __init__(
            self,
            original_module: nn.Module,
            intervention_strategy: Union[BaseConceptInterventionStrategy, BaseModuleInterventionStrategy],
            intervention_policy: BaseInterventionPolicy,
            out_concepts_to_intervene_on: Union[List[str], List[int]] = None,
            quantile: float = 1.0,
            eps: float = 1e-12,
            build_context: Optional[Callable] = None,
            extra_modules: Optional[Dict[str, nn.Module]] = None,
            *args,
            **kwargs
    ):
        super().__init__()
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
        """
        Patches ``self.forward`` at instance level to expose the same named
        arguments as ``original_module.forward``, with ``extra_tensors`` added
        as a keyword-only argument.

        This means IDEs and ``help()`` will show the correct call signature for
        this :class:`InterventionModule` instance, e.g.::

            forward(embeddings: Tensor, *, extra_tensors: Optional[Dict[str, Tensor]] = None)

        The same named arguments are also visible inside ``build_context`` when
        calling ``module.original_module.forward``, because ``original_module``
        is a concrete type with its own declared signature.
        """
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

            original_forward = InterventionModule.forward

            @functools.wraps(original_forward)
            def patched_forward(*args, **kwargs):
                return original_forward(self, *args, **kwargs)

            patched_forward.__signature__ = new_sig
            self.forward = patched_forward
        except (ValueError, TypeError):
            pass  # silently skip if signature cannot be determined

    @property
    def sel_idx(self):
        if self.out_concepts_to_intervene_on is not None:
            if isinstance(self.out_concepts_to_intervene_on[0], int):
                return torch.tensor(self.out_concepts_to_intervene_on, dtype=torch.long)
            elif isinstance(self.out_concepts_to_intervene_on[0], str):
                original_annotations = getattr(self.original_module, "out_concepts", None)
                if original_annotations is None and not isinstance(original_annotations, Annotations):
                    raise ValueError("To use string-based concept selection, the original module must have an "
                                     "'out_concepts' attribute of type Annotations.")
                indices = list(chain.from_iterable(
                    range(s.start, s.stop, s.step or 1)
                    for s in (original_annotations.concept_slices[item] for item in self.out_concepts_to_intervene_on)
                ))
                return torch.tensor(indices, dtype=torch.long)
            else:
                raise ValueError("out_concepts_to_intervene_on must be a list of integers (indices) or strings (names)")

        return None

    @abstractmethod
    def build_context(
            self,
            original_module_inputs: Dict[str, torch.Tensor],
            original_module: nn.Module,
            original_module_predictions: torch.Tensor,
            extra_tensors: Dict[str, torch.Tensor] = None,
            extra_modules: Dict[str, nn.Module] = None,
    ) -> dict:
        raise NotImplementedError("Subclasses must implement build_context method "
                                  "to provide extra context for policy and strategy.")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        extra_tensors: Dict[str, torch.Tensor] = kwargs.pop('extra_tensors', None) or {}

        # bind positional and keyword args to parameter names of the wrapped module
        try:
            sig = inspect.signature(self.original_module.forward)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            original_module_inputs = dict(bound.arguments)
        except TypeError:
            original_module_inputs = {}

        original_module_predictions = self.original_module(*args, **kwargs)  # [B, F]
        assert original_module_predictions.dim() == 2, (
            f"BaseConceptInterventionStrategy expects 2-D tensors [Batch, N_concepts]. "
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

        if isinstance(self.intervention_strategy, BaseConceptInterventionStrategy):
            intervened_predictions = self.intervention_strategy(original_module_predictions, *args, **kwargs, **context)

        elif isinstance(self.intervention_strategy, BaseModuleInterventionStrategy):
            intervened_module = self.intervention_strategy.transform(self.original_module, *args, **kwargs)
            intervened_predictions = intervened_module(*args, **kwargs)

        else:
            raise ValueError("Intervention strategy must be an instance of "
                             "BaseConceptInterventionStrategy or BaseModuleInterventionStrategy.")

        return (original_module_predictions * intervention_mask +
                intervened_predictions * (1.0 - intervention_mask))


class InterventionModule(BaseInterventionModule):
    def __init__(
            self,
            original_module: nn.Module,
            intervention_strategy: Union[BaseConceptInterventionStrategy, BaseModuleInterventionStrategy],
            intervention_policy: BaseInterventionPolicy,
            out_concepts_to_intervene_on: Union[List[str], List[int]] = None,
            quantile: float = 1.0,
            eps: float = 1e-12,
            build_context: Optional[Callable] = None,
            extra_modules: Optional[Dict[str, nn.Module]] = None,
            *args,
            **kwargs
    ):
        super().__init__(
            original_module,
            intervention_strategy,
            intervention_policy,
            out_concepts_to_intervene_on,
            quantile,
            eps,
            build_context=build_context,
            extra_modules=extra_modules,
            *args,
            **kwargs
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
        Build extra context passed as kwargs to the policy and strategy.

        Override this method in a subclass, or supply a ``build_context``
        callable at construction time. The callable receives::

            build_context(
                original_module,
                original_module_predictions,
                original_module_inputs,
                extra_tensors,
                extra_modules,
            )

        where:
        - ``original_module``            — the wrapped encoder module
        - ``original_module_predictions`` — encoder output ``[B, F]``, with grad_fn intact
        - ``original_module_inputs``      — encoder inputs bound by name via ``inspect.signature``
                                           (e.g. ``{"embeddings": tensor}``)
        - ``extra_tensors``              — dict of tensors passed by the caller at call time
                                           (e.g. pre-computed ``y_pred``, ``c_pred``)
        - ``extra_modules``              — dict of registered extra modules passed at construction
                                           (e.g. ``{"task_head": task_head}``)

        Returns an empty dict by default (zero overhead).
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


@contextmanager
def intervention(
        original_module: nn.Module,
        intervention_strategy: Union[BaseConceptInterventionStrategy, BaseModuleInterventionStrategy],
        intervention_policy: BaseInterventionPolicy,
        out_concepts_to_intervene_on: Union[List[str], List[int]] = None,
        quantile: float = 1.0,
        eps: float = 1e-12,
        build_context: Optional[Callable] = None,
        extra_modules: Optional[Dict[str, nn.Module]] = None,
        *args,
        **kwargs
):
    """
    Context manager to automatically apply a policy and strategy
    to a concept encoder during execution.
    """
    try:
        yield InterventionModule(
            original_module,
            intervention_strategy,
            intervention_policy,
            out_concepts_to_intervene_on,
            quantile,
            eps,
            build_context=build_context,
            extra_modules=extra_modules,
            *args,
            **kwargs
        )
    finally:
        pass


def intervene(
        original_module: nn.Module,
        intervention_strategy: Union[BaseConceptInterventionStrategy, BaseModuleInterventionStrategy],
        intervention_policy: BaseInterventionPolicy,
        out_concepts_to_intervene_on: Union[List[str], List[int]] = None,
        quantile: float = 1.0,
        eps: float = 1e-12,
        build_context: Optional[Callable] = None,
        extra_modules: Optional[Dict[str, nn.Module]] = None,
        *args,
        **kwargs
) -> nn.Module:
    """
    Wrap a concept encoder module with an intervention strategy and policy.

    Args:
        original_module: The original module to wrap.
        intervention_strategy: The intervention strategy to apply.
        intervention_policy: The intervention policy to determine which concepts to intervene on.
        out_concepts_to_intervene_on: A list of concept names or indices to intervene on.
            If None, all concepts are considered for intervention.
        quantile: Fraction of selected concepts to intervene on (default 1.0 = all selected).
        eps: Small epsilon for numerical stability.
        build_context: Optional callable ``(module, predictions, extra_tensors, *args, **kwargs) -> dict``
            that produces extra kwargs forwarded to the policy and strategy.
            Receives the :class:`InterventionModule` as first argument so it can
            access any registered ``extra_modules``, ``predictions`` as the
            un-intervened encoder output, and ``extra_tensors`` as a dict of
            tensors passed at call time.
        extra_modules: Optional dict of ``{name: nn.Module}`` to register inside
            the :class:`InterventionModule` (e.g. a task head needed by ``build_context``).
            Registered modules are fully tracked by PyTorch (parameters, device, state_dict).

    Returns:
        nn.Module: A new module that applies the specified intervention strategy and policy.
    """
    return InterventionModule(
        original_module,
        intervention_strategy,
        intervention_policy,
        out_concepts_to_intervene_on,
        quantile,
        eps,
        build_context=build_context,
        extra_modules=extra_modules,
        *args,
        **kwargs
    )
