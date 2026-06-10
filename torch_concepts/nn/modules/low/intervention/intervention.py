import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import List, Union

from torch_concepts import AxisAnnotation
from ..base.intervention import (
    BaseConceptInterventionStrategy,
    BaseModuleInterventionStrategy,
    BaseInterventionPolicy
)


class InterventionModule(nn.Module):
    def __init__(
            self,
            original_module: nn.Module,
            intervention_strategy: Union[BaseConceptInterventionStrategy, BaseModuleInterventionStrategy],
            intervention_policy: BaseInterventionPolicy,
            out_concepts_to_intervene_on: Union[List[str], List[int]] = None,
            quantile: float = 1.0,
            eps: float = 1e-12,
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

    @property
    def sel_idx(self):
        if self.out_concepts_to_intervene_on is not None:
            if isinstance(self.out_concepts_to_intervene_on[0], int):
                return torch.tensor(self.out_concepts_to_intervene_on, dtype=torch.long)
            elif isinstance(self.out_concepts_to_intervene_on[0], str):
                original_annotations = getattr(self.original_module, "out_concepts", None)
                if original_annotations is None and not isinstance(original_annotations, AxisAnnotation):
                    raise ValueError("To use string-based concept selection, the original module must have an "
                                     "'out_concepts' attribute of type AxisAnnotation.")
                indices = [original_annotations.label_to_index[item] for item in self.out_concepts_to_intervene_on]
                return torch.tensor(indices, dtype=torch.long)
            else:
                raise ValueError("out_concepts_to_intervene_on must be a list of integers (indices) or strings (names)")

        return None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        original_predictions = self.original_module(*args, **kwargs)  # [B, F]
        assert original_predictions.dim() == 2, (f"BaseConceptInterventionStrategy expects 2-D "
                                                 f"tensors [Batch, N_concepts]. "
                                                 f"Got shape: {original_predictions.shape}")

        policy_scores = self.intervention_policy(original_predictions, *args, **kwargs)
        intervention_mask = self.intervention_policy.build_mask(
            policy_scores,
            sel_idx=self.sel_idx,
            quantile=self.quantile,
            eps=self.eps
        ).to(dtype=original_predictions.dtype)

        if isinstance(self.intervention_strategy, BaseConceptInterventionStrategy):
            intervened_predictions = self.intervention_strategy(original_predictions, *args, **kwargs)

        elif isinstance(self.intervention_strategy, BaseModuleInterventionStrategy):
            intervened_module = self.intervention_strategy.transform(self.original_module, *args, **kwargs)
            intervened_predictions = intervened_module(*args, **kwargs)

        else:
            raise ValueError("Intervention strategy must be an instance of "
                             "BaseConceptInterventionStrategy or BaseModuleInterventionStrategy.")

        return (original_predictions * intervention_mask +
                intervened_predictions * (1.0 - intervention_mask))


def intervene(
        original_module: nn.Module,
        intervention_strategy: Union[BaseConceptInterventionStrategy, BaseModuleInterventionStrategy],
        intervention_policy: BaseInterventionPolicy,
        out_concepts_to_intervene_on: Union[List[str], List[int]] = None,
        quantile: float = 1.0,
        eps: float = 1e-12,
        *args,
        **kwargs
) -> nn.Module:
    """
    This method must be implemented by subclasses to define the
    specific intervention strategy.

    Args:
        original_module: The original module to wrap.
        intervention_strategy: The intervention strategy to apply.
        intervention_policy: The intervention policy to determine which concepts to intervene on.
        out_concepts: The number of output concepts or an AxisAnnotation describing them.
        out_concepts_to_intervene_on: A list of concept names or indices to intervene on. If None, all concepts are considered for intervention.

    Returns:
        torch.nn.Module: A new module that applies the specified intervention strategy and policy.
    """
    return InterventionModule(
        original_module,
        intervention_strategy,
        intervention_policy,
        out_concepts_to_intervene_on,
        quantile,
        eps,
        *args,
        **kwargs
    )

@contextmanager
def intervention(
        original_module: nn.Module,
        intervention_strategy: Union[BaseConceptInterventionStrategy, BaseModuleInterventionStrategy],
        intervention_policy: BaseInterventionPolicy,
        out_concepts_to_intervene_on: Union[List[str], List[int]] = None,
        quantile: float = 1.0,
        eps: float = 1e-12,
        *args,
        **kwargs
):
    """
    Context manager to automatically apply a policy and strategy
    to a concept encoder during execution.
    """
    try:
        # 2. Hand control back to the 'with' block
        yield intervene(
            original_module,
            intervention_strategy,
            intervention_policy,
            out_concepts_to_intervene_on,
            quantile,
            eps,
            *args,
            **kwargs
        )
    finally:
        # 3. Teardown phase (Runs after the 'with' block finishes or errors out)
        # If you need to clean up or reset the model, do it here.
        pass
