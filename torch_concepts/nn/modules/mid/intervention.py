from operator import itemgetter
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Union

from ..low.base.intervention import (
    BaseConceptInterventionStrategy,
    BaseModuleInterventionStrategy,
    BaseInterventionPolicy,
)
from ..low.intervention.intervention import InterventionModule
from .models.probabilistic_model import ProbabilisticModel


@contextmanager
def intervention(
        probabilistic_model: ProbabilisticModel,
        intervention_strategy: Union[BaseConceptInterventionStrategy, BaseModuleInterventionStrategy],
        intervention_policy: BaseInterventionPolicy,
        variable_to_intervene_on: str,
        parameter_to_intervene_on: str,
        members_to_intervene_on: Union[List[int], List[str]] = None,
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
        original_module = probabilistic_model.factors[variable_to_intervene_on].parametrization[parameter_to_intervene_on]
        if members_to_intervene_on is not None:
            if isinstance(members_to_intervene_on[0], str):
                members_to_intervene_on = itemgetter(*members_to_intervene_on)(probabilistic_model.members_to_idx[variable_to_intervene_on])
                if isinstance(members_to_intervene_on, int):
                    members_to_intervene_on = [members_to_intervene_on]
                else:
                    members_to_intervene_on = list(members_to_intervene_on)

        intervened_module = InterventionModule(
            original_module,
            intervention_strategy,
            intervention_policy,
            members_to_intervene_on,
            quantile,
            eps,
            build_context=build_context,
            extra_modules=extra_modules,
            *args,
            **kwargs
        )
        probabilistic_model.factors[variable_to_intervene_on].parametrization[parameter_to_intervene_on] = intervened_module
        yield

    finally:
        probabilistic_model.factors[variable_to_intervene_on].parametrization[parameter_to_intervene_on] = original_module
        pass
