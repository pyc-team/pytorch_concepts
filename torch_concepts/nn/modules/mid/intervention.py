import torch.nn as nn
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Union

from ..low.base.intervention import (
    BaseConceptInterventionStrategy,
    BaseModuleInterventionStrategy,
    BaseInterventionPolicy,
)
from ..low.intervention.intervention import InterventionModule
from .graph.probabilistic_model import ProbabilisticModel


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
    """Temporarily wrap one factor's parametrization module with an intervention.

    Inside the ``with`` block, ``factors[variable_to_intervene_on]
    .parametrization[parameter_to_intervene_on]`` is replaced by an
    :class:`~torch_concepts.nn.modules.low.intervention.intervention.InterventionModule`
    applying ``intervention_strategy`` under ``intervention_policy``. The
    original module is always restored on exit, including when the body raises.

    ``members_to_intervene_on`` restricts the intervention to a subset of the
    variable's event columns; member *names* are resolved to column indices via
    :meth:`~torch_concepts.nn.modules.mid.variable.Variable.get_slice`.
    ``None`` intervenes on every column.
    """
    # Resolve the target before entering the try block: a bad variable or
    # parameter name must surface as its own KeyError, not as a NameError from
    # a finally clause restoring a module that was never captured.
    try:
        factor = probabilistic_model.factors[variable_to_intervene_on]
    except KeyError:
        raise KeyError(
            f"intervention: no factor named {variable_to_intervene_on!r}; "
            f"available factors are {sorted(probabilistic_model.factors.keys())}."
        ) from None
    if parameter_to_intervene_on not in factor.parametrization:
        raise KeyError(
            f"intervention: factor {variable_to_intervene_on!r} has no "
            f"parametrization entry {parameter_to_intervene_on!r}; available "
            f"parameters are {sorted(factor.parametrization.keys())}."
        )
    original_module = factor.parametrization[parameter_to_intervene_on]

    if members_to_intervene_on is not None and members_to_intervene_on:
        if isinstance(members_to_intervene_on[0], str):
            members_to_intervene_on = factor.variable.get_slice(members_to_intervene_on)

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

    factor.parametrization[parameter_to_intervene_on] = intervened_module
    try:
        yield
    finally:
        factor.parametrization[parameter_to_intervene_on] = original_module
