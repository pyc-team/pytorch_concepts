"""DeterministicInference — forward inference that evaluates MAP estimates."""
from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional

import torch

from ...distributions import spec_for
from ...graph.bayesian_network import BayesianNetwork
from ...variable import Variable
from .forward import ForwardInference
from .utils import propagated_value


class DeterministicInference(ForwardInference):
    """Forward inference engine that returns MAP (deterministic) estimates.

    All continuous variables are evaluated at their distribution mean; discrete
    variables use the mode.  No sampling is performed.

    Parameters
    ----------
    pgm : BayesianNetwork
        The probabilistic graphical model to query.
    p_int : float
        Teacher-forcing probability used when a query variable has a known
        ground-truth value.  Defaults to ``0.0`` (never teacher-force).
    parallelize_levels : bool
        Evaluate conditionally independent variables in the same topological
        level concurrently (see :meth:`ForwardInference.predict_level`).
        Defaults to ``False``.
    activations : dict, optional
        ``{variable name: callable}`` overriding the default activation that
        maps a ``logits``-parametrized variable to its primary parameter before
        propagation. Logits variables left out get the default, with a warning.
    """

    name = "DeterministicInference"
    is_stochastic = False

    #: Map a variable parametrized by ``logits`` through its default activation
    #: (see :attr:`~torch_concepts.nn.modules.mid.distributions.DistributionSpec.param_activations`)
    #: before feeding it to child CPDs, so it propagates *probabilities*
    #: downstream; ``out.params`` still reports the raw values.
    activate_before_propagation = True

    def __init__(
            self,
            pgm: BayesianNetwork,
            p_int: float = 0.,
            parallelize_levels: bool = False,
            activations: Optional[Dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
            **temperature_kwargs,
    ):
        # Accepted and ignored: this engine never samples, but a config that
        # sets a schedule must not break when it swaps the engine out.
        super().__init__(
            pgm,
            p_int=p_int,
            parallelize_levels=parallelize_levels,
            **temperature_kwargs,
        )
        self.activations = dict(activations or {})
        if self.activate_before_propagation:
            defaults = {}
            for name, cpd in pgm.factors.items():
                if "logits" not in cpd.parametrization or name in self.activations:
                    continue
                v, spec = cpd.variable, spec_for(cpd.variable.distribution)
                factory = spec.param_activations.get(spec.primary_param)
                if factory is not None:
                    defaults[v.distribution.__name__] = factory(v.member_size, v.member_size)
            if defaults:
                warnings.warn(
                    f"{self.name}: logits are activated internally before propagation "
                    f"({', '.join(f'{d} -> {a}' for d, a in defaults.items())}). "
                    "Pass `activations={variable_name: fn}` to override.",
                    stacklevel=2,
                )

    def _resolve(
        self,
        variable: Variable,
        params: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """The family's canonical parameter — no sampling (``temperature`` unused)."""
        fn = self.activations.get(variable.name)
        if fn is not None and self.activate_before_propagation and "logits" in params:
            variable._require_member_layout(params["logits"], "logits")
            return fn(params["logits"])
        return propagated_value(
            variable,
            params,
            activate=self.activate_before_propagation,
        )
