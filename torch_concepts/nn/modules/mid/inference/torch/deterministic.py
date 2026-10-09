"""DeterministicInference — forward inference that evaluates MAP estimates."""
from __future__ import annotations

from typing import Dict

import torch

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

    def _resolve(
        self,
        variable: Variable,
        params: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """The family's canonical parameter — no sampling (``temperature`` unused)."""
        return propagated_value(
            variable,
            params,
            activate=self.activate_before_propagation,
        )
