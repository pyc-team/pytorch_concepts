"""DeterministicInference — forward inference that evaluates MAP estimates."""
from __future__ import annotations

from typing import Dict

import torch

from ...models.bayesian_network import BayesianNetwork
from ...models.variable import Variable
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
        ground-truth value.  Defaults to ``1.0`` (always teacher-force).
    activate_before_propagation : bool
        When ``True``, each variable's propagated parameter is passed
        through its default activation (see
        :attr:`~torch_concepts.nn.modules.mid.models.distributions.DistributionSpec.activations`)
        before being fed to child CPDs — e.g. a CPD producing ``logits``
        propagates probabilities downstream. The parameters returned in the
        inference output remain the raw (non-activated) values. When ``False``,
        the raw parameter is propagated unchanged.
    parallelize_levels : bool
        Evaluate conditionally independent variables in the same topological
        level concurrently (see :meth:`ForwardInference.predict_level`).
        Defaults to ``False``.
    """

    name = "DeterministicInference"
    is_stochastic = False

    def __init__(
            self,
            pgm: BayesianNetwork,
            activate_before_propagation: bool = True,
            p_int: float = 0.,
            parallelize_levels: bool = False,
    ):
        super().__init__(
            pgm,
            p_int=p_int,
            parallelize_levels=parallelize_levels,
            activate_before_propagation=activate_before_propagation,
        )

    def _resolve(
        self,
        variable: Variable,
        params: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """The family's canonical parameter — no sampling (``temperature`` unused)."""
        return propagated_value(
            variable.distribution,
            params,
            activate=self.activate_before_propagation,
        )
