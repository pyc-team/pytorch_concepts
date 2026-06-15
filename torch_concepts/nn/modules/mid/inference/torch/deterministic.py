"""DeterministicInference — forward inference that evaluates MAP estimates."""
from __future__ import annotations

from ...models.bayesian_network import BayesianNetwork
from .forward import ForwardInference


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
    parallelize_levels : bool
        Evaluate conditionally independent variables in the same topological
        level concurrently (see :meth:`ForwardInference.predict_level`).
        Defaults to ``False``.
    """

    name = "DeterministicInference"

    def __init__(
            self,
            pgm: BayesianNetwork,
            p_int: float = 0.,
            parallelize_levels: bool = False,
    ):
        super().__init__(
            pgm,
            mode="deterministic",
            p_int=p_int,
            parallelize_levels=parallelize_levels,
        )
