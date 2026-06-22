"""AncestralSamplingInference — forward inference that samples ancestrally."""
from __future__ import annotations

from typing import Callable, Union

from ...models.bayesian_network import BayesianNetwork
from .forward import ForwardInference


class AncestralSamplingInference(ForwardInference):
    """Forward inference engine that draws samples ancestrally.

    Discrete variables are sampled via the straight-through (ST) estimator so
    gradients can flow.  A temperature schedule controls the sharpness of the
    relaxed distributions over the course of training.

    Parameters
    ----------
    pgm : BayesianNetwork
        The probabilistic graphical model to query.
    p_int : float
        Teacher-forcing probability used when a query variable has a known
        ground-truth value.  Defaults to ``1.0`` (always teacher-force).
    initial_temperature : float
        Starting temperature for relaxed-discrete samplers.  Defaults to
        ``1.0`` (uniform-ish).
    annealing : str or callable
        Temperature schedule.  Built-in options: ``"constant"``,
        ``"exponential"``, ``"linear"``.  A custom callable
        ``f(step) -> float`` is also accepted.
    annealing_rate : float
        Decay rate passed to the built-in annealing schedules.
    parallelize_levels : bool
        Evaluate conditionally independent variables in the same topological
        level concurrently (see :meth:`ForwardInference.predict_level`). Because
        sampling consumes the global RNG, enabling this makes the draw order
        across a level non-deterministic. Defaults to ``False``.
    """

    name = "AncestralSamplingInference"

    def __init__(
        self,
        pgm: BayesianNetwork,
        p_int: float = 1.0,
        initial_temperature: float = 1.0,
        annealing: Union[str, Callable[[int], float]] = "constant",
        annealing_rate: float = 0.0,
        parallelize_levels: bool = False,
    ):
        super().__init__(
            pgm,
            mode="ancestral",
            p_int=p_int,
            initial_temperature=initial_temperature,
            annealing=annealing,
            annealing_rate=annealing_rate,
            parallelize_levels=parallelize_levels,
        )
