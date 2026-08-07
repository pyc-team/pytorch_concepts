"""AncestralSamplingInference — forward inference that samples ancestrally."""
from __future__ import annotations

from typing import Callable, Dict, Union

import torch

from ...graph.bayesian_network import BayesianNetwork
from ...variable import Variable
from .forward import ForwardInference
from .utils import sample_from


class AncestralSamplingInference(ForwardInference):
    """Forward inference engine that draws samples ancestrally.

    By default, discrete variables are drawn from their relaxed (Concrete /
    Gumbel-Softmax) surrogate and the *soft* sample is what propagates to
    descendants — no straight-through estimator is applied unless ``hard=True``
    (see below). A temperature schedule controls the sharpness of the relaxed
    distributions over the course of training.

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
    hard : bool
        If ``True``, a discrete variable's draw is quantized to its exact mode (a
        ``0.``/``1.`` bit, a one-hot row) by a straight-through estimator before
        it propagates: hard forward value, soft gradient. Continuous families are
        unaffected. Set it when a descendant mixes by the sampled score (e.g. a
        concept bottleneck), where a soft draw decodes a blend of every state
        rather than a real assignment. Defaults to ``False``.
    """

    name = "AncestralSamplingInference"
    is_stochastic = True

    def __init__(
        self,
        pgm: BayesianNetwork,
        p_int: float = 1.0,
        parallelize_levels: bool = False,
        hard: bool = False,
        **temperature_kwargs,
    ):
        # The temperature schedule is not re-declared here: it belongs to every
        # engine (see BaseInference), so it passes straight through.
        super().__init__(
            pgm,
            p_int=p_int,
            parallelize_levels=parallelize_levels,
            **temperature_kwargs,
        )
        self.hard = bool(hard)

    def _resolve(
        self,
        variable: Variable,
        params: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """A reparameterised draw from the variable's relaxed distribution."""
        return sample_from(variable, params, temperature, hard=self.hard)
