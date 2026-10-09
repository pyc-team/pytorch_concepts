"""AncestralSamplingInference — forward inference that samples ancestrally."""
from __future__ import annotations

from typing import Dict

import torch
import torch.distributions as td

from ...graph.bayesian_network import BayesianNetwork
from ...variable import Variable
from .forward import ForwardInference
from ..utils import EXACT_FAMILY, build_distribution
from .utils import sample_from


class AncestralSamplingInference(ForwardInference):
    """Forward inference engine that draws samples ancestrally.

    Discrete variables are drawn from their relaxed (Concrete / Gumbel-Softmax)
    surrogate, and a temperature schedule controls its sharpness over the course
    of training. Whether the propagated draw is **soft or hard** is a property of
    the variable's declared family, not of this engine: declare it
    ``Bernoulli`` / ``RelaxedBernoulli`` for a soft Concrete sample, or
    ``RelaxedBernoulliStraightThrough`` for an exact bit with a soft gradient
    (likewise ``OneHotCategorical`` / ``RelaxedOneHotCategoricalStraightThrough``).

    Parameters
    ----------
    pgm : BayesianNetwork
        The probabilistic graphical model to query.
    p_int : float
        Teacher-forcing probability used when a query variable has a known
        ground-truth value.  Defaults to ``0.0`` (never teacher-force).
    initial_temperature : float
        Starting temperature for relaxed-discrete samplers.  Defaults to
        ``1.0`` (uniform-ish).
    annealing : str or callable
        Temperature schedule.  Built-in options: ``"constant"``,
        ``"exponential"``, ``"linear"``.  A custom callable
        ``f(step) -> float`` is also accepted.
    annealing_rate : float
        Decay rate passed to the built-in annealing schedules.
    exact : bool
        Draw from each variable's **exact** family with a hard, non-reparameterised
        sample instead of from the relaxed surrogate. Ignores a
        relaxed declaration (``EXACT_FAMILY`` maps it to its hard counterpart) so
        the draw does not depend on how the model was written. Not differentiable;
        defaults to ``False``.
    parallelize_levels : bool
        Evaluate conditionally independent variables in the same topological
        level concurrently (see :meth:`ForwardInference.predict_level`). Because
        sampling consumes the global RNG, enabling this makes the draw order
        across a level non-deterministic. Defaults to ``False``.
    """

    name = "AncestralSamplingInference"
    is_stochastic = True

    def __init__(
        self,
        pgm: BayesianNetwork,
        p_int: float = 0.0,
        exact: bool = False,
        parallelize_levels: bool = False,
        **temperature_kwargs,
    ):
        self.exact = bool(exact)
        # The temperature schedule is not re-declared here: it belongs to every
        # engine (see BaseInference), so it passes straight through.
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
        """A draw from the variable: relaxed and reparameterised, or exact and hard."""
        if not self.exact:
            return sample_from(variable, params, temperature)
        D = variable.distribution
        if issubclass(D, td.Categorical):
            # Plain Categorical samples *indices*, not a one-hot, so it is built
            # on the flat parameters and read back into the member layout.
            flat = {k: variable.to_event(v, k) for k, v in params.items()}
            return variable.to_member(td.Categorical(**flat).sample())
        return build_distribution(
            variable, params, family=EXACT_FAMILY.get(D)
        ).sample()
