"""Independent training inference."""

import logging

from ...models.bayesian_network import BayesianNetwork
from .deterministic import DeterministicInference

logger = logging.getLogger(__name__)


class IndependentInference(DeterministicInference):
    """Independent (sequential) training inference.

    A convenience subclass of :class:`DeterministicInference` that pins
    ``p_int=1.0``, so ground-truth concepts are always propagated to downstream
    predictors instead of the model's own predictions. Equivalent to
    ``DeterministicInference(..., p_int=1.0)``.

    Parameters
    ----------
    pgm : BayesianNetwork
        The model to query.
    activate_before_propagation : bool, optional
        Forwarded to :class:`DeterministicInference`; see there.
    """
    def __init__(self, pgm: BayesianNetwork, activate_before_propagation: bool = True):
        super().__init__(
            pgm,
            activate_before_propagation=activate_before_propagation,
            p_int=1.0,
        )
