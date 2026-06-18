"""Independent training inference."""

import logging

from ...models.bayesian_network import BayesianNetwork
from .deterministic import DeterministicInference

logger = logging.getLogger(__name__)


class IndependentInference(DeterministicInference):
    """
    Independent training inference.

    This is a convenience subclass of :class:`DeterministicInference` that
    forces ``p_int=1``, so ground truth concepts are always propagated to
    downstream predictors during training.

    Equivalent to ``DeterministicInference(..., p_int=1.0)``.

    ``activate_before_propagation`` is mandatory and forwarded to
    :class:`DeterministicInference`.
    """
    def __init__(self, pgm: BayesianNetwork, activate_before_propagation: True):
        super().__init__(
            pgm,
            activate_before_propagation=activate_before_propagation,
            p_int=1.0,
        )
