"""Independent training inference."""

import logging

from ...models.bayesian_network import BayesianNetwork
from .deterministic import TorchDeterministicInference

logger = logging.getLogger(__name__)


class IndependentInference(TorchDeterministicInference):
    """
    Independent training inference.

    This is a convenience subclass of :class:`TorchDeterministicInference` that
    forces ``p_int=1``, so ground truth concepts are always propagated to
    downstream predictors during training.

    Equivalent to ``TorchDeterministicInference(..., p_int=1.0)``.
    """
    def __init__(self, pgm: BayesianNetwork):
        super().__init__(pgm, p_int=1.0)
