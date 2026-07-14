"""TorchBaseInference — base class for pure-PyTorch inference engines.

PyTorch-backed engines hold a reference to the user's
:class:`ProbabilisticModel` and run inference using only ``torch`` and
``torch.distributions`` (no Pyro). Parameter sharing with the wrapped PGM is
inherited from :class:`BaseInference`.
"""
from __future__ import annotations

from ..base import BaseInference


class TorchBaseInference(BaseInference):
    """Marker base for pure-PyTorch inference engines.

    Inherits all engine scaffolding from :class:`BaseInference` (including
    reference-based parameter sharing with the wrapped PGM).
    """

    name = "TorchBaseInference"