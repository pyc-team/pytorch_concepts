"""TorchBaseInference — base class for pure-PyTorch inference engines."""

from __future__ import annotations

from ..base import BaseInference


class TorchBaseInference(BaseInference):
    """Marker base for pure-PyTorch inference engines.

    PyTorch-backed engines hold a reference to the user's
    :class:`ProbabilisticModel` and run inference using only ``torch`` and
    ``torch.distributions``. Parameter sharing with the wrapped PGM is
    inherited from :class:`BaseInference`.
    """

    name = "TorchBaseInference"