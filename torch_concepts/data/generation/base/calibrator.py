from abc import ABC, abstractmethod

from torch import Tensor

from torch_concepts import Annotations


class Calibrator(ABC):
    """Change the semantics of raw annotation scores."""

    @abstractmethod
    def calibrate(self, scores: Tensor, concepts: Annotations) -> Tensor:
        """Return calibrated scores with the same shape as the input."""
