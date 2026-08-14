from abc import ABC, abstractmethod

from torch_concepts.tensor import AnnotatedTensor


class Calibrator(ABC):
    """Change the semantics of raw annotation scores."""

    @abstractmethod
    def calibrate(self, scores: AnnotatedTensor) -> AnnotatedTensor:
        """Return calibrated scores with the same shape and annotation."""
