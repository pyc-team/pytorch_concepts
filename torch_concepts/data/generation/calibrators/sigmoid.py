from torch_concepts.data.generation.base.calibrator import Calibrator
from torch_concepts.tensor import AnnotatedTensor


class SigmoidCalibrator(Calibrator):
    """Map raw annotation scores through a scaled sigmoid function."""

    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        self.scale = scale
        self.bias = bias

    def calibrate(self, scores: AnnotatedTensor) -> AnnotatedTensor:
        return (scores * self.scale + self.bias).sigmoid()
