import torch
from torch import Tensor

from torch_concepts import Annotations
from torch_concepts.data.base.calibrator import Calibrator


class SigmoidCalibrator(Calibrator):
    """Map raw annotation scores through a scaled sigmoid function."""

    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        self.scale = scale
        self.bias = bias

    def calibrate(self, scores: Tensor, concepts: Annotations) -> Tensor:
        del concepts
        return torch.sigmoid(scores * self.scale + self.bias)

