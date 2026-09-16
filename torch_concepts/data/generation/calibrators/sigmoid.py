from torch_concepts.data.generation.base.calibrator import Calibrator
from torch_concepts.tensor import AnnotatedTensor


class SigmoidCalibrator(Calibrator):
    """Map raw annotation scores through a scaled sigmoid function.

    ``scale`` controls the steepness of the mapping and ``bias`` shifts its
    midpoint. The annotation metadata is preserved on the calibrated tensor.

    Examples
    --------
    >>> import torch
    >>> from torch_concepts import Annotations
    >>> from torch_concepts.tensor import AnnotatedTensor
    >>> scores = AnnotatedTensor(
    ...     torch.tensor([[-1.0], [0.0], [1.0]]),
    ...     Annotations(labels=["is_red"]),
    ...     axis=1,
    ... )
    >>> calibrated = SigmoidCalibrator(scale=2.0).calibrate(scores)
    >>> torch.testing.assert_close(
    ...     calibrated.tensor,
    ...     torch.sigmoid(torch.tensor([[-2.0], [0.0], [2.0]])),
    ... )
    >>> calibrated.annotation.labels
    ['is_red']
    """

    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        self.scale = scale
        self.bias = bias

    def calibrate(self, scores: AnnotatedTensor) -> AnnotatedTensor:
        return (scores * self.scale + self.bias).sigmoid()
