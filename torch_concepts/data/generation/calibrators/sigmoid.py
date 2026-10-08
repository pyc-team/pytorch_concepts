from torch_concepts.data.generation.base.calibrator import Calibrator
from torch_concepts.tensor import AnnotatedTensor


class SigmoidCalibrator(Calibrator):
    """Map raw annotation scores through a scaled sigmoid function.

    ``scale`` controls the steepness of the mapping and ``bias`` shifts its
    midpoint. With ``standardize=True``, each concept's scores are first
    rescaled to zero mean and unit variance across the calibrated samples, so
    that concepts whose raw scores sit on different baselines (as CLIP
    similarities do) become comparable. The annotation metadata is preserved
    on the calibrated tensor.

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
    >>> calibrated.annotations.labels
    ['is_red']
    """

    def __init__(self, scale: float = 1.0, bias: float = 0.0, standardize: bool = False):
        self.scale = scale
        self.bias = bias
        self.standardize = standardize

    def calibrate(self, scores: AnnotatedTensor) -> AnnotatedTensor:
        if self.standardize:
            # Statistics per concept: reduce over every dim but the concept axis.
            values = scores.tensor
            dims = [d for d in range(values.dim()) if d != scores.axis]
            std = values.std(dims, keepdim=True, correction=0).clamp_min(1e-12)
            scores = (scores - values.mean(dims, keepdim=True)) / std
        return (scores * self.scale + self.bias).sigmoid()
