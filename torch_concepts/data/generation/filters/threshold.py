from torch_concepts.data.generation.base.filter_annotator import FilterAnnotator
from torch_concepts.tensor import AnnotatedTensor


class ThresholdAnnotationFilter(FilterAnnotator):
    """Filter sample-concept scores below a fixed threshold.

    Examples
    --------
    >>> scores = AnnotatedTensor(
    ...     torch.tensor([[0.2, 0.8], [0.6, 0.4]]),
    ...     Annotations(labels=["color", "shape"]),
    ...     axis=1,
    ... )
    >>> filtered = ThresholdAnnotationFilter(threshold=0.5).filter(scores)
    >>> torch.testing.assert_close(
    ...     filtered.tensor,
    ...     torch.tensor([[0.0, 0.8], [0.6, 0.0]]),
    ... )
    >>> filtered.annotation.labels
    ['color', 'shape']
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def filter(self, scores: AnnotatedTensor) -> AnnotatedTensor:
        if not scores.is_floating_point():
            raise TypeError(
                "ThresholdAnnotationFilter requires floating-point scores."
            )
        return scores.masked_fill(scores < self.threshold, 0.0)
