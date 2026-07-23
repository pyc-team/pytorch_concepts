from torch import Tensor

from torch_concepts import Annotations
from torch_concepts.data.base.annotation_filter import AnnotationFilter


class ThresholdAnnotationFilter(AnnotationFilter):
    """Filter sample-concept scores below a fixed threshold."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def filter(self, scores: Tensor, concepts: Annotations) -> Tensor:
        del concepts
        if not scores.is_floating_point():
            raise TypeError(
                "ThresholdAnnotationFilter requires floating-point scores."
            )
        return scores.masked_fill(scores < self.threshold, float("nan"))

