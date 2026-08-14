from torch_concepts.data.generation.base.filter_annotator import FilterAnnotator
from torch_concepts.tensor import AnnotatedTensor


class ThresholdAnnotationFilter(FilterAnnotator):
    """Filter sample-concept scores below a fixed threshold."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def filter(self, scores: AnnotatedTensor) -> AnnotatedTensor:
        if not scores.is_floating_point():
            raise TypeError(
                "ThresholdAnnotationFilter requires floating-point scores."
            )
        return scores.masked_fill(scores < self.threshold, float("nan"))
