from typing import Any

import torch
from torch.utils.data import Dataset

from torch_concepts import Annotations
from torch_concepts.data.generation.base.annotator import Annotator
from torch_concepts.tensor import AnnotatedTensor


class RandomAnnotator(Annotator):
    """Baseline annotator that scores every sample uniformly at random in [0, 1).

    Args:
        seed: Seed for the scores. If ``None``, the global torch RNG is used.
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def annotate(
        self,
        dataset: Dataset,
        concepts: Annotations,
        **kwargs: Any,
    ) -> AnnotatedTensor:
        del kwargs
        generator = None if self.seed is None else torch.Generator().manual_seed(self.seed)
        scores = torch.rand(len(dataset), concepts.size, generator=generator)
        return AnnotatedTensor(scores, concepts, axis=-1)
