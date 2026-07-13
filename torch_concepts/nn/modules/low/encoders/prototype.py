import torch
import torch.nn.functional as F
from typing import Union

from .....annotations import Annotations
from ..base.layer import BaseConceptLayer


class MonotonicScoresEmbeddingToConcept(BaseConceptLayer):
    """
    Generates monotonic concept scores.

    This encoder produces concept scores where each sample is represented by a cumulative sum of positive scores,
    ensuring monotonicity across samples.

    Args:
        in_embeddings: Dimension of the input embeddings (in_embeddings).
        out_concepts: Number of concepts to generate (num_concepts).

    Example:
        >>> embeddings = torch.randn(32, 10)  # Batch of 32 embeddings with 10 features
        >>> encoder = MonotonicScoresEmbeddingToConcept(out_concepts=100, in_embeddings=10)
        >>> concepts = encoder(embeddings)  # [32, 100]
    """
    def __init__(
        self,
        in_embeddings: Union[int, Annotations],
        out_concepts: Union[int, Annotations],
        *args,
        **kwargs
    ):
        super().__init__(
            out_concepts=out_concepts,
            in_concepts=None,
            in_embeddings=in_embeddings
        )
        self.projection = torch.nn.Linear(
            self.in_embeddings_shape,
            self.out_concepts_shape
        )

    def forward(
            self,
            embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate cumulative weights for all concepts.

        Returns:
            torch.Tensor: Tensor of shape [max_prototypes, num_concepts]
        """
        scores = self.projection(embeddings)  # [max_prototypes, num_concepts]

        # Ensure positivity with softplus (numerically stable)
        positive_scores = F.softplus(scores)
        # Cumulative sum ensures monotonicity
        cumulative_output = torch.cumsum(positive_scores, dim=0)  # [max_prototypes, num_concepts]

        return cumulative_output
