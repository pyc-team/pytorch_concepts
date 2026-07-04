import torch
import torch.nn.functional as F
from typing import Union

from .....annotations import Annotations
from ..base.layer import BaseConceptLayer


class CumulativeWeightsToConcept(BaseConceptLayer):
    """
    Generates cumulative prototype weights as concept representations.

    This encoder produces static concept features (prototype weights) that are
    independent of input data. Each concept gets a set of cumulative weights
    across prototypes, ensuring monotonicity.

    Args:
        out_concepts: Number of concepts to generate (num_concepts).
        max_prototypes: Number of prototypes per concept (feature dimension).
        rank_dim: Dimension of the low-rank embedding for concepts.

    Example:
        >>> encoder = CumulativeWeightsToConcept(out_concepts=100, max_prototypes=10, rank_dim=32)
        >>> concepts = encoder()  # [1, 100, 10]

    Forward:
        Returns: Tensor of shape [1, num_concepts, max_prototypes] - static concept weights.
    """
    def __init__(
        self,
        out_concepts: Union[int, Annotations],
        max_prototypes: int,
        rank_dim: int = 32,
        *args,
        **kwargs
    ):
        super().__init__(
            out_concepts=out_concepts,
            in_concepts=None,
            in_embeddings=None
        )

        self.max_prototypes = max_prototypes
        self.rank_dim = rank_dim

        # Low-rank embedding for concepts
        self.embedding = torch.nn.Embedding(self.out_concepts_shape, rank_dim)
        # Linear projection to prototypes
        self.projection = torch.nn.Linear(rank_dim, max_prototypes)

    def forward(self) -> torch.Tensor:
        """
        Generate cumulative prototype weights for all concepts.

        Returns:
            torch.Tensor: Tensor of shape [1, num_concepts, max_prototypes]
                - dim 0: batch (size 1, static weights)
                - dim 1: num_concepts (concept dimension)
                - dim 2: max_prototypes (features per concept)
        """
        # Get concept embeddings and project to prototype scores
        concept_emb = self.embedding.weight  # [num_concepts, rank_dim]
        scores = self.projection(concept_emb)  # [num_concepts, max_prototypes]

        # Ensure positivity with softplus (numerically stable)
        positive_scores = F.softplus(scores)
        # Cumulative sum ensures monotonicity
        cumulative_output = torch.cumsum(positive_scores, dim=1)  # [num_concepts, max_prototypes]

        # Add batch dimension: [num_concepts, max_prototypes] -> [1, num_concepts, max_prototypes]
        return cumulative_output.unsqueeze(0)
