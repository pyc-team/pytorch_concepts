import torch
from typing import Optional, Union
import torch.nn.functional as F

from .....annotations import Annotations
from ..base .layer import BaseConceptLayer
from ..ops import StraightThroughSoftmax


class PrototypeConceptEmbeddingToConcept(BaseConceptLayer):
    """
    Aggregates prototype-based concept weights with input embeddings.

    This predictor computes similarity between input embeddings and stored prototypes,
    then uses concept weights from the encoder to produce final concept predictions.

    Args:
        out_concepts: Number of output concepts (num_concepts).
        proto_samples: Tensor of shape [max_prototypes, num_concepts, n_features] - prototype feature vectors.
        proto_scores: Tensor of shape [max_prototypes, num_concepts] - scores for sorting prototypes.
        learnable_prototypes: Whether prototypes should be learnable parameters.
        temperature: Temperature for backward pass (default: 1.0).
        temp_forward: Temperature for forward pass (default: 0.01).
        use_straight_through: Use straight-through estimator for peaked forward, soft backward.

    Example:
        >>> proto_samples = torch.randn(10, 100, 50)  # 10 prototypes, 100 concepts, 50 features
        >>> proto_scores = torch.randn(10, 100)
        >>> predictor = PrototypeConceptEmbeddingToConcept(
        ...     out_concepts=100,
        ...     proto_samples=proto_samples,
        ...     proto_scores=proto_scores,
        ... )
        >>> concepts = torch.randn(10, 100)  # From encoder
        >>> embeddings = torch.randn(32, 50)  # Batch of 32
        >>> output = predictor(concepts, embeddings)  # [32, 100]
    """
    def __init__(
        self,
        out_concepts: Union[int, Annotations],
        proto_samples: torch.Tensor,
        proto_scores: torch.Tensor,
        learnable_prototypes: bool = False,
        temperature: float = 1.0,
        temp_forward: Optional[float] = None,
        use_straight_through: bool = True,
        *args,
        **kwargs
    ):
        max_prototypes, num_concepts, n_features = proto_samples.shape
        # Infer in_embeddings
        in_embeddings = n_features

        super().__init__(
            out_concepts=out_concepts,
            in_concepts=None,  # Concepts come from encoder, not traditional input
            in_embeddings=in_embeddings
        )

        assert self.out_concepts_shape == num_concepts, \
            f"out_concepts ({self.out_concepts_shape}) must match num_concepts from proto_samples ({num_concepts})"

        self.num_concepts = num_concepts
        self.max_prototypes = max_prototypes
        self.n_features = n_features

        # Temperature settings
        self.use_straight_through = use_straight_through
        if use_straight_through:
            self.temp_forward = temp_forward if temp_forward is not None else 0.01
            self.temp_backward = temperature
        else:
            self.temp_forward = temperature
            self.temp_backward = temperature

        self.register_buffer('temperature_forward', torch.tensor(self.temp_forward))
        self.register_buffer('temperature_backward', torch.tensor(self.temp_backward))

        # Sort prototypes by scores (ascending order for monotonicity)
        sorted_indices = torch.stack([
            torch.argsort(proto_scores[:, i], descending=False)
            for i in range(num_concepts)
        ])

        # Reorder prototypes: [num_concepts, max_prototypes, n_features]
        prototypes = torch.stack([
            proto_samples[sorted_indices[i], i, :]
            for i in range(num_concepts)
        ])

        # Store or register as parameter
        if learnable_prototypes:
            self.prototypes = torch.nn.Parameter(prototypes)
        else:
            self.register_buffer('prototypes', prototypes)

    def forward(
        self,
        concepts: torch.Tensor,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate concept weights with embeddings via prototype similarity.

        Args:
            concepts: Tensor of shape [max_prototypes, num_concepts].
            embeddings: Tensor of shape [batch, n_features].

        Returns:
            torch.Tensor: Tensor of shape [batch, num_concepts] - final concept predictions.
        """
        similarity = self.similarity_scores(concepts, embeddings)

        # Weighted sum over prototypes
        # similarity: [batch, num_concepts, max_prototypes]
        # cumulative_weights: [num_concepts, max_prototypes]
        output = (similarity * concepts.T.unsqueeze(0)).sum(dim=2)  # [batch, num_concepts]

        return output

    def similarity_scores(
            self,
            concepts: torch.Tensor,
            embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity scores between embeddings and prototypes.
        """

        assert self.max_prototypes == concepts.shape[0], \
            f"Expected concepts to have shape [{self.max_prototypes}, {self.num_concepts}], got {concepts.shape}"

        batch_size = embeddings.shape[0]

        # Compute similarities between embeddings and prototypes
        # embeddings: [batch, n_features] -> [batch, 1, 1, n_features]
        # prototypes: [num_concepts, max_prototypes, n_features] -> [1, num_concepts, max_prototypes, n_features]
        x_expanded = embeddings.view(batch_size, 1, 1, self.n_features)
        proto_expanded = self.prototypes.unsqueeze(0)

        # Compute squared distances: [batch, num_concepts, max_prototypes]
        squared_distances = torch.sum((x_expanded - proto_expanded) ** 2, dim=-1)

        # Use negative squared distances as logits
        logits = -squared_distances

        # Compute similarities with appropriate method
        if self.use_straight_through:
            # Straight-through: peaked forward, smooth backward
            similarity = StraightThroughSoftmax.apply(
                logits,
                self.temperature_forward,
                self.temperature_backward,
                2  # dim for softmax
            )
        else:
            # Standard softmax with single temperature
            temp = torch.clamp(self.temperature_forward, min=0.1)
            similarity = F.softmax(logits / temp, dim=2)

        return similarity
