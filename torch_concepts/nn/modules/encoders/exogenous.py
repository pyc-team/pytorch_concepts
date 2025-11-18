"""
Exogenous encoder module for concept embeddings.

This module provides encoders that transform embeddings into exogenous variables
for concept-based models, supporting the Concept Embedding Models architecture.
"""
import numpy as np
import torch

from ... import BaseEncoder
from typing import Tuple


class ExogEncoder(BaseEncoder):
    """
    Exogenous encoder that creates supervised concept embeddings.

    Transforms input embeddings into exogenous variables (external features) for
    each concept, producing a 2D output of shape (out_features, embedding_size).
    Implements the 'embedding generators' from Concept Embedding Models (Zarlenga et al., 2022).

    Attributes:
        embedding_size (int): Dimension of each concept's embedding.
        out_logits_dim (int): Number of output concepts.
        encoder (nn.Sequential): The encoding network.

    Args:
        in_features_embedding: Number of input embedding features.
        out_features: Number of output concepts.
        embedding_size: Dimension of each concept's embedding.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import ExogEncoder
        >>>
        >>> # Create exogenous encoder
        >>> encoder = ExogEncoder(
        ...     in_features_embedding=128,
        ...     out_features=5,
        ...     embedding_size=16
        ... )
        >>>
        >>> # Forward pass
        >>> embeddings = torch.randn(4, 128)  # batch_size=4
        >>> exog = encoder(embeddings)
        >>> print(exog.shape)
        torch.Size([4, 5, 16])
        >>>
        >>> # Each concept has its own 16-dimensional embedding
        >>> print(f"Concept 0 embedding shape: {exog[:, 0, :].shape}")
        Concept 0 embedding shape: torch.Size([4, 16])

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
        Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056
    """

    def __init__(
        self,
        in_features_embedding: int,
        out_features: int,
        embedding_size: int
    ):
        """
        Initialize the exogenous encoder.

        Args:
            in_features_embedding: Number of input embedding features.
            out_features: Number of output concepts.
            embedding_size: Dimension of each concept's embedding.
        """
        super().__init__(
            in_features_embedding=in_features_embedding,
            out_features=out_features,
        )
        self.embedding_size = embedding_size

        self.out_logits_dim = out_features
        self.out_exogenous_shape = (self.out_logits_dim, embedding_size)
        self.out_encoder_dim = np.prod(self.out_exogenous_shape).item()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_embedding,
                self.out_encoder_dim
            ),
            torch.nn.Unflatten(-1, self.out_exogenous_shape),
            torch.nn.LeakyReLU(),
        )

    def forward(
        self,
        embedding: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Encode embeddings into exogenous variables.

        Args:
            embedding: Input embeddings of shape (batch_size, in_features_embedding).

        Returns:
            Tuple[torch.Tensor]: Exogenous variables of shape
                                (batch_size, out_features, embedding_size).
        """
        return self.encoder(embedding)
