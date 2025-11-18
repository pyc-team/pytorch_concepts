"""
Linear encoder modules for concept prediction from latent features.

This module provides encoder layers that transform embeddings or exogenous
variables into concept representations.
"""
import torch

from ...base.layer import BaseEncoder
from typing import List, Union


class ProbEncoderFromEmb(BaseEncoder):
    """
    Encoder that predicts concept activations from embeddings.

    This encoder transforms input embeddings into concept logits using a
    linear layer. It's typically used as the first layer in concept bottleneck
    models to extract concepts from neural network embeddings.

    Attributes:
        in_features_embedding (int): Number of input embedding features.
        out_features (int): Number of output concept features.
        encoder (nn.Sequential): The encoding network.

    Args:
        in_features_embedding: Number of input embedding features.
        out_features: Number of output concept features.
        *args: Additional arguments for torch.nn.Linear.
        **kwargs: Additional keyword arguments for torch.nn.Linear.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import ProbEncoderFromEmb
        >>>
        >>> # Create encoder
        >>> encoder = ProbEncoderFromEmb(
        ...     in_features_embedding=128,
        ...     out_features=10
        ... )
        >>>
        >>> # Forward pass with embeddings from a neural network
        >>> embeddings = torch.randn(4, 128)  # batch_size=4, embedding_dim=128
        >>> concept_logits = encoder(embeddings)
        >>> print(concept_logits.shape)
        torch.Size([4, 10])
        >>>
        >>> # Apply sigmoid to get probabilities
        >>> concept_probs = torch.sigmoid(concept_logits)
        >>> print(concept_probs.shape)
        torch.Size([4, 10])

    References:
        Koh et al. "Concept Bottleneck Models", ICML 2020.
        https://arxiv.org/pdf/2007.04612
    """
    def __init__(
        self,
        in_features_embedding: int,
        out_features: int,
        *args,
        **kwargs,
    ):
        """
        Initialize the embedding encoder.

        Args:
            in_features_embedding: Number of input embedding features.
            out_features: Number of output concept features.
            *args: Additional arguments for torch.nn.Linear.
            **kwargs: Additional keyword arguments for torch.nn.Linear.
        """
        super().__init__(
            in_features_embedding=in_features_embedding,
            out_features=out_features,
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_embedding,
                out_features,
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, (out_features,)),
        )

    def forward(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode embeddings into concept logits.

        Args:
            embedding: Input embeddings of shape (batch_size, in_features_embedding).

        Returns:
            torch.Tensor: Concept logits of shape (batch_size, out_features).
        """
        return self.encoder(embedding)


class ProbEncoderFromExog(BaseEncoder):
    """
    Encoder that extracts concepts from exogenous variables.

    This encoder processes exogenous latent variables to produce
    concept representations. It requires at least one exogenous variable per concept.

    Attributes:
        in_features_exogenous (int): Number of exogenous input features.
        n_exogenous_per_concept (int): Number of exogenous vars per concept.
        encoder (nn.Sequential): The encoding network.

    Args:
        in_features_exogenous: Number of exogenous input features.
        n_exogenous_per_concept: Number of exogenous variables per concept (default: 1).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import ProbEncoderFromExog
        >>>
        >>> # Create encoder with 2 exogenous vars per concept
        >>> encoder = ProbEncoderFromExog(
        ...     in_features_exogenous=5,
        ...     n_exogenous_per_concept=2
        ... )
        >>>
        >>> # Forward pass with exogenous variables
        >>> # Expected input shape: (batch, out_features, in_features * n_exogenous_per_concept)
        >>> exog_vars = torch.randn(4, 3, 10)  # batch=4, concepts=3, exog_features=5*2
        >>> concept_logits = encoder(exog_vars)
        >>> print(concept_logits.shape)
        torch.Size([4, 3])

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056
    """
    def __init__(
        self,
        in_features_exogenous: int,
        n_exogenous_per_concept: int = 1
    ):
        """
        Initialize the exogenous encoder.

        Args:
            in_features_exogenous: Number of exogenous input features.
            out_features: Number of output concept features.
            n_exogenous_per_concept: Number of exogenous variables per concept.
        """
        self.n_exogenous_per_concept = n_exogenous_per_concept
        in_features_exogenous = in_features_exogenous * n_exogenous_per_concept
        super().__init__(
            in_features_exogenous=in_features_exogenous,
            out_features=-1,
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_exogenous,
                1
            ),
            torch.nn.Flatten(),
        )

    def forward(
        self,
        exogenous: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode exogenous variables into concept logits.

        Args:
            exogenous: Exogenous variables of shape
                      (batch_size, out_features, in_features_exogenous).

        Returns:
            torch.Tensor: Concept logits of shape (batch_size, out_features).
        """
        return self.encoder(exogenous)
