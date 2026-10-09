"""
MLP encoder modules for concept prediction from embeddings.

These modules provide encoder layers that transform embeddings into concept
representations.
"""
from typing import Union

import torch

from torch_concepts import Annotations
from ..base.layer import BaseConceptLayer
from ..dense_layers import MLP


class MLPEmbeddingToConcept(BaseConceptLayer):
    """
    Encoder that predicts concept representations from embeddings.

    The nonlinear counterpart of :class:`LinearEmbeddingToConcept`: the same
    ``embeddings -> concepts`` interface, with an :class:`MLP` in place of the
    single linear layer, for a latent that does not explain its concepts
    linearly.

    Attributes:
        in_embeddings (int): Number of input embedding features.
        out_concepts (int): Number of output concept representations.
        encoder (MLP): The encoding network.

    Args:
        in_embeddings: Number of input embedding features.
        out_concepts: Number of output concept representations.
        hidden_size: Units in each hidden layer.
        n_layers: Number of hidden layers. Defaults to ``1``.
        activation: Activation function. Defaults to ``'relu'``.
        dropout: Dropout probability. Defaults to ``0.``.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import MLPEmbeddingToConcept
        >>>
        >>> encoder = MLPEmbeddingToConcept(
        ...     in_embeddings=128,
        ...     out_concepts=10,
        ...     hidden_size=64,
        ... )
        >>> embeddings = torch.randn(4, 128)  # batch_size=4, embedding_dim=128
        >>> concepts = encoder(embeddings)
        >>> print(concepts.shape)
        torch.Size([4, 10])
    """
    def __init__(
        self,
        in_embeddings: int,
        out_concepts: Union[int, Annotations],
        hidden_size: int,
        n_layers: int = 1,
        activation: str = 'relu',
        dropout: float = 0.,
    ):
        """
        Initialize the encoder.

        Args:
            in_embeddings: Number of input embedding features.
            out_concepts: Number of output concept representations.
            hidden_size: Units in each hidden layer.
            n_layers: Number of hidden layers.
            activation: Activation function.
            dropout: Dropout probability.
        """
        super().__init__(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        # (..., in_embeddings) -> (..., out_concepts)
        self.encoder = MLP(
            self.in_embeddings_shape,
            hidden_size,
            self.out_concepts_shape,
            n_layers,
            activation,
            dropout,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode embeddings into concept representations.

        Args:
            embeddings: Input embeddings of shape (..., in_embeddings).

        Returns:
            torch.Tensor: Concept representations of shape (..., out_concepts).
        """
        return self.encoder(embeddings)
