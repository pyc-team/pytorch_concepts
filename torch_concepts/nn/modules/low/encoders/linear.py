"""
Linear encoder modules for concept prediction from embeddings.

These modules provide encoder layers that transform embeddings into concept representations.
"""
from typing import Union

import torch

from torch_concepts import Annotations
from ..base.layer import BaseConceptLayer


class LinearEmbeddingToConcept(BaseConceptLayer):
    """
    Encoder that predicts concept representations from embeddings.

    This encoder transforms an embedding into concept representations using 
    a linear layer.

    Attributes:
        in_embeddings (int): Number of input embedding features.
        out_concepts (int): Number of output concept representations.

    Args:
        in_embeddings: Number of input embedding features.
        out_concepts: Number of output concept representations.
        *args: Additional arguments for torch.nn.Linear.
        **kwargs: Additional keyword arguments for torch.nn.Linear.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import LinearEmbeddingToConcept
        >>>
        >>> encoder = LinearEmbeddingToConcept(
        ...     in_embeddings=128,
        ...     out_concepts=10
        ... )
        >>> embeddings = torch.randn(4, 128)  # batch_size=4, embedding_dim=128
        >>> concepts = encoder(embeddings)
        >>> print(concepts.shape)
        torch.Size([4, 10])

    References:
        Koh et al. "Concept Bottleneck Models", ICML 2020.
        https://arxiv.org/pdf/2007.04612
    """
    def __init__(
        self,
        in_embeddings: Union[int, Annotations],
        out_concepts: Union[int, Annotations],
        *args,
        **kwargs,
    ):
        """
        Initialize the encoder.

        Args:
            in_embeddings: Number of input embedding features.
            out_concepts: Number of output concept representations.
            *args: Additional arguments for torch.nn.Linear.
            **kwargs: Additional keyword arguments for torch.nn.Linear.
        """
        super().__init__(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        # (..., in_embeddings) -> (..., out_concepts)
        self.encoder = torch.nn.Linear(
            self.in_embeddings_shape,
            self.out_concepts_shape,
            *args,
            **kwargs,
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
