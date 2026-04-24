"""
Exogenous encoder module.

This module provides encoders that transform latent into exogenous variables
for concept-based models.
"""
import numpy as np
import torch

from ..base.layer import BaseEncoder
from typing import Tuple


class LinearLatentToExogenous(BaseEncoder):
    """
    Exogenous encoder that creates exogenous variables from latent features.

    Transforms latent input into exogenous variables for each concept state, 
    producing a 2D output of shape (out_concepts, out_exogenous). Implements 
    the 'embedding generators' from Concept Embedding Models (Zarlenga et al., 2022).

    Attributes:
        exogenous_dim (int): Dimension of each concept's exogenous.
        n_concepts (int): Number of output concepts.
        encoder (nn.Sequential): The encoding network.

    Args:
        in_latent: Number of input latent features.
        out_concepts: Number of output concept representations.
        out_exogenous: Dimension of each exogenous variable.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import LinearLatentToExogenous
        >>>
        >>> # Create exogenous encoder
        >>> encoder = LinearLatentToExogenous(
        ...     in_latent=128,
        ...     out_concepts=5,
        ...     out_exogenous=16
        ... )
        >>>
        >>> # Forward pass
        >>> latent = torch.randn(4, 128)  # batch_size=4
        >>> exog = encoder(latent)
        >>> print(exog.shape)
        torch.Size([4, 5, 16])
        >>>
        >>> # Each concept has its own 16-dimensional exogenous
        >>> print(f"Concept 0 exogenous shape: {exog[:, 0, :].shape}")
        Concept 0 exogenous shape: torch.Size([4, 16])

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
        Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056
    """

    def __init__(
        self,
        in_latent: int,
        out_concepts: int,
        out_exogenous: int
    ):
        """
        Initialize the exogenous encoder.
    
        Args:
            in_latent: Number of input latent features.
            out_concepts: Number of output concept representations.
            out_exogenous: Dimension of each exogenous variable.
        """
        super().__init__(
            in_latent=in_latent,
            out_concepts=out_concepts,
        )

        self.out_exogenous = out_exogenous
        self.out_exogenous_shape = (out_concepts, out_exogenous)
        self.out_encoder_dim = np.prod(self.out_exogenous_shape).item()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_latent,
                self.out_encoder_dim
            ),
            torch.nn.Unflatten(-1, self.out_exogenous_shape),
            torch.nn.LeakyReLU(),
        )

    def forward(
        self,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Encode latent into exogenous variables.

        Args:
            latent: Input latent of shape (batch_size, in_latent).

        Returns:
            Tuple[torch.Tensor]: Exogenous variables of shape
                                (batch_size, out_concepts, out_exogenous).
        """
        return self.encoder(latent)
