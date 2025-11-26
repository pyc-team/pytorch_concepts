"""
Exogenous encoder module.

This module provides encoders that transform latent into exogenous variables
for concept-based models, supporting the Concept Embedding Models architecture.
"""
import numpy as np
import torch

from ..base.layer import BaseEncoder
from typing import Tuple


class LinearZU(BaseEncoder):
    """
    Exogenous encoder that creates concept exogenous.

    Transforms input input into exogenous variables (external features) for
    each concept, producing a 2D output of shape (out_features, exogenous_size).
    Implements the 'embedding generators' from Concept Embedding Models (Zarlenga et al., 2022).

    Attributes:
        exogenous_size (int): Dimension of each concept's exogenous.
        out_endogenous_dim (int): Number of output concepts.
        encoder (nn.Sequential): The encoding network.

    Args:
        in_features: Number of input latent features.
        out_features: Number of output concepts.
        exogenous_size: Dimension of each concept's exogenous.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import LinearZU
        >>>
        >>> # Create exogenous encoder
        >>> encoder = LinearZU(
        ...     in_features=128,
        ...     out_features=5,
        ...     exogenous_size=16
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
        in_features: int,
        out_features: int,
        exogenous_size: int
    ):
        """
        Initialize the exogenous encoder.

        Args:
            in_features: Number of input latent features.
            out_features: Number of output concepts.
            exogenous_size: Dimension of each concept's exogenous.
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
        )
        self.exogenous_size = exogenous_size

        self.out_endogenous_dim = out_features
        self.out_exogenous_shape = (self.out_endogenous_dim, exogenous_size)
        self.out_encoder_dim = np.prod(self.out_exogenous_shape).item()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features,
                self.out_encoder_dim
            ),
            torch.nn.Unflatten(-1, self.out_exogenous_shape),
            torch.nn.LeakyReLU(),
        )

    def forward(
        self,
        input: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Encode latent into exogenous variables.

        Args:
            input: Input latent of shape (batch_size, in_features).

        Returns:
            Tuple[torch.Tensor]: Exogenous variables of shape
                                (batch_size, out_features, exogenous_size).
        """
        return self.encoder(input)
