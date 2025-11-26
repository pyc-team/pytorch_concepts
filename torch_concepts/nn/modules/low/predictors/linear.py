"""
Linear predictor modules for concept-based models.

This module provides linear prediction layers that transform concept
representations into new concept representations using a linear layer.
"""
import torch

from ..base.layer import BasePredictor
from typing import Callable

from ....functional import prune_linear_layer


class LinearCC(BasePredictor):
    """
    Linear concept predictor.

    This predictor transforms input concept endogenous into other concept
    endogenous using a linear layer followed by activation.

    Attributes:
        in_features_endogenous (int): Number of input logit features.
        out_features (int): Number of output concept features.
        in_activation (Callable): Activation function for inputs (default: sigmoid).
        predictor (nn.Sequential): The prediction network.

    Args:
        in_features_endogenous: Number of input logit features.
        out_features: Number of output concept features.
        in_activation: Activation function to apply to input endogenous (default: torch.sigmoid).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import LinearCC
        >>>
        >>> # Create predictor
        >>> predictor = LinearCC(
        ...     in_features_endogenous=10,
        ...     out_features=5
        ... )
        >>>
        >>> # Forward pass
        >>> in_endogenous = torch.randn(2, 10)  # batch_size=2, in_features=10
        >>> out_endogenous = predictor(in_endogenous)
        >>> print(out_endogenous.shape)
        torch.Size([2, 5])

    References:
        Koh et al. "Concept Bottleneck Models", ICML 2020.
        https://arxiv.org/pdf/2007.04612
    """

    def __init__(
        self,
        in_features_endogenous: int,
        out_features: int,
        in_activation: Callable = torch.sigmoid
    ):
        """
        Initialize the probabilistic predictor.

        Args:
            in_features_endogenous: Number of input logit features.
            out_features: Number of output concept features.
            in_activation: Activation function for inputs (default: torch.sigmoid).
        """
        super().__init__(
            in_features_endogenous=in_features_endogenous,
            out_features=out_features,
            in_activation=in_activation,
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_endogenous,
                out_features
            ),
            torch.nn.Unflatten(-1, (out_features,)),
        )

    def forward(
        self,
        endogenous: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the predictor.

        Args:
            endogenous: Input endogenous of shape (batch_size, in_features_endogenous).

        Returns:
            torch.Tensor: Predicted concept probabilities of shape (batch_size, out_features).
        """
        in_probs = self.in_activation(endogenous)
        probs = self.predictor(in_probs)
        return probs

    def prune(self, mask: torch.Tensor):
        """
        Prune input features based on a binary mask.

        Removes input features where mask is False/0, reducing model complexity.

        Args:
            mask: Binary mask of shape (in_features_endogenous,) indicating which
                  features to keep (True/1) or remove (False/0).

        Example:
            >>> import torch
            >>> from torch_concepts.nn import LinearCC
            >>>
            >>> predictor = LinearCC(in_features_endogenous=10, out_features=5)
            >>>
            >>> # Prune first 3 features
            >>> mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool)
            >>> predictor.prune(mask)
            >>>
            >>> # Now only accepts 7 input features
            >>> endogenous = torch.randn(2, 7)
            >>> probs = predictor(endogenous)
            >>> print(probs.shape)
            torch.Size([2, 5])
        """
        self.in_features_endogenous = sum(mask.int())
        self.predictor[0] = prune_linear_layer(self.predictor[0], mask, dim=0)
