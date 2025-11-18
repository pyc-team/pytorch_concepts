"""
Linear predictor modules for concept-based models.

This module provides linear prediction layers that transform concept
representations into new concept representations using a linear layer.
"""
import torch

from ..base.layer import BasePredictor
from typing import Callable

from ....functional import prune_linear_layer


class ProbPredictor(BasePredictor):
    """
    Linear concept predictor.

    This predictor transforms input concept logits into other concept
    logits using a linear layer followed by activation.

    Attributes:
        in_features_logits (int): Number of input logit features.
        out_features (int): Number of output concept features.
        in_activation (Callable): Activation function for inputs (default: sigmoid).
        predictor (nn.Sequential): The prediction network.

    Args:
        in_features_logits: Number of input logit features.
        out_features: Number of output concept features.
        in_activation: Activation function to apply to input logits (default: torch.sigmoid).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import ProbPredictor
        >>>
        >>> # Create predictor
        >>> predictor = ProbPredictor(
        ...     in_features_logits=10,
        ...     out_features=5
        ... )
        >>>
        >>> # Forward pass
        >>> in_logits = torch.randn(2, 10)  # batch_size=2, in_features=10
        >>> out_logits = predictor(in_logits)
        >>> print(out_logits.shape)
        torch.Size([2, 5])

    References:
        Koh et al. "Concept Bottleneck Models", ICML 2020.
        https://arxiv.org/pdf/2007.04612
    """

    def __init__(
        self,
        in_features_logits: int,
        out_features: int,
        in_activation: Callable = torch.sigmoid
    ):
        """
        Initialize the probabilistic predictor.

        Args:
            in_features_logits: Number of input logit features.
            out_features: Number of output concept features.
            in_activation: Activation function for inputs (default: torch.sigmoid).
        """
        super().__init__(
            in_features_logits=in_features_logits,
            out_features=out_features,
            in_activation=in_activation,
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_logits,
                out_features
            ),
            torch.nn.Unflatten(-1, (out_features,)),
        )

    def forward(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the predictor.

        Args:
            logits: Input logits of shape (batch_size, in_features_logits).

        Returns:
            torch.Tensor: Predicted concept probabilities of shape (batch_size, out_features).
        """
        in_probs = self.in_activation(logits)
        probs = self.predictor(in_probs)
        return probs

    def prune(self, mask: torch.Tensor):
        """
        Prune input features based on a binary mask.

        Removes input features where mask is False/0, reducing model complexity.

        Args:
            mask: Binary mask of shape (in_features_logits,) indicating which
                  features to keep (True/1) or remove (False/0).

        Example:
            >>> import torch
            >>> from torch_concepts.nn import ProbPredictor
            >>>
            >>> predictor = ProbPredictor(in_features_logits=10, out_features=5)
            >>>
            >>> # Prune first 3 features
            >>> mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool)
            >>> predictor.prune(mask)
            >>>
            >>> # Now only accepts 7 input features
            >>> logits = torch.randn(2, 7)
            >>> probs = predictor(logits)
            >>> print(probs.shape)
            torch.Size([2, 5])
        """
        self.in_features_logits = sum(mask.int())
        self.predictor[0] = prune_linear_layer(self.predictor[0], mask, dim=0)
