"""
Base layer classes for concept-based neural networks.

This module provides abstract base classes for building concept layers,
including encoders and predictors.
"""
from typing import Callable

import torch

from abc import ABC


class BaseConceptLayer(ABC, torch.nn.Module):
    """
    Abstract base class for concept layers.

    This class provides the foundation for all concept-based layers,
    defining the interface and basic structure for concept encoders
    and predictors.

    Attributes:
        in_features_logits (int): Number of input logit features.
        in_features_embedding (int): Number of input embedding features.
        in_features_exogenous (int): Number of exogenous input features.
        out_features (int): Number of output features.

    Args:
        out_features: Number of output features.
        in_features_logits: Number of input logit features (optional).
        in_features_embedding: Number of input embedding features (optional).
        in_features_exogenous: Number of exogenous input features (optional).
    """

    def __init__(
        self,
        out_features: int,
        in_features_logits: int = None,
        in_features_embedding: int = None,
        in_features_exogenous: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_features_logits = in_features_logits
        self.in_features_embedding = in_features_embedding
        self.in_features_exogenous = in_features_exogenous
        self.out_features = out_features

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the concept layer.

        Must be implemented by subclasses.

        Returns:
            torch.Tensor: Output tensor.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError


class BaseEncoder(BaseConceptLayer):
    """
    Abstract base class for concept encoder layers.

    Encoders transform input features (embeddings or exogenous variables)
    into concept representations.

    Args:
        out_features: Number of output concept features.
        in_features_embedding: Number of input embedding features (optional).
        in_features_exogenous: Number of exogenous input features (optional).
    """

    def __init__(self,
                 out_features: int,
                 in_features_embedding: int = None,
                 in_features_exogenous: int = None):
        super().__init__(
            in_features_logits=None,
            in_features_embedding=in_features_embedding,
            in_features_exogenous=in_features_exogenous,
            out_features=out_features
        )


class BasePredictor(BaseConceptLayer):
    """
    Abstract base class for concept predictor layers.

    Predictors take concept representations (plus embeddings or exogenous
    variables) and predict other concept representations.

    Attributes:
        in_activation (Callable): Activation function for input (default: sigmoid).

    Args:
        out_features: Number of output concept features.
        in_features_logits: Number of input logit features.
        in_features_embedding: Number of input embedding features (optional).
        in_features_exogenous: Number of exogenous input features (optional).
        in_activation: Activation function for input (default: torch.sigmoid).
    """

    def __init__(self,
                 out_features: int,
                 in_features_logits: int,
                 in_features_embedding: int = None,
                 in_features_exogenous: int = None,
                 in_activation: Callable = torch.sigmoid):
        super().__init__(
            in_features_logits=in_features_logits,
            in_features_embedding=in_features_embedding,
            in_features_exogenous=in_features_exogenous,
            out_features=out_features,
        )
        self.in_activation = in_activation

    def prune(self, mask: torch.Tensor):
        """
        Prune the predictor by removing connections based on the given mask.

        This method removes unnecessary connections in the predictor layer
        based on a binary mask, which can help reduce model complexity and
        improve interpretability.

        Args:
            mask: A binary mask indicating which connections to keep (1) or remove (0).

        Raises:
            NotImplementedError: Must be implemented by subclasses that support pruning.
        """
        raise NotImplementedError(f"Pruning is not yet supported for {self.__class__.__name__}.")
