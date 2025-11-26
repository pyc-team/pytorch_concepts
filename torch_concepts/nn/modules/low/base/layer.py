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
        in_features_endogenous (int): Number of input logit features.
        in_features (int): Number of input latent features.
        in_features_exogenous (int): Number of exogenous input features.
        out_features (int): Number of output features.

    Args:
        out_features: Number of output features.
        in_features_endogenous: Number of input logit features (optional).
        in_features: Number of input latent features (optional).
        in_features_exogenous: Number of exogenous input features (optional).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import BaseConceptLayer
        >>>
        >>> # Create a custom concept layer
        >>> class MyConceptLayer(BaseConceptLayer):
        ...     def __init__(self, out_features, in_features_endogenous):
        ...         super().__init__(
        ...             out_features=out_features,
        ...             in_features_endogenous=in_features_endogenous
        ...         )
        ...         self.linear = torch.nn.Linear(in_features_endogenous, out_features)
        ...
        ...     def forward(self, endogenous):
        ...         return torch.sigmoid(self.linear(endogenous))
        >>>
        >>> # Example usage
        >>> layer = MyConceptLayer(out_features=5, in_features_endogenous=10)
        >>>
        >>> # Generate random input
        >>> endogenous = torch.randn(2, 10)  # batch_size=2, in_features=10
        >>>
        >>> # Forward pass
        >>> output = layer(endogenous)
        >>> print(output.shape)  # torch.Size([2, 5])
    """

    def __init__(
        self,
        out_features: int,
        in_features_endogenous: int = None,
        in_features: int = None,
        in_features_exogenous: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_features_endogenous = in_features_endogenous
        self.in_features = in_features
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

    Encoders transform input features (latent or exogenous variables)
    into concept representations.

    Args:
        out_features: Number of output concept features.
        in_features: Number of input latent features (optional).
        in_features_exogenous: Number of exogenous input features (optional).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import BaseEncoder
        >>>
        >>> # Create a custom encoder
        >>> class MyEncoder(BaseEncoder):
        ...     def __init__(self, out_features, in_features):
        ...         super().__init__(
        ...             out_features=out_features,
        ...             in_features=in_features
        ...         )
        ...         self.net = torch.nn.Sequential(
        ...             torch.nn.Linear(in_features, 128),
        ...             torch.nn.ReLU(),
        ...             torch.nn.Linear(128, out_features)
        ...         )
        ...
        ...     def forward(self, latent):
        ...         return self.net(latent)
        >>>
        >>> # Example usage
        >>> encoder = MyEncoder(out_features=10, in_features=784)
        >>>
        >>> # Generate random image latent (e.g., flattened MNIST)
        >>> x = torch.randn(4, 784)  # batch_size=4, pixels=784
        >>>
        >>> # Encode to concepts
        >>> concepts = encoder(x)
        >>> print(concepts.shape)  # torch.Size([4, 10])
    """

    def __init__(self,
                 out_features: int,
                 in_features: int = None,
                 in_features_exogenous: int = None):
        super().__init__(
            in_features_endogenous=None,
            in_features=in_features,
            in_features_exogenous=in_features_exogenous,
            out_features=out_features
        )


class BasePredictor(BaseConceptLayer):
    """
    Abstract base class for concept predictor layers.

    Predictors take concept representations (plus latent or exogenous
    variables) and predict other concept representations.

    Attributes:
        in_activation (Callable): Activation function for input (default: sigmoid).

    Args:
        out_features: Number of output concept features.
        in_features_endogenous: Number of input logit features.
        in_features: Number of input latent features (optional).
        in_features_exogenous: Number of exogenous input features (optional).
        in_activation: Activation function for input (default: torch.sigmoid).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import BasePredictor
        >>>
        >>> # Create a custom predictor
        >>> class MyPredictor(BasePredictor):
        ...     def __init__(self, out_features, in_features_endogenous):
        ...         super().__init__(
        ...             out_features=out_features,
        ...             in_features_endogenous=in_features_endogenous,
        ...             in_activation=torch.sigmoid
        ...         )
        ...         self.linear = torch.nn.Linear(in_features_endogenous, out_features)
        ...
        ...     def forward(self, endogenous):
        ...         # Apply activation to input endogenous
        ...         probs = self.in_activation(endogenous)
        ...         # Predict next concepts
        ...         return self.linear(probs)
        >>>
        >>> # Example usage
        >>> predictor = MyPredictor(out_features=3, in_features_endogenous=10)
        >>>
        >>> # Generate random concept endogenous
        >>> concept_endogenous = torch.randn(4, 10)  # batch_size=4, n_concepts=10
        >>>
        >>> # Predict task labels from concepts
        >>> task_endogenous = predictor(concept_endogenous)
        >>> print(task_endogenous.shape)  # torch.Size([4, 3])
        >>>
        >>> # Get task predictions
        >>> task_probs = torch.sigmoid(task_endogenous)
        >>> print(task_probs.shape)  # torch.Size([4, 3])
    """

    def __init__(self,
                 out_features: int,
                 in_features_endogenous: int,
                 in_features: int = None,
                 in_features_exogenous: int = None,
                 in_activation: Callable = torch.sigmoid):
        super().__init__(
            in_features_endogenous=in_features_endogenous,
            in_features=in_features,
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
