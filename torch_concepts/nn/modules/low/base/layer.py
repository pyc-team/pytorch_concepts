"""
Base layer classes for concept-based neural networks.

This module provides abstract base classes for building concept layers,
including encoders and predictors.
"""

import torch

from abc import ABC


class BaseConceptLayer(ABC, torch.nn.Module):
    """
    Abstract base class for concept layers.

    This class provides the foundation for all concept-based layers,
    defining the interface and basic structure for concept encoders
    and predictors.

    Attributes:
        in_concepts (int): Number of input concept features.
        in_latent (int): Number of input latent features.
        in_exogenous (int): Number of exogenous input features.
        out_concepts (int): Number of output concept features.

    Args:
        out_concepts: Number of output concept features.
        in_concepts: Number of input concept features (optional).
        in_latent: Number of input latent features (optional).
        in_exogenous: Number of exogenous input features (optional).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import BaseConceptLayer
        >>>
        >>> # Create a custom concept layer
        >>> class MyConceptLayer(BaseConceptLayer):
        ...     def __init__(self, out_concepts, in_concepts):
        ...         super().__init__(
        ...             out_concepts=out_concepts,
        ...             in_concepts=in_concepts
        ...         )
        ...         self.linear = torch.nn.Linear(in_concepts, out_concepts)
        ...
        ...     def forward(self, concepts):
        ...         return torch.sigmoid(self.linear(concepts))
        >>>
        >>> # Example usage
        >>> layer = MyConceptLayer(out_concepts=5, in_concepts=10)
        >>>
        >>> # Generate random input
        >>> concepts = torch.randn(2, 10)  # batch_size=2, in_concepts=10
        >>>
        >>> # Forward pass
        >>> output = layer(concepts)
        >>> print(output.shape)  # torch.Size([2, 5])
    """

    def __init__(
        self,
        out_concepts: int,
        in_concepts: int = None,
        in_latent: int = None,
        in_exogenous: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_concepts = in_concepts
        self.in_latent = in_latent
        self.in_exogenous = in_exogenous
        self.out_concepts = out_concepts

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
        out_concepts: Number of output concept features.
        in_latent: Number of input latent features (optional).
        in_exogenous: Number of exogenous input features (optional).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import BaseEncoder
        >>>
        >>> # Create a custom encoder
        >>> class MyEncoder(BaseEncoder):
        ...     def __init__(self, out_concepts, in_latent):
        ...         super().__init__(
        ...             out_concepts=out_concepts,
        ...             in_latent=in_latent
        ...         )
        ...         self.net = torch.nn.Sequential(
        ...             torch.nn.Linear(in_latent, 128),
        ...             torch.nn.ReLU(),
        ...             torch.nn.Linear(128, out_concepts)
        ...         )
        ...
        ...     def forward(self, latent):
        ...         return self.net(latent)
        >>>
        >>> # Example usage
        >>> encoder = MyEncoder(out_concepts=10, in_latent=784)
        >>>
        >>> # Generate random image latent (e.g., flattened MNIST)
        >>> x = torch.randn(4, 784)  # batch_size=4, pixels=784
        >>>
        >>> # Encode to concepts
        >>> concepts = encoder(x)
        >>> print(concepts.shape)  # torch.Size([4, 10])
    """

    def __init__(self,
                 out_concepts: int,
                 in_latent: int = None,
                 in_exogenous: int = None):
        super().__init__(
            in_concepts=None,
            in_latent=in_latent,
            in_exogenous=in_exogenous,
            out_concepts=out_concepts
        )


class BasePredictor(BaseConceptLayer):
    """
    Abstract base class for concept predictor layers.

    Predictors take concept representations (plus latent or exogenous
    variables) and predict other concept representations.

    Args:
        out_concepts: Number of output concept features.
        in_concepts: Number of input concept features.
        in_latent: Number of input latent features (optional).
        in_exogenous: Number of exogenous input features (optional).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import BasePredictor
        >>>
        >>> # Create a custom predictor
        >>> class MyPredictor(BasePredictor):
        ...     def __init__(self, out_concepts, in_concepts):
        ...         super().__init__(
        ...             out_concepts=out_concepts,
        ...             in_concepts=in_concepts,
        ...         )
        ...         self.linear = torch.nn.Linear(in_concepts, out_concepts)
        ...
        ...     def forward(self, concepts):
        ...         return self.linear(concepts)
        >>>
        >>> # Example usage
        >>> predictor = MyPredictor(out_concepts=3, in_concepts=10)
        >>>
        >>> # Generate random concept probabilities
        >>> concept_probs = torch.rand(4, 10)  # batch_size=4, n_concepts=10
        >>>
        >>> # Predict task labels from concepts
        >>> task_logits = predictor(concept_probs)
        >>> print(task_logits.shape)  # torch.Size([4, 3])
    """

    def __init__(self,
                 out_concepts: int,
                 in_concepts: int,
                 in_latent: int = None,
                 in_exogenous: int = None,
                 **kwargs):
        super().__init__(
            in_concepts=in_concepts,
            in_latent=in_latent,
            in_exogenous=in_exogenous,
            out_concepts=out_concepts,
        )

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
