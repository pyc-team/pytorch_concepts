"""
Base layer classes for concept-based neural networks.

This module provides abstract base classes for building concept layers,
including encoders and predictors.
"""
from typing import Union

import torch

from abc import ABC

from torch_concepts import AxisAnnotation, AnnotatedTensor


class BaseConceptLayer(ABC, torch.nn.Module):
    """
    Abstract base class for concept layers.

    This class provides the foundation for all concept-based layers,
    defining the interface and basic structure for concept encoders
    and predictors.

    Attributes:
        in_concepts (int): Number of input concept features.
        in_embeddings (int): Number of input embedding features.
        out_concepts (int): Number of output concept features.

    Args:
        out_concepts: Number of output concept features.
        in_concepts: Number of input concept features (optional).
        in_embeddings: Number of input embedding features (optional).

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
        >>> print(output.shape)
        torch.Size([2, 5])
    """

    def __init__(
        self,
        out_concepts: Union[int, AxisAnnotation],
        in_concepts: Union[int, AxisAnnotation] = None,
        in_embeddings: Union[int, AxisAnnotation] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_concepts = in_concepts
        self.in_embeddings = in_embeddings
        self.out_concepts = out_concepts

        self.in_concepts_shape = None
        if in_concepts is not None:
            self.in_concepts_shape = in_concepts if isinstance(in_concepts, int) else in_concepts.shape

        self.in_embeddings_shape = None
        if in_embeddings is not None:
            self.in_embeddings_shape = in_embeddings if isinstance(in_embeddings, int) else in_embeddings.shape

        self.out_concepts_shape = out_concepts if isinstance(out_concepts, int) else out_concepts.shape

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

    def annotate(self, x, out_concepts: AxisAnnotation = None) -> AnnotatedTensor:
        if out_concepts is None:
            if isinstance(self.out_concepts, AxisAnnotation):
                out_concepts = self.out_concepts
            else:
                return x
        return AnnotatedTensor(x, out_concepts)

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
