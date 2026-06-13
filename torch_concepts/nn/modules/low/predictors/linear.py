"""
Linear predictor modules for concept-based models.

This module provides linear prediction layers that transform concept
representations into new concept representations using a linear layer.
"""
from typing import Union

import torch

from torch_concepts import AxisAnnotation
from ..base.layer import BaseConceptLayer

from ....functional import prune_linear_layer


class LinearConceptToConcept(BaseConceptLayer):
    """
    Linear concept predictor.

    This predictor transforms input concept representations into other concept
    representations using a linear layer.

    Attributes:
        in_concepts (int): Number of input concept representations.
        out_concepts (int): Number of output concept representations.
        predictor (nn.Sequential): The prediction network.

    Args:
        in_concepts: Number of input concept representations.
        out_concepts: Number of output concept representations.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import LinearConceptToConcept
        >>>
        >>> # Create predictor
        >>> predictor = LinearConceptToConcept(
        ...     in_concepts=10,
        ...     out_concepts=5
        ... )
        >>>
        >>> # Forward pass
        >>> in_concepts = torch.rand(2, 10)  # batch_size=2, in_concepts=10
        >>> out_concepts = predictor(in_concepts)
        >>> print(out_concepts.shape)
        torch.Size([2, 5])

    References:
        Koh et al. "Concept Bottleneck Models", ICML 2020.
        https://arxiv.org/pdf/2007.04612
    """

    def __init__(
        self,
        in_concepts: Union[int, AxisAnnotation],
        out_concepts: Union[int, AxisAnnotation],
        *args,
        **kwargs,
    ):
        """
        Initialize the predictor.

        Args:
            in_concepts: Number of input concept representations.
            out_concepts: Number of output concept representations.
        """
        super().__init__(
            in_concepts=in_concepts,
            out_concepts=out_concepts,
        )
        self.predictor = torch.nn.Linear(
            in_concepts,
            out_concepts,
            *args,
            **kwargs,
        )

    def forward(
        self,
        concepts: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the predictor.

        Args:
            concepts: Input concepts of shape (..., in_concepts).

        Returns:
            torch.Tensor: Predicted concept probabilities of shape (..., out_concepts).
        """
        return self.predictor(concepts)
        

    def prune(self, mask: torch.Tensor):
        """
        Prune input features based on a binary mask.

        Removes input features where mask is False/0, reducing model complexity.

        Args:
            mask: Binary mask of shape (in_concepts,) indicating which
                  features to keep (True/1) or remove (False/0).

        Example:
            >>> import torch
            >>> from torch_concepts.nn import LinearConceptToConcept
            >>>
            >>> predictor = LinearConceptToConcept(in_concepts=10, out_concepts=5)
            >>>
            >>> # Prune first 3 features
            >>> mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool)
            >>> predictor.prune(mask)
            >>>
            >>> # Now only accepts 7 input features
            >>> concepts = torch.randn(2, 7)
            >>> probs = predictor(concepts)
            >>> print(probs.shape)
            torch.Size([2, 5])
        """
        self.in_concepts = sum(mask.int())
        self.predictor = prune_linear_layer(self.predictor, mask, dim=0)
