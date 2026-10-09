"""
MLP predictor modules for concept-based models.

This module provides a prediction layer that transforms concept
representations into new concept representations using an MLP.
"""
from typing import Union

import torch

from torch_concepts import Annotations
from ..base.layer import BaseConceptLayer
from ..dense_layers import MLP


class MLPConceptToConcept(BaseConceptLayer):
    """
    MLP concept predictor.

    The nonlinear counterpart of :class:`LinearConceptToConcept`: the same
    ``concepts -> concepts`` interface, with an :class:`MLP` in place of the
    single linear layer, for a level whose parents do not explain their child
    linearly.

    Attributes:
        in_concepts (int): Number of input concept representations.
        out_concepts (int): Number of output concept representations.
        predictor (MLP): The prediction network.

    Args:
        in_concepts: Number of input concept representations.
        out_concepts: Number of output concept representations.
        hidden_size: Units in each hidden layer.
        n_layers: Number of hidden layers. Defaults to ``1``.
        activation: Activation function. Defaults to ``'relu'``.
        dropout: Dropout probability. Defaults to ``0.``.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import MLPConceptToConcept
        >>>
        >>> # Create predictor
        >>> predictor = MLPConceptToConcept(
        ...     in_concepts=10,
        ...     out_concepts=5,
        ...     hidden_size=16,
        ... )
        >>>
        >>> # Forward pass
        >>> in_concepts = torch.rand(2, 10)  # batch_size=2, in_concepts=10
        >>> out_concepts = predictor(in_concepts)
        >>> print(out_concepts.shape)
        torch.Size([2, 5])
    """

    def __init__(
        self,
        in_concepts: Union[int, Annotations],
        out_concepts: Union[int, Annotations],
        hidden_size: int,
        n_layers: int = 1,
        activation: str = 'relu',
        dropout: float = 0.,
    ):
        """
        Initialize the predictor.

        Args:
            in_concepts: Number of input concept representations.
            out_concepts: Number of output concept representations.
            hidden_size: Units in each hidden layer.
            n_layers: Number of hidden layers.
            activation: Activation function.
            dropout: Dropout probability.
        """
        super().__init__(
            in_concepts=in_concepts,
            out_concepts=out_concepts,
        )
        self.predictor = MLP(
            self.in_concepts_shape,
            hidden_size,
            self.out_concepts_shape,
            n_layers,
            activation,
            dropout,
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
            torch.Tensor: Predicted concepts of shape (..., out_concepts).
        """
        return self.predictor(concepts)
