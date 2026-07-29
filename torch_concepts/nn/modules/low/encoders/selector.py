"""
Memory selector module for memory selection.

This module provides a memory-based selector that learns to attend over
a memory bank of concept exogenous.
"""
import numpy as np
import torch
import torch.nn.functional as F


from ..base.layer import BaseEncoder


class CategoricalSelector(BaseEncoder):
    """
    Categorical selector that outputs concept-wise assignment probabilities.

    This module maps latent inputs to logits of shape
    ``(batch_size, out_concepts, out_exogenous)`` and applies a softmax over
    the exogenous dimension to produce normalized mixing probabilities.

    Attributes:
        out_exogenous (int): Hidden width used in the selector MLP.
        out_concepts (int): Number of output concepts.
        selector_hidden_layers (int): Number of hidden layers in the selector MLP.
        selector (nn.Sequential): Attention network for memory selection.

    Args:
        in_latent: Number of input latent features.
        out_exogenous: Number of output exogenous features.
        out_concepts: Number of output concept representations.
        selector_hidden_layers: Number of hidden layers in the selector MLP.
            Must be >= 0.
        *args: Additional positional arguments for linear layers in the selector.
        **kwargs: Additional keyword arguments for linear layers in the selector.

    References:
        Debot et al. "Interpretable Concept-Based Memory Reasoning", NeurIPS 2024. https://arxiv.org/abs/2407.15527
    """
    def __init__(
        self,
        in_latent: int,
        out_exogenous: int,  # nb_rules
        out_concepts: int,  # nb_tasks
        selector_hidden_layers: int = 1,
        *args,
        **kwargs,
    ):
        """
        Initialize the categorical selector.

        Args:
            in_latent: Number of input latent features.
            out_exogenous: Number of output exogenous features.
            out_concepts: Number of output concepts.
            selector_hidden_layers: Number of hidden layers in the selector
                MLP. Must be >= 0.
            *args: Additional positional arguments for linear layers in the selector.
            **kwargs: Additional keyword arguments for linear layers in the selector.
        """
        super().__init__(
            in_latent=in_latent,
            out_concepts=out_concepts,
        )
        if selector_hidden_layers < 0:
            raise ValueError("selector_hidden_layers must be >= 0")

        self.out_exogenous = out_exogenous
        self.out_concepts = out_concepts
        self.selector_hidden_layers = selector_hidden_layers
        self._selector_out_shape = (out_concepts, out_exogenous)
        self._selector_out_dim = np.prod(self._selector_out_shape).item()

        selector_layers = []
        in_features = in_latent
        for _ in range(selector_hidden_layers):
            selector_layers.extend([
                torch.nn.Linear(in_features, in_latent, *args, **kwargs),
                torch.nn.ReLU(),
            ])
            in_features = in_latent
        selector_layers.extend([
            torch.nn.Linear(in_features, self._selector_out_dim, *args, **kwargs),
            torch.nn.Unflatten(-1, self._selector_out_shape),
        ])
        self.selector = torch.nn.Sequential(*selector_layers)

    def forward(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute concept-wise mixing probabilities from latent input.

        Applies the selector MLP and normalizes logits with softmax over
        the exogenous axis.

        Args:
            latent: Input latent of shape (batch_size, in_latent).

        Returns:
            torch.Tensor: Mixing probabilities of shape
                         (batch_size, out_concepts, out_exogenous).
        """
        mixing_coeff = self.selector(latent)
        return torch.softmax(mixing_coeff, dim=-1)  # [Batch x Task x Memory]
