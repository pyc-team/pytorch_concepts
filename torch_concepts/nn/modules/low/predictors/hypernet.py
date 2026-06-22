from optparse import Option
from typing import Union, Optional

import torch

from torch_concepts import AxisAnnotation
from ..base.layer import BaseConceptLayer
from ..dense_layers import MLP
from ....functional import prune_linear_layer


class HyperlinearConceptEmbeddingToConcept(BaseConceptLayer):
    """
    Hypernetwork-based linear predictor for concept-based models.

    This predictor uses a (nonlinear) hypernetwork to generate per-sample weights
    from embeddings. These weights are then used in a linear layer to predict concept representations.
    It also supports stochastic biases with learnable mean and standard deviation.

    Attributes:
        in_concepts (int): Number of input concept representations.
        in_embeddings (int): Number of embedding features.
        hidden_size (int): Hidden size of the hypernetwork.
        out_concepts (int): Number of output concept representations.
        use_bias (bool): Whether to use stochastic bias.
        hypernet (nn.Module): Hypernetwork that generates weights.

    Args:
        in_concepts: Number of input concept representations.
        in_embeddings: Number of embedding input features.
        hidden_size: Hidden dimension of hypernetwork.
        activation: Activation function for hypernetwork output (default: identity).
        use_bias: Whether to add stochastic bias (default: False).
        init_bias_mean: Initial mean for bias distribution (default: 0.0).
        init_bias_std: Initial std for bias distribution (default: 0.01).
        min_std: Minimum std to ensure stability (default: 1e-6).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import HyperlinearConceptEmbeddingToConcept
        >>>
        >>> # Create hypernetwork predictor
        >>> predictor = HyperlinearConceptEmbeddingToConcept(
        ...     in_concepts=10,      # 10 concept states
        ...     in_embeddings=128,   # 128-dim embedding features
        ...     hidden_size=64,   # Hidden dim of hypernet
        ...     use_bias=False
        ... )
        >>>
        >>> # Generate random inputs
        >>> concepts = torch.randn(4, 10)   # batch_size=4, n_concepts=10
        >>> embeddings = torch.randn(4, 3, 128)        # batch_size=4, n_tasks=3, embedding_dim=128
        >>>
        >>> # Forward pass
        >>> output = predictor(concepts=concepts, embeddings=embeddings)
        >>> print(output.shape)
        torch.Size([4, 3])

    References:
        De Felice et al. "Causally Reliable Concept Bottleneck Models", NeurIPS 2025. https://arxiv.org/pdf/2503.04363
    """
    def __init__(
        self,
        in_concepts: Union[int, AxisAnnotation],
        in_embeddings: int,
        out_concepts: Optional[Union[int, AxisAnnotation]] = None,
        hidden_size: int = 32,
        activation='relu',
        use_bias : bool = True,
        init_bias_mean: float = 0.0,
        init_bias_std: float = 0.01,
        min_std: float = 1e-6,
        **kwargs,
    ):
        # Output size is inferred from the embeddings at forward time, so the
        # stored value is just a sentinel: default to -1 when not given.
        out_concepts = out_concepts if out_concepts is not None else -1
        super().__init__(
            in_concepts=in_concepts,
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.min_std = min_std
        self.init_bias_mean = init_bias_mean
        self.init_bias_std = init_bias_std

        self.hypernet = MLP(
            input_size=in_embeddings,
            hidden_size=hidden_size,
            output_size=in_concepts,
            activation=activation,
        )

        # Learnable distribution params for the stochastic bias (scalar, broadcasts to (B, Y))
        if self.use_bias:
            self.bias_mean = torch.nn.Parameter(torch.tensor(float(init_bias_mean)))
            # raw_std is unconstrained; softplus(raw_std) -> positive std
            # initialize so that softplus(raw_std) ~= init_bias_std
            init_raw_std = torch.log(torch.exp(torch.tensor(float(init_bias_std))) - 1.0).item()
            self.bias_raw_std = torch.nn.Parameter(torch.tensor(init_raw_std))
        else:
            # Keep attributes for shape/device consistency even if unused
            self.register_buffer("bias_mean", torch.tensor(0.0))
            self.register_buffer("bias_raw_std", torch.tensor(0.0))

    def _bias_std(self) -> torch.Tensor:
        # softplus to ensure positivity; add small floor for stability
        return torch.nn.functional.softplus(self.bias_raw_std) + self.min_std

    def forward(
            self,
            concepts: torch.Tensor,
            embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through hypernetwork predictor.

        Args:
            concepts: Concept representations of shape (batch_size, in_concepts).
            embeddings: Embedding features of shape (batch_size, out_concepts, in_embeddings).

        Returns:
            torch.Tensor: Output concepts of shape (batch_size, out_concepts).
        """
        weights = self.hypernet(embeddings)

        out_concepts = torch.einsum('bc,bnc->bn', concepts, weights)

        if self.use_bias:
            # Reparameterized sampling so mean/std are learnable
            eps = torch.randn_like(out_concepts)              # ~ N(0,1)
            std = self._bias_std().to(out_concepts.dtype).to(out_concepts.device)  # scalar -> broadcast
            mean = self.bias_mean.to(out_concepts.dtype).to(out_concepts.device)   # scalar -> broadcast
            out_concepts = out_concepts + mean + std * eps

        return out_concepts

    def prune(self, mask: torch.Tensor):
        """
        Prune the predictor based on a concept mask.

        Args:
            mask: Binary mask of shape (n_concepts,) indicating which concepts to keep.
        """
        self.in_concepts = mask.int().sum().item()
        self.hypernet[-1] = prune_linear_layer(self.hypernet[-1], mask, dim=1)
