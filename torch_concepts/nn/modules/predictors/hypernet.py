import torch

from ...base.layer import BasePredictor
from typing import Callable

from ...functional import prune_linear_layer


class HyperLinearPredictor(BasePredictor):
    """
    Hypernetwork-based linear predictor for concept-based models.

    This predictor uses a hypernetwork to generate per-sample weights from
    exogenous features, enabling sample-adaptive predictions. It also supports
    stochastic biases with learnable mean and standard deviation.

    Attributes:
        in_features_logits (int): Number of input concept logits.
        in_features_exogenous (int): Number of exogenous features.
        embedding_size (int): Hidden size of the hypernetwork.
        out_features (int): Number of output features.
        use_bias (bool): Whether to use stochastic bias.
        hypernet (nn.Module): Hypernetwork that generates weights.

    Args:
        in_features_logits: Number of input concept logits.
        in_features_exogenous: Number of exogenous input features.
        embedding_size: Hidden dimension of hypernetwork.
        out_features: Number of output task features.
        in_activation: Activation function for concepts (default: identity).
        use_bias: Whether to add stochastic bias (default: True).
        init_bias_mean: Initial mean for bias distribution (default: 0.0).
        init_bias_std: Initial std for bias distribution (default: 0.01).
        min_std: Minimum std to ensure stability (default: 1e-6).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import HyperLinearPredictor
        >>>
        >>> # Create hypernetwork predictor
        >>> predictor = HyperLinearPredictor(
        ...     in_features_logits=10,      # 10 concepts
        ...     in_features_exogenous=128,   # 128-dim context features
        ...     embedding_size=64,           # Hidden dim of hypernet
        ...     use_bias=True
        ... )
        >>>
        >>> # Generate random inputs
        >>> concept_logits = torch.randn(4, 10)   # batch_size=4, n_concepts=10
        >>> exogenous = torch.randn(4, 3, 128)         # batch_size=4, n_tasks=3, exogenous_dim=128
        >>>
        >>> # Forward pass - generates per-sample weights via hypernetwork
        >>> task_logits = predictor(logits=concept_logits, exogenous=exogenous)
        >>> print(task_logits.shape)  # torch.Size([4, 3])
        >>>
        >>> # The hypernetwork generates different weights for each sample
        >>> # This enables sample-adaptive predictions
        >>>
        >>> # Example without bias
        >>> predictor_no_bias = HyperLinearPredictor(
        ...     in_features_logits=10,
        ...     in_features_exogenous=128,
        ...     embedding_size=64,
        ...     use_bias=False
        ... )
        >>>
        >>> task_logits = predictor_no_bias(logits=concept_logits, exogenous=exogenous)
        >>> print(task_logits.shape)  # torch.Size([4, 3])

    References:
        Debot et al. "Interpretable Concept-Based Memory Reasoning", NeurIPS 2024. https://arxiv.org/abs/2407.15527
    """
    def __init__(
        self,
        in_features_logits: int,
        in_features_exogenous: int,
        embedding_size: int,
        in_activation: Callable = lambda x: x,
        use_bias : bool = True,
        init_bias_mean: float = 0.0,
        init_bias_std: float = 0.01,
        min_std: float = 1e-6
    ):
        in_features_exogenous = in_features_exogenous
        super().__init__(
            in_features_logits=in_features_logits,
            in_features_exogenous=in_features_exogenous,
            out_features=-1,
            in_activation=in_activation,
        )
        self.embedding_size = embedding_size
        self.use_bias = use_bias
        self.min_std = float(min_std)

        self.hypernet = torch.nn.Sequential(
            torch.nn.Linear(in_features_exogenous, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(
                embedding_size,
                in_features_logits
            ),
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
            logits: torch.Tensor,
            exogenous: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through hypernetwork predictor.

        Args:
            logits: Concept logits of shape (batch_size, n_concepts).
            exogenous: Exogenous features of shape (batch_size, exog_dim).

        Returns:
            torch.Tensor: Task predictions of shape (batch_size, out_features).
        """
        weights = self.hypernet(exogenous)

        in_probs = self.in_activation(logits)
        out_logits = torch.einsum('bc,byc->by', in_probs, weights)

        if self.use_bias:
            # Reparameterized sampling so mean/std are learnable
            eps = torch.randn_like(out_logits)              # ~ N(0,1)
            std = self._bias_std().to(out_logits.dtype).to(out_logits.device)  # scalar -> broadcast
            mean = self.bias_mean.to(out_logits.dtype).to(out_logits.device)   # scalar -> broadcast
            out_logits = out_logits + mean + std * eps

        return out_logits

    def prune(self, mask: torch.Tensor):
        """
        Prune the predictor based on a concept mask.

        Args:
            mask: Binary mask of shape (n_concepts,) indicating which concepts to keep.
        """
        self.in_features_logits = mask.int().sum().item()
        self.hypernet[-1] = prune_linear_layer(self.hypernet[-1], mask, dim=1)
