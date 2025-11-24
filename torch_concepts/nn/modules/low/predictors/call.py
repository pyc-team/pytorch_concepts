import torch

from ..base.layer import BasePredictor
from typing import Callable


class CallableCC(BasePredictor):
    """
    A predictor that applies a custom callable function to concept representations.

    This predictor allows flexible task prediction by accepting any callable function
    that operates on concept representations. It optionally includes learnable stochastic
    bias parameters (mean and standard deviation) that are added to the output using
    the reparameterization trick for gradient-based learning.

    The module can be used to write custom layers for standard Structural Causal Models (SCMs).

    Args:
        func: Callable function that takes concept probabilities and returns task predictions.
              Should accept a tensor of shape (batch_size, n_concepts) and return
              a tensor of shape (batch_size, out_features).
        in_activation: Activation function to apply to input endogenous before passing to func.
                      Default is identity (lambda x: x).
        use_bias: Whether to add learnable stochastic bias to the output. Default is True.
        init_bias_mean: Initial value for the bias mean parameter. Default is 0.0.
        init_bias_std: Initial value for the bias standard deviation. Default is 0.01.
        min_std: Minimum standard deviation floor for numerical stability. Default is 1e-6.

    Examples:
        >>> import torch
        >>> from torch_concepts.nn import CallableCC
        >>>
        >>> # Generate sample data
        >>> batch_size = 32
        >>> n_concepts = 3
        >>> endogenous = torch.randn(batch_size, n_concepts)
        >>>
        >>> # Define a polynomial function with fixed weights for 3 inputs, 2 outputs
        >>> def quadratic_predictor(probs):
        ...     c0, c1, c2 = probs[:, 0:1], probs[:, 1:2], probs[:, 2:3]
        ...     output1 = 0.5*c0**2 + 1.0*c1**2 + 1.5*c2
        ...     output2 = 2.0*c0 - 1.0*c1**2 + 0.5*c2**3
        ...     return torch.cat([output1, output2], dim=1)
        >>>
        >>> predictor = CallableCC(
        ...     func=quadratic_predictor,
        ...     use_bias=True
        ... )
        >>> predictions = predictor(endogenous)
        >>> print(predictions.shape)  # torch.Size([32, 2])

        References
            Pearl, J. "Causality", Cambridge University Press (2009).
    """

    def __init__(
        self,
        func: Callable,
        in_activation: Callable = lambda x: x,
        use_bias : bool = True,
        init_bias_mean: float = 0.0,
        init_bias_std: float = 0.01,
        min_std: float = 1e-6
    ):
        super().__init__(
            in_features_endogenous=-1,
            out_features=-1,
            in_activation=in_activation,
        )
        self.use_bias = use_bias
        self.min_std = float(min_std)
        self.func = func

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
        """
        Compute the bias standard deviation using softplus activation.

        Returns:
            torch.Tensor: Positive standard deviation value with minimum floor applied.
        """
        # softplus to ensure positivity; add small floor for stability
        return torch.nn.functional.softplus(self.bias_raw_std) + self.min_std

    def forward(
            self,
            endogenous: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        in_probs = self.in_activation(endogenous)
        out_endogenous = self.func(in_probs, *args, **kwargs)

        if self.use_bias:
            # Reparameterized sampling so mean/std are learnable
            eps = torch.randn_like(out_endogenous)              # ~ N(0,1)
            std = self._bias_std().to(out_endogenous.dtype).to(out_endogenous.device)  # scalar -> broadcast
            mean = self.bias_mean.to(out_endogenous.dtype).to(out_endogenous.device)   # scalar -> broadcast
            out_endogenous = out_endogenous + mean + std * eps

        return out_endogenous
