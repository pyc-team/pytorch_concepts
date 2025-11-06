import torch

from ...base.layer import BasePredictor
from typing import Callable


class HyperLinearPredictor(BasePredictor):
    """
    """
    def __init__(
        self,
        in_features_logits: int,
        in_features_exogenous: int,
        embedding_size: int,
        out_features: int,
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
            out_features=out_features,
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
