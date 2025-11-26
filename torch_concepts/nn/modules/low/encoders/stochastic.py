"""
Stochastic encoder module for probabilistic concept representations.

This module provides encoders that predict both mean and covariance for concepts,
enabling uncertainty quantification in concept-based models.
"""
import torch
import torch.nn.functional as F

from ..base.layer import BaseEncoder
from torch.distributions import MultivariateNormal


class StochasticZC(BaseEncoder):
    """
    Stochastic encoder that predicts concept distributions with uncertainty.

    Encodes input latent into concept distributions by predicting both mean
    and covariance matrices. Uses Monte Carlo sampling from the predicted
    multivariate normal distribution to generate concept representations.

    Attributes:
        num_monte_carlo (int): Number of Monte Carlo samples.
        mu (nn.Sequential): Network for predicting concept means.
        sigma (nn.Linear): Network for predicting covariance lower triangle.

    Args:
        in_features: Number of input latent features.
        out_features: Number of output concepts.
        num_monte_carlo: Number of Monte Carlo samples for uncertainty (default: 200).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import StochasticZC
        >>>
        >>> # Create stochastic encoder
        >>> encoder = StochasticZC(
        ...     in_features=128,
        ...     out_features=5,
        ...     num_monte_carlo=100
        ... )
        >>>
        >>> # Forward pass with mean reduction
        >>> latent = torch.randn(4, 128)
        >>> concept_endogenous = encoder(latent, reduce=True)
        >>> print(concept_endogenous.shape)
        torch.Size([4, 5])
        >>>
        >>> # Forward pass keeping all MC samples
        >>> concept_samples = encoder(latent, reduce=False)
        >>> print(concept_samples.shape)
        torch.Size([4, 5, 100])

    References:
        Vandenhirtz et al. "Stochastic Concept Bottleneck Models", 2024.
        https://arxiv.org/pdf/2406.19272
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_monte_carlo: int = 200,
        eps: float = 1e-6,
    ):
        """
        Initialize the stochastic encoder.

        Args:
            in_features: Number of input latent features.
            out_features: Number of output concepts.
            num_monte_carlo: Number of Monte Carlo samples (default: 200).
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
        )
        self.num_monte_carlo = num_monte_carlo
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(
                in_features,
                out_features,
            ),
            torch.nn.Unflatten(-1, (out_features,)),
        )
        self.sigma = torch.nn.Linear(
            in_features,
            int(out_features * (out_features + 1) / 2),
        )
        # Prevent exploding precision matrix at initialization
        self.sigma.weight.data *= (0.01)
        self.eps = eps

    def _predict_sigma(self, x):
        """
        Predict lower triangular covariance matrix.

        Args:
            x: Input embeddings.

        Returns:
            torch.Tensor: Lower triangular covariance matrix.
        """
        c_sigma = self.sigma(x)
        # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
        c_triang_cov = torch.zeros((c_sigma.shape[0], self.out_features, self.out_features), device=c_sigma.device)
        rows, cols = torch.tril_indices(row=self.out_features, col=self.out_features, offset=0)
        diag_idx = rows == cols
        c_triang_cov[:, rows, cols] = c_sigma
        c_sigma_activated = F.softplus(c_sigma[:, diag_idx])
        c_triang_cov[:, range(self.out_features), range(self.out_features)] = (c_sigma_activated + self.eps)
        return c_triang_cov

    def forward(self,
        input: torch.Tensor,
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Predict concept scores with uncertainty via Monte Carlo sampling.

        Predicts a multivariate normal distribution over concepts and samples
        from it using the reparameterization trick.

        Args:
            input: Input input of shape (batch_size, in_features).
            reduce: If True, return mean over MC samples; if False, return all samples
                   (default: True).

        Returns:
            torch.Tensor: Concept endogenous of shape (batch_size, out_features) if reduce=True,
                         or (batch_size, out_features, num_monte_carlo) if reduce=False.
        """
        c_mu = self.mu(input)
        c_triang_cov = self._predict_sigma(input)
        # Sample from predicted normal distribution
        c_dist = MultivariateNormal(c_mu, scale_tril=c_triang_cov)
        c_mcmc_logit = c_dist.rsample([self.num_monte_carlo]).movedim(0, -1)  # [batch_size,num_concepts,mcmc_size]
        if reduce:
            c_mcmc_logit = c_mcmc_logit.mean(dim=-1)  # [batch_size,num_concepts]
        return c_mcmc_logit
