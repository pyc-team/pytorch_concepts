import torch
import torch.nn.functional as F

from ... import BaseEncoder
from torch.distributions import MultivariateNormal


class StochasticEncoderFromEmb(BaseEncoder):
    """
    StochasticEncoderFromEmb creates a bottleneck of supervised concepts with their covariance matrix.
    Main reference: `"Stochastic Concept Layer
    Models" <https://arxiv.org/pdf/2406.19272>`_

    Attributes:
        in_features_embedding (int): Number of input features.
        out_features (int): Number of output concepts.
        num_monte_carlo (int): Number of Monte Carlo samples.
    """

    def __init__(
        self,
        in_features_embedding: int,
        out_features: int,
        num_monte_carlo: int = 200,
    ):
        super().__init__(
            in_features_embedding=in_features_embedding,
            out_features=out_features,
        )
        self.num_monte_carlo = num_monte_carlo
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_embedding,
                out_features,
            ),
            torch.nn.Unflatten(-1, (out_features,)),
        )
        self.sigma = torch.nn.Linear(
            in_features_embedding,
            int(out_features * (out_features + 1) / 2),
        )
        # Prevent exploding precision matrix at initialization
        self.sigma.weight.data *= (0.01)

    def _predict_sigma(self, x):
        c_sigma = self.sigma(x)
        # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
        c_triang_cov = torch.zeros((c_sigma.shape[0], self.out_features, self.out_features), device=c_sigma.device)
        rows, cols = torch.tril_indices(row=self.out_features, col=self.out_features, offset=0)
        diag_idx = rows == cols
        c_triang_cov[:, rows, cols] = c_sigma
        c_triang_cov[:, range(self.out_features), range(self.out_features)] = (F.softplus(c_sigma[:, diag_idx]) + 1e-6)
        return c_triang_cov

    def forward(self,
        embedding: torch.Tensor,
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Predict concept scores.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted concept scores.
        """
        c_mu = self.mu(embedding)
        c_triang_cov = self._predict_sigma(embedding)
        # Sample from predicted normal distribution
        c_dist = MultivariateNormal(c_mu, scale_tril=c_triang_cov)
        c_mcmc_logit = c_dist.rsample([self.num_monte_carlo]).movedim(0, -1)  # [batch_size,num_concepts,mcmc_size]
        if reduce:
            c_mcmc_logit = c_mcmc_logit.mean(dim=-1)  # [batch_size,num_concepts]
        return c_mcmc_logit
