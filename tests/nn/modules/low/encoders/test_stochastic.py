"""
Tests for StochasticEmbeddingToConcept (formerly StochasticLatentToConcept).
"""
import unittest
import torch
from torch_concepts.nn.modules.low.encoders.stochastic import StochasticEmbeddingToConcept


class TestStochasticEmbeddingToConcept(unittest.TestCase):
    """Test StochasticEmbeddingToConcept."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = StochasticEmbeddingToConcept(
            in_embeddings=128,
            out_concepts=5,
            num_monte_carlo=100
        )
        self.assertEqual(encoder.in_embeddings, 128)
        self.assertEqual(encoder.out_concepts, 5)
        self.assertEqual(encoder.num_monte_carlo, 100)
        self.assertIsNotNone(encoder.mu)
        self.assertIsNotNone(encoder.sigma)

    def test_forward_with_reduce(self):
        """Test forward pass with reduce=True."""
        encoder = StochasticEmbeddingToConcept(
            in_embeddings=64,
            out_concepts=5,
            num_monte_carlo=50
        )
        embeddings = torch.randn(4, 64)
        output = encoder(embeddings, reduce=True)
        self.assertEqual(output.shape, (4, 5))

    def test_forward_without_reduce(self):
        """Test forward pass with reduce=False."""
        encoder = StochasticEmbeddingToConcept(
            in_embeddings=32,
            out_concepts=3,
            num_monte_carlo=20
        )
        embeddings = torch.randn(2, 32)
        output = encoder(embeddings, reduce=False)
        self.assertEqual(output.shape, (2, 3, 20))

    def test_gradient_flow(self):
        """Test gradient flow through stochastic encoder."""
        encoder = StochasticEmbeddingToConcept(
            in_embeddings=16,
            out_concepts=4,
            num_monte_carlo=10
        )
        embeddings = torch.randn(2, 16, requires_grad=True)
        output = encoder(embeddings, reduce=True)
        output.sum().backward()
        self.assertIsNotNone(embeddings.grad)

    def test_predict_sigma(self):
        """Test internal _predict_sigma method."""
        encoder = StochasticEmbeddingToConcept(
            in_embeddings=16,
            out_concepts=3,
            num_monte_carlo=10
        )
        embeddings = torch.randn(2, 16)
        sigma = encoder._predict_sigma(embeddings)
        self.assertEqual(sigma.shape, (2, 3, 3))
        # Check lower triangular
        for i in range(2):
            for row in range(3):
                for col in range(row + 1, 3):
                    self.assertEqual(sigma[i, row, col].item(), 0.0)

    def test_positive_diagonal_covariance(self):
        """Test that diagonal of covariance is positive."""
        encoder = StochasticEmbeddingToConcept(
            in_embeddings=16,
            out_concepts=3,
            num_monte_carlo=10
        )
        embeddings = torch.randn(2, 16)
        sigma = encoder._predict_sigma(embeddings)
        for i in range(2):
            for j in range(3):
                self.assertGreater(sigma[i, j, j].item(), 0.0)

    def test_monte_carlo_samples_variability(self):
        """Test that MC samples show variability."""
        encoder = StochasticEmbeddingToConcept(
            in_embeddings=16,
            out_concepts=2,
            num_monte_carlo=100
        )
        embeddings = torch.randn(1, 16)
        output = encoder(embeddings, reduce=False)
        std = output.std(dim=2)
        self.assertTrue(torch.any(std > 0.01))

    def test_different_monte_carlo_sizes(self):
        """Test various MC sample sizes."""
        for mc_size in [10, 50, 200]:
            encoder = StochasticEmbeddingToConcept(
                in_embeddings=16,
                out_concepts=3,
                num_monte_carlo=mc_size
            )
            embeddings = torch.randn(2, 16)
            output = encoder(embeddings, reduce=False)
            self.assertEqual(output.shape[2], mc_size)

    def test_mean_consistency(self):
        """Test that mean of samples approximates mu."""
        torch.manual_seed(42)
        encoder = StochasticEmbeddingToConcept(
            in_embeddings=16,
            out_concepts=2,
            num_monte_carlo=1000
        )
        embeddings = torch.randn(1, 16)
        mu = encoder.mu(embeddings)
        samples = encoder(embeddings, reduce=False)
        mc_mean = samples.mean(dim=2)
        self.assertTrue(torch.allclose(mu, mc_mean, atol=0.3))

    def test_batch_processing(self):
        """Test different batch sizes."""
        encoder = StochasticEmbeddingToConcept(
            in_embeddings=32,
            out_concepts=4,
            num_monte_carlo=20
        )
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 32)
            output_reduced = encoder(embeddings, reduce=True)
            output_full = encoder(embeddings, reduce=False)
            self.assertEqual(output_reduced.shape, (batch_size, 4))
            self.assertEqual(output_full.shape, (batch_size, 4, 20))

    def test_sigma_weight_initialization(self):
        """Test that sigma weights are scaled down at init."""
        encoder = StochasticEmbeddingToConcept(
            in_embeddings=16,
            out_concepts=3,
            num_monte_carlo=10
        )
        sigma_weight_norm = encoder.sigma.weight.data.norm().item()
        self.assertLess(sigma_weight_norm, 1.0)


if __name__ == '__main__':
    unittest.main()
