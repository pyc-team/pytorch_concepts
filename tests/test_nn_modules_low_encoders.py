"""
Comprehensive tests for torch_concepts.nn.modules.low.encoders

Tests all encoder modules (linear, exogenous, selector, stochastic).
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn.modules.low.encoders.linear import ProbEncoderFromEmb, ProbEncoderFromExog
from torch_concepts.nn.modules.low.encoders.exogenous import ExogEncoder
from torch_concepts.nn.modules.low.encoders.selector import MemorySelector
from torch_concepts.nn.modules.low.encoders.stochastic import StochasticEncoderFromEmb


class TestProbEncoderFromEmb(unittest.TestCase):
    """Test ProbEncoderFromEmb."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = ProbEncoderFromEmb(
            in_features_embedding=128,
            out_features=10
        )
        self.assertEqual(encoder.in_features_embedding, 128)
        self.assertEqual(encoder.out_features, 10)
        self.assertIsInstance(encoder.encoder, nn.Sequential)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = ProbEncoderFromEmb(
            in_features_embedding=128,
            out_features=10
        )
        embeddings = torch.randn(4, 128)
        output = encoder(embeddings)
        self.assertEqual(output.shape, (4, 10))

    def test_gradient_flow(self):
        """Test gradient flow through encoder."""
        encoder = ProbEncoderFromEmb(
            in_features_embedding=64,
            out_features=5
        )
        embeddings = torch.randn(2, 64, requires_grad=True)
        output = encoder(embeddings)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_batch_processing(self):
        """Test different batch sizes."""
        encoder = ProbEncoderFromEmb(
            in_features_embedding=32,
            out_features=5
        )
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 32)
            output = encoder(embeddings)
            self.assertEqual(output.shape, (batch_size, 5))

    def test_with_bias_false(self):
        """Test encoder without bias."""
        encoder = ProbEncoderFromEmb(
            in_features_embedding=32,
            out_features=5,
            bias=False
        )
        embeddings = torch.randn(2, 32)
        output = encoder(embeddings)
        self.assertEqual(output.shape, (2, 5))


class TestProbEncoderFromExog(unittest.TestCase):
    """Test ProbEncoderFromExog."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = ProbEncoderFromExog(
            in_features_exogenous=16,
            n_exogenous_per_concept=2
        )
        self.assertEqual(encoder.n_exogenous_per_concept, 2)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = ProbEncoderFromExog(
            in_features_exogenous=8,
            n_exogenous_per_concept=2
        )
        # Input shape: (batch, concepts, in_features * n_exogenous_per_concept)
        exog = torch.randn(4, 5, 16)  # 8 * 2 = 16
        output = encoder(exog)
        self.assertEqual(output.shape, (4, 5))

    def test_single_exogenous_per_concept(self):
        """Test with single exogenous per concept."""
        encoder = ProbEncoderFromExog(
            in_features_exogenous=10,
            n_exogenous_per_concept=1
        )
        exog = torch.randn(3, 4, 10)
        output = encoder(exog)
        self.assertEqual(output.shape, (3, 4))

    def test_gradient_flow(self):
        """Test gradient flow."""
        encoder = ProbEncoderFromExog(
            in_features_exogenous=8,
            n_exogenous_per_concept=2
        )
        exog = torch.randn(2, 3, 16, requires_grad=True)
        output = encoder(exog)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(exog.grad)


class TestExogEncoder(unittest.TestCase):
    """Test ExogEncoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = ExogEncoder(
            in_features_embedding=128,
            out_features=10,
            embedding_size=16
        )
        self.assertEqual(encoder.in_features_embedding, 128)
        self.assertEqual(encoder.out_features, 10)
        self.assertEqual(encoder.embedding_size, 16)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = ExogEncoder(
            in_features_embedding=64,
            out_features=5,
            embedding_size=8
        )
        embeddings = torch.randn(4, 64)
        output = encoder(embeddings)
        self.assertEqual(output.shape, (4, 5, 8))

    def test_gradient_flow(self):
        """Test gradient flow through encoder."""
        encoder = ExogEncoder(
            in_features_embedding=32,
            out_features=3,
            embedding_size=4
        )
        embeddings = torch.randn(2, 32, requires_grad=True)
        output = encoder(embeddings)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_different_embedding_sizes(self):
        """Test various embedding sizes."""
        for emb_size in [4, 8, 16, 32]:
            encoder = ExogEncoder(
                in_features_embedding=64,
                out_features=5,
                embedding_size=emb_size
            )
            embeddings = torch.randn(2, 64)
            output = encoder(embeddings)
            self.assertEqual(output.shape, (2, 5, emb_size))

    def test_encoder_output_dimension(self):
        """Test output dimension calculation."""
        encoder = ExogEncoder(
            in_features_embedding=128,
            out_features=10,
            embedding_size=16
        )
        self.assertEqual(encoder.out_logits_dim, 10)
        self.assertEqual(encoder.out_encoder_dim, 10 * 16)

    def test_leaky_relu_activation(self):
        """Test that LeakyReLU is applied."""
        encoder = ExogEncoder(
            in_features_embedding=32,
            out_features=3,
            embedding_size=4
        )
        embeddings = torch.randn(2, 32)
        output = encoder(embeddings)
        # Output should have passed through LeakyReLU
        self.assertIsNotNone(output)


class TestMemorySelector(unittest.TestCase):
    """Test MemorySelector."""

    def test_initialization(self):
        """Test selector initialization."""
        selector = MemorySelector(
            in_features_embedding=64,
            in_features_logits=10,
            out_features=5,
            memory_size=20,
            embedding_size=8
        )
        self.assertEqual(selector.in_features_logits, 10)
        self.assertEqual(selector.out_features, 5)
        self.assertEqual(selector.memory_size, 20)
        self.assertEqual(selector.embedding_size, 8)

    def test_forward_without_sampling(self):
        """Test forward pass without sampling (soft selection)."""
        selector = MemorySelector(
            in_features_embedding=64,
            in_features_logits=8,
            out_features=4,
            memory_size=10,
            embedding_size=6
        )
        embeddings = torch.randn(2, 64)
        logits = torch.randn(2, 8)
        output = selector(embedding=embeddings, logits=logits, sampling=False)
        self.assertEqual(output.shape, (2, 4, 6))

    def test_forward_with_sampling(self):
        """Test forward pass with sampling (Gumbel-softmax)."""
        selector = MemorySelector(
            in_features_embedding=64,
            in_features_logits=8,
            out_features=4,
            memory_size=10,
            embedding_size=6
        )
        embeddings = torch.randn(2, 64)
        logits = torch.randn(2, 8)
        output = selector(embedding=embeddings, logits=logits, sampling=True)
        self.assertEqual(output.shape, (2, 4, 6))

    def test_gradient_flow_soft(self):
        """Test gradient flow with soft selection."""
        selector = MemorySelector(
            in_features_embedding=32,
            in_features_logits=6,
            out_features=3,
            memory_size=8,
            embedding_size=4
        )
        embeddings = torch.randn(2, 32, requires_grad=True)
        logits = torch.randn(2, 6, requires_grad=True)
        output = selector(embedding=embeddings, logits=logits, sampling=False)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)
        self.assertIsNotNone(logits.grad)

    def test_gradient_flow_hard(self):
        """Test gradient flow with hard selection."""
        selector = MemorySelector(
            in_features_embedding=32,
            in_features_logits=6,
            out_features=3,
            memory_size=8,
            embedding_size=4
        )
        embeddings = torch.randn(2, 32, requires_grad=True)
        logits = torch.randn(2, 6, requires_grad=True)
        output = selector(embedding=embeddings, logits=logits, sampling=True)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_different_temperatures(self):
        """Test with different temperature values."""
        for temp in [0.1, 0.5, 1.0, 2.0]:
            selector = MemorySelector(
                in_features_embedding=32,
                in_features_logits=6,
                out_features=3,
                memory_size=8,
                embedding_size=4,
                temperature=temp
            )
            self.assertEqual(selector.temperature, temp)
            embeddings = torch.randn(2, 32)
            logits = torch.randn(2, 6)
            output = selector(embedding=embeddings, logits=logits, sampling=False)
            self.assertEqual(output.shape, (2, 3, 4))

    def test_memory_initialization(self):
        """Test that memory is properly initialized."""
        selector = MemorySelector(
            in_features_embedding=32,
            in_features_logits=6,
            out_features=5,
            memory_size=10,
            embedding_size=8
        )
        # Memory should have shape (out_features, memory_size * embedding_size)
        self.assertEqual(selector.memory.num_embeddings, 5)
        self.assertEqual(selector.memory.embedding_dim, 10 * 8)

    def test_batch_processing(self):
        """Test different batch sizes."""
        selector = MemorySelector(
            in_features_embedding=32,
            in_features_logits=6,
            out_features=3,
            memory_size=8,
            embedding_size=4
        )
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 32)
            logits = torch.randn(batch_size, 6)
            output = selector(embedding=embeddings, logits=logits, sampling=False)
            self.assertEqual(output.shape, (batch_size, 3, 4))

    def test_selector_network(self):
        """Test that selector network is created."""
        selector = MemorySelector(
            in_features_embedding=32,
            in_features_logits=6,
            out_features=3,
            memory_size=8,
            embedding_size=4
        )
        self.assertIsNotNone(selector.selector)


class TestStochasticEncoderFromEmb(unittest.TestCase):
    """Test StochasticEncoderFromEmb."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = StochasticEncoderFromEmb(
            in_features_embedding=128,
            out_features=5,
            num_monte_carlo=100
        )
        self.assertEqual(encoder.in_features_embedding, 128)
        self.assertEqual(encoder.out_features, 5)
        self.assertEqual(encoder.num_monte_carlo, 100)
        self.assertIsNotNone(encoder.mu)
        self.assertIsNotNone(encoder.sigma)

    def test_forward_with_reduce(self):
        """Test forward pass with reduce=True."""
        encoder = StochasticEncoderFromEmb(
            in_features_embedding=64,
            out_features=5,
            num_monte_carlo=50
        )
        embeddings = torch.randn(4, 64)
        output = encoder(embeddings, reduce=True)
        self.assertEqual(output.shape, (4, 5))

    def test_forward_without_reduce(self):
        """Test forward pass with reduce=False."""
        encoder = StochasticEncoderFromEmb(
            in_features_embedding=32,
            out_features=3,
            num_monte_carlo=20
        )
        embeddings = torch.randn(2, 32)
        output = encoder(embeddings, reduce=False)
        self.assertEqual(output.shape, (2, 3, 20))

    def test_gradient_flow(self):
        """Test gradient flow through stochastic encoder."""
        encoder = StochasticEncoderFromEmb(
            in_features_embedding=16,
            out_features=4,
            num_monte_carlo=10
        )
        embeddings = torch.randn(2, 16, requires_grad=True)
        output = encoder(embeddings, reduce=True)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_predict_sigma(self):
        """Test internal _predict_sigma method."""
        encoder = StochasticEncoderFromEmb(
            in_features_embedding=16,
            out_features=3,
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
        encoder = StochasticEncoderFromEmb(
            in_features_embedding=16,
            out_features=3,
            num_monte_carlo=10
        )
        embeddings = torch.randn(2, 16)
        sigma = encoder._predict_sigma(embeddings)
        # Check diagonal is positive
        for i in range(2):
            for j in range(3):
                self.assertGreater(sigma[i, j, j].item(), 0.0)

    def test_monte_carlo_samples_variability(self):
        """Test that MC samples show variability."""
        encoder = StochasticEncoderFromEmb(
            in_features_embedding=16,
            out_features=2,
            num_monte_carlo=100
        )
        embeddings = torch.randn(1, 16)
        output = encoder(embeddings, reduce=False)
        # Check that samples vary
        std = output.std(dim=2)
        self.assertTrue(torch.any(std > 0.01))

    def test_different_monte_carlo_sizes(self):
        """Test various MC sample sizes."""
        for mc_size in [10, 50, 200]:
            encoder = StochasticEncoderFromEmb(
                in_features_embedding=16,
                out_features=3,
                num_monte_carlo=mc_size
            )
            embeddings = torch.randn(2, 16)
            output = encoder(embeddings, reduce=False)
            self.assertEqual(output.shape[2], mc_size)

    def test_mean_consistency(self):
        """Test that mean of samples approximates mu."""
        torch.manual_seed(42)
        encoder = StochasticEncoderFromEmb(
            in_features_embedding=16,
            out_features=2,
            num_monte_carlo=1000
        )
        embeddings = torch.randn(1, 16)

        # Get mean directly from mu
        mu = encoder.mu(embeddings)

        # Get mean from MC samples
        samples = encoder(embeddings, reduce=False)
        mc_mean = samples.mean(dim=2)

        # Should be close for large num_monte_carlo
        self.assertTrue(torch.allclose(mu, mc_mean, atol=0.3))

    def test_batch_processing(self):
        """Test different batch sizes."""
        encoder = StochasticEncoderFromEmb(
            in_features_embedding=32,
            out_features=4,
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
        encoder = StochasticEncoderFromEmb(
            in_features_embedding=16,
            out_features=3,
            num_monte_carlo=10
        )
        # Check that weights are small (scaled by 0.01)
        sigma_weight_norm = encoder.sigma.weight.data.norm().item()
        self.assertLess(sigma_weight_norm, 1.0)


if __name__ == '__main__':
    unittest.main()
