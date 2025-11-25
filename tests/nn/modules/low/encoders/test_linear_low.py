"""
Comprehensive tests for torch_concepts.nn.modules.low.encoders

Tests all encoder modules (linear, exogenous, selector, stochastic).
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn.modules.low.encoders.linear import LinearZC, LinearUC


class TestLinearZC(unittest.TestCase):
    """Test LinearZC."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = LinearZC(
            in_features=128,
            out_features=10
        )
        self.assertEqual(encoder.in_features, 128)
        self.assertEqual(encoder.out_features, 10)
        self.assertIsInstance(encoder.encoder, nn.Sequential)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = LinearZC(
            in_features=128,
            out_features=10
        )
        embeddings = torch.randn(4, 128)
        output = encoder(embeddings)
        self.assertEqual(output.shape, (4, 10))

    def test_gradient_flow(self):
        """Test gradient flow through encoder."""
        encoder = LinearZC(
            in_features=64,
            out_features=5
        )
        embeddings = torch.randn(2, 64, requires_grad=True)
        output = encoder(embeddings)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_batch_processing(self):
        """Test different batch sizes."""
        encoder = LinearZC(
            in_features=32,
            out_features=5
        )
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 32)
            output = encoder(embeddings)
            self.assertEqual(output.shape, (batch_size, 5))

    def test_with_bias_false(self):
        """Test encoder without bias."""
        encoder = LinearZC(
            in_features=32,
            out_features=5,
            bias=False
        )
        embeddings = torch.randn(2, 32)
        output = encoder(embeddings)
        self.assertEqual(output.shape, (2, 5))


class TestLinearUC(unittest.TestCase):
    """Test LinearUC."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = LinearUC(
            in_features_exogenous=16,
            n_exogenous_per_concept=2
        )
        self.assertEqual(encoder.n_exogenous_per_concept, 2)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = LinearUC(
            in_features_exogenous=8,
            n_exogenous_per_concept=2
        )
        # Input shape: (batch, concepts, in_features * n_exogenous_per_concept)
        exog = torch.randn(4, 5, 16)  # 8 * 2 = 16
        output = encoder(exog)
        self.assertEqual(output.shape, (4, 5))

    def test_single_exogenous_per_concept(self):
        """Test with single exogenous per concept."""
        encoder = LinearUC(
            in_features_exogenous=10,
            n_exogenous_per_concept=1
        )
        exog = torch.randn(3, 4, 10)
        output = encoder(exog)
        self.assertEqual(output.shape, (3, 4))

    def test_gradient_flow(self):
        """Test gradient flow."""
        encoder = LinearUC(
            in_features_exogenous=8,
            n_exogenous_per_concept=2
        )
        exog = torch.randn(2, 3, 16, requires_grad=True)
        output = encoder(exog)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(exog.grad)


if __name__ == '__main__':
    unittest.main()
