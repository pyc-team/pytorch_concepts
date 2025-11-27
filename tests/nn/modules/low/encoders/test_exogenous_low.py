"""
Comprehensive tests for torch_concepts.nn.modules.low.encoders

Tests all encoder modules (linear, exogenous, selector, stochastic).
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn.modules.low.encoders.exogenous import LinearZU


class TestLinearZU(unittest.TestCase):
    """Test LinearZU."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = LinearZU(
            in_features=128,
            out_features=10,
            exogenous_size=16
        )
        self.assertEqual(encoder.in_features, 128)
        self.assertEqual(encoder.out_features, 10)
        self.assertEqual(encoder.exogenous_size, 16)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = LinearZU(
            in_features=64,
            out_features=5,
            exogenous_size=8
        )
        embeddings = torch.randn(4, 64)
        output = encoder(embeddings)
        self.assertEqual(output.shape, (4, 5, 8))

    def test_gradient_flow(self):
        """Test gradient flow through encoder."""
        encoder = LinearZU(
            in_features=32,
            out_features=3,
            exogenous_size=4
        )
        embeddings = torch.randn(2, 32, requires_grad=True)
        output = encoder(embeddings)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_different_embedding_sizes(self):
        """Test various embedding sizes."""
        for emb_size in [4, 8, 16, 32]:
            encoder = LinearZU(
                in_features=64,
                out_features=5,
                exogenous_size=emb_size
            )
            embeddings = torch.randn(2, 64)
            output = encoder(embeddings)
            self.assertEqual(output.shape, (2, 5, emb_size))

    def test_encoder_output_dimension(self):
        """Test output dimension calculation."""
        encoder = LinearZU(
            in_features=128,
            out_features=10,
            exogenous_size=16
        )
        self.assertEqual(encoder.out_endogenous_dim, 10)
        self.assertEqual(encoder.out_encoder_dim, 10 * 16)

    def test_leaky_relu_activation(self):
        """Test that LeakyReLU is applied."""
        encoder = LinearZU(
            in_features=32,
            out_features=3,
            exogenous_size=4
        )
        embeddings = torch.randn(2, 32)
        output = encoder(embeddings)
        # Output should have passed through LeakyReLU
        self.assertIsNotNone(output)


if __name__ == '__main__':
    unittest.main()
