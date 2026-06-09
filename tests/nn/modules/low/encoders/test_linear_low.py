"""
Tests for LinearLatentToConcept (alias) and LinearEmbeddingToConcept.
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn.modules.low.encoders.linear import LinearLatentToConcept, LinearEmbeddingToConcept


class TestLinearLatentToConcept(unittest.TestCase):
    """Test LinearLatentToConcept (alias for LinearEmbeddingToConcept)."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = LinearLatentToConcept(
            in_embeddings=128,
            out_concepts=10
        )
        self.assertEqual(encoder.in_embeddings, 128)
        self.assertEqual(encoder.out_concepts, 10)
        self.assertIsInstance(encoder.encoder, nn.Linear)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = LinearLatentToConcept(
            in_embeddings=128,
            out_concepts=10
        )
        embeddings = torch.randn(4, 128)
        output = encoder(embeddings)
        self.assertEqual(output.shape, (4, 10))

    def test_gradient_flow(self):
        """Test gradient flow through encoder."""
        encoder = LinearLatentToConcept(
            in_embeddings=64,
            out_concepts=5
        )
        embeddings = torch.randn(2, 64, requires_grad=True)
        output = encoder(embeddings)
        output.sum().backward()
        self.assertIsNotNone(embeddings.grad)

    def test_batch_processing(self):
        """Test different batch sizes."""
        encoder = LinearLatentToConcept(
            in_embeddings=32,
            out_concepts=5
        )
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 32)
            output = encoder(embeddings)
            self.assertEqual(output.shape, (batch_size, 5))

    def test_with_bias_false(self):
        """Test encoder without bias."""
        encoder = LinearLatentToConcept(
            in_embeddings=32,
            out_concepts=5,
            bias=False
        )
        embeddings = torch.randn(2, 32)
        output = encoder(embeddings)
        self.assertEqual(output.shape, (2, 5))


class TestLinearEmbeddingToConcept(unittest.TestCase):
    """Test LinearEmbeddingToConcept with canonical parameter names."""

    def test_initialization(self):
        """Test encoder stores correct attributes."""
        encoder = LinearEmbeddingToConcept(in_embeddings=16, out_concepts=5)
        self.assertEqual(encoder.in_embeddings, 16)
        self.assertEqual(encoder.out_concepts, 5)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = LinearEmbeddingToConcept(in_embeddings=8, out_concepts=3)
        x = torch.randn(4, 8)
        self.assertEqual(encoder(x).shape, (4, 3))

    def test_gradient_flow(self):
        """Test gradient flow."""
        encoder = LinearEmbeddingToConcept(in_embeddings=8, out_concepts=3)
        x = torch.randn(2, 8, requires_grad=True)
        encoder(x).sum().backward()
        self.assertIsNotNone(x.grad)

    def test_single_concept(self):
        """Test with a single output concept."""
        encoder = LinearEmbeddingToConcept(in_embeddings=10, out_concepts=1)
        x = torch.randn(3, 10)
        self.assertEqual(encoder(x).shape, (3, 1))

    def test_is_alias_for_latent(self):
        """LinearLatentToConcept must be the same class."""
        self.assertIs(LinearLatentToConcept, LinearEmbeddingToConcept)


if __name__ == '__main__':
    unittest.main()
