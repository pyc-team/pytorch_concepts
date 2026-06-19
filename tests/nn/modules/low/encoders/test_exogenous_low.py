"""Tests for LinearEmbeddingEncoder."""
import unittest
import torch
from torch_concepts.nn.modules.low.dense_layers import LinearEmbeddingEncoder


class TestLinearEmbeddingEncoder(unittest.TestCase):
    """Test LinearEmbeddingEncoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = LinearEmbeddingEncoder(
            in_features=128,
            n_embeddings=10,
            out_features=16,
        )
        self.assertEqual(encoder.out_shape, (10, 16))

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = LinearEmbeddingEncoder(
            in_features=64,
            n_embeddings=5,
            out_features=8,
        )
        x = torch.randn(4, 64)
        output = encoder(x)
        self.assertEqual(output.shape, (4, 5, 8))

    def test_gradient_flow(self):
        """Test gradient flow through encoder."""
        encoder = LinearEmbeddingEncoder(
            in_features=32,
            n_embeddings=3,
            out_features=4,
        )
        x = torch.randn(2, 32, requires_grad=True)
        output = encoder(x)
        output.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_different_embedding_sizes(self):
        """Test various out_features values."""
        for emb_size in [4, 8, 16, 32]:
            encoder = LinearEmbeddingEncoder(
                in_features=64,
                n_embeddings=5,
                out_features=emb_size,
            )
            x = torch.randn(2, 64)
            output = encoder(x)
            self.assertEqual(output.shape, (2, 5, emb_size))

    def test_single_embedding(self):
        """Test default n_embeddings=1."""
        encoder = LinearEmbeddingEncoder(in_features=32, out_features=8)
        x = torch.randn(3, 32)
        self.assertEqual(encoder(x).shape, (3, 1, 8))

    def test_out_shape_attribute(self):
        """out_shape should be (n_embeddings, out_features)."""
        encoder = LinearEmbeddingEncoder(in_features=128, n_embeddings=10, out_features=16)
        self.assertEqual(encoder.out_shape, (10, 16))


if __name__ == '__main__':
    unittest.main()
