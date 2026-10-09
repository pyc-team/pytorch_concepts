"""
Tests for torch_concepts.nn.modules.low.encoders.mlp
"""
import unittest
import torch

from torch_concepts import Annotations
from torch_concepts.nn import LazyConstructor, MLPEmbeddingToConcept


class TestMLPEmbeddingToConcept(unittest.TestCase):
    """Test MLPEmbeddingToConcept."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = MLPEmbeddingToConcept(
            in_embeddings=128,
            out_concepts=10,
            hidden_size=64,
        )
        self.assertEqual(encoder.in_embeddings, 128)
        self.assertEqual(encoder.out_concepts, 10)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = MLPEmbeddingToConcept(
            in_embeddings=16,
            out_concepts=5,
            hidden_size=32,
            n_layers=2,
        )
        embeddings = torch.randn(4, 16)
        output = encoder(embeddings=embeddings)
        self.assertEqual(output.shape, (4, 5))

    def test_leading_dims(self):
        """Extra leading (batch-like) axes pass through untouched."""
        encoder = MLPEmbeddingToConcept(
            in_embeddings=16, out_concepts=5, hidden_size=32,
        )
        self.assertEqual(encoder(torch.randn(3, 4, 16)).shape, (3, 4, 5))

    def test_annotations_are_sized(self):
        """Annotations are accepted like any other concept layer's sizes."""
        axis = Annotations(labels=['c1', 'c2'], cardinalities=[1, 3])
        encoder = MLPEmbeddingToConcept(
            in_embeddings=16, out_concepts=axis, hidden_size=8,
        )
        self.assertEqual(encoder(torch.randn(2, 16)).shape, (2, axis.size))

    def test_gradient_flow(self):
        """Test gradient flow through encoder."""
        encoder = MLPEmbeddingToConcept(
            in_embeddings=8, out_concepts=3, hidden_size=8,
        )
        embeddings = torch.randn(2, 8, requires_grad=True)
        encoder(embeddings=embeddings).sum().backward()
        self.assertIsNotNone(embeddings.grad)

    def test_lazy_construction(self):
        """A CPD sizes it from its parents and its variable."""
        module = LazyConstructor(MLPEmbeddingToConcept, hidden_size=8).build(
            out_concepts=3, in_concepts=None, in_embeddings=16,
        )
        self.assertIsInstance(module, MLPEmbeddingToConcept)
        self.assertEqual(module(torch.randn(2, 16)).shape, (2, 3))


if __name__ == '__main__':
    unittest.main()
