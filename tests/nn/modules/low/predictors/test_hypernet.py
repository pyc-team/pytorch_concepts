"""
Comprehensive tests for HyperlinearConceptEmbeddingToConcept.
"""
import unittest
import torch
from torch_concepts.nn import HyperlinearConceptEmbeddingToConcept


class TestHyperlinearConceptEmbeddingToConcept(unittest.TestCase):
    """Test HyperlinearConceptEmbeddingToConcept."""

    def test_initialization(self):
        """Test hypernetwork predictor initialization."""
        predictor = HyperlinearConceptEmbeddingToConcept(
            in_concepts=10,
            in_embeddings=128,
            hidden_size=64,
        )
        self.assertEqual(predictor.in_concepts, 10)
        self.assertEqual(predictor.in_embeddings, 128)
        self.assertEqual(predictor.hidden_size, 64)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictor = HyperlinearConceptEmbeddingToConcept(
            in_concepts=10,
            in_embeddings=128,
            hidden_size=64,
        )
        concepts = torch.randn(4, 10)
        embeddings = torch.randn(4, 3, 128)
        output = predictor(concepts=concepts, embeddings=embeddings)
        self.assertEqual(output.shape, (4, 3))

    def test_without_bias(self):
        """Test hypernetwork without stochastic bias."""
        predictor = HyperlinearConceptEmbeddingToConcept(
            in_concepts=10,
            in_embeddings=128,
            hidden_size=64,
            use_bias=False,
        )
        concepts = torch.randn(4, 10)
        embeddings = torch.randn(4, 3, 128)
        output = predictor(concepts=concepts, embeddings=embeddings)
        self.assertEqual(output.shape, (4, 3))

    def test_with_bias(self):
        """Test hypernetwork with stochastic bias."""
        predictor = HyperlinearConceptEmbeddingToConcept(
            in_concepts=10,
            in_embeddings=128,
            hidden_size=64,
            use_bias=True,
            init_bias_mean=0.0,
            init_bias_std=0.01,
        )
        concepts = torch.randn(4, 10)
        embeddings = torch.randn(4, 3, 128)
        output = predictor(concepts=concepts, embeddings=embeddings)
        self.assertEqual(output.shape, (4, 3))

    def test_gradient_flow(self):
        """Test gradient flow through hypernetwork."""
        predictor = HyperlinearConceptEmbeddingToConcept(
            in_concepts=8,
            in_embeddings=64,
            hidden_size=32,
        )
        concepts = torch.randn(2, 8, requires_grad=True)
        embeddings = torch.randn(2, 2, 64, requires_grad=True)
        output = predictor(concepts=concepts, embeddings=embeddings)
        output.sum().backward()
        self.assertIsNotNone(concepts.grad)
        self.assertIsNotNone(embeddings.grad)

    def test_sample_adaptive_weights(self):
        """Different embedding inputs should produce different outputs."""
        predictor = HyperlinearConceptEmbeddingToConcept(
            in_concepts=5,
            in_embeddings=32,
            hidden_size=16,
            use_bias=False,
        )
        concepts = torch.ones(1, 5)
        embeddings1 = torch.randn(1, 1, 32)
        embeddings2 = torch.randn(1, 1, 32)
        out1 = predictor(concepts=concepts, embeddings=embeddings1)
        out2 = predictor(concepts=concepts, embeddings=embeddings2)
        self.assertFalse(torch.allclose(out1, out2))

    def test_single_task(self):
        """Test with a single output task."""
        predictor = HyperlinearConceptEmbeddingToConcept(
            in_concepts=6,
            in_embeddings=32,
            hidden_size=16,
            use_bias=False,
        )
        concepts = torch.randn(3, 6)
        embeddings = torch.randn(3, 1, 32)
        output = predictor(concepts=concepts, embeddings=embeddings)
        self.assertEqual(output.shape, (3, 1))


if __name__ == '__main__':
    unittest.main()
