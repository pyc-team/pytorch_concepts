"""
Comprehensive tests for MixConceptEmbeddingToConcept and MixSumConceptEmbeddingToConcept.
"""
import unittest
import torch
from torch_concepts.nn import MixConceptEmbeddingToConcept, MixSumConceptEmbeddingToConcept


class TestMixConceptEmbeddingToConcept(unittest.TestCase):
    """Test MixConceptEmbeddingToConcept."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = MixConceptEmbeddingToConcept(
            in_concepts=10,
            in_embeddings=20,
            out_concepts=3,
            cardinalities=[1] * 10,
        )
        self.assertEqual(predictor.in_concepts, 10)
        self.assertEqual(predictor.in_embeddings, 20)
        self.assertEqual(predictor.out_concepts, 3)

    def test_forward_shape_all_binary(self):
        """Test forward pass output shape with all binary concepts."""
        predictor = MixConceptEmbeddingToConcept(
            in_concepts=10,
            in_embeddings=20,
            out_concepts=3,
            cardinalities=[1] * 10,
        )
        concepts = torch.randn(4, 10)
        embeddings = torch.randn(4, 10, 20)
        output = predictor(concepts=concepts, embeddings=embeddings)
        self.assertEqual(output.shape, (4, 3))

    def test_forward_shape_categorical(self):
        """Test forward pass with categorical concepts."""
        predictor = MixConceptEmbeddingToConcept(
            in_concepts=10,
            in_embeddings=20,
            out_concepts=3,
            cardinalities=[3, 4, 3],
        )
        concepts = torch.randn(4, 10)
        embeddings = torch.randn(4, 10, 20)
        output = predictor(concepts=concepts, embeddings=embeddings)
        self.assertEqual(output.shape, (4, 3))

    def test_forward_shape_mixed(self):
        """Test with a mix of binary and categorical concepts."""
        # 1 binary + 1 categorical(3) + 2 binary + 1 categorical(4) = 10
        predictor = MixConceptEmbeddingToConcept(
            in_concepts=10,
            in_embeddings=16,
            out_concepts=5,
            cardinalities=[1, 3, 1, 1, 4],
        )
        concepts = torch.randn(2, 10)
        embeddings = torch.randn(2, 10, 16)
        output = predictor(concepts=concepts, embeddings=embeddings)
        self.assertEqual(output.shape, (2, 5))

    def test_cardinalities_must_sum_to_in_concepts(self):
        """cardinalities that don't sum to in_concepts should raise."""
        with self.assertRaises(AssertionError):
            MixConceptEmbeddingToConcept(
                in_concepts=10,
                in_embeddings=20,
                out_concepts=3,
                cardinalities=[3, 3],  # sums to 6, not 10
            )

    def test_requires_cardinalities(self):
        """None cardinalities should raise ValueError."""
        with self.assertRaises((ValueError, TypeError)):
            MixConceptEmbeddingToConcept(
                in_concepts=10,
                in_embeddings=20,
                out_concepts=3,
                cardinalities=None,
            )

    def test_gradient_flow(self):
        """Test gradient flow through predictor."""
        predictor = MixConceptEmbeddingToConcept(
            in_concepts=8,
            in_embeddings=16,
            out_concepts=2,
            cardinalities=[1] * 8,
        )
        concepts = torch.randn(2, 8, requires_grad=True)
        embeddings = torch.randn(2, 8, 16, requires_grad=True)
        output = predictor(concepts=concepts, embeddings=embeddings)
        output.sum().backward()
        self.assertIsNotNone(concepts.grad)
        self.assertIsNotNone(embeddings.grad)

    def test_predictor_is_linear(self):
        """predictor should be a plain nn.Linear (no wrapping Sequential)."""
        import torch.nn as nn
        predictor = MixConceptEmbeddingToConcept(
            in_concepts=6,
            in_embeddings=10,
            out_concepts=3,
            cardinalities=[2, 2, 2],
        )
        self.assertIsInstance(predictor.predictor, nn.Linear)


class TestMixSumConceptEmbeddingToConcept(unittest.TestCase):
    """Test MixSumConceptEmbeddingToConcept."""

    def test_initialization_with_cardinalities(self):
        """Test initialization with explicit cardinalities."""
        predictor = MixSumConceptEmbeddingToConcept(
            in_concepts=10,
            in_embeddings=20,
            out_concepts=3,
            cardinalities=[3, 4, 3],
        )
        self.assertEqual(predictor.in_concepts, 10)
        self.assertEqual(predictor.out_concepts, 3)

    def test_initialization_defaults_all_binary(self):
        """Default cardinalities=[1]*in_concepts for all-binary case."""
        predictor = MixSumConceptEmbeddingToConcept(
            in_concepts=8,
            in_embeddings=16,
            out_concepts=4,
        )
        self.assertEqual(predictor.cardinalities, [1] * 8)

    def test_forward_shape(self):
        """Test output shape."""
        predictor = MixSumConceptEmbeddingToConcept(
            in_concepts=10,
            in_embeddings=20,
            out_concepts=3,
            cardinalities=[3, 4, 3],
        )
        concepts = torch.randn(4, 10)
        embeddings = torch.randn(4, 10, 20)
        output = predictor(concepts=concepts, embeddings=embeddings)
        self.assertEqual(output.shape, (4, 3))

    def test_forward_shape_all_binary(self):
        """Test output shape with default all-binary cardinalities."""
        predictor = MixSumConceptEmbeddingToConcept(
            in_concepts=6,
            in_embeddings=12,
            out_concepts=2,
        )
        concepts = torch.randn(3, 6)
        embeddings = torch.randn(3, 6, 12)
        output = predictor(concepts=concepts, embeddings=embeddings)
        self.assertEqual(output.shape, (3, 2))

    def test_predictor_is_linear(self):
        """predictor should be a plain nn.Linear."""
        import torch.nn as nn
        predictor = MixSumConceptEmbeddingToConcept(
            in_concepts=4,
            in_embeddings=8,
            out_concepts=2,
        )
        self.assertIsInstance(predictor.predictor, nn.Linear)

    def test_group_count_invariance(self):
        """Sum aggregation: predictor weight shape depends only on in_embeddings."""
        p1 = MixSumConceptEmbeddingToConcept(
            in_concepts=4, in_embeddings=8, out_concepts=2, cardinalities=[1]*4
        )
        p2 = MixSumConceptEmbeddingToConcept(
            in_concepts=6, in_embeddings=8, out_concepts=2, cardinalities=[2, 2, 2]
        )
        self.assertEqual(
            p1.predictor.weight.shape,
            p2.predictor.weight.shape,
        )

    def test_gradient_flow(self):
        """Test gradient flow."""
        predictor = MixSumConceptEmbeddingToConcept(
            in_concepts=6,
            in_embeddings=10,
            out_concepts=2,
            cardinalities=[2, 2, 2],
        )
        concepts = torch.randn(2, 6, requires_grad=True)
        embeddings = torch.randn(2, 6, 10, requires_grad=True)
        output = predictor(concepts=concepts, embeddings=embeddings)
        output.sum().backward()
        self.assertIsNotNone(concepts.grad)
        self.assertIsNotNone(embeddings.grad)


if __name__ == '__main__':
    unittest.main()
