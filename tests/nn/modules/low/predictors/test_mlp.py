"""
Tests for torch_concepts.nn.modules.low.predictors.mlp
"""
import unittest
import torch

from torch_concepts import Annotations
from torch_concepts.nn import LazyConstructor, MLPConceptToConcept


class TestMLPConceptToConcept(unittest.TestCase):
    """Test MLPConceptToConcept."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = MLPConceptToConcept(
            in_concepts=10,
            out_concepts=5,
            hidden_size=16,
        )
        self.assertEqual(predictor.in_concepts, 10)
        self.assertEqual(predictor.out_concepts, 5)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictor = MLPConceptToConcept(
            in_concepts=10,
            out_concepts=5,
            hidden_size=16,
            n_layers=2,
        )
        concepts = torch.randn(4, 10)
        output = predictor(concepts=concepts)
        self.assertEqual(output.shape, (4, 5))

    def test_leading_dims(self):
        """Extra leading (batch-like) axes pass through untouched."""
        predictor = MLPConceptToConcept(
            in_concepts=10, out_concepts=5, hidden_size=16,
        )
        self.assertEqual(predictor(torch.randn(3, 4, 10)).shape, (3, 4, 5))

    def test_annotations_are_sized(self):
        """Annotations are accepted like any other concept layer's sizes."""
        axis = Annotations(labels=['c1', 'c2'], cardinalities=[1, 3])
        predictor = MLPConceptToConcept(
            in_concepts=axis, out_concepts=axis, hidden_size=8,
        )
        self.assertEqual(predictor(torch.randn(2, axis.size)).shape, (2, axis.size))

    def test_gradient_flow(self):
        """Test gradient flow through predictor."""
        predictor = MLPConceptToConcept(
            in_concepts=8, out_concepts=3, hidden_size=8,
        )
        concepts = torch.randn(2, 8, requires_grad=True)
        predictor(concepts=concepts).sum().backward()
        self.assertIsNotNone(concepts.grad)

    def test_lazy_construction(self):
        """A CPD sizes it from its parents and its variable."""
        module = LazyConstructor(MLPConceptToConcept, hidden_size=8).build(
            out_concepts=3, in_concepts=10, in_embeddings=None,
        )
        self.assertIsInstance(module, MLPConceptToConcept)
        self.assertEqual(module(torch.randn(2, 10)).shape, (2, 3))

    def test_prune_not_supported(self):
        """Pruning is a linear-layer operation; the base says so."""
        predictor = MLPConceptToConcept(
            in_concepts=4, out_concepts=2, hidden_size=8,
        )
        with self.assertRaises(NotImplementedError):
            predictor.prune(torch.ones(4, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
