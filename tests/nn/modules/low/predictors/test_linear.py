"""
Comprehensive tests for torch_concepts.nn.modules.low.predictors

Tests all predictor modules (linear, embedding, hypernet).
"""
import unittest
import torch
from torch_concepts.nn import LinearConceptToConcept


class TestLinearConceptToConcept(unittest.TestCase):
    """Test LinearConceptToConcept."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = LinearConceptToConcept(
            in_concepts=10,
            out_concepts=5
        )
        self.assertEqual(predictor.in_concepts, 10)
        self.assertEqual(predictor.out_concepts, 5)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictor = LinearConceptToConcept(
            in_concepts=10,
            out_concepts=5
        )
        concepts = torch.randn(4, 10)
        output = predictor(concepts=concepts)
        self.assertEqual(output.shape, (4, 5))

    def test_gradient_flow(self):
        """Test gradient flow through predictor."""
        predictor = LinearConceptToConcept(
            in_concepts=8,
            out_concepts=3
        )
        concepts = torch.randn(2, 8, requires_grad=True)
        output = predictor(concepts=concepts)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(concepts.grad)

    def test_custom_activation(self):
        """Test with custom activation function."""
        predictor = LinearConceptToConcept(
            in_concepts=10,
            out_concepts=5,
            activation=torch.tanh
        )
        concepts = torch.randn(2, 10)
        output = predictor(concepts=concepts)
        self.assertEqual(output.shape, (2, 5))

    def test_prune_functionality(self):
        """Test pruning of input features."""
        predictor = LinearConceptToConcept(
            in_concepts=10,
            out_concepts=5
        )
        # Prune to keep only first 5 features
        mask = torch.zeros(10, dtype=torch.bool)
        mask[:5] = True
        predictor.prune(mask)

        # Should now work with 5 input features
        concepts = torch.randn(2, 5)
        output = predictor(concepts=concepts)
        self.assertEqual(output.shape, (2, 5))


if __name__ == '__main__':
    unittest.main()
