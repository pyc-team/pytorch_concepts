"""
Comprehensive tests for torch_concepts.nn.modules.low.predictors

Tests all predictor modules (linear, embedding, hypernet).
"""
import unittest
import torch
from torch_concepts.nn import HyperlinearConceptExogenousToConcept


class TestHyperlinearConceptExogenousToConcept(unittest.TestCase):
    """Test HyperlinearConceptExogenousToConcept."""

    def test_initialization(self):
        """Test hypernetwork predictor initialization."""
        predictor = HyperlinearConceptExogenousToConcept(
            in_concepts=10,
            in_exogenous=128,
            hidden_size=64
        )
        self.assertEqual(predictor.in_concepts, 10)
        self.assertEqual(predictor.in_exogenous, 128)
        self.assertEqual(predictor.hidden_size, 64)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictor = HyperlinearConceptExogenousToConcept(
            in_concepts=10,
            in_exogenous=128,
            hidden_size=64
        )
        concepts = torch.randn(4, 10)
        exogenous = torch.randn(4, 3, 128)
        output = predictor(concepts=concepts, exogenous=exogenous)
        self.assertEqual(output.shape, (4, 3))

    def test_without_bias(self):
        """Test hypernetwork without bias."""
        predictor = HyperlinearConceptExogenousToConcept(
            in_concepts=10,
            in_exogenous=128,
            hidden_size=64,
            use_bias=False
        )
        concepts = torch.randn(4, 10)
        exogenous = torch.randn(4, 3, 128)
        output = predictor(concepts=concepts, exogenous=exogenous)
        self.assertEqual(output.shape, (4, 3))

    def test_gradient_flow(self):
        """Test gradient flow through hypernetwork."""
        predictor = HyperlinearConceptExogenousToConcept(
            in_concepts=8,
            in_exogenous=64,
            hidden_size=32
        )
        concepts = torch.randn(2, 8, requires_grad=True)
        exogenous = torch.randn(2, 2, 64, requires_grad=True)
        output = predictor(concepts=concepts, exogenous=exogenous)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(concepts.grad)
        self.assertIsNotNone(exogenous.grad)

    def test_sample_adaptive_weights(self):
        """Test that different samples get different weights."""
        predictor = HyperlinearConceptExogenousToConcept(
            in_concepts=5,
            in_exogenous=32,
            hidden_size=16
        )
        # Different exogenous features should produce different predictions
        concepts = torch.ones(2, 5)  # Same concepts
        exogenous1 = torch.randn(1, 1, 32)
        exogenous2 = torch.randn(1, 1, 32)

        output1 = predictor(concepts=concepts[:1], exogenous=exogenous1)
        output2 = predictor(concepts=concepts[:1], exogenous=exogenous2)

        # Different exogenous should produce different outputs
        self.assertFalse(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()
