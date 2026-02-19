"""
Comprehensive tests for torch_concepts.nn.modules.low.predictors

Tests all predictor modules (linear, embedding, hypernet).
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn import MixConceptExogegnousToConcept


class TestMixConceptExogegnousToConcept(unittest.TestCase):
    """Test MixConceptExogegnousToConcept."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = MixConceptExogegnousToConcept(
            in_concepts=10,
            in_exogenous=20,
            out_concepts=3,
            cardinalities=[1]*10
        )
        self.assertEqual(predictor.in_concepts, 10)
        self.assertEqual(predictor.in_exogenous, 20)
        self.assertEqual(predictor.out_concepts, 3)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictor = MixConceptExogegnousToConcept(
            in_concepts=10,
            in_exogenous=20,
            out_concepts=3,
            cardinalities=[1]*10
        )
        concepts = torch.randn(4, 10)
        exogenous = torch.randn(4, 10, 20)
        output = predictor(concepts=concepts, exogenous=exogenous)
        self.assertEqual(output.shape, (4, 3))

    def test_with_cardinalities(self):
        """Test with concept cardinalities."""
        predictor = MixConceptExogegnousToConcept(
            in_concepts=10,
            in_exogenous=20,
            out_concepts=3,
            cardinalities=[3, 4, 3]
        )
        concepts = torch.randn(4, 10)
        exogenous = torch.randn(4, 10, 20)
        output = predictor(concepts=concepts, exogenous=exogenous)
        self.assertEqual(output.shape, (4, 3))

    def test_gradient_flow(self):
        """Test gradient flow."""
        predictor = MixConceptExogegnousToConcept(
            in_concepts=8,
            in_exogenous=16,
            out_concepts=2,
            cardinalities=[1]*8
        )
        concepts = torch.randn(2, 8, requires_grad=True)
        exogenous = torch.randn(2, 8, 16, requires_grad=True)
        output = predictor(concepts=concepts, exogenous=exogenous)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(concepts.grad)
        self.assertIsNotNone(exogenous.grad)

    def test_even_exogenous_requirement(self):
        """Test that exogenous features must be even."""
        with self.assertRaises(AssertionError):
            MixConceptExogegnousToConcept(
                in_concepts=10,
                in_exogenous=15,  # Odd number
                out_concepts=3,
                cardinalities=[1]*10
            )


if __name__ == '__main__':
    unittest.main()
