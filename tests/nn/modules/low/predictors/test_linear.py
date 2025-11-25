"""
Comprehensive tests for torch_concepts.nn.modules.low.predictors

Tests all predictor modules (linear, embedding, hypernet).
"""
import unittest
import torch
from torch_concepts.nn import LinearCC


class TestLinearCC(unittest.TestCase):
    """Test LinearCC."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = LinearCC(
            in_features_endogenous=10,
            out_features=5
        )
        self.assertEqual(predictor.in_features_endogenous, 10)
        self.assertEqual(predictor.out_features, 5)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictor = LinearCC(
            in_features_endogenous=10,
            out_features=5
        )
        endogenous = torch.randn(4, 10)
        output = predictor(endogenous)
        self.assertEqual(output.shape, (4, 5))

    def test_gradient_flow(self):
        """Test gradient flow through predictor."""
        predictor = LinearCC(
            in_features_endogenous=8,
            out_features=3
        )
        endogenous = torch.randn(2, 8, requires_grad=True)
        output = predictor(endogenous)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(endogenous.grad)

    def test_custom_activation(self):
        """Test with custom activation function."""
        predictor = LinearCC(
            in_features_endogenous=10,
            out_features=5,
            in_activation=torch.tanh
        )
        endogenous = torch.randn(2, 10)
        output = predictor(endogenous)
        self.assertEqual(output.shape, (2, 5))

    def test_prune_functionality(self):
        """Test pruning of input features."""
        predictor = LinearCC(
            in_features_endogenous=10,
            out_features=5
        )
        # Prune to keep only first 5 features
        mask = torch.zeros(10, dtype=torch.bool)
        mask[:5] = True
        predictor.prune(mask)

        # Should now work with 5 input features
        endogenous = torch.randn(2, 5)
        output = predictor(endogenous)
        self.assertEqual(output.shape, (2, 5))


if __name__ == '__main__':
    unittest.main()
