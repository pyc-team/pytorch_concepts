"""
Comprehensive tests for torch_concepts.nn.modules.low.predictors

Tests all predictor modules (linear, embedding, hypernet).
"""
import unittest
import torch
from torch_concepts.nn import HyperLinearCUC


class TestHyperLinearCUC(unittest.TestCase):
    """Test HyperLinearCUC."""

    def test_initialization(self):
        """Test hypernetwork predictor initialization."""
        predictor = HyperLinearCUC(
            in_features_endogenous=10,
            in_features_exogenous=128,
            embedding_size=64
        )
        self.assertEqual(predictor.in_features_endogenous, 10)
        self.assertEqual(predictor.in_features_exogenous, 128)
        self.assertEqual(predictor.embedding_size, 64)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictor = HyperLinearCUC(
            in_features_endogenous=10,
            in_features_exogenous=128,
            embedding_size=64
        )
        concept_endogenous = torch.randn(4, 10)
        exogenous = torch.randn(4, 3, 128)
        output = predictor(endogenous=concept_endogenous, exogenous=exogenous)
        self.assertEqual(output.shape, (4, 3))

    def test_without_bias(self):
        """Test hypernetwork without bias."""
        predictor = HyperLinearCUC(
            in_features_endogenous=10,
            in_features_exogenous=128,
            embedding_size=64,
            use_bias=False
        )
        concept_endogenous = torch.randn(4, 10)
        exogenous = torch.randn(4, 3, 128)
        output = predictor(endogenous=concept_endogenous, exogenous=exogenous)
        self.assertEqual(output.shape, (4, 3))

    def test_gradient_flow(self):
        """Test gradient flow through hypernetwork."""
        predictor = HyperLinearCUC(
            in_features_endogenous=8,
            in_features_exogenous=64,
            embedding_size=32
        )
        concept_endogenous = torch.randn(2, 8, requires_grad=True)
        exogenous = torch.randn(2, 2, 64, requires_grad=True)
        output = predictor(endogenous=concept_endogenous, exogenous=exogenous)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(concept_endogenous.grad)
        self.assertIsNotNone(exogenous.grad)

    def test_custom_activation(self):
        """Test with custom activation."""
        predictor = HyperLinearCUC(
            in_features_endogenous=10,
            in_features_exogenous=128,
            embedding_size=64,
            in_activation=torch.sigmoid
        )
        concept_endogenous = torch.randn(2, 10)
        exogenous = torch.randn(2, 3, 128)
        output = predictor(endogenous=concept_endogenous, exogenous=exogenous)
        self.assertEqual(output.shape, (2, 3))

    def test_sample_adaptive_weights(self):
        """Test that different samples get different weights."""
        predictor = HyperLinearCUC(
            in_features_endogenous=5,
            in_features_exogenous=32,
            embedding_size=16
        )
        # Different exogenous features should produce different predictions
        concept_endogenous = torch.ones(2, 5)  # Same concepts
        exogenous1 = torch.randn(1, 1, 32)
        exogenous2 = torch.randn(1, 1, 32)

        output1 = predictor(endogenous=concept_endogenous[:1], exogenous=exogenous1)
        output2 = predictor(endogenous=concept_endogenous[:1], exogenous=exogenous2)

        # Different exogenous should produce different outputs
        self.assertFalse(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()
