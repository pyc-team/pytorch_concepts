"""
Comprehensive tests for torch_concepts.nn.modules.low.predictors

Tests all predictor modules (linear, embedding, hypernet).
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn import MixCUC


class TestMixCUC(unittest.TestCase):
    """Test MixCUC."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = MixCUC(
            in_features_endogenous=10,
            in_features_exogenous=20,
            out_features=3
        )
        self.assertEqual(predictor.in_features_endogenous, 10)
        self.assertEqual(predictor.in_features_exogenous, 20)
        self.assertEqual(predictor.out_features, 3)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictor = MixCUC(
            in_features_endogenous=10,
            in_features_exogenous=10,
            out_features=3
        )
        concept_endogenous = torch.randn(4, 10)
        exogenous = torch.randn(4, 10, 20)
        output = predictor(endogenous=concept_endogenous, exogenous=exogenous)
        self.assertEqual(output.shape, (4, 3))

    def test_with_cardinalities(self):
        """Test with concept cardinalities."""
        predictor = MixCUC(
            in_features_endogenous=10,
            in_features_exogenous=20,
            out_features=3,
            cardinalities=[3, 4, 3]
        )
        concept_endogenous = torch.randn(4, 10)
        exogenous = torch.randn(4, 10, 20)
        output = predictor(endogenous=concept_endogenous, exogenous=exogenous)
        self.assertEqual(output.shape, (4, 3))

    def test_gradient_flow(self):
        """Test gradient flow."""
        predictor = MixCUC(
            in_features_endogenous=8,
            in_features_exogenous=16,
            out_features=2
        )
        concept_endogenous = torch.randn(2, 8, requires_grad=True)
        # Exogenous should have shape (batch, n_concepts, emb_size)
        # where emb_size = in_features_exogenous * 2 (for no cardinalities case)
        exogenous = torch.randn(2, 8, 32, requires_grad=True)  # 32 = 16 * 2
        output = predictor(endogenous=concept_endogenous, exogenous=exogenous)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(concept_endogenous.grad)
        self.assertIsNotNone(exogenous.grad)

    def test_even_exogenous_requirement(self):
        """Test that exogenous features must be even."""
        with self.assertRaises(AssertionError):
            MixCUC(
                in_features_endogenous=10,
                in_features_exogenous=15,  # Odd number
                out_features=3
            )


if __name__ == '__main__':
    unittest.main()
