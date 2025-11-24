"""
Comprehensive tests for torch_concepts.nn.modules.low.predictors

Tests all predictor modules (linear, embedding, hypernet).
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn import ProbPredictor
from torch_concepts.nn import MixProbExogPredictor
from torch_concepts.nn import HyperLinearPredictor


class TestProbPredictor(unittest.TestCase):
    """Test ProbPredictor."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = ProbPredictor(
            in_features_endogenous=10,
            out_features=5
        )
        self.assertEqual(predictor.in_features_endogenous, 10)
        self.assertEqual(predictor.out_features, 5)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictor = ProbPredictor(
            in_features_endogenous=10,
            out_features=5
        )
        endogenous = torch.randn(4, 10)
        output = predictor(endogenous)
        self.assertEqual(output.shape, (4, 5))

    def test_gradient_flow(self):
        """Test gradient flow through predictor."""
        predictor = ProbPredictor(
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
        predictor = ProbPredictor(
            in_features_endogenous=10,
            out_features=5,
            in_activation=torch.tanh
        )
        endogenous = torch.randn(2, 10)
        output = predictor(endogenous)
        self.assertEqual(output.shape, (2, 5))

    def test_prune_functionality(self):
        """Test pruning of input features."""
        predictor = ProbPredictor(
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


class TestMixProbExogPredictor(unittest.TestCase):
    """Test MixProbExogPredictor."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = MixProbExogPredictor(
            in_features_endogenous=10,
            in_features_exogenous=20,
            out_features=3
        )
        self.assertEqual(predictor.in_features_endogenous, 10)
        self.assertEqual(predictor.in_features_exogenous, 20)
        self.assertEqual(predictor.out_features, 3)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictor = MixProbExogPredictor(
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
        predictor = MixProbExogPredictor(
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
        predictor = MixProbExogPredictor(
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
            MixProbExogPredictor(
                in_features_endogenous=10,
                in_features_exogenous=15,  # Odd number
                out_features=3
            )


class TestHyperLinearPredictor(unittest.TestCase):
    """Test HyperLinearPredictor."""

    def test_initialization(self):
        """Test hypernetwork predictor initialization."""
        predictor = HyperLinearPredictor(
            in_features_endogenous=10,
            in_features_exogenous=128,
            embedding_size=64
        )
        self.assertEqual(predictor.in_features_endogenous, 10)
        self.assertEqual(predictor.in_features_exogenous, 128)
        self.assertEqual(predictor.embedding_size, 64)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictor = HyperLinearPredictor(
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
        predictor = HyperLinearPredictor(
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
        predictor = HyperLinearPredictor(
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
        predictor = HyperLinearPredictor(
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
        predictor = HyperLinearPredictor(
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
