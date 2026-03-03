"""
Comprehensive tests for torch_concepts.nn.modules.low.predictors

Tests all predictor modules (linear, embedding, hypernet).
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn import MixConceptExogegnousToConcept, MixMemoryConceptExogenousToConcept


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


class TestMixMemoryConceptExogenousToConcept(unittest.TestCase):
    """Test MixMemoryConceptExogenousToConcept."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = MixMemoryConceptExogenousToConcept(
            in_concepts=10,
            in_exogenous=5,
            out_concepts=3,
            memory_latent_size=64,
            memory_decoder_hidden_layers=2,
            eps=1e-3,
        )
        self.assertEqual(predictor.in_concepts, 10)
        self.assertEqual(predictor.in_exogenous, 5)
        self.assertEqual(predictor.out_concepts, 3)
        self.assertEqual(predictor.memory_decoder_hidden_layers, 2)
        self.assertEqual(predictor.memory.weight.shape, (3, 64))
        self.assertEqual(predictor.memory_network_shape, (5, 10, 3))

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictor = MixMemoryConceptExogenousToConcept(
            in_concepts=8,
            in_exogenous=4,
            out_concepts=2,
        )
        concepts = torch.randn(4, 8)
        exogenous = torch.softmax(torch.randn(4, 2, 4), dim=-1)
        output = predictor(concepts=concepts, exogenous=exogenous)
        self.assertEqual(output.shape, (4, 2))

    def test_forward_with_optional_flags(self):
        """Test forward pass with include_rec and hard_roles options."""
        predictor = MixMemoryConceptExogenousToConcept(
            in_concepts=6,
            in_exogenous=3,
            out_concepts=2,
        )
        concepts = torch.randn(3, 6)
        exogenous = torch.softmax(torch.randn(3, 2, 3), dim=-1)

        output_rec = predictor(
            concepts=concepts,
            exogenous=exogenous,
            include_rec=True,
            rec_weight=1.0,
        )
        output_hard = predictor(
            concepts=concepts,
            exogenous=exogenous,
            hard_roles=True,
        )
        self.assertEqual(output_rec.shape, (3, 2))
        self.assertEqual(output_hard.shape, (3, 2))

    def test_memory_decoder_hidden_layers_config(self):
        """Test configurable number of hidden layers in memory decoder."""
        predictor_zero = MixMemoryConceptExogenousToConcept(
            in_concepts=4,
            in_exogenous=3,
            out_concepts=2,
            memory_decoder_hidden_layers=0,
        )
        predictor_two = MixMemoryConceptExogenousToConcept(
            in_concepts=4,
            in_exogenous=3,
            out_concepts=2,
            memory_decoder_hidden_layers=2,
        )

        linear_zero = sum(isinstance(layer, torch.nn.Linear) for layer in predictor_zero.memory_decoder)
        linear_two = sum(isinstance(layer, torch.nn.Linear) for layer in predictor_two.memory_decoder)

        self.assertEqual(linear_zero, 1)
        self.assertEqual(linear_two, 3)

    def test_memory_decoder_hidden_layers_validation(self):
        """Test hidden layer argument validation."""
        with self.assertRaises(ValueError):
            MixMemoryConceptExogenousToConcept(
                in_concepts=4,
                in_exogenous=2,
                out_concepts=1,
                memory_decoder_hidden_layers=-1,
            )

    def test_gradient_flow(self):
        """Test gradient flow to exogenous and memory while concepts are detached."""
        predictor = MixMemoryConceptExogenousToConcept(
            in_concepts=5,
            in_exogenous=4,
            out_concepts=2,
        )
        concepts = torch.randn(2, 5, requires_grad=True)
        exogenous = torch.softmax(torch.randn(2, 2, 4), dim=-1).requires_grad_()
        output = predictor(concepts=concepts, exogenous=exogenous)
        loss = output.sum()
        loss.backward()

        self.assertIsNone(concepts.grad)
        self.assertIsNotNone(exogenous.grad)
        self.assertIsNotNone(predictor.memory.weight.grad)


if __name__ == '__main__':
    unittest.main()
