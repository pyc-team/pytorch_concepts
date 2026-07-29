"""Comprehensive tests for torch_concepts.nn.modules.low.predictors."""
import unittest

import torch

from torch_concepts.nn import (
    RuleMemory,
    RuleReconstructionPredictor,
    RuleTaskPredictor,
    MixConceptExogegnousToConcept,
)


class TestMixConceptExogegnousToConcept(unittest.TestCase):
    """Test MixConceptExogegnousToConcept."""

    def test_initialization(self):
        predictor = MixConceptExogegnousToConcept(
            in_concepts=10,
            in_exogenous=20,
            out_concepts=3,
            cardinalities=[1] * 10,
        )
        self.assertEqual(predictor.in_concepts, 10)
        self.assertEqual(predictor.in_exogenous, 20)
        self.assertEqual(predictor.out_concepts, 3)

    def test_forward_shape(self):
        predictor = MixConceptExogegnousToConcept(
            in_concepts=10,
            in_exogenous=20,
            out_concepts=3,
            cardinalities=[1] * 10,
        )
        concepts = torch.randn(4, 10)
        exogenous = torch.randn(4, 10, 20)
        output = predictor(concepts=concepts, exogenous=exogenous)
        self.assertEqual(output.shape, (4, 3))

    def test_with_cardinalities(self):
        predictor = MixConceptExogegnousToConcept(
            in_concepts=10,
            in_exogenous=20,
            out_concepts=3,
            cardinalities=[3, 4, 3],
        )
        concepts = torch.randn(4, 10)
        exogenous = torch.randn(4, 10, 20)
        output = predictor(concepts=concepts, exogenous=exogenous)
        self.assertEqual(output.shape, (4, 3))

    def test_gradient_flow(self):
        predictor = MixConceptExogegnousToConcept(
            in_concepts=8,
            in_exogenous=16,
            out_concepts=2,
            cardinalities=[1] * 8,
        )
        concepts = torch.randn(2, 8, requires_grad=True)
        exogenous = torch.randn(2, 8, 16, requires_grad=True)
        output = predictor(concepts=concepts, exogenous=exogenous)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(concepts.grad)
        self.assertIsNotNone(exogenous.grad)


class TestRuleMemory(unittest.TestCase):
    def test_initialization(self):
        memory = RuleMemory(n_tasks=3, n_rules=5, n_concepts=10, latent_size=64, hidden_layers=2)
        self.assertEqual(memory.shape, (3, 5, 10, 3))
        self.assertEqual(memory.memory.weight.shape, (3, 64))

    def test_forward_shape(self):
        memory = RuleMemory(n_tasks=2, n_rules=4, n_concepts=6)
        roles = memory()
        self.assertEqual(roles.shape, (2, 4, 6, 3))
        self.assertTrue(torch.all((roles >= 0) & (roles <= 1)))
        self.assertTrue(torch.allclose(roles.sum(dim=-1), torch.ones_like(roles.sum(dim=-1))))

    def test_hidden_layers_config(self):
        memory_zero = RuleMemory(n_tasks=2, n_rules=3, n_concepts=4, hidden_layers=0)
        memory_two = RuleMemory(n_tasks=2, n_rules=3, n_concepts=4, hidden_layers=2)
        linear_zero = sum(isinstance(layer, torch.nn.Linear) for layer in memory_zero.decoder)
        linear_two = sum(isinstance(layer, torch.nn.Linear) for layer in memory_two.decoder)
        self.assertEqual(linear_zero, 1)
        self.assertEqual(linear_two, 3)

    def test_gradient_flow(self):
        memory = RuleMemory(n_tasks=2, n_rules=3, n_concepts=4)
        loss = memory().sum()
        loss.backward()
        self.assertIsNotNone(memory.memory.weight.grad)


class TestRuleTaskPredictor(unittest.TestCase):
    def test_forward_shape(self):
        predictor = RuleTaskPredictor(in_concepts=6, in_exogenous=3, out_concepts=2)
        concepts = torch.rand(4, 6)
        selector = torch.softmax(torch.randn(4, 2, 3), dim=-1)
        roles = torch.softmax(torch.randn(2, 3, 6, 3), dim=-1)
        output = predictor(concepts=concepts, selector=selector, roles=roles)
        self.assertEqual(output.shape, (4, 2))

    def test_gradient_flow_detaches_concepts(self):
        predictor = RuleTaskPredictor(in_concepts=5, in_exogenous=4, out_concepts=2)
        concepts = torch.rand(2, 5, requires_grad=True)
        selector = torch.softmax(torch.randn(2, 2, 4), dim=-1).requires_grad_()
        roles = torch.softmax(torch.randn(2, 4, 5, 3), dim=-1).requires_grad_()
        output = predictor(concepts=concepts, selector=selector, roles=roles)
        output.sum().backward()
        self.assertIsNone(concepts.grad)
        self.assertIsNotNone(selector.grad)
        self.assertIsNotNone(roles.grad)


class TestRuleReconstructionPredictor(unittest.TestCase):
    def test_forward_shape(self):
        predictor = RuleReconstructionPredictor(in_concepts=6, in_exogenous=3, out_concepts=2, rec_weight=0.5)
        concepts = torch.rand(4, 6)
        selector = torch.softmax(torch.randn(4, 2, 3), dim=-1)
        roles = torch.softmax(torch.randn(2, 3, 6, 3), dim=-1)
        output = predictor(concepts=concepts, selector=selector, roles=roles)
        self.assertEqual(output.shape, (4, 2))

    def test_rec_weight_changes_output(self):
        concepts = torch.rand(3, 4)
        selector = torch.softmax(torch.randn(3, 2, 2), dim=-1)
        roles = torch.softmax(torch.randn(2, 2, 4, 3), dim=-1)
        low = RuleReconstructionPredictor(in_concepts=4, in_exogenous=2, out_concepts=2, rec_weight=0.0)
        high = RuleReconstructionPredictor(in_concepts=4, in_exogenous=2, out_concepts=2, rec_weight=1.0)
        out_low = low(concepts=concepts, selector=selector, roles=roles)
        out_high = high(concepts=concepts, selector=selector, roles=roles)
        self.assertFalse(torch.allclose(out_low, out_high))

    def test_gradient_flow_detaches_concepts(self):
        predictor = RuleReconstructionPredictor(in_concepts=5, in_exogenous=4, out_concepts=2)
        concepts = torch.rand(2, 5, requires_grad=True)
        selector = torch.softmax(torch.randn(2, 2, 4), dim=-1).requires_grad_()
        roles = torch.softmax(torch.randn(2, 4, 5, 3), dim=-1).requires_grad_()
        output = predictor(concepts=concepts, selector=selector, roles=roles)
        output.sum().backward()
        self.assertIsNone(concepts.grad)
        self.assertIsNotNone(selector.grad)
        self.assertIsNotNone(roles.grad)


if __name__ == '__main__':
    unittest.main()
