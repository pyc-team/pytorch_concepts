"""Tests for CMR rule-memory and concept-embedding predictors."""
import unittest

import torch

from torch_concepts import Annotations, AnnotatedTensor
from torch_concepts.nn import (
    ReconstructionRuleConceptEmbeddingToConcept,
    RuleConceptEmbeddingToConcept,
    RuleMemory,
)


def pack_rule_embeddings(selector, roles):
    """Flatten selector and roles into the standard embedding input."""
    return torch.cat(
        [selector.flatten(start_dim=-2), roles.flatten(start_dim=-4)],
        dim=-1,
    )


class TestRuleMemory(unittest.TestCase):
    def test_initialization(self):
        memory = RuleMemory(
            n_tasks=3,
            n_rules=5,
            n_concepts=10,
            latent_size=64,
            hidden_layers=2,
        )
        self.assertEqual(memory.shape, (3, 5, 10, 3))
        self.assertEqual(memory.memory.weight.shape, (3, 64))

    def test_forward_shape(self):
        memory = RuleMemory(n_tasks=2, n_rules=4, n_concepts=6)
        roles = memory()
        self.assertEqual(roles.shape, (2, 4, 6, 3))
        self.assertTrue(torch.all((roles >= 0) & (roles <= 1)))
        self.assertTrue(
            torch.allclose(
                roles.sum(dim=-1), torch.ones_like(roles.sum(dim=-1))
            )
        )

    def test_hidden_layers_config(self):
        memory_zero = RuleMemory(
            n_tasks=2, n_rules=3, n_concepts=4, hidden_layers=0
        )
        memory_two = RuleMemory(
            n_tasks=2, n_rules=3, n_concepts=4, hidden_layers=2
        )
        linear_zero = sum(
            isinstance(layer, torch.nn.Linear)
            for layer in memory_zero.decoder.modules()
        )
        linear_two = sum(
            isinstance(layer, torch.nn.Linear)
            for layer in memory_two.decoder.modules()
        )
        self.assertEqual(linear_zero, 1)
        self.assertEqual(linear_two, 3)

    def test_gradient_flow(self):
        memory = RuleMemory(n_tasks=2, n_rules=3, n_concepts=4)
        loss = memory().sum()
        loss.backward()
        self.assertIsNotNone(memory.memory.weight.grad)


class TestRuleConceptEmbeddingToConcept(unittest.TestCase):
    def test_forward_shape(self):
        n_tasks, n_rules, n_concepts = 2, 3, 6
        in_embeddings = n_tasks * n_rules * (1 + 3 * n_concepts)
        predictor = RuleConceptEmbeddingToConcept(
            out_concepts=n_tasks,
            in_concepts=n_concepts,
            in_embeddings=in_embeddings,
            n_rules=n_rules,
        )
        concepts = torch.rand(4, n_concepts)
        selector = torch.softmax(
            torch.randn(4, n_tasks, n_rules), dim=-1
        )
        roles = torch.softmax(
            torch.randn(4, n_tasks, n_rules, n_concepts, 3), dim=-1
        )
        output = predictor(
            concepts=concepts,
            embeddings=pack_rule_embeddings(selector, roles),
        )
        self.assertEqual(output.shape, (4, n_tasks))

    def test_annotation_output(self):
        tasks = Annotations(
            labels=["y1", "y2"], cardinalities=[1, 1]
        )
        n_rules, n_concepts = 2, 4
        predictor = RuleConceptEmbeddingToConcept(
            out_concepts=tasks,
            in_concepts=n_concepts,
            in_embeddings=2 * n_rules * (1 + 3 * n_concepts),
            n_rules=n_rules,
        )
        selector = torch.softmax(torch.randn(3, 2, n_rules), dim=-1)
        roles = torch.softmax(
            torch.randn(3, 2, n_rules, n_concepts, 3), dim=-1
        )
        output = predictor(
            concepts=torch.rand(3, n_concepts),
            embeddings=pack_rule_embeddings(selector, roles),
        )
        self.assertIsInstance(output, AnnotatedTensor)
        self.assertEqual(output.annotation.labels, ["y1", "y2"])

    def test_gradient_flow_detaches_concepts(self):
        n_tasks, n_rules, n_concepts = 2, 4, 5
        predictor = RuleConceptEmbeddingToConcept(
            out_concepts=n_tasks,
            in_concepts=n_concepts,
            in_embeddings=n_tasks * n_rules * (1 + 3 * n_concepts),
            n_rules=n_rules,
        )
        concepts = torch.rand(2, n_concepts, requires_grad=True)
        selector_logits = torch.randn(
            2, n_tasks, n_rules, requires_grad=True
        )
        roles_logits = torch.randn(
            2, n_tasks, n_rules, n_concepts, 3, requires_grad=True
        )
        embeddings = pack_rule_embeddings(
            torch.softmax(selector_logits, dim=-1),
            torch.softmax(roles_logits, dim=-1),
        )
        predictor(concepts=concepts, embeddings=embeddings).sum().backward()
        self.assertIsNone(concepts.grad)
        self.assertIsNotNone(selector_logits.grad)
        self.assertIsNotNone(roles_logits.grad)


class TestReconstructionRuleConceptEmbeddingToConcept(unittest.TestCase):
    def test_forward_shape(self):
        n_tasks, n_rules, n_concepts = 2, 3, 6
        predictor = ReconstructionRuleConceptEmbeddingToConcept(
            out_concepts=n_tasks,
            in_concepts=n_concepts,
            in_embeddings=n_tasks * n_rules * (1 + 3 * n_concepts),
            n_rules=n_rules,
            rec_weight=0.5,
        )
        selector = torch.softmax(
            torch.randn(4, n_tasks, n_rules), dim=-1
        )
        roles = torch.softmax(
            torch.randn(4, n_tasks, n_rules, n_concepts, 3), dim=-1
        )
        output = predictor(
            concepts=torch.rand(4, n_concepts),
            embeddings=pack_rule_embeddings(selector, roles),
        )
        self.assertEqual(output.shape, (4, n_tasks))

    def test_rec_weight_changes_output(self):
        n_tasks, n_rules, n_concepts = 2, 2, 4
        kwargs = dict(
            out_concepts=n_tasks,
            in_concepts=n_concepts,
            in_embeddings=n_tasks * n_rules * (1 + 3 * n_concepts),
            n_rules=n_rules,
        )
        low = ReconstructionRuleConceptEmbeddingToConcept(
            **kwargs, rec_weight=0.0
        )
        high = ReconstructionRuleConceptEmbeddingToConcept(
            **kwargs, rec_weight=1.0
        )
        concepts = torch.rand(3, n_concepts)
        selector = torch.softmax(
            torch.randn(3, n_tasks, n_rules), dim=-1
        )
        roles = torch.softmax(
            torch.randn(3, n_tasks, n_rules, n_concepts, 3), dim=-1
        )
        embeddings = pack_rule_embeddings(selector, roles)
        out_low = low(concepts=concepts, embeddings=embeddings)
        out_high = high(concepts=concepts, embeddings=embeddings)
        self.assertFalse(torch.allclose(out_low, out_high))


if __name__ == "__main__":
    unittest.main()
