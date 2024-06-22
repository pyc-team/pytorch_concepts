import unittest
import torch
from torch import nn
from torch_concepts import BaseConcept, ConceptLinear


class TestBaseConcept(unittest.TestCase):
    def test_base_concept_init(self):
        in_features = 10
        n_concepts = 5
        base_concept = BaseConcept(in_features, n_concepts)
        self.assertEqual(base_concept.in_features, in_features)
        self.assertEqual(base_concept.n_concepts, n_concepts)

    def test_base_concept_forward(self):
        base_concept = BaseConcept(10, 5)
        with self.assertRaises(NotImplementedError):
            base_concept.forward(torch.randn(1, 10))

    def test_base_concept_intervene(self):
        base_concept = BaseConcept(10, 5)
        with self.assertRaises(NotImplementedError):
            base_concept.intervene(torch.randn(1, 10))

    def test_base_concept_call(self):
        base_concept = BaseConcept(10, 5)
        with self.assertRaises(NotImplementedError):
            base_concept(torch.randn(1, 10))


class TestConceptLinear(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.n_concepts = 5
        self.model = ConceptLinear(self.in_features, self.n_concepts)
        self.input = torch.randn(1, self.in_features)

    def test_concept_linear_init(self):
        self.assertIsInstance(self.model.fc, nn.Linear)
        self.assertEqual(self.model.fc.in_features, self.in_features)
        self.assertEqual(self.model.fc.out_features, self.n_concepts)

    def test_concept_linear_forward(self):
        output = self.model.forward(self.input)
        self.assertEqual(output.shape, (1, self.n_concepts))

    def test_concept_linear_intervene_no_intervention(self):
        output = self.model.intervene(self.input)
        self.assertEqual(output.shape, (1, self.n_concepts))

    def test_concept_linear_intervene_with_intervention(self):
        c = torch.randn(1, self.n_concepts)
        intervention_idxs = [0, 2, 4]
        output = self.model.intervene(self.input, c, intervention_idxs)
        self.assertTrue(torch.equal(output[:, intervention_idxs], c[:, intervention_idxs]))

    def test_concept_linear_call(self):
        output = self.model(self.input)
        self.assertEqual(output.shape, (1, self.n_concepts))


if __name__ == '__main__':
    unittest.main()
