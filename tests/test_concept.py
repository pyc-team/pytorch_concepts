import unittest
import torch
from torch import nn
from torch_concepts.nn import BaseConcept, ConceptLinear, ConceptEmbedding


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


class TestConceptEmbedding(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.n_concepts = 5
        self.emb_size = 20
        self.model = ConceptEmbedding(
            in_features=self.in_features,
            n_concepts=self.n_concepts,
            emb_size=self.emb_size
        )

    def test_initialization(self):
        self.assertEqual(self.model.in_features, self.in_features)
        self.assertEqual(self.model.n_concepts, self.n_concepts)
        self.assertEqual(self.model.emb_size, self.emb_size)
        self.assertEqual(len(self.model.concept_context_generators), self.n_concepts)
        self.assertIsInstance(self.model.concept_prob_predictor, torch.nn.Sequential)

    def test_forward_output_shapes(self):
        x = torch.randn(42, self.in_features)
        c_emb, c_pred, c_int = self.model.forward(x)
        self.assertEqual(c_emb.shape, (42, self.n_concepts, self.emb_size))
        self.assertEqual(c_pred.shape, (42, self.n_concepts))

    def test_interventions(self):
        x = torch.randn(1, self.in_features)
        c = torch.ones(1, self.n_concepts)
        intervention_idxs = [0, 2, 4]
        c_emb, c_pred, c_int = self.model.forward(x, c=c, intervention_idxs=intervention_idxs, train=False)
        for idx in intervention_idxs:
            self.assertTrue(c_int[0, idx].item() == 1.0)

    def test_call_method(self):
        x = torch.randn(42, self.in_features)
        c_emb = self.model(x)
        self.assertEqual(c_emb.shape, (42, self.n_concepts, self.emb_size))
        self.assertIsNotNone(self.model.saved_c_pred)
        self.assertEqual(self.model.saved_c_pred.shape, (42, self.n_concepts))
        self.assertEqual(self.model.saved_c_int.shape, (42, self.n_concepts))
        self.assertTrue(torch.all(self.model.saved_c_pred.ravel() == self.model.saved_c_int.ravel()))


if __name__ == '__main__':
    unittest.main()
