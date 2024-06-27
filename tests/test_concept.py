import unittest
import torch
from torch_concepts.nn import ConceptLinear, ConceptEmbedding, ConceptEmbeddingResidual
from torch_concepts.nn import Sequential


class TestConceptLinear(unittest.TestCase):

    def setUp(self):
        self.in_features = 10
        self.n_concepts = 5
        self.emb_size = 8
        self.model = ConceptLinear(self.in_features, self.n_concepts, emb_size=self.emb_size)
        self.x = torch.randn(2, self.in_features)
        self.c = torch.randn(2, self.n_concepts)
        self.intervention_idxs = torch.LongTensor([0, 2])

    def test_forward_shape(self):
        output = self.model(self.x)
        self.assertIn('c_pred', output)
        self.assertIn('c_int', output)
        self.assertIn('emb_pre', output)

        self.assertEqual(output['c_pred'].shape, (2, self.n_concepts))
        self.assertEqual(output['c_int'].shape, (2, self.n_concepts))
        self.assertEqual(output['emb_pre'].shape, (2, self.emb_size))

    def test_forward_intervention(self):
        output = self.model(self.x, self.c, self.intervention_idxs)
        self.assertTrue(torch.equal(output['c_int'][:, self.intervention_idxs], self.c[:, self.intervention_idxs]))

    def test_no_intervention(self):
        output = self.model(self.x)
        self.assertTrue(torch.equal(output['c_pred'], output['c_int']))

    def test_intervention_rate(self):
        # Assuming intervention_rate is not currently used in the model
        output = self.model(self.x, self.c, self.intervention_idxs, intervention_rate=0.5)
        self.assertTrue(torch.equal(output['c_int'][:, self.intervention_idxs], self.c[:, self.intervention_idxs]))


class TestConceptEmbeddingResidual(unittest.TestCase):

    def setUp(self):
        self.in_features = 10
        self.n_concepts = 5
        self.emb_size = 8
        self.n_residuals = 3
        self.model = ConceptEmbeddingResidual(self.in_features, self.n_concepts,
                                              emb_size=self.emb_size, n_residuals=self.n_residuals)
        self.x = torch.randn(2, self.in_features)
        self.c = torch.randn(2, self.n_concepts)
        self.intervention_idxs = [0, 2]

    def test_forward_shape(self):
        output = self.model(self.x)
        self.assertIn('residuals', output)
        self.assertEqual(output['residuals'].shape, (2, self.n_residuals))


class TestConceptEmbedding(unittest.TestCase):

    def setUp(self):
        self.in_features = 10
        self.n_concepts = 5
        self.emb_size = 8
        self.active_intervention_values = torch.tensor([1.0] * self.n_concepts)
        self.inactive_intervention_values = torch.tensor([0.0] * self.n_concepts)
        self.model = ConceptEmbedding(
            self.in_features,
            self.n_concepts,
            torch.sigmoid,
            self.emb_size,
            self.active_intervention_values,
            self.inactive_intervention_values
        )
        self.x = torch.randn(2, self.in_features)
        self.c = torch.randn(2, self.n_concepts)
        self.intervention_idxs = torch.LongTensor([0, 2])

    def test_forward_shape(self):
        output = self.model(self.x)
        self.assertIn('c_emb', output)
        self.assertIn('c_pred', output)
        self.assertIn('c_int', output)
        self.assertIn('emb_pre', output)

        self.assertEqual(output['c_emb'].shape, (2, self.n_concepts, self.emb_size))
        self.assertEqual(output['c_pred'].shape, (2, self.n_concepts))
        self.assertEqual(output['c_int'].shape, (2, self.n_concepts))
        self.assertEqual(output['emb_pre'].shape, (2, self.emb_size))

    def test_forward_intervention(self):
        output = self.model(self.x, self.c, self.intervention_idxs)
        self.assertTrue(torch.equal(output['c_int'][:, self.intervention_idxs], self.c[:, self.intervention_idxs]))

    def test_no_intervention(self):
        output = self.model(self.x)
        self.assertTrue(torch.equal(output['c_pred'], output['c_int']))

    def test_intervention_rate(self):
        # Assuming intervention_rate is not currently used in the model
        output = self.model(self.x, self.c, self.intervention_idxs, intervention_rate=0.5, train=True)
        self.assertTrue(torch.equal(output['c_int'][:, self.intervention_idxs], self.c[:, self.intervention_idxs]))

    def test_invalid_intervention_index(self):
        invalid_idxs = torch.LongTensor([0, self.n_concepts])  # This should raise an error
        with self.assertRaises(ValueError):
            self.model(self.x, self.c, invalid_idxs)

    def test_after_interventions(self):
        prob = torch.tensor([[0.5], [0.5]])
        concept_idx = torch.LongTensor([0])
        c_true = torch.tensor([[1.0], [0.0]])
        output = self.model._after_interventions(prob, concept_idx, self.intervention_idxs, c_true)
        expected = torch.tensor([[1.0], [0.0]])
        self.assertTrue(torch.equal(output, expected))


class TestSequential(unittest.TestCase):

    def setUp(self):
        self.in_features = 10
        self.n_concepts = 5
        self.emb_size = 8
        self.n_residuals = 3
        self.batch_size = 2

        self.concept_linear = ConceptLinear(self.in_features, self.n_concepts, emb_size=self.emb_size)
        self.concept_embedding = ConceptEmbedding(self.in_features, self.n_concepts, emb_size=self.emb_size)
        self.concept_residual = ConceptEmbeddingResidual(self.in_features, self.n_concepts, torch.sigmoid,
                                                         self.emb_size, self.n_residuals)

    def test_sequential_with_concept_linear(self):
        model = Sequential(
            torch.nn.Linear(self.in_features, self.in_features),
            self.concept_linear,
        )

        x = torch.randn(self.batch_size, self.in_features)
        output = model(x)
        self.assertEqual(output['c_pred'].shape, (2, self.n_concepts))
        self.assertEqual(output['c_int'].shape, (2, self.n_concepts))
        self.assertEqual(output['emb_pre'].shape, (2, self.emb_size))

    def test_sequential_with_concept_embedding(self):
        model = Sequential(
            torch.nn.Linear(self.in_features, self.in_features),
            self.concept_embedding,
        )

        x = torch.randn(self.batch_size, self.in_features)
        c = torch.randn(self.batch_size, self.n_concepts)
        output = model(x, concept_c=c)
        self.assertEqual(output['c_emb'].shape, (2, self.n_concepts, self.emb_size))
        self.assertEqual(output['c_pred'].shape, (2, self.n_concepts))
        self.assertEqual(output['c_int'].shape, (2, self.n_concepts))
        self.assertEqual(output['emb_pre'].shape, (2, self.emb_size))

    def test_sequential_with_concept_residual(self):
        model = Sequential(
            torch.nn.Linear(self.in_features, self.in_features),
            self.concept_residual,
        )

        x = torch.randn(self.batch_size, self.in_features)
        c = torch.randn(self.batch_size, self.n_concepts)
        output = model(x, concept_c=c)
        self.assertEqual(output['residuals'].shape, (2, self.n_residuals))


if __name__ == '__main__':
    unittest.main()
