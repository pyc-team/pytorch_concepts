import unittest
import torch
from torch.nn import functional as F
from torch_concepts.nn import ConceptEncoder, ConceptScorer
from torch_concepts.base import ConceptTensor
from torch_concepts.nn.functional import intervene, concept_embedding_mixture

from torch_concepts.nn import BaseBottleneck, ConceptBottleneck, ConceptResidualBottleneck, MixConceptEmbeddingBottleneck


class TestConceptBottlenecks(unittest.TestCase):

    def setUp(self):
        self.in_features = 10
        self.n_concepts = 5
        self.emb_size = 4
        self.batch_size = 3
        self.concept_names = ["A", "B", "C", "D", "E"]
        self.x = torch.randn(self.batch_size, self.in_features)
        self.c_true = ConceptTensor.concept(torch.randn(self.batch_size, self.n_concepts), self.concept_names)
        self.intervention_idxs = ConceptTensor.concept(torch.randint(0, 2, (self.batch_size, self.n_concepts)).bool(), self.concept_names)
        self.intervention_rate = 0.1

    def test_concept_bottleneck_forward(self):
        bottleneck = ConceptBottleneck(self.in_features, self.n_concepts, concept_names=self.concept_names)
        result = bottleneck(self.x, self.c_true, self.intervention_idxs, self.intervention_rate)

        # Test output keys
        self.assertIn('next', result)
        self.assertIn('c_pred', result)
        self.assertIn('c_int', result)
        self.assertIn('emb', result)

        # Test output shapes
        self.assertEqual(result['c_pred'].shape, (self.batch_size, self.n_concepts))
        self.assertEqual(result['c_int'].shape, (self.batch_size, self.n_concepts))

    def test_concept_residual_bottleneck_forward(self):
        bottleneck = ConceptResidualBottleneck(self.in_features, self.n_concepts, self.emb_size, concept_names=self.concept_names)
        result = bottleneck(self.x, self.c_true, self.intervention_idxs, self.intervention_rate)

        # Test output keys
        self.assertIn('next', result)
        self.assertIn('c_pred', result)
        self.assertIn('c_int', result)
        self.assertIn('emb', result)

        # Test output shapes
        self.assertEqual(result['c_pred'].shape, (self.batch_size, self.n_concepts))
        self.assertEqual(result['c_int'].shape, (self.batch_size, self.n_concepts))
        self.assertEqual(result['emb'].shape, (self.batch_size, self.emb_size))
        self.assertEqual(result['next'].shape, (self.batch_size, self.n_concepts + self.emb_size))

    def test_mix_concept_embedding_bottleneck_forward(self):
        bottleneck = MixConceptEmbeddingBottleneck(self.in_features, self.n_concepts, self.emb_size, concept_names=self.concept_names)
        result = bottleneck(self.x, self.c_true, self.intervention_idxs, self.intervention_rate)

        # Test output keys
        self.assertIn('next', result)
        self.assertIn('c_pred', result)
        self.assertIn('c_int', result)
        self.assertIn('emb', result)
        self.assertIn('context', result)

        # Test output shapes
        self.assertEqual(result['c_pred'].shape, (self.batch_size, self.n_concepts))
        self.assertEqual(result['c_int'].shape, (self.batch_size, self.n_concepts))
        self.assertIsNone(result['emb'])
        self.assertEqual(result['context'].shape, (self.batch_size, self.n_concepts, 2 * self.emb_size))


if __name__ == "__main__":
    unittest.main()
