import unittest
import torch

from torch_concepts.base import ConceptTensor
from torch_concepts.nn.concept import ConceptEncoder, ConceptScorer


class TestConceptClasses(unittest.TestCase):

    def setUp(self):
        self.in_features = 10
        self.n_concepts = 5
        self.emb_size = 4
        self.batch_size = 3

    def test_concept_encoder(self):
        encoder = ConceptEncoder(self.in_features, self.n_concepts, self.emb_size)
        x = torch.randn(self.batch_size, self.in_features)
        result = encoder(x)

        # Test output shape
        self.assertEqual(result.shape, (self.batch_size, self.n_concepts, self.emb_size))

        # Test emb_size=1 case
        encoder = ConceptEncoder(self.in_features, self.n_concepts, emb_size=1)
        result = encoder(x)
        self.assertEqual(result.shape, (self.batch_size, self.n_concepts))

    def test_concept_scorer(self):
        scorer = ConceptScorer(self.emb_size)
        x = ConceptTensor.concept(torch.randn(self.batch_size, self.n_concepts, self.emb_size))
        result = scorer(x)

        # Test output shape
        self.assertEqual(result.shape, (self.batch_size, self.n_concepts))


if __name__ == '__main__':
    unittest.main()
