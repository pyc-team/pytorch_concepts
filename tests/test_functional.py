import unittest
import torch
from torch_concepts.base import ConceptTensor
import torch_concepts.nn.functional as CF


class TestConceptFunctions(unittest.TestCase):

    def setUp(self):
        self.c_pred = ConceptTensor.concept(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
        self.c_true = ConceptTensor.concept(torch.tensor([[0.9, 0.8], [0.7, 0.6]]))
        self.indexes = ConceptTensor.concept(torch.tensor([[True, False], [False, True]]))

    def test_intervene(self):
        result = CF.intervene(self.c_pred, self.c_true, self.indexes)
        expected = ConceptTensor.concept(torch.tensor([[0.9, 0.2], [0.3, 0.6]]))
        self.assertTrue(torch.equal(result, expected), f"Expected {expected}, but got {result}")

    def test_concept_embedding_mixture(self):
        c_emb = ConceptTensor.concept(torch.randn(5, 4, 6))
        c_scores = ConceptTensor.concept(torch.randint(0, 2, (5, 4)))
        result = CF.concept_embedding_mixture(c_emb, c_scores)
        self.assertTrue(result.shape == (5, 4, 3), f"Expected shape (5, 4, 3), but got {result.shape}")


if __name__ == '__main__':
    unittest.main()
