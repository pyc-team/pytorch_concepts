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

    def test_intervene_on_concept_graph(self):
        # Create a ConceptTensor adjacency matrix
        adj_matrix_data = torch.tensor([[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0]], dtype=torch.float)
        concept_names = ["concept_1", "concept_2", "concept_3"]
        c_adj = ConceptTensor(adj_matrix_data, concept_names)

        # Intervene by zeroing out specific columns
        intervened_c_adj = CF.intervene_on_concept_graph(c_adj, ["concept_2"])

        # Verify the shape of the output
        self.assertEqual(intervened_c_adj.shape, c_adj.shape)

        # Verify that the specified columns are zeroed out
        expected_data = torch.tensor([[0, 0, 0],
                                      [1, 0, 1],
                                      [0, 0, 0]], dtype=torch.float)
        self.assertTrue(torch.equal(intervened_c_adj, expected_data))

        # Verify that the concept names remain unchanged
        self.assertEqual(intervened_c_adj.concept_names, concept_names)


if __name__ == '__main__':
    unittest.main()
