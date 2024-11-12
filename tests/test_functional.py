import unittest
import torch
from torch_concepts.base import ConceptTensor
import torch_concepts.nn.functional as CF


class TestConceptFunctions(unittest.TestCase):

    def setUp(self):
        self.c_pred = ConceptTensor.concept(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
        self.c_true = ConceptTensor.concept(torch.tensor([[0.9, 0.8], [0.7, 0.6]]))
        self.indexes = ConceptTensor.concept(torch.tensor([[True, False], [False, True]]))
        self.concept_names = {1: ['concept1', 'concept2', 'concept3']}
        self.c_confidence = ConceptTensor(torch.tensor([[0.8, 0.1, 0.6], [0.9, 0.2, 0.4], [0.7, 0.3, 0.5]]), self.concept_names)
        self.target_confidence = 0.5

    def test_intervene(self):
        concept_names = {1: ['concept1', 'concept2']}
        result = CF.intervene(self.c_pred, self.c_true, self.indexes)
        expected = ConceptTensor.concept(torch.tensor([[0.9, 0.2], [0.3, 0.6]]))
        self.assertTrue(torch.equal(result, expected), f"Expected {expected}, but got {result}")

        # repeat with standard tensor objects
        result = CF.intervene(self.c_pred.to_standard_tensor(), self.c_true.to_standard_tensor(), self.indexes.to_standard_tensor())
        expected = ConceptTensor.concept(torch.tensor([[0.9, 0.2], [0.3, 0.6]]))
        self.assertTrue(torch.equal(result, expected), f"Expected {expected}, but got {result}")

        # repeat with standard tensor objects giving concept names as input
        result = CF.intervene(self.c_pred.to_standard_tensor(), self.c_true.to_standard_tensor(), self.indexes.to_standard_tensor(), concept_names)
        expected = ConceptTensor.concept(torch.tensor([[0.9, 0.2], [0.3, 0.6]]), concept_names)
        self.assertTrue(torch.equal(result, expected), f"Expected {expected}, but got {result}")

    def test_concept_embedding_mixture(self):
        c_emb = ConceptTensor.concept(torch.randn(5, 4, 6))
        c_scores = ConceptTensor.concept(torch.randint(0, 2, (5, 4)))
        result = CF.concept_embedding_mixture(c_emb, c_scores)
        self.assertTrue(result.shape == (5, 4, 3), f"Expected shape (5, 4, 3), but got {result.shape}")

        # repeat with standard tensor objects
        result = CF.concept_embedding_mixture(c_emb.to_standard_tensor(), c_scores.to_standard_tensor())
        self.assertTrue(result.shape == (5, 4, 3), f"Expected shape (5, 4, 3), but got {result.shape}")

        # repeat with standard tensor objects giving concept names as input
        concept_names = {1: ['concept1', 'concept2', 'concept3', 'concept4']}
        result = CF.concept_embedding_mixture(c_emb.to_standard_tensor(), c_scores.to_standard_tensor(), concept_names)
        self.assertTrue(result.shape == (5, 4, 3), f"Expected shape (5, 4, 3), but got {result.shape}")

    def test_intervene_on_concept_graph(self):
        # Create a ConceptTensor adjacency matrix
        adj_matrix_data = torch.tensor([[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0]], dtype=torch.float)
        concept_names = ["concept_1", "concept_2", "concept_3"]
        c_adj = ConceptTensor(adj_matrix_data, {0: concept_names, 1: concept_names})

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
        self.assertEqual(intervened_c_adj.concept_names[1], concept_names)

        # repeat with standard tensor objects
        intervened_c_adj = CF.intervene_on_concept_graph(c_adj.to_standard_tensor(), ["concept_1_1"])
        # Verify the shape of the output
        self.assertEqual(intervened_c_adj.shape, c_adj.shape)
        # Verify that the specified columns are zeroed out
        self.assertTrue(torch.equal(intervened_c_adj, expected_data))
        # Verify that the concept names remain unchanged
        self.assertEqual(intervened_c_adj.concept_names[1], ['concept_1_0', 'concept_1_1', 'concept_1_2'])

        # repeat with standard tensor objects giving concept names as input
        intervened_c_adj = CF.intervene_on_concept_graph(c_adj.to_standard_tensor(), ["concept_2"], {1: concept_names})
        # Verify the shape of the output
        self.assertEqual(intervened_c_adj.shape, c_adj.shape)
        # Verify that the specified columns are zeroed out
        self.assertTrue(torch.equal(intervened_c_adj, expected_data))
        # Verify that the concept names remain unchanged
        self.assertEqual(intervened_c_adj.concept_names[1], concept_names)


    def test_selective_calibration(self):
        expected_theta = torch.tensor([[0.8, 0.2, 0.5]])
        expected_result = ConceptTensor(expected_theta)
        result = CF.selective_calibration(self.c_confidence, self.target_confidence)
        self.assertEqual(torch.all(result == expected_result).item(), True)

        # repeat with standard tensor objects
        result = CF.selective_calibration(self.c_confidence.to_standard_tensor(), self.target_confidence)
        self.assertEqual(torch.all(result == expected_result).item(), True)

        # repeat with standard tensor objects giving concept names as input
        concept_names = {1: ['concept1', 'concept2', 'concept3']}
        result = CF.selective_calibration(self.c_confidence.to_standard_tensor(), self.target_confidence, concept_names)
        self.assertEqual(torch.all(result == expected_result).item(), True)

    def test_confidence_selection(self):
        theta = ConceptTensor(torch.tensor([[0.8, 0.3, 0.5]]))
        expected_mask = torch.tensor([[False, False, True],
                                      [True, False, False],
                                      [False, False, False]])
        expected_result = ConceptTensor(expected_mask)
        result = CF.confidence_selection(self.c_confidence, theta)
        self.assertEqual(torch.all(result == expected_result).item(), True)

        # repeat with standard tensor objects
        result = CF.confidence_selection(self.c_confidence.to_standard_tensor(), theta.to_standard_tensor())
        self.assertEqual(torch.all(result == expected_result).item(), True)

        # repeat with standard tensor objects giving concept names as input
        concept_names = {1: ['concept1', 'concept2', 'concept3']}
        result = CF.confidence_selection(self.c_confidence.to_standard_tensor(), theta.to_standard_tensor(), concept_names)
        self.assertEqual(torch.all(result == expected_result).item(), True)


if __name__ == '__main__':
    unittest.main()
