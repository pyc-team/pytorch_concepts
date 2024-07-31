import unittest
import torch

from torch_concepts.base import ConceptTensor
from torch_concepts.utils import prepare_pyg_data


class TestPreparePygData(unittest.TestCase):
    def test_prepare_pyg_data(self):
        batch_size = 2
        n_nodes = 3
        emb_size = 4

        # Create a ConceptTensor
        tensor_data = torch.randn(batch_size, n_nodes, emb_size)
        concept_names = ["concept_1", "concept_2", "concept_3"]
        concept_tensor = ConceptTensor(tensor_data, {1: concept_names})

        # Create an adjacency matrix
        adjacency_matrix = torch.tensor([[0, 1, 0],
                                         [1, 0, 1],
                                         [0, 1, 0]], dtype=torch.float)
        adjacency_matrix = ConceptTensor(adjacency_matrix, {0: concept_names, 1: concept_names})

        # Call the function
        node_features, edge_index, batch = prepare_pyg_data(concept_tensor, adjacency_matrix)

        # Verify the shapes of the outputs
        self.assertEqual(node_features.shape, (batch_size * n_nodes, emb_size))
        self.assertEqual(edge_index.shape[0], 2)  # should be 2 rows for edge_index
        self.assertEqual(edge_index.shape[1], adjacency_matrix.sum().item() * batch_size)
        self.assertEqual(batch.shape, (batch_size * n_nodes,))

        # Check some values for correctness
        self.assertTrue(torch.equal(node_features[:n_nodes], tensor_data[0]))
        self.assertTrue(torch.equal(node_features[n_nodes:], tensor_data[1]))
        self.assertTrue(torch.equal(batch[:n_nodes], torch.zeros(n_nodes, dtype=torch.long)))
        self.assertTrue(torch.equal(batch[n_nodes:], torch.ones(n_nodes, dtype=torch.long)))


if __name__ == '__main__':
    unittest.main()
