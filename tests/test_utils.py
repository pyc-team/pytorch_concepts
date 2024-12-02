import importlib
import torch
import unittest

from torch_concepts.base import ConceptTensor

################################
## Taken from our causal_concept_embedding_model example
################################

def prepare_pyg_data(
    tensor: ConceptTensor,
    adjacency_matrix: ConceptTensor,
):
    """
    Prepare PyG data from a ConceptTensor and an adjacency matrix.

    Args:
        tensor: ConceptTensor of shape (batch_size, n_nodes, emb_size).
        adjacency_matrix: Adjacency matrix of shape (n_nodes, n_nodes).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: node_features,
            edge_index, batch
    """
    # Lazy import as we will only import it if the module is installed
    import torch_geometric
    adjacency_matrix = (
        adjacency_matrix.to_standard_tensor()
        if isinstance(adjacency_matrix, ConceptTensor) else adjacency_matrix
    )

    batch_size, n_nodes, emb_size = tensor.size()

    # Convert adjacency matrix to edge_index
    edge_index, _ = torch_geometric.utils.dense_to_sparse(adjacency_matrix)

    # Prepare node_features
    # Shape (batch_size * n_nodes, emb_size)
    node_features = tensor.view(-1, emb_size)

    # Create batch tensor
    # Shape (batch_size * n_nodes)
    batch = torch.arange(batch_size).repeat_interleave(n_nodes)

    # Calculate offsets
    # Shape (batch_size, 1)
    offsets = torch.arange(batch_size).view(-1, 1) * n_nodes
    # Shape (1, batch_size * num_edges)
    offsets = offsets.repeat(1, edge_index.size(1)).view(1, -1)

    # Repeat edge_index and add offsets
    edge_index = edge_index.repeat(1, batch_size) + offsets

    return node_features, edge_index, batch

class TestPreparePygData(unittest.TestCase):
    def test_prepare_pyg_data(self):
        if importlib.util.find_spec('torch_geometric') is None:
            # Then we will not test anything as we have not installed it
            return
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
