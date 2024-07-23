from typing import Tuple

import torch
from torch_geometric.utils import dense_to_sparse

from torch_concepts.base import ConceptTensor


def prepare_pyg_data(tensor: ConceptTensor,
                     adjacency_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare PyG data from a ConceptTensor and an adjacency matrix.

    Args:
        tensor: ConceptTensor of shape (batch_size, n_nodes, emb_size).
        adjacency_matrix: Adjacency matrix of shape (n_nodes, n_nodes).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: node_features, edge_index, batch
    """
    batch_size, n_nodes, emb_size = tensor.size()

    # Convert adjacency matrix to edge_index
    edge_index, _ = dense_to_sparse(adjacency_matrix)

    # Prepare node_features
    node_features = tensor.view(-1, emb_size)  # Shape (batch_size * n_nodes, emb_size)

    # Create batch tensor
    batch = torch.arange(batch_size).repeat_interleave(n_nodes)  # Shape (batch_size * n_nodes,)

    # Calculate offsets
    offsets = torch.arange(batch_size).view(-1, 1) * n_nodes  # Shape (batch_size, 1)
    offsets = offsets.repeat(1, edge_index.size(1)).view(1, -1)  # Shape (1, batch_size * num_edges)

    # Repeat edge_index and add offsets
    edge_index = edge_index.repeat(1, batch_size) + offsets

    return node_features, edge_index, batch
