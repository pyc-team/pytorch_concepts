
import torch

from torch_geometric.utils import dense_to_sparse
from typing import Tuple, Dict, Union, List

from torch_concepts.base import ConceptTensor


def validate_and_generate_concept_names(
    concept_names: Dict[int, Union[int, List[str]]]
) -> Dict[int, List[str]]:
    """
    Validate and generate concept names based on the provided dictionary.

    Args:
        concept_names: Dictionary where keys are dimension indices and values
            are either integers (indicating the size of the dimension) or lists
            of strings (concept names).

    Returns:
        Dict[int, List[str]]: Processed dictionary with concept names.
    """
    processed_concept_names = {}
    for dim, value in concept_names.items():
        if dim == 0:
            # Batch size dimension is expected to be empty
            processed_concept_names[dim] = []
        elif isinstance(value, int):
            processed_concept_names[dim] = [
                f"concept_{dim}_{i}" for i in range(value)
            ]
        elif isinstance(value, list):
            processed_concept_names[dim] = value
        else:
            raise ValueError(
                f"Invalid value for dimension {dim}: must be either int or "
                "list of strings."
            )
    return processed_concept_names


def compute_output_size(concept_names: Dict[int, Union[int, List[str]]]) -> int:
    """
    Compute the output size of the linear layer based on the concept names.

    Args:
        concept_names: Dictionary where keys are dimension indices and values
            are either integers (indicating the size of the dimension) or lists
            of strings (concept names).

    Returns:
        int: Computed output size.
    """
    output_size = 1
    for dim, value in concept_names.items():
        if dim != 0:  # Skip batch size dimension
            if isinstance(value, int):
                output_size *= value
            elif isinstance(value, list):
                output_size *= len(value)
    return output_size


def prepare_pyg_data(
    tensor: ConceptTensor,
    adjacency_matrix: ConceptTensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare PyG data from a ConceptTensor and an adjacency matrix.

    Args:
        tensor: ConceptTensor of shape (batch_size, n_nodes, emb_size).
        adjacency_matrix: Adjacency matrix of shape (n_nodes, n_nodes).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: node_features,
            edge_index, batch
    """
    adjacency_matrix = (
        adjacency_matrix.to_standard_tensor()
        if isinstance(adjacency_matrix, ConceptTensor) else adjacency_matrix
    )

    batch_size, n_nodes, emb_size = tensor.size()

    # Convert adjacency matrix to edge_index
    edge_index, _ = dense_to_sparse(adjacency_matrix)

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
