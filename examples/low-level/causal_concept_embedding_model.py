"""
IMPORTANT NOTE: for this specific example to work, you will need to manually
install torch_geometric and causal-learn
"""

import torch
import torch_geometric as pyg

from causallearn.search.PermutationBased.GRaSP import grasp
from sklearn.metrics import accuracy_score
from torch_geometric.utils import dense_to_sparse
from typing import Tuple

from torch_concepts.base import ConceptTensor

import torch_concepts.nn.functional as CF

from torch_concepts.data import ToyDataset
from torch_concepts.nn import ConceptLayer
from torch_concepts.utils import prepare_pyg_data



################################################################################
## Utility Functions
################################################################################

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


################################################################################
## Main Entry Point
################################################################################


def main():
    latent_dims = 6
    concept_emb_size = 2*latent_dims
    n_epochs = 500
    n_samples = 1000
    iterations = 100
    concept_reg = 0.5

    # load data
    data = ToyDataset('checkmark', size=n_samples, random_state=42)
    x_train, c_train, y_train, dag, concept_names, task_names = (
        data.data,
        data.concept_labels,
        data.target_labels,
        data.dag,
        data.concept_attr_names,
        data.task_attr_names,
    )
    c_train = torch.cat([c_train, y_train], dim=1)
    concept_names += task_names
    y_train = c_train.clone()
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    intervention_indexes = torch.ones_like(c_train).bool()

    # define model
    encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
    )
    c_encoder = ConceptLayer(
        in_features=latent_dims,
        out_concept_dimensions={1: concept_names, 2: concept_emb_size},
    )
    c_scorer = ConceptLayer(
        in_features=concept_emb_size,
        out_concept_dimensions={1: concept_names},
        reduce_dim=2,
    )
    gcn = pyg.nn.GCNConv(
        latent_dims,
        latent_dims,
        add_self_loops=False,
        normalize=False,
    )
    mpnn = pyg.nn.Sequential('x, edge_index', [
        (gcn, 'x, edge_index -> x'),
        torch.nn.LeakyReLU(inplace=True),
    ])
    y_predictors = torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Linear(latent_dims, 1), torch.nn.Sigmoid())
    ] * n_concepts)
    model = torch.nn.Sequential(encoder, c_encoder, c_scorer, mpnn, y_predictors)

    # learn the causal graph structure (it could be replaced by a known graph or
    # differentiable graph learning)
    G = grasp(c_train.numpy(), score_func='local_score_BDeu').graph
    graph = torch.abs(torch.tensor(G))
    print(graph)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_emb = c_encoder(emb)
        c_pred = c_scorer(c_emb).sigmoid()
        # concept interventions make training faster and reduce leakage
        c_intervened = CF.intervene(c_pred, c_train, intervention_indexes)
        c_mix = CF.concept_embedding_mixture(c_emb, c_intervened)
        # use the graph to make each concept dependent on parent concepts only
        node_features, edge_index, batch = prepare_pyg_data(c_mix, graph)
        c_emb_post = mpnn(node_features, edge_index).reshape(
            -1,
            n_concepts,
            latent_dims,
        )
        # update task predictions using the updated concept embeddings
        y_pred = torch.zeros_like(y_train)
        for node, concept_name in enumerate(concept_names):
            y_pred[:, node] = y_predictors[node](c_emb_post[:, node]).squeeze()

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.2f}")

    non_zero_y = graph.sum(0) > 0
    task_accuracy = accuracy_score(
        y_train[:, non_zero_y], y_pred[:, non_zero_y] > 0.5
    )
    concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")

    # at inference time, we can intervene on the causal graph by:
    # removing some edges and replacing concept values with ground truth
    intervened_idxs = ['A']
    intervened_graph = CF.intervene_on_concept_graph(
        graph,
        indexes=intervened_idxs,
        concept_names={1: concept_names},
    )
    c_pred[:, 0] = c_train[:, 0].clone()

    # at inference time, we can predict each concept using its parents only by
    # propagating concepts through the graph we first generate initial concept
    # embeddings/predictions
    emb = encoder(x_train)
    c_emb = c_encoder(emb)
    y_pred = c_pred = c_scorer(c_emb)
    # we then look for the fixed point of the graph convolutional network
    for epoch in range(iterations):
        # compute concept embeddings
        c_mix = CF.concept_embedding_mixture(c_emb, y_pred)

        # use the graph to generate new concept embeddings using parent
        # concepts only
        node_features, edge_index, batch = prepare_pyg_data(
            c_mix,
            intervened_graph,
        )
        c_emb_post = mpnn(node_features, edge_index).reshape(
            -1,
            n_concepts,
            latent_dims,
        )

        # update concept predictions using the updated concept embeddings (for
        # non-intervened concepts only)
        for node, concept_name in enumerate(concept_names):
            if concept_name not in intervened_idxs:
                y_pred[:, node] = y_predictors[node](
                    c_emb_post[:, node]
                ).squeeze()

    non_zero_y = intervened_graph.sum(0) > 0
    task_accuracy = accuracy_score(
        y_train[:, non_zero_y], y_pred[:, non_zero_y] > 0.5
    )
    concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")

    return


if __name__ == "__main__":
    main()
