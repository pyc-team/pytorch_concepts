import torch
from causallearn.search.PermutationBased.GRaSP import grasp
from sklearn.metrics import accuracy_score
import torch_geometric as pyg

from torch_concepts.base import ConceptTensor
from torch_concepts.data import ToyDataset
from torch_concepts.nn import ConceptEncoder
from torch_concepts.utils import prepare_pyg_data
import torch_concepts.nn.functional as CF


def main():
    latent_dims = 6
    concept_emb_size = 2*latent_dims
    n_epochs = 500
    n_samples = 1000
    iterations = 100
    concept_reg = 0.5

    # load data
    data = ToyDataset('checkmark', size=n_samples, random_state=42)
    x_train, c_train, y_train, dag, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.dag, data.concept_attr_names, data.task_attr_names
    c_train = torch.cat([c_train, y_train], dim=1)
    concept_names += task_names
    y_train = c_train.clone()
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    concepts_train = ConceptTensor.concept(c_train, {1: concept_names})
    intervention_indexes = ConceptTensor.concept(torch.ones_like(c_train).bool(), {1: concept_names})

    # define model
    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, latent_dims), torch.nn.LeakyReLU())
    c_encoder = ConceptEncoder(in_features=latent_dims, out_concept_dimensions={1: concept_names, 2: concept_emb_size})
    c_scorer = ConceptEncoder(in_features=concept_emb_size, out_concept_dimensions={1: concept_names}, reduce_dim=2)
    mpnn = pyg.nn.Sequential('x, edge_index', [
        (pyg.nn.GCNConv(latent_dims, latent_dims, add_self_loops=False, normalize=False), 'x, edge_index -> x'),
        torch.nn.LeakyReLU(inplace=True),
    ])
    y_predictors = torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Linear(latent_dims, 1), torch.nn.Sigmoid())
    ] * n_concepts)
    model = torch.nn.Sequential(encoder, c_encoder, c_scorer, mpnn, y_predictors)

    # learn the causal graph structure (it could be replaced by a known graph or differentiable graph learning)
    G = grasp(c_train.numpy(), score_func='local_score_BDeu').graph
    graph = ConceptTensor(torch.abs(torch.tensor(G)), {0: concept_names, 1: concept_names})
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
        c_intervened = CF.intervene(c_pred, concepts_train, intervention_indexes)
        c_mix = CF.concept_embedding_mixture(c_emb, c_intervened)
        # use the graph to make each concept dependent on parent concepts only
        node_features, edge_index, batch = prepare_pyg_data(c_mix, graph)
        c_emb_post = mpnn(node_features, edge_index).reshape(-1, n_concepts, latent_dims)
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
    task_accuracy = accuracy_score(y_train[:, non_zero_y], y_pred[:, non_zero_y] > 0.5)
    concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")

    # at inference time, we can intervene on the causal graph by:
    # removing some edges and replacing concept values with ground truth
    intervened_idxs = ['A']
    intervened_graph = CF.intervene_on_concept_graph(graph, indexes=intervened_idxs)
    c_pred[:, 0] = c_train[:, 0].clone()

    # at inference time, we can predict each concept using its parents only by propagating concepts through the graph
    # we first generate initial concept embeddings/predictions
    emb = encoder(x_train)
    c_emb = c_encoder(emb)
    y_pred = c_pred = c_scorer(c_emb)
    # we then look for the fixed point of the graph convolutional network
    for epoch in range(iterations):
        # compute concept embeddings
        c_mix = CF.concept_embedding_mixture(c_emb, ConceptTensor(y_pred, {1: concept_names}))

        # use the graph to generate new concept embeddings using parent concepts only
        node_features, edge_index, batch = prepare_pyg_data(c_mix, intervened_graph)
        c_emb_post = mpnn(node_features, edge_index).reshape(-1, n_concepts, latent_dims)

        # update concept predictions using the updated concept embeddings (for non-intervened concepts only)
        for node, concept_name in enumerate(concept_names):
            if concept_name not in intervened_idxs:
                y_pred[:, node] = y_predictors[node](c_emb_post[:, node]).squeeze()

    non_zero_y = intervened_graph.sum(0) > 0
    task_accuracy = accuracy_score(y_train[:, non_zero_y], y_pred[:, non_zero_y] > 0.5)
    concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")

    return


if __name__ == "__main__":
    main()
