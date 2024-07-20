import torch
from sklearn.metrics import accuracy_score
import networkx as nx

from torch_concepts.base import ConceptTensor
from torch_concepts.data import checkmark
from torch_concepts.nn import ConceptScorer, ConceptEncoder
import torch_concepts.nn.functional as CF


def main():
    emb_size = 6
    n_epochs = 500
    n_samples = 1000

    # load data
    x_train, c_train, dag = checkmark(n_samples)
    y_train = c_train.clone()
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    concept_names = ["C1", "C2", "C3", "C4"]
    concepts_train = ConceptTensor.concept(c_train, concept_names=concept_names)
    intervention_indexes = ConceptTensor.concept(torch.ones_like(c_train).bool(), concept_names=concept_names)

    # define model
    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, emb_size), torch.nn.LeakyReLU())
    c_encoder = ConceptEncoder(emb_size, n_concepts, 2*emb_size, concept_names=concept_names)
    c_scorer = torch.nn.Sequential(
        ConceptScorer(2*emb_size, concept_names=concept_names),
        torch.nn.Sigmoid()
    )
    y_predictors = torch.nn.ModuleList(
        [torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size),
                             torch.nn.LeakyReLU(),
                             torch.nn.Linear(emb_size, 1),
                             torch.nn.Sigmoid())
         for _ in range(n_concepts)]
    )
    model = torch.nn.Sequential(encoder, c_encoder, c_scorer, y_predictors)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form = torch.nn.BCELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_emb = c_encoder(emb)
        c_pred = ConceptTensor.concept(c_scorer(c_emb), concept_names)
        # concept interventions make training faster and reduce leakage
        c_intervened = CF.intervene(c_pred, concepts_train, intervention_indexes)
        c_mix = CF.concept_embedding_mixture(c_emb, c_intervened)
        # use the DAG to make each concept dependent on parent concepts only
        c_masked = torch.einsum('ijk,jl->ilk', c_mix, dag)
        # predict each concept using its parents only
        y_pred = torch.zeros_like(y_train)
        for c in range(n_concepts):
            y_pred[:, c] = y_predictors[c](c_masked[:, c]).squeeze()

        # compute loss
        concept_loss = loss_form(c_pred, c_train)
        task_loss = loss_form(y_pred, y_train)
        loss = concept_loss + 0.5 * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.2f}")

    non_zero_y = dag.sum(0) > 0
    task_accuracy = accuracy_score(y_train[:, non_zero_y], y_pred[:, non_zero_y] > 0.5)
    concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")

    # at inference time, we can predict each concept using its parents only
    emb = encoder(x_train)
    c_emb = c_encoder(emb)
    c_pred = ConceptTensor.concept(c_scorer(c_emb), concept_names)
    c_mix = CF.concept_embedding_mixture(c_emb, c_pred)
    c_masked = torch.einsum('ijk,jl->ilk', c_mix, dag)
    digraph = nx.from_numpy_array(dag.numpy(), create_using=nx.DiGraph)
    y_pred = torch.zeros_like(y_train)
    for node in nx.topological_sort(digraph):
        print(f"Node {node} ({concept_names[node]})")
        print(f"Parents: {list(digraph.predecessors(node))}")
        y_pred[:, node] = y_predictors[node](c_masked[:, node]).squeeze()
        c_pred[:, node] = y_pred[:, node].clone()
        c_mix = CF.concept_embedding_mixture(c_emb, c_pred)
        c_masked = torch.einsum('ijk,jl->ilk', c_mix, dag)

    non_zero_y = dag.sum(0) > 0
    task_accuracy = accuracy_score(y_train[:, non_zero_y], y_pred[:, non_zero_y] > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")

    return


if __name__ == "__main__":
    main()
