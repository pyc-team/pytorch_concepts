"""
Example: Concept Embedding Model with Low-Level API

This example demonstrates how to build a Concept Embedding Model (CEM)
using the low-level encoder and predictor layers.
"""
import torch
from sklearn.metrics import accuracy_score
from torch.nn import ModuleDict

from torch_concepts import seed_everything
from torch_concepts.data import ToyDataset
from torch_concepts.nn import LinearEmbeddingToConcept, MixConceptEmbeddingToConcept, MLP


def main():
    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5
    embedding_size = 7
    
    seed_everything(42)

    # Load dataset
    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]

    # Get dimensions
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_tasks = y_train.shape[1]

    model = ModuleDict({
        # input encoding: (batch, n_features) -> (batch, latent_dims)
        "encoder": MLP(
            input_size=n_features,
            hidden_size=latent_dims,
            n_layers=1,
            activation='leaky_relu',
        ),
        # embedding encoder: (batch, latent_dims) -> (batch, n_concepts, embedding_size)
        "emb_encoder": torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_concepts * embedding_size),
            torch.nn.Unflatten(unflattened_size=(n_concepts, embedding_size), dim=1),
        ),
        # concept encoder: (batch, n_concepts, embedding_size) -> (batch, n_concepts)
        "concept_encoder": torch.nn.Sequential(
            LinearEmbeddingToConcept(in_embeddings=embedding_size, out_concepts=1),
            torch.nn.Flatten(),
        ),
        # predictor: (batch, n_concepts) + (batch, n_concepts, embedding_size) -> (batch, n_tasks)
        "task_predictor": MixConceptEmbeddingToConcept(
            in_concepts=n_concepts,
            in_embeddings=embedding_size,
            out_concepts=n_tasks,
            cardinalities=[1, 1],
        ),
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Generate concept and task predictions
        latent = model["encoder"](x_train)                          # (batch, latent_dims)
        embeddings = model["emb_encoder"](latent)                  # (batch, n_concepts, embedding_size)
        c_pred = model["concept_encoder"](embeddings)               # (batch, n_concepts)
        y_pred = model["task_predictor"](concepts=c_pred, embeddings=embeddings)  # (batch, n_tasks)

        # Compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.detach() > 0.)
            concept_accuracy = accuracy_score(c_train, c_pred.detach() > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    return


if __name__ == "__main__":
    main()
