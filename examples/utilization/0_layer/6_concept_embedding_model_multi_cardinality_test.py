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
from torch_concepts.nn import MixConceptExogegnousToConcept, LinearLatentToExogenous, \
    LinearExogenousToConcept


def main():
    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5
    exogenous_size = 7
    
    seed_everything(42)

    # Load dataset
    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]
    c_new = torch.randint(0, 5, size=(len(c_train), 1))
    c_new_one_hot = torch.nn.functional.one_hot(c_new.squeeze(), 5).float()
    c_train = torch.cat([
        c_new_one_hot,
        c_train[:, 0].unsqueeze(-1),
        1-c_train[:, 0].unsqueeze(-1),
        c_train[:, 1].unsqueeze(-1),
        c_new_one_hot,
        c_train[:, 0].unsqueeze(-1),
        c_train[:, 1].unsqueeze(-1),
    ], dim=1)

    # Get dimensions
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_tasks = y_train.shape[1]

    # Build model using low-level layers
    latent_encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
    )
    # Exogenous encoder: latent -> per-concept exogenous
    exog_encoder = LinearLatentToExogenous(
        in_latent=latent_dims,
        out_concepts=n_concepts,
        out_exogenous=exogenous_size
    )
    # Concept encoder: exogenous -> concepts
    c_encoder = LinearExogenousToConcept(
        in_exogenous=exogenous_size,
    )
    # Predictor: concepts + exogenous -> tasks
    y_predictor = MixConceptExogegnousToConcept(
        in_concepts=n_concepts,
        in_exogenous=exogenous_size,
        out_concepts=n_tasks,
        cardinalities=[5, 2, 1, 5, 1, 1]
    )
    model = ModuleDict(
        {"latent_encoder": latent_encoder,
         "exog_encoder": exog_encoder,
         "concept_encoder": c_encoder,
         "task_predictor": y_predictor}
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Generate concept and task predictions
        emb = latent_encoder(x_train)
        exog = exog_encoder(latent=emb)
        c_pred = c_encoder(exogenous=exog)
        y_pred = y_predictor(concepts=c_pred, exogenous=exog)

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
