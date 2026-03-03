"""
Example: Concept Memory Reasoner with Low-Level API

This example demonstrates how to build a Concept Memory Reasoner (CMR)
using the low-level encoder and predictor layers.
"""
import torch
from sklearn.metrics import accuracy_score
from torch.nn import ModuleDict

from torch_concepts import seed_everything
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.nn import (
    CategoricalSelectorLatentToExogenous,
    LinearLatentToConcept,
    MixMemoryConceptExogenousToConcept,
)


def main():
    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    nb_rules = 10
    memory_latent_size = 100
    rec_weight = 0.1
    
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

    # Build model using low-level layers
    latent_encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
    )

    exog_encoder = CategoricalSelectorLatentToExogenous(
        in_latent=latent_dims,
        out_concepts=n_tasks,
        out_exogenous=nb_rules,
    )
    
    c_encoder = LinearLatentToConcept(in_latent=latent_dims, out_concepts=n_concepts)

    y_predictor = MixMemoryConceptExogenousToConcept(
        in_concepts=n_concepts,
        in_exogenous=nb_rules,
        out_concepts=n_tasks,
        memory_latent_size=memory_latent_size,
        eps=0.001,
    )
    model = ModuleDict(
        {"latent_encoder": latent_encoder,
         "exog_encoder": exog_encoder,
         "concept_encoder": c_encoder,
         "task_predictor": y_predictor}
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn_y = torch.nn.BCELoss(reduction='none')
    loss_fn_c = torch.nn.BCEWithLogitsLoss()
    model.train()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Generate concept and task predictions
        emb = latent_encoder(x_train)
        exog = exog_encoder(latent=emb)
        c_pred = c_encoder(latent=emb)
        y_pred = y_predictor(concepts=c_pred, exogenous=exog)
        y_pred_with_rec = y_predictor(concepts=c_pred, exogenous=exog, include_rec=True, rec_weight=rec_weight)

        # Compute loss
        concept_loss = loss_fn_c(c_pred, c_train)
        task_loss_no_rec = loss_fn_y(y_pred, y_train)
        task_loss_rec = loss_fn_y(y_pred_with_rec, y_train)
        task_loss = (y_train * task_loss_rec + (1 - y_train) * task_loss_no_rec).mean()  # only apply rec loss to positive samples
        loss = concept_loss + task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.detach() > 0.5)
            concept_accuracy = accuracy_score(c_train, c_pred.detach() > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    return


if __name__ == "__main__":
    main()
