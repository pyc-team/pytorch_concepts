"""
Concept Bottleneck Model (Low-Level Interface)
===============================================

This example demonstrates how to implement a Concept Bottleneck Model (CBM) using
the low-level interface of PyC, which provides pure PyTorch syntax.

Key Components:
- LinearLatentToConcept: Maps latent embeddings (Z) to concept predictions (C)
- LinearConceptToConcept: Maps concept predictions (C) to task predictions (C)
- Intervention API: Allows concept interventions at inference time

This low-level approach gives you full control over:
- Model architecture and layer composition
- Training loop and optimization
- Loss computation and weighting
- Intervention strategies during inference

Dataset: XOR toy dataset with 2 binary concepts and 1 binary task
"""
import torch
from sklearn.metrics import accuracy_score
from torch.nn import ModuleDict

from torch_concepts import seed_everything
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.nn import LinearLatentToConcept, LinearConceptToConcept
from torch_concepts.nn import RandomPolicy, DoIntervention, intervention


def main():
    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5

    seed_everything(42)
    
    dataset = ToyDataset(dataset='xor', n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]
    
    # Get dimensions
    n_features = x_train.shape[1]
    concept_dims = c_train.shape[1]
    task_dims = y_train.shape[1]

    latent_encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
    )

    # PyC layers
    c_encoder = LinearLatentToConcept(in_latent=latent_dims, out_concepts=concept_dims)
    y_predictor = LinearConceptToConcept(in_concepts=concept_dims, out_concepts=task_dims)

    # these are equivalent to the following torch layers
    # c_encoder = torch.nn.Linear(latent_dims, concept_dims)
    # y_predictor = torch.nn.Linear(concept_dims, task_dims)
    
    model = ModuleDict(
        {"latent_encoder": latent_encoder,
         "concept_encoder": c_encoder,
         "task_predictor": y_predictor}
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        latent = model["latent_encoder"](x_train)
        c_pred = model["concept_encoder"](latent=latent)
        y_pred = model["task_predictor"](concepts=c_pred)

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.detach() > 0.)
            concept_accuracy = accuracy_score(c_train, c_pred.detach() > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    int_policy_c = RandomPolicy(out_concepts=c_train.shape[1], scale=100)
    int_strategy_c = DoIntervention(model=model["concept_encoder"], constants=-10)
    with intervention(
        policies=int_policy_c,
        strategies=int_strategy_c,
        target_concepts=[1],
        quantiles=1
    ) as new_encoder:
        latent = model["latent_encoder"](x_train)
        c_pred = new_encoder(latent=latent)
        y_pred = model["task_predictor"](concepts=c_pred)
        cy_pred = torch.cat([c_pred, y_pred], dim=1)
        print('intervened output: \n', cy_pred[:5])

    return


if __name__ == "__main__":
    main()
