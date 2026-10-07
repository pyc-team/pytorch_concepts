"""
Concept Bottleneck Model (Low-Level Interface)
===============================================

This example demonstrates how to implement a Concept Bottleneck Model (CBM) using
the low-level interface of PyC, which provides pure PyTorch syntax.

Key Components:
- LinearEmbeddingToConcept: Maps latent embeddings (Z) to concept predictions (C)
- LinearConceptToConcept: Maps concept predictions (C) to task predictions (Y)
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
from torch_concepts.data import ToyDataset
from torch_concepts.nn import LinearEmbeddingToConcept, LinearConceptToConcept

# data params
N_SAMPLES = 1000

# model params
TASK_LABEL = 'xor'
LATENT_DIMS = 10

# training params
N_EPOCHS = 500
LEARNING_RATE = 0.01
TASK_LOSS_WEIGHT = 0.5

def main():
    seed_everything(42)

    dataset = ToyDataset(dataset='xor', n_gen=N_SAMPLES)
    
    x_train = dataset.input_data
    c_train = dataset.concepts[[l for l in dataset.concept_names if l not in TASK_LABEL]]
    y_train = dataset.concepts[TASK_LABEL]

    latent_encoder = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], LATENT_DIMS),
        torch.nn.LeakyReLU(),
    )
    # PyC layers. Equivalent to torch.nn.Linear
    c_encoder = LinearEmbeddingToConcept(
        in_embeddings=LATENT_DIMS, 
        out_concepts=c_train.shape[1]
    )
    y_predictor = LinearConceptToConcept(
        in_concepts=c_train.shape[1], 
        out_concepts=y_train.shape[1]
    )
    
    model = ModuleDict(
        {"latent_encoder": latent_encoder,
         "concept_encoder": c_encoder,
         "task_predictor": y_predictor}
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()

        # generate concept and task predictions
        latent = model["latent_encoder"](x_train)
        c_pred = model["concept_encoder"](embeddings=latent)
        y_pred = model["task_predictor"](concepts=c_pred)

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + TASK_LOSS_WEIGHT * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.detach() > 0.)
            concept_accuracy = accuracy_score(c_train, c_pred.detach() > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    return


if __name__ == "__main__":
    main()
