"""
Example: Concept Embedding Model with Low-Level API

This example demonstrates how to build a Concept Embedding Model (CEM)
using the low-level encoder and predictor layers.
"""
import torch
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.nn import ModuleDict

import torch_concepts as pyc
from torch_concepts import seed_everything
from torch_concepts.data import ToyDataset
from torch_concepts.nn import MixConceptEmbeddingToConcept


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

    # Get dimensions
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_tasks = y_train.shape[1]
    concept_annotations = pyc.AxisAnnotation.empty(n_concepts, types=['discrete', 'continuous'])

    # Build model using low-level layers
    latent_encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
    )
    # Exogenous encoder: latent -> per-concept exogenous
    exog_encoder = torch.nn.Sequential(
        torch.nn.Linear(latent_dims, exogenous_size*n_concepts),
        torch.nn.Unflatten(dim=1, unflattened_size=(n_concepts, exogenous_size)),
    )
    # Concept encoder: exogenous -> concepts
    c_encoder = pyc.nn.ConceptSequential(
        torch.nn.Linear(exogenous_size, 1),
        torch.nn.Flatten(),
        out_concepts=concept_annotations
    )
    # Predictor: concepts + exogenous -> tasks
    y_predictor = MixConceptEmbeddingToConcept(
        in_concepts=concept_annotations,
        in_embeddings=exogenous_size,
        out_concepts=n_tasks,
    )
    model = ModuleDict({
        "latent_encoder": latent_encoder,
         "exog_encoder": exog_encoder,
         "concept_encoder": c_encoder,
         "task_predictor": y_predictor
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn_discrete = torch.nn.BCEWithLogitsLoss()
    loss_fn_continuous = torch.nn.MSELoss()
    model.train()

    c_train = pyc.AnnotatedTensor(c_train, annotation=concept_annotations)
    c_train_dict = c_train.split_by_type()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Generate concept and task predictions
        emb = latent_encoder(x_train)
        exog = exog_encoder(emb)
        c_pred = c_encoder(exog)
        y_pred = y_predictor(concepts=c_pred, embeddings=exog)

        # Compute loss
        c_pred_dict = c_encoder.annotate(c_pred, concept_annotations).split_by_type()
        concept_loss = (loss_fn_discrete(c_pred_dict['discrete'], c_train_dict['discrete']) +
                        loss_fn_continuous(c_pred_dict['continuous'], c_train_dict['continuous']))
        task_loss = loss_fn_discrete(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.detach() > 0.)
            concept_accuracy = accuracy_score(c_train_dict['discrete'], c_pred_dict['discrete'].detach() > 0.)
            concept_mse = mean_squared_error(c_train_dict['continuous'], c_pred_dict['continuous'].detach())
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | "
                  f"Concept Acc: {concept_accuracy:.2f} | Concept MSE: {concept_mse:.2f}")

    return


if __name__ == "__main__":
    main()











