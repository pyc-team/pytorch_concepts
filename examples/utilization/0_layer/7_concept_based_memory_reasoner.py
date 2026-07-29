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
    RuleMemory,
    RuleReconstructionPredictor,
    RuleTaskPredictor,
    CategoricalSelectorLatentToExogenous,
    LinearEmbeddingToConcept,
)


def main():
    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    nb_rules = 10
    memory_latent_size = 100

    seed_everything(42)

    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]

    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_tasks = y_train.shape[1]

    latent_encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
    )
    selector_encoder = CategoricalSelectorLatentToExogenous(
        in_latent=latent_dims,
        out_concepts=n_tasks,
        out_exogenous=nb_rules,
    )
    concept_encoder = LinearEmbeddingToConcept(in_embeddings=latent_dims, out_concepts=n_concepts)
    memory = RuleMemory(
        n_tasks=n_tasks,
        n_rules=nb_rules,
        n_concepts=n_concepts,
        latent_size=memory_latent_size,
    )
    task_predictor = RuleTaskPredictor(
        in_concepts=n_concepts,
        in_exogenous=nb_rules,
        out_concepts=n_tasks,
    )
    reconstruction_predictor = RuleReconstructionPredictor(
        in_concepts=n_concepts,
        in_exogenous=nb_rules,
        out_concepts=n_tasks,
        rec_weight=0.1,
    )

    model = ModuleDict({
        'latent_encoder': latent_encoder,
        'selector_encoder': selector_encoder,
        'concept_encoder': concept_encoder,
        'memory': memory,
        'task_predictor': task_predictor,
        'reconstruction_predictor': reconstruction_predictor,
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    concept_loss_fn = torch.nn.BCEWithLogitsLoss()
    task_loss_fn = torch.nn.BCELoss(reduction='none')
    model.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        emb = latent_encoder(x_train)
        selector = selector_encoder(latent=emb)
        c_logits = concept_encoder(embeddings=emb)
        c_probs = c_logits.sigmoid()
        roles = memory()

        y_pred = task_predictor(concepts=c_probs, selector=selector, roles=roles)
        y_pred_with_rec = reconstruction_predictor(concepts=c_probs, selector=selector, roles=roles)

        concept_loss = concept_loss_fn(c_logits, c_train)
        task_loss_no_rec = task_loss_fn(y_pred, y_train)
        task_loss_with_rec = task_loss_fn(y_pred_with_rec, y_train)
        switched_task_loss = ((1.0 - y_train) * task_loss_no_rec + y_train * task_loss_with_rec).mean()
        loss = concept_loss + switched_task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train.cpu(), (y_pred.detach() > 0.5).cpu())
            concept_accuracy = accuracy_score(c_train.cpu(), (c_logits.detach() > 0.0).cpu())
            print(
                f'Epoch {epoch}: Loss {loss.item():.2f} | '
                f'Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}'
            )


if __name__ == '__main__':
    main()
