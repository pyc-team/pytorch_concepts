"""
Example: Concept Memory Reasoner with Low-Level API

This example demonstrates how to build a Concept Memory Reasoner (CMR)
using PyC's low-level building blocks and standard two-input concept layers.
"""
import torch
from sklearn.metrics import accuracy_score
from torch.nn import ModuleDict

from torch_concepts import seed_everything
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.nn import (
    LinearEmbeddingToConcept,
    MLP,
    ReconstructionRuleConceptEmbeddingToConcept,
    RuleConceptEmbeddingToConcept,
    RuleMemory,
)


def main():
    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    n_rules = 10
    memory_latent_size = 100

    seed_everything(42)

    dataset = ToyDataset(dataset="xor", seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]

    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_tasks = y_train.shape[1]
    rule_embedding_size = n_tasks * n_rules * (1 + 3 * n_concepts)

    latent_encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
    )
    # The selector is an ordinary embedding network. It emits logits; the
    # categorical activation is composed explicitly below.
    selector_encoder = torch.nn.Sequential(
        MLP(
            input_size=latent_dims,
            hidden_size=latent_dims,
            output_size=n_tasks * n_rules,
            n_layers=1,
            activation="relu",
        ),
        torch.nn.Unflatten(-1, (n_tasks, n_rules)),
    )
    concept_encoder = LinearEmbeddingToConcept(
        in_embeddings=latent_dims, out_concepts=n_concepts
    )
    memory = RuleMemory(
        n_tasks=n_tasks,
        n_rules=n_rules,
        n_concepts=n_concepts,
        latent_size=memory_latent_size,
    )
    task_predictor = RuleConceptEmbeddingToConcept(
        out_concepts=n_tasks,
        in_concepts=n_concepts,
        in_embeddings=rule_embedding_size,
        n_rules=n_rules,
    )
    reconstruction_predictor = (
        ReconstructionRuleConceptEmbeddingToConcept(
            out_concepts=n_tasks,
            in_concepts=n_concepts,
            in_embeddings=rule_embedding_size,
            n_rules=n_rules,
            rec_weight=0.1,
        )
    )

    model = ModuleDict(
        {
            "latent_encoder": latent_encoder,
            "selector_encoder": selector_encoder,
            "concept_encoder": concept_encoder,
            "memory": memory,
            "task_predictor": task_predictor,
            "reconstruction_predictor": reconstruction_predictor,
        }
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    concept_loss_fn = torch.nn.BCEWithLogitsLoss()
    task_loss_fn = torch.nn.BCELoss(reduction="none")
    model.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        emb = latent_encoder(x_train)
        selector = torch.softmax(selector_encoder(emb), dim=-1)
        c_logits = concept_encoder(embeddings=emb)
        c_probs = c_logits.sigmoid()
        roles = memory()
        batched_roles = roles.unsqueeze(0).expand(
            x_train.shape[0], *roles.shape
        )
        rule_embeddings = torch.cat(
            [
                selector.flatten(start_dim=-2),
                batched_roles.flatten(start_dim=-4),
            ],
            dim=-1,
        )

        y_pred = task_predictor(
            concepts=c_probs, embeddings=rule_embeddings
        )
        y_pred_with_rec = reconstruction_predictor(
            concepts=c_probs, embeddings=rule_embeddings
        )

        concept_loss = concept_loss_fn(c_logits, c_train)
        task_loss_no_rec = task_loss_fn(y_pred, y_train)
        task_loss_with_rec = task_loss_fn(y_pred_with_rec, y_train)
        switched_task_loss = (
            (1.0 - y_train) * task_loss_no_rec
            + y_train * task_loss_with_rec
        ).mean()
        loss = concept_loss + switched_task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(
                y_train.cpu(), (y_pred.detach() > 0.5).cpu()
            )
            concept_accuracy = accuracy_score(
                c_train.cpu(), (c_logits.detach() > 0.0).cpu()
            )
            print(
                f"Epoch {epoch}: Loss {loss.item():.2f} | "
                f"Task Acc: {task_accuracy:.2f} | "
                f"Concept Acc: {concept_accuracy:.2f}"
            )


if __name__ == "__main__":
    main()
