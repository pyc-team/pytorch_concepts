"""
Example: Concept Embedding Model with Multi-Cardinality Concepts (Low-Level API)

Uses the insurance Bayesian-network dataset, which has concepts of mixed cardinality
(binary and multi-class), to demonstrate MixConceptEmbeddingToConcept with real
heterogeneous concept data.

The dataset is built on first run (autoencoder training + BN sampling) and cached.

Concept/task split:
  - Concepts: non-leaf nodes in the insurance DAG (21 nodes, mixed cardinality)
  - Task: PropCost (leaf node, 4 classes) predicted from concept embeddings
"""
import torch
import warnings
from torch.nn import ModuleDict
import torch.nn.functional as F

from torch_concepts import seed_everything, AnnotatedTensor
from torch_concepts.data import BnLearnDataset
from torch_concepts.nn import LinearEmbeddingToConcept, MixConceptEmbeddingToConcept
from torch_concepts.nn import MLP


def main():
    latent_dims = 128
    n_epochs = 1000
    n_samples = 10000
    concept_reg = 0.2
    embedding_size = 4
    task_node = 'PropCost'  # leaf node, cardinality=4

    warnings.filterwarnings('ignore')
    seed_everything(42)

    # Load dataset: input_data are autoencoder embeddings of the BN samples
    # concept cardinalities:
    # [4, 1, 3, 4, 1, 4, 4, 3, 5, 1, 4, 3, 3, 1, 4, 5, 1, 1, 4, 4, 1, 4, 4, 1, 4, 3]
    dataset = BnLearnDataset('insurance', n_gen=n_samples, seed=42)
    x_train = dataset.input_data    # (n_samples, 32)
    c_raw = dataset.concepts        # (n_samples, 27) integer class indices

    # Concept/task split from the DAG structure: every node but the task, keeping
    # each one's cardinality and type.
    axis = dataset.annotations
    concept_names = [node for node in axis.labels if node != task_node]
    concept_annotations = axis.subset(concept_names)

    n_concepts_classes = concept_annotations.size    # one column per cardinality class
    n_task_classes = axis.concept(task_node).cardinality     # 4

    # `c_raw` holds one integer class column per node, in `axis.labels` order.
    c_all = AnnotatedTensor(c_raw, axis.to_concept_space())
    c_train = c_all[concept_names]          # (n_samples, 26) integer class indices
    y_train = c_all[task_node].squeeze(-1).long()   # (n_samples,) integer 0-3

    n_features = x_train.shape[1]           # 32

    model = ModuleDict({
        # input encoding: (batch, n_features) -> (batch, latent_dims)
        "encoder": MLP(
            input_size=n_features,
            hidden_size=latent_dims,
            n_layers=2,
            activation='leaky_relu',
        ),
        # one embedding vector per cardinality-class slot
        # (batch, latent_dims) -> (batch, n_concepts_expanded, embedding_size)
        "emb_encoder": torch.nn.Sequential(
            torch.nn.Linear(latent_dims, n_concepts_classes * embedding_size),
            torch.nn.Unflatten(unflattened_size=(n_concepts_classes, embedding_size), dim=1),
        ),
        # score each embedding: (batch, n_concepts_expanded, emb) -> (batch, n_concepts_expanded)
        "concept_encoder": torch.nn.Sequential(
            LinearEmbeddingToConcept(in_embeddings=embedding_size, out_concepts=1),
            torch.nn.Flatten(),
        ),
        # mix concept activations with embeddings -> task prediction
        "task_predictor": MixConceptEmbeddingToConcept(
            in_concepts=concept_annotations,
            in_embeddings=embedding_size,
            out_concepts=n_task_classes,
        ),
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        latent = model["encoder"](x_train)
        embeddings = model["emb_encoder"](latent)       # (batch, n_concepts_expanded, embedding_size)
        c_pred = model["concept_encoder"](embeddings)   # (batch, n_concepts_expanded)
        y_pred = model["task_predictor"](concepts=c_pred, embeddings=embeddings)  # (batch, n_task_classes)

        # per-group concept loss + task cross-entropy. Both tensors are annotated,
        # so every concept is addressed by name: `c_pred[name]` is its score
        # column(s), `c_train[name]` its integer class column.
        c_pred = AnnotatedTensor(c_pred, concept_annotations)
        binary_names = c_pred.binary().annotation.labels
        categorical_names = c_pred.categorical().annotation.labels

        binary_loss = F.binary_cross_entropy_with_logits(
            c_pred[binary_names], c_train[binary_names].float()
        )
        categorical_loss = sum(
            F.cross_entropy(c_pred[name], c_train[name].squeeze(-1).long())
            for name in categorical_names
        )
        task_loss = F.cross_entropy(y_pred, y_train)
        loss = binary_loss + categorical_loss + concept_reg * task_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 100 == 0:
            task_acc = (y_pred.detach().argmax(dim=1) == y_train).float().mean().item()
            bin_acc = ((c_pred[binary_names].detach() > 0).float()
                       == c_train[binary_names].float()).float().mean().item()
            cat_acc = sum(
                (c_pred[name].detach().argmax(dim=1) == c_train[name].squeeze(-1).long())
                .float().mean().item()
                for name in categorical_names
            ) / len(categorical_names)
            print(f"Epoch {epoch}: Loss {loss.item():.4f} | Task Acc: {task_acc:.4f} | Binary C. Acc: {bin_acc:.4f} | Categorical C. Acc: {cat_acc:.4f}")

    return


if __name__ == "__main__":
    main()
