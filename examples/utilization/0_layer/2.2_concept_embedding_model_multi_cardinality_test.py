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

from torch_concepts import seed_everything, AxisAnnotation, AnnotatedTensor
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

    # Concept/task split from the DAG structure
    axis = dataset.annotations.get_axis_annotation(1)
    task_idx = axis.labels.index(task_node)
    concept_idx = [axis.labels.index(node) for node in axis.labels if node != task_node]

    concept_cardinalities = [axis.cardinalities[i] for i in concept_idx]
    n_concepts_expanded = sum(concept_cardinalities)  # one column per cardinality class

    # Annotation describing concepts' semantics.
    concept_annotations = AxisAnnotation(
        labels=[axis.labels[i] for i in concept_idx],
        cardinalities=concept_cardinalities,
        types=['binary' if c == 1 else 'categorical' for c in concept_cardinalities],
    )

    n_task_classes = axis.cardinalities[task_idx]     # 4

    c_train = c_raw[:, concept_idx]         # (n_samples, 26) integer class indices
    y_train = c_raw[:, task_idx].long()     # (n_samples,) integer 0-3

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
            torch.nn.Linear(latent_dims, n_concepts_expanded * embedding_size),
            torch.nn.Unflatten(unflattened_size=(n_concepts_expanded, embedding_size), dim=1),
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

        # per-group concept loss + task cross-entropy
        c_pred = AnnotatedTensor(c_pred, annotation=concept_annotations)
        splitted = c_pred.split_by_type()
        binary, categorical = splitted['binary'], splitted['categorical']
        bin_g = concept_annotations.type_groups['binary']
        cat_g = concept_annotations.type_groups['categorical']
        cat_ann = categorical.annotation
        cum = cat_ann.cumulative_cardinalities

        binary_loss = F.binary_cross_entropy_with_logits(
            binary.tensor,
            c_train[:, bin_g['concept_idx']].float()
        )
        categorical_loss = sum(
            F.cross_entropy(categorical.tensor[:, cum[i]:cum[i+1]], c_train[:, cat_g['concept_idx'][i]].long())
            for i in range(len(cat_ann.labels))
        )
        task_loss = F.cross_entropy(y_pred, y_train)
        loss = binary_loss + categorical_loss + concept_reg * task_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 100 == 0:
            task_acc = (y_pred.detach().argmax(dim=1) == y_train).float().mean().item()
            bin_acc = ((binary.tensor.detach().sigmoid() > 0.5).float() == c_train[:, bin_g['concept_idx']].float()).float().mean().item()
            cat_acc = sum(
                (categorical.tensor.detach()[:, cum[i]:cum[i+1]].argmax(dim=1) == c_train[:, cat_g['concept_idx'][i]].long()).float().mean().item()
                for i in range(len(cat_ann.labels))
            ) / len(cat_ann.labels)
            print(f"Epoch {epoch}: Loss {loss.item():.4f} | Task Acc: {task_acc:.4f} | Binary C. Acc: {bin_acc:.4f} | Categorical C. Acc: {cat_acc:.4f}")

    return


if __name__ == "__main__":
    main()
