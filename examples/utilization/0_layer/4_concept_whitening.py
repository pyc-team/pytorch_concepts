"""
Concept Whitening (Low-Level Interface)
========================================

Concept Whitening (Chen, Bei & Rudin, Nature Machine Intelligence 2020)
replaces a normalization layer with one that (1) whitens the representation
and (2) rotates it so that chosen axes align with human concepts.

This example reproduces the paper's core claim on a CelebA subset: swapping
BatchNorm for ConceptWhitening makes single axes readable as concept
detectors at no cost to task accuracy. Both models are identical except for
the normalization layer; the CW model additionally runs a concept-alignment
step every 30 batches (the paper's schedule) using each concept's positive
examples -- no concept labels ever enter the task loss.

We measure interpretability as each axis's correlation with its assigned
concept label ("purity"): high for CW's aligned axes, arbitrary for the
same axis indices under BatchNorm.

Key Components:
- ConceptWhitening: whitens + rotates a fixed-size embedding, preserving its
  dimension (drop-in normalization layer, PyC's ``torch_concepts.nn``)
- WhitenedEmbeddingToConcept: BaseConceptLayer wrapper of the above, for
  CBM-style bottleneck pipelines (not used here -- this example keeps the
  full-width, paper-faithful setting where the task head sees everything)

Dataset: CelebA (a 4000-image subset, frozen ResNet18 embeddings)
Concepts aligned: Smiling (axis 0), Male (axis 1), Blond_Hair (axis 2)
Task: predict 'Attractive' from the full 128-dim whitened representation
"""
import torch

from torch_concepts import ImageBackbone, seed_everything
from torch_concepts.data import CelebADataModule
from torch_concepts.nn import ConceptWhitening

CONCEPTS = ["Smiling", "Male", "Blond_Hair"]
TASK = "Attractive"
LATENT = 128


def load_embeddings(n_samples=4000):
    """Frozen ResNet18 embeddings + binary attributes for a CelebA subset."""
    dm = CelebADataModule(
        root="./data/celeba",
        concept_subset=CONCEPTS + [TASK],
        max_samples=n_samples,
        splitter=None,   # required alongside max_samples (see CelebADataModule docs)
    )
    dm.precompute_embeddings(ImageBackbone("resnet18"), cache=False)
    return dm.dataset.input_data, dm.dataset.concepts.float()


def purity(axis_activations, concept_labels):
    """Pearson correlation between one axis and one binary concept label."""
    return torch.corrcoef(torch.stack([axis_activations, concept_labels]))[0, 1].item()


def train(norm, x_train, c_train, y_train, epochs=30, batch_size=256):
    """Train projection -> `norm` -> task head; align concepts if `norm` is CW.

    The normalization layer preserves the embedding dimension, so the task
    head always sees the full representation -- no bottleneck.
    """
    is_cw = isinstance(norm, ConceptWhitening)
    proj = torch.nn.Linear(x_train.shape[1], LATENT)
    head = torch.nn.Linear(LATENT, 1)
    optimizer = torch.optim.AdamW(
        list(proj.parameters()) + list(norm.parameters()) + list(head.parameters()),
        lr=0.001,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    step = 0
    model = torch.nn.Sequential(proj, norm, head)
    for _ in range(epochs):
        for batch in torch.randperm(len(x_train)).split(batch_size):
            model.train()
            optimizer.zero_grad()
            loss_fn(model(x_train[batch]), y_train[batch]).backward()
            optimizer.step()

            # CW-specific step (paper: every 30 batches): align each
            # concept's axis on that concept's positive examples. Just a
            # forward pass per concept inside `align` -- no gradient step,
            # no concept term in the task loss.
            step += 1
            if is_cw and step % 30 == 0:
                with torch.no_grad():
                    for axis in range(len(CONCEPTS)):
                        with norm.align(axis):
                            norm(proj(x_train[c_train[:, axis] == 1]))
                norm.update_rotation_matrix()

    return torch.nn.Sequential(proj, norm), head


def main():
    seed_everything(7)

    print("Loading CelebA subset + extracting frozen ResNet18 embeddings...")
    embeddings, attrs = load_embeddings()
    n_train = int(0.8 * len(embeddings))
    x_train, x_test = embeddings[:n_train], embeddings[n_train:]
    c_train, c_test = attrs[:n_train, :-1], attrs[n_train:, :-1]
    y_train, y_test = attrs[:n_train, -1:], attrs[n_train:, -1:]

    results = {}
    for name, norm in [
        ("BatchNorm", torch.nn.BatchNorm1d(LATENT)),
        ("ConceptWhitening", ConceptWhitening(in_features=LATENT)),
    ]:
        print(f"Training with {name}...")
        seed_everything(7)  # identical init/batches for a fair comparison
        encoder, head = train(norm, x_train, c_train, y_train)
        encoder.eval(), head.eval()
        with torch.no_grad():
            z = encoder(x_test)
            acc = ((head(z) > 0).float() == y_test).float().mean().item()
        results[name] = (acc, z)

    print(f"\nTask accuracy ('{TASK}'):")
    for name, (acc, _) in results.items():
        print(f"  {name:<18} {acc:.2f}")

    print(f"\nConcept purity (correlation of the concept's assigned axis with "
          f"its label):\n{'axis':<8}{'concept':<14}{'BatchNorm':<12}ConceptWhitening")
    for axis, concept in enumerate(CONCEPTS):
        row = [purity(results[name][1][:, axis], c_test[:, axis])
               for name in results]
        print(f"{axis:<8}{concept:<14}{row[0]:+.2f}        {row[1]:+.2f}")

    print(
        "\nSame architecture, same task accuracy -- but under CW each aligned "
        "axis is a\nreadable detector for its concept, while BatchNorm axes "
        "carry no assigned\nmeaning. The task head sees the full "
        f"{LATENT}-dim representation throughout:\nCW adds interpretability "
        "without a concept bottleneck."
    )


if __name__ == "__main__":
    main()
