"""
Concept Whitening
=================

Concept Whitening (CW) replaces a normalization layer with one that whitens the
representation and rotates it, so that chosen axes align with human concepts.
Swapping BatchNorm for CW makes single axes readable as concept detectors, at
no cost in task accuracy and without a concept bottleneck.

The two models below are identical except for the normalization layer. The CW
model also aligns its axes every 30 batches, on each concept's positive
examples; concept labels never enter the task loss. An axis is interpretable
if it correlates with its concept ("purity").

Data: 4,000 CelebA images (the dataset, ~1.4 GB, is downloaded on first run),
embedded by a frozen ResNet18. Concepts: Smiling, Male, Blond_Hair (axes 0, 1,
2). Task: Attractive.

Reference: Chen, Bei & Rudin, "Concept Whitening for Interpretable Image
Recognition", Nature Machine Intelligence 2020.
"""

# %%
import torch

from torch_concepts import ImageBackbone, seed_everything
from torch_concepts.data import CelebADataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import ConceptWhitening

seed_everything(7)

# %%
# Data
# ----
# Frozen ResNet18 embeddings, computed once and cached next to the dataset.
concepts, task = ["Smiling", "Male", "Blond_Hair"], "Attractive"
datamodule = CelebADataModule(
    root=str(DATA_ROOT / "celeba"),
    concept_subset=concepts + [task],
    max_samples=4000,
    seed=7,
    splitter=None,  # required with max_samples
)
datamodule.precompute_embeddings(ImageBackbone("resnet18"), cache=True)

x = datamodule.dataset.input_data
c = datamodule.dataset.concepts[concepts].float()
y = datamodule.dataset.concepts[[task]].float()
n_train = int(0.8 * len(x))
x_train, x_test = x[:n_train], x[n_train:]
c_train, c_test = c[:n_train], c[n_train:]
y_train, y_test = y[:n_train], y[n_train:]


# %%
# Training
# --------
# Projection -> normalization -> task head. The normalization layer keeps the
# size of the representation, so the head always sees all of it.
def train(norm, epochs=30, batch_size=256):
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 128),
        norm,
        torch.nn.Linear(128, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    step = 0
    for _ in range(epochs):
        for batch in torch.randperm(n_train).split(batch_size):
            model.train()
            optimizer.zero_grad()
            loss_fn(model(x_train[batch]), y_train[batch]).backward()
            optimizer.step()
            step += 1
            # CW only, the paper's schedule: every 30 batches, align each
            # concept's axis on that concept's positive examples (a forward pass
            # inside ``align``, no gradient step).
            if isinstance(norm, ConceptWhitening) and step % 30 == 0:
                with torch.no_grad():
                    for axis in range(len(concepts)):
                        with norm.align(axis):
                            norm(model[0](x_train[c_train[:, axis] == 1]))
                norm.update_rotation_matrix()
    return model.eval()


results = {}
for name, norm in [
    ("BatchNorm", torch.nn.BatchNorm1d(128)),
    ("ConceptWhitening", ConceptWhitening(128)),
]:
    seed_everything(7)  # same initialization and batches for both models
    model = train(norm)
    with torch.no_grad():
        z = model[:2](x_test)  # the normalized representation
        accuracy = ((model[2](z) > 0).float() == y_test).float().mean().item()
    results[name] = accuracy, z

# %%
# Evaluation
# ----------
print(f"task accuracy ({task}):")
for name, (accuracy, _) in results.items():
    print(f"  {name:<18} {accuracy:.2f}")

print("purity of the aligned axes (correlation with their concept):")
print(f"  {'concept':<12} {'BatchNorm':>10} {'ConceptWhitening':>17}")
for axis, concept in enumerate(concepts):
    purity = [
        torch.corrcoef(torch.stack([z[:, axis], c_test[:, axis]]))[0, 1].item()
        for _, z in results.values()
    ]
    print(f"  {concept:<12} {purity[0]:>+10.2f} {purity[1]:>+17.2f}")
