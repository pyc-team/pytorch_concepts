"""
Single-cell Comparison with scCBGM
=================================

``SingleCellDataset`` generates synthetic single-cell gene-expression data
following scCBGM. Each row is a cell, each column is a gene, and the values are
raw Poisson counts. Binary concepts describe simulated cellular programs that
affect gene expression alongside cell type, tissue, batch, library size and
noise. The binary concepts are the training labels and intervention targets;
the other values form the cell background, held fixed during interventions.

``SingleCellDataModule`` provides loading, splitting and batching like any
other PyC dataset.

Original: notebooks/demo/2025-09-11-cbm-datagen-tutorial.ipynb,
section "Counterfactual Generation", at:
https://github.com/almaan/scCBGM/blob/5436a2bc3f0e341e755cfa6de2c9636095f2854c/notebooks/demo/2025-09-11-cbm-datagen-tutorial.ipynb

Generate the same 250 cells and 50 genes, then turn on the concept with the
largest variance. Save raw arrays for numerical comparison with upstream,
and plot expression before/after intervention per cell and on average.
The plots use the same genes and affected-cell indices shown in the saved
tutorial figures; its original plotting selections were not seeded.
A small CBM trains an MLP backbone and its concept/task heads together on the
original counts. The last concept is a toy prediction target; the short
training run checks the complete pipeline.

Data: 250 synthetic cells, 50 genes and 5 binary concepts, generated locally.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_concepts import seed_everything
from torch_concepts.data import SingleCellDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import ConceptBottleneckModel, WeightedConceptLoss

seed_everything(42)

# %%
# Data
# ----
# Same generation parameters as the original tutorial; other settings keep
# their defaults, including coefficient sparsity and independent concept logits.
dm = SingleCellDataModule(
    root=DATA_ROOT / "single_cell_comparison",
    seed=42,
    generation_seed=42,
    batch_size=64,
    generation_config={
        "cells": {"n": 250},
        "genes": {"n": 50},
        "tissue": {"n": 1},
        "celltype": {"n": 1},
        "batch": {"n": 1},
        "original_concept": {"n": 5, "std": 1},
    },
)
dm.setup()
original = dm.dataset
print(f"Dataset size: {dm.n_samples} cells")
print(f"Genes: {dm.n_features[0]}")
print(f"Concepts: {dm.concept_names}")

batch = next(iter(dm.train_dataloader()))
print(f"inputs:   {tuple(batch['inputs']['x'].shape)}")
print(f"concepts: {tuple(batch['concepts']['c'].shape)}")

counts_before = original.input_data.numpy()
concepts_before = original.concepts.tensor.numpy()

# %%
# Intervention
# ------------
# As in the tutorial, turn 0 into 1 for the concept with the largest variance.
concept_index = int(np.argmax(np.var(concepts_before, axis=0)))
concept_name = original.concept_names[concept_index]
changed_cells = concepts_before[:, concept_index] == 0
edited = original.generate_interventions({concept_name: 1})
counts_after = edited.input_data.numpy()
concepts_after = edited.concepts.tensor.numpy()
print(f"Generated {counts_before.shape[0]} cells and {counts_before.shape[1]} genes")
print(f"Intervention: {concept_name}, 0 -> 1 in {changed_cells.sum()} cells")

# Cells where the concept was already 1 retain exactly the same counts.
np.testing.assert_array_equal(counts_before[~changed_cells], counts_after[~changed_cells])

# %%
# Save the comparison
# -------------------
# Compare only the cells whose concept value changed, as in the tutorial.
coefficients = original.model_parameters["concept_effect"][concept_index].numpy()
rate_multiplier = np.exp(coefficients)
pre_pert_cells = counts_before[changed_cells]
post_pert_cells = counts_after[changed_cells]
mean_before = pre_pert_cells.mean(axis=0, dtype=np.float64)
mean_after = post_pert_cells.mean(axis=0, dtype=np.float64)
# exp(coefficient) scales Poisson rates; sampled counts need not scale exactly.
scaled_mean = mean_before * rate_multiplier

output = Path("outputs/single_cell_comparison")
output.mkdir(parents=True, exist_ok=True)
np.savez(
    output / "comparison.npz",
    counts_before=counts_before, counts_after=counts_after,
    concepts_before=concepts_before, concepts_after=concepts_after,
    concept_name=concept_name, changed_cells=changed_cells,
    concept_coefficients=coefficients,
    mean_before=mean_before, mean_after=mean_after, scaled_mean=scaled_mean,
)

# %%
# Expression per cell
# -------------------
# Match the selections displayed in the pinned notebook's saved figure.
# Observation indices refer to the affected-cell subset, as in the tutorial.
genes = [44, 18, 20, 12, 47, 10, 33, 17, 15, 29]
observations = [80, 56, 26, 13, 39]
y = np.arange(len(genes))
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for observation, ax in zip(observations, axes):
    before = pre_pert_cells[observation, genes]
    after = post_pert_cells[observation, genes]
    for j, (pre, post) in enumerate(zip(before, after)):
        if abs(post - pre) > 0.05:
            ax.annotate("", xy=(post, j), xytext=(pre, j),
                        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5})
        ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.4)
    ax.scatter(before, y, color="red", zorder=3)
    ax.scatter(after, y, color="blue", zorder=3)
    ax.set(title=f"Observation: {observation}", xlabel="Gene", ylabel="Feature")
    ax.set_yticks(y, [f"gene_{g} ({rate_multiplier[g]:.2f})" for g in genes])
fig.tight_layout()
fig.savefig(output / "per_cell.pdf", bbox_inches="tight")

# %%
# Average expression
# ------------------
# Average comparison across affected cells, including the coefficient multiplier.
fig, ax = plt.subplots(figsize=(15, 4))
x = np.arange(counts_before.shape[1])
for i in x:
    if abs(rate_multiplier[i] - 1) > 0.05:
        ax.annotate("", xy=(i, mean_after[i]), xytext=(i, mean_before[i]),
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5})
        ax.annotate("", xy=(i + 0.2, scaled_mean[i]), xytext=(i + 0.2, mean_before[i]),
                    arrowprops={"arrowstyle": "->", "color": "darkgreen", "lw": 1.5,
                                "linestyle": "dotted"})
ax.scatter(x, mean_before, color="red", label="pre", zorder=3)
ax.scatter(x, mean_after, color="blue", label="post", zorder=3)
ax.set_xticks(x, [f"var_{g}" for g in x], rotation=90)
ax.set_title("Average Change Pre to Post perturbation")
ax.legend()
fig.savefig(output / "average.pdf", bbox_inches="tight")
print(f"Saved arrays and figures to {output}")

# %%
# Model
# -----
# Predict the first four concepts from genes, then the last concept from them.
n_genes = dm.n_features[0]
task_names = [dm.concept_names[-1]]
backbone = torch.nn.Sequential(torch.nn.Linear(n_genes, 32), torch.nn.ReLU())
model = ConceptBottleneckModel(
    input_size=n_genes,
    annotations=dm.annotations,
    task_names=task_names,
    backbone=backbone,
    latent_size=32,
)
loss_fn = WeightedConceptLoss(
    concept_weight=1.0,
    task_weight=1.0,
    task_names=task_names,
    binary=torch.nn.BCEWithLogitsLoss(),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
initial_weight = backbone[0].weight.detach().clone()

# %%
# Training
# --------
# Train the backbone and heads together directly on raw counts.
for epoch in range(5):
    model.train()
    total_loss = 0.0
    for batch in dm.train_dataloader():
        inputs = batch["inputs"]["x"]
        optimizer.zero_grad()
        out = model(query=dm.concept_names, input=inputs)
        loss = loss_fn(out, batch["concepts"]["c"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(inputs)
    print(f"Epoch {epoch + 1}: loss = {total_loss / len(dm.trainset):.3f}")
print(f"Backbone updated: {not torch.equal(initial_weight, backbone[0].weight)}")

# %%
# Evaluation
# ----------
indices = dm.testset.indices
inputs = original.input_data[indices]
targets = original.concepts[indices]
model.eval()
with torch.no_grad():
    out = model(query=dm.concept_names, input=inputs)
for name in dm.concept_names:
    accuracy = ((out.logits[name] > 0) == targets[name]).float().mean()
    print(f"Test accuracy of {name}: {accuracy.item():.3f}")

plt.show()
