"""Generate single-cell data, intervene on concepts and train a small CBM.

Generate 250 cells, 50 genes and 5 binary concepts. Normalize each cell to
10,000 total counts, then apply log1p for training.

Turn on the most variable concept to generate a paired dataset with the same
cells and background. Train the CBM on the original data for five epochs,
using the first four concepts to predict the last one as a toy task.
"""

# %%
import numpy as np
import torch

from torch_concepts import seed_everything
from torch_concepts.data import SingleCellDataModule
from torch_concepts.env import DATA_ROOT
from torch_concepts.nn import ConceptBottleneckModel, WeightedConceptLoss

seed_everything(42)

# %%
# Data
dm = SingleCellDataModule(
    root=DATA_ROOT / "single_cell_comparison",
    seed=42,
    generation_seed=42,
    batch_size=64,
    normalize=True,  # False keeps raw counts for training.
    generation_config={
        "n_cells": 250,
        "n_genes": 50,
        "n_tissue": 1,
        "n_celltype": 1,
        "n_batch": 1,
        "n_concepts": 5,
        "std_dev_concept": 1,
    },
)
dm.setup()
original = dm.dataset

# %%
# Intervention
concepts_before = original.native_concepts.tensor.numpy()
concept_index = int(np.argmax(np.var(concepts_before, axis=0)))
concept_name = original.concept_names[concept_index]

# Generate the counterfactual dataset with the intervention applied.
edited = original.generate_interventions({concept_name: 1})

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
