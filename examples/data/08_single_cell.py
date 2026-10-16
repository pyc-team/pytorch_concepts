"""
Single-cell Generation and Caching
==================================

Internal checks for ``SingleCellDataset``. The standalone public example is
``07_single_cell_comparison.py``.

Generate synthetic counts, reuse saved datasets, change concepts and model
parameters, and select cells and training concepts. Assertions check that the
source is preserved and that cached files are reused. A frozen toy backbone
also demonstrates separate embedding caches for original and intervened data.

Data: 600 synthetic cells, 80 genes and 3 binary concepts, generated locally.
Set ``PLOT_UMAP = True`` for UMAP plots; these require the optional
Scanpy package (``pip install scanpy``).
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from torch_concepts import seed_everything
from torch_concepts.data import SingleCellDataModule, SingleCellDataset
from torch_concepts.env import DATA_ROOT

seed_everything(42)
PLOT_UMAP = False


# %%
# Data
# ----
root = DATA_ROOT / "single_cell"
config = {
    "cells": {"n": 600}, "genes": {"n": 80}, "celltype": {"n": 3},
    "original_concept": {"n": 3, "std": 0.45, "zero_inflate": False},
}
original = SingleCellDataset(root=root, seed=42, generation_config=config)
print(f"Counts: {tuple(original.input_data.shape)}")
print(f"Concepts: {original.concept_names}")

# %%
# Dataset cache
# -------------
# An identical request reuses the same files without rewriting them.
saved_at = {Path(path): Path(path).stat().st_mtime_ns for path in original.processed_paths}
cached = SingleCellDataset(root=root, seed=42, generation_config=config)
assert cached.root == original.root
torch.testing.assert_close(cached.input_data, original.input_data)
assert all(path.stat().st_mtime_ns == timestamp for path, timestamp in saved_at.items())
print(f"Dataset cache reused: {original.root}")

# %%
# Interventions
# -------------
# Change only the named concepts, retaining the same cells and background.
edited = original.generate_interventions({"concept_0": 1, "concept_1": 0})
assert torch.all(edited.concepts.tensor[:, 0] == 1)
assert torch.all(edited.concepts.tensor[:, 1] == 0)
torch.testing.assert_close(edited.concepts.tensor[:, 2], original.concepts.tensor[:, 2])
for name, value in original.cell_background.items():
    torch.testing.assert_close(edited.cell_background[name], value)
torch.testing.assert_close(original.input_data, cached.input_data)

saved_at = Path(edited.generation_state_path).stat().st_mtime_ns
cached_edit = original.generate_interventions({"concept_0": 1, "concept_1": 0})
assert cached_edit.root == edited.root
assert Path(cached_edit.generation_state_path).stat().st_mtime_ns == saved_at
torch.testing.assert_close(cached_edit.input_data, edited.input_data)
print(f"Intervention cache reused: {edited.root}")

# %%
# New cells and parameter overrides
# --------------------------------
new_cells = original.generate_from_dataset(n_cells=200, seed=7)
for name, value in original.model_parameters.items():
    torch.testing.assert_close(new_cells.model_parameters[name], value)
print(f"Original: {len(original)} cells; new sample: {len(new_cells)} cells")

# With the same draws, a higher baseline can only increase the counts.
stronger = original.generate_from_dataset(
    n_cells=200, seed=7,
    model_parameters={"baseline_expr": original.model_parameters["baseline_expr"] + 0.5},
)
assert stronger.root != new_cells.root
assert torch.all(stronger.input_data >= new_cells.input_data)

# %%
# Row and concept selection
# -------------------------
dm = SingleCellDataModule(
    root=original.root, max_samples=12, concept_subset=["concept_1"],
    batch_size=4, seed=12,
)
dm.setup()
cell_indices = dm.dataset.cell_background["cell_indices"]
print(f"Selected cells: {cell_indices.tolist()}; training concepts: {dm.concept_names}")

# A partial DataFrame can also edit a concept excluded from training.
changes = pd.DataFrame({"concept_2": [1]}, index=[f"cell_{cell_indices[0].item()}"])
partial = dm.dataset.generate_interventions(changes)
assert len(partial) == 12
assert partial.load_raw()[1][0, 2] == 1
torch.testing.assert_close(partial.cell_background["cell_indices"], cell_indices)

# A saved variant loads independently, exposing all concepts unless selected.
restored = SingleCellDataset(root=partial.root)
assert restored.concepts.shape == (12, 3)
torch.testing.assert_close(restored.input_data, partial.input_data)
torch.testing.assert_close(restored.cell_background["cell_indices"], cell_indices)

# %%
# Backbone embedding cache
# ------------------------
# The seed fixes this toy backbone's weights across runs. Keep them frozen
# when reusing embeddings; changed weights require force=True on raw data.
backbone = torch.nn.Sequential(torch.nn.Linear(original.n_features[0], 16), torch.nn.ReLU())
backbone.requires_grad_(False)
backbone.source = "local"
backbone.filename = "single_cell_demo_mlp.pt"

# Use fresh instances so original and edited retain their raw gene counts.
# Each variant's directory keeps its backbone embeddings separate.
for dataset in (SingleCellDataset(root=original.root), SingleCellDataset(root=edited.root)):
    expected_embeddings = backbone(dataset.input_data)
    dataset.precompute_embeddings(backbone)
    torch.testing.assert_close(dataset.input_data, expected_embeddings)
    cache_path = Path(dataset.root) / f"single_cell_demo_mlp_n{len(dataset)}.pt"
    saved_at = cache_path.stat().st_mtime_ns
    reloaded = SingleCellDataset(root=dataset.root)
    reloaded.precompute_embeddings(backbone)
    assert reloaded.input_data.shape == (len(reloaded), 16)
    torch.testing.assert_close(reloaded.input_data, expected_embeddings)
    assert cache_path.stat().st_mtime_ns == saved_at
    print(f"Embedding cache reused: {cache_path}")

# %%
# Optional UMAP plots
# -------------------
# As in the original notebook: normalize, log1p, PCA, neighbors, then UMAP.
# Binary concepts use numeric colors; fixed background categories use legends.
if PLOT_UMAP:
    output = Path("outputs/single_cell")
    output.mkdir(parents=True, exist_ok=True)
    for name, dataset in [("original", original), ("intervention", edited)]:
        axes = dataset.plot(ncols=3, wspace=0.5)
        axes[0].figure.savefig(output / f"{name}_umap.pdf", bbox_inches="tight")
    plt.show()
