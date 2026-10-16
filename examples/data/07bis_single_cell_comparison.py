"""
Single-cell Dataset Generation Compared with scCBGM
==================================================

Reproduce the initial generation and new-cell generation in the scCBGM tutorial:
https://github.com/almaan/scCBGM/blob/main/notebooks/demo/2025-09-11-cbm-datagen-tutorial.ipynb

Compare complete count matrices, concepts, coefficients and cell categories
against the original generator. Requires ``conceptlab`` from a scCBGM checkout,
as in the original notebook, including its ``xarray`` dependency.

scCBGM provides ``conceptlab.utils.set_seed``, but neither this tutorial nor
``scripts/generate_synthetic_data.py`` calls it. The generator's ``seed=42``
only seeds ``default_rng``; categories use NumPy's separate global RNG.
The reproducibility guide does not record that RNG's state for the tutorial.

Here we seed both streams and copy the global state for PyC. For the same run
in the original notebook, call ``clab.utils.set_seed(42)`` before generation,
then run through ``generate_from_dataset`` without resetting the RNG.
This compares exact outputs of a reproducible run; the historical saved
notebook tables cannot be guaranteed without their original global RNG state.

Data: 5,000 cells, 1,000 genes, 3 tissues, 10 cell types, 2 batches and 8 binary
concepts, followed by 10 new cells with the same global parameters.
"""

# %%
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from conceptlab.datagen.omics import OmicsDataGenerator

from torch_concepts import seed_everything
from torch_concepts.data import SingleCellDataset
from torch_concepts.data.datasets import single_cell

seed_everything(42)
categorical_rng = np.random.RandomState()
categorical_rng.set_state(np.random.get_state())
pyc_model = single_cell.model

# %%
# RNG alignment
# -------------
# scCBGM draws categories with global np.random, and all other quantities
# with default_rng. Preserve that split without changing either generator.
# This initial tutorial section uses independent concept logits.
def rng_aligned_model(*args, rng=None, **kwargs):
    rng = np.random.default_rng(42) if rng is None else rng
    aligned = SimpleNamespace(
        multinomial=categorical_rng.multinomial,
        multivariate_normal=rng.multivariate_normal,
        binomial=rng.binomial,
        normal=rng.normal,
        uniform=rng.uniform,
    )
    return pyc_model(*args, rng=aligned, **kwargs)


# %%
# Data
# ----
# A fresh cache ensures neither generation skips any RNG draws on reruns.
cache = TemporaryDirectory(prefix="pyc_sccbGM_comparison_")
reference = OmicsDataGenerator.generate(
    n_obs=5000, n_vars=1000, n_tissues=3, n_celltypes=10,
    n_batches=2, n_concepts=8, seed=42,
)
with patch.object(single_cell, "model", rng_aligned_model):
    dataset = SingleCellDataset(
        root=cache.name, seed=42,
        generation_config={
            "cells": {"n": 5000}, "genes": {"n": 1000},
            "tissue": {"n": 3}, "celltype": {"n": 10},
            "batch": {"n": 2}, "original_concept": {"n": 8},
        },
    )
print(reference)
print(dataset)

# %%
# Gene expression
# ---------------
# Same layout and labels as dataset.data.to_dataframe().unstack() upstream.
reference_data = reference.data.to_dataframe().unstack()
data = pd.DataFrame(dataset.input_data.numpy(),
                    index=reference_data.index, columns=reference_data.columns)
# Counts use float32 in PyC and float64 upstream; compare their exact values.
pd.testing.assert_frame_equal(data, reference_data, check_dtype=False, check_exact=True)
print("scCBGM counts:")
print(reference_data)
print("PyC counts:")
print(data)

# %%
# Concepts and coefficients
# -------------------------
parameters = dataset.model_parameters
for label, values, expected in (
    ("Concept indicators", dataset.concepts.tensor.numpy(),
     reference.concepts.to_dataframe().unstack()),
    ("Concept coefficients", parameters["concept_effect"].numpy(),
     reference.concept_coef.to_dataframe().unstack()),
    ("Tissue coefficients", parameters["tissue_effect"].numpy(),
     reference.tissue_coef.to_dataframe().unstack()),
):
    frame = pd.DataFrame(values, index=expected.index, columns=expected.columns)
    pd.testing.assert_frame_equal(frame, expected, check_dtype=False, check_exact=True)
    print(f"{label} (identical to scCBGM):")
    print(frame)

tissues = pd.DataFrame(
    {"tissues": [f"tissue_{i}" for i in dataset.cell_background["tissue"].tolist()]},
    index=reference_data.index,
).unstack()
pd.testing.assert_series_equal(tissues, reference.tissues.to_dataframe().unstack())
print(tissues)

# %%
# New cells
# ---------
# Do not reset the categorical RNG: upstream continues its global stream.
# Both implementations restart the other RNG at seed 42 for the new cells.
new_reference = OmicsDataGenerator.generate_from_dataset(n_obs=10, dataset=reference)
with patch.object(single_cell, "model", rng_aligned_model):
    new_dataset = dataset.generate_from_dataset(n_cells=10, seed=42)

new_reference_data = new_reference.data.to_dataframe().unstack()
new_data = pd.DataFrame(new_dataset.input_data.numpy(),
                        index=new_reference_data.index, columns=new_reference_data.columns)
pd.testing.assert_frame_equal(new_data, new_reference_data, check_dtype=False, check_exact=True)
np.testing.assert_array_equal(new_dataset.concepts.tensor.numpy(), new_reference.concepts.values)
print("scCBGM new counts:")
print(new_reference_data)
print("PyC new counts:")
print(new_data)

# %%
# Cell identities and background
# -----------------------------
for actual, expected in ((dataset, reference), (new_dataset, new_reference)):
    background = actual.cell_background
    np.testing.assert_array_equal(
        [f"obs_{i}" for i in background["cell_indices"].tolist()], expected.obs.values,
    )
    for name, upstream_name in (("batch", "batches"), ("tissue", "tissues"),
                                ("celltype", "celltypes")):
        np.testing.assert_array_equal(
            [f"{name}_{i}" for i in background[name].tolist()], expected[upstream_name].values,
        )
    for name, upstream_name in (("libsize", "libsize"), ("noise", "noise"),
                                ("exog_noise", "exogenous_noise"),
                                ("bernoulli_prob", "bernoulli_prob")):
        np.testing.assert_array_equal(background[name].numpy(), expected[upstream_name].values)
print("Exact comparison passed for all initial and new counts, concepts and background.")
