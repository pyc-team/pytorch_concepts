"""Synthetic single-cell counts following scCBGM's OmicsDataGenerator.

Reference: https://github.com/almaan/scCBGM/blob/main/conceptlab/datagen/omics.py
"""

import hashlib
import json
import os
from collections.abc import Mapping
from numbers import Real

import numpy as np
import torch
from scipy.stats import poisson, wishart

from ...annotations import Annotations
from ..base.dataset import ConceptDataset


_COUNT_QUANTITIES = ("cells", "genes", "batch", "celltype", "tissue", "concept")

# Supported quantities and their default counts and distribution parameters.
_DEFAULT_GENERATION_CONFIG = {
    "cells": {"n": 10000},
    "genes": {"n": 1000},
    "batch": {"n": 3, "std": 0.08},
    "celltype": {"n": 4, "std": 0.08},
    "tissue": {"n": 2, "std": 0.07},
    "concept": {
        "n": 8,
        "std": 0.05,
        "zero_inflate": True,
        "beta_a": 1,
        "beta_b": 0.5,
        "use_dependency": False,
        "cov_prior_range": 5,
    },
    "baseline": {"lower": 1, "upper": 5},
    "library": {"std_lower": 0.01, "std_upper": 0.03},
    "noise": {"std": 0.01},
}


def _resolve_generation_config(generation_config):
    """Merge partial quantity settings into independent defaults and validate them."""
    config = {quantity: params.copy() for quantity, params in _DEFAULT_GENERATION_CONFIG.items()}
    if generation_config is not None:
        if not isinstance(generation_config, Mapping):
            raise TypeError("generation_config must be a mapping of quantities to parameter mappings.")
        for quantity, params in generation_config.items():
            if quantity not in config:
                raise ValueError(f"Unknown generation quantity: {quantity!r}.")
            if not isinstance(params, Mapping):
                raise TypeError(f"generation_config[{quantity!r}] must be a parameter mapping.")
            for name in params:
                if name not in config[quantity]:
                    raise ValueError(f"Unknown parameter {name!r} for generation quantity {quantity!r}.")
            config[quantity].update(params)

    for quantity, params in config.items():
        for name, value in params.items():
            if name == "n":
                continue
            if name in ("zero_inflate", "use_dependency"):
                if not isinstance(value, (bool, np.bool_)):
                    raise TypeError(f"Expected {quantity}.{name} to be a boolean.")
                params[name] = bool(value)
            else:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                    raise TypeError(f"Expected {quantity}.{name} to be a real number.")
                if not np.isfinite(value):
                    raise ValueError(f"Expected {quantity}.{name} to be finite.")
                params[name] = float(value)
    for quantity in _COUNT_QUANTITIES:
        n = config[quantity]["n"]
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"Expected {quantity}.n to be a positive integer.")
        config[quantity]["n"] = int(n)
    baseline_config = config["baseline"]
    library_config = config["library"]
    concept_config = config["concept"]
    if not 0 < baseline_config["lower"] <= baseline_config["upper"]:
        raise ValueError("Expected 0 < baseline.lower <= baseline.upper.")
    if not 0 <= library_config["std_lower"] <= library_config["std_upper"]:
        raise ValueError("Expected 0 <= library.std_lower <= library.std_upper.")
    if min(config[factor]["std"] for factor in ("batch", "celltype", "tissue", "concept", "noise")) < 0:
        raise ValueError("Noise and coefficient standard deviations must be nonnegative.")
    if concept_config["beta_a"] <= 0 or concept_config["beta_b"] <= 0 or concept_config["cov_prior_range"] < 0:
        raise ValueError("Beta parameters must be positive and cov_prior_range nonnegative.")
    return config


def _generate_single_cell(generation_config=None, seed=42):
    """Sample counts and concepts using grouped generation settings and a local RNG."""
    config = _resolve_generation_config(generation_config)
    n_obs = config["cells"]["n"]
    n_vars = config["genes"]["n"]
    n_batches = config["batch"]["n"]
    n_tissues = config["tissue"]["n"]
    n_celltypes = config["celltype"]["n"]
    n_concepts = config["concept"]["n"]
    baseline_config = config["baseline"]
    library_config = config["library"]
    concept_config = config["concept"]

    rng = np.random.default_rng(seed)
    # Sample parameters for batches, tissues, cell types, and concepts
    p_batch = rng.dirichlet(10 * np.ones(n_batches)) # probability of sampling each batch
    p_tissue = rng.dirichlet(10 * np.ones(n_tissues)) # probability of sampling each tissue
    p_celltype = rng.dirichlet(np.ones(n_celltypes), size=n_tissues) # probability of sampling each cell type within a tissue
    p_concept = rng.dirichlet(np.ones(n_concepts) / 5, size=n_celltypes) # probability of sampling each concept within a cell type
    std_libsize = rng.uniform(library_config["std_lower"], library_config["std_upper"], size=n_batches) # standard deviation of library size for each batch

    # Sample effect coefficients for batches, cell types, tissues, and concepts
    batch_effect = rng.normal(0, config["batch"]["std"], size=(n_batches, n_vars)) # batch effect coefficients for each batch and gene
    celltype_effect = rng.normal(0, config["celltype"]["std"], size=(n_celltypes, n_vars)) # cell type effect coefficients for each cell type and gene
    tissue_effect = rng.normal(0, config["tissue"]["std"], size=(n_tissues, n_vars)) # tissue effect coefficients for each tissue and gene
    concept_effect = rng.normal(0, concept_config["std"], size=(n_concepts, n_vars)) # concept effect coefficients for each concept and gene

    baseline_expr = rng.uniform(np.log(baseline_config["lower"]), np.log(baseline_config["upper"]), size=n_vars)

    covariance = np.zeros((n_concepts, n_concepts))
    if concept_config["use_dependency"]:
        scale = np.zeros_like(covariance)
        indices = np.tril_indices_from(scale)
        cov_prior_range = concept_config["cov_prior_range"]
        scale[indices] = rng.uniform(-cov_prior_range, cov_prior_range, size=(n_concepts * (n_concepts + 1) // 2))
        scale = scale @ scale.T + np.eye(n_concepts) * 1e-6
        covariance = np.atleast_2d(wishart.rvs(df=n_concepts, scale=scale, random_state=rng))
    

    if concept_config["zero_inflate"]:
        # As in scCBGM, sparsity masks concept coefficients, not observed counts.
        probability = 1 - rng.beta(concept_config["beta_a"], concept_config["beta_b"], size=n_concepts)
        concept_effect *= rng.binomial(1, probability[:, None], size=concept_effect.shape)

    inputs = np.empty((n_obs, n_vars), dtype=np.float32)
    concepts = np.empty((n_obs, n_concepts + 3), dtype=np.float32)
    for i in range(n_obs):

        # Sample indices for batch, tissue, and cell type for the current observation
        batch = rng.multinomial(1, p_batch).argmax() # Sample a batch index for the current observation
        tissue = rng.multinomial(1, p_tissue).argmax() # Sample a tissue index for the current observation
        celltype = rng.multinomial(1, p_celltype[tissue]).argmax() # Sample a cell type index for the current observation based on the tissue
        
        # Sample concept indicators for the current observation
        probability = p_concept[celltype] # Get the concept probabilities for the sampled cell type
        mean = np.log((probability + 1e-8) / (1 - probability + 1e-8))
        logits = np.clip(rng.multivariate_normal(mean, covariance), -8, 8) # Sample concept logits with dependencies
        concept_indicators = rng.binomial(1, 1 / (1 + np.exp(-logits))) # Convert logits to prob again, after considering dependencies

        # Get library size for the current observation
        libsize = rng.normal(0, std_libsize[batch])
        # Get noise for the current observation
        noise = rng.normal(0, config["noise"]["std"], size=n_vars)

        # Compute the log Poisson rate for the current observation
        log_rate = (
            baseline_expr + concept_indicators @ concept_effect + celltype_effect[celltype]
            + batch_effect[batch] + tissue_effect[tissue] + libsize + noise
        )
        inputs[i] = poisson.ppf(rng.uniform(0, 1, size=n_vars), np.exp(log_rate))
        concepts[i] = np.concatenate([concept_indicators, [celltype, tissue, batch]])
    return torch.from_numpy(inputs), torch.from_numpy(concepts)


class SingleCellDataset(ConceptDataset):
    """Dataset class for synthetic single-cell data from scCBGM.

    This dataset models the relationship between binary concepts, cell type,
    tissue, batch effects, and gene expression as raw Poisson counts.
    Concept annotations include binary indicators and class indices for
    celltype, tissue, and batch. No directed concept graph is assigned.

    ``generation_config`` groups counts and distribution parameters by quantity.
    ``n`` counts cells, genes, batches, cell types, tissues, or binary concepts;
    the concept count excludes the celltype, tissue, and batch annotations.
    Effect contributions sum to the log Poisson rate of each gene. Omitted
    quantities and parameters use these defaults::

        {
            "cells": {"n": 10000},
            "genes": {"n": 1000},
            "batch": {"n": 3, "std": 0.08},
            "celltype": {"n": 4, "std": 0.08},
            "tissue": {"n": 2, "std": 0.07},
            "concept": {
                "n": 8,
                "std": 0.05,
                "zero_inflate": True,
                "beta_a": 1,
                "beta_b": 0.5,
                "use_dependency": False,
                "cov_prior_range": 5,
            },
            "baseline": {"lower": 1, "upper": 5},
            "library": {"std_lower": 0.01, "std_upper": 0.03},
            "noise": {"std": 0.01},
        }

    Baseline bounds are expression rates, sampled uniformly in log space.
    The ``std`` parameters control effect coefficients and gene noise in log
    space; library size adds a cell-level log offset with a batch-specific
    standard deviation drawn between ``std_lower`` and ``std_upper``.
    ``zero_inflate`` masks concept coefficients using the beta parameters.
    ``use_dependency`` introduces covariance between concept logits, using
    uniform draws in [-cov_prior_range, cov_prior_range] to construct the
    Wishart scale. Unknown quantities or parameter names raise ``ValueError``.

    Example::

        dataset = SingleCellDataset(
            seed=42,
            generation_config={"cells": {"n": 1000}, "batch": {"n": 2, "std": 0.15}},
        )
    """

    def __init__(
        self,
        root: str | None = None,  # root directory to store/load the dataset
        seed: int = 42,  # seed for data generation
        concept_subset: list | None = None,  # subset of concept labels (concept_i, celltype, tissue, batch)
        generation_config: dict | None = None,  # partial counts and distribution parameters grouped by quantity
        label_descriptions: dict | None = None,  # mapping from concept names to descriptions
    ):
        # If root is not provided, use a local folder automatically
        self.root = os.fspath(root) if root is not None else os.path.join(os.getcwd(), "data", "single_cell")
        self.seed = seed
        self.label_descriptions = label_descriptions
        self.generation_config = _resolve_generation_config(generation_config)
        self.n_gen = self.generation_config["cells"]["n"]

        # inputs is a torch tensor of gene counts (cells.n, genes.n)
        # concepts is a torch tensor of binary indicators and context class indices
        inputs, concepts = self.load()

        # Get concept annotations: binary indicators followed by celltype, tissue, batch
        n_concepts = self.generation_config["concept"]["n"]
        context_sizes = [self.generation_config[quantity]["n"] for quantity in ("celltype", "tissue", "batch")]
        annotations = Annotations(
            labels=[f"concept_{i}" for i in range(n_concepts)] + ["celltype", "tissue", "batch"],
            cardinalities=[1] * n_concepts + context_sizes,
            types=["binary"] * n_concepts + ["categorical" if n > 1 else "binary" for n in context_sizes],
        )

        # Initialize parent class
        super().__init__(
            input_data=inputs,
            concepts=concepts,
            annotations=annotations,
            concept_names_subset=concept_subset,  # subset of concept names
            name="SingleCellDataset",
        )

    @property
    def raw_filenames(self) -> list[str]:
        """List of raw filenames that need to be present in the raw directory
        for the dataset to be considered present."""
        return []  # nothing to download, data is generated locally

    @property
    def processed_filenames(self) -> list[str]:
        """List of processed filenames that will be created during build step."""
        # Include all generation settings in the cache key
        settings = json.dumps({"generation_config": self.generation_config, "seed": self.seed}, sort_keys=True)
        key = hashlib.sha256(settings.encode()).hexdigest()[:16]
        return [f"single_cell_N_{self.n_gen}_seed_{self.seed}_{key}.pt"]

    def download(self):
        """No download needed; data is generated locally."""
        pass

    def build(self):
        # Generate gene counts and concept values
        inputs, concepts = _generate_single_cell(generation_config=self.generation_config, seed=self.seed)

        # Save inputs and concepts together in the processed file
        os.makedirs(self.root_dir, exist_ok=True)
        torch.save({"inputs": inputs, "concepts": concepts}, self.processed_paths[0])

    def load_raw(self):
        self.maybe_build()

        # Load the generated gene counts and concept values
        data = torch.load(self.processed_paths[0], weights_only=True)
        return data["inputs"], data["concepts"]
