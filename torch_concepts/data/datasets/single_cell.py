"""Synthetic single-cell counts following scCBGM's OmicsDataGenerator.

Reference: https://github.com/almaan/scCBGM/blob/main/conceptlab/datagen/omics.py
"""

import hashlib
import json
import os
from collections.abc import Mapping
from numbers import Real
from venv import logger

import numpy as np
import pandas as pd
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
    if not 0 < config["baseline"]["lower"] <= config["baseline"]["upper"]:
        raise ValueError("Expected 0 < baseline.lower <= baseline.upper.")
    if not 0 <= config["library"]["std_lower"] <= config["library"]["std_upper"]:
        raise ValueError("Expected 0 <= library.std_lower <= library.std_upper.")
    if min(config[factor]["std"] for factor in ("batch", "celltype", "tissue", "concept", "noise")) < 0:
        raise ValueError("Noise and coefficient standard deviations must be nonnegative.")
    if config["concept"]["beta_a"] <= 0 or config["concept"]["beta_b"] <= 0 or config["concept"]["cov_prior_range"] < 0:
        raise ValueError("Beta parameters must be positive and cov_prior_range nonnegative.")
    return config


def _prepare_exog_noise(exog_noise, n_obs, n_vars):
    """Normalize optional Poisson quantiles without modifying the caller's array."""
    if exog_noise is None:
        return None
    values = np.array(exog_noise, dtype=float, copy=True)
    if values.shape != (n_obs, n_vars):
        raise ValueError("exog_noise must have shape (cells.n, genes.n).")
    if not np.all(np.isfinite(values) & (values > 0) & (values < 1)):
        raise ValueError("exog_noise must contain finite quantiles strictly between 0 and 1.")
    return values


def _generate_single_cell(generation_config=None, seed=42, exog_noise=None):
    """Return counts and concepts using a local RNG."""
    config = _resolve_generation_config(generation_config)
    n_obs = config["cells"]["n"]
    n_vars = config["genes"]["n"]
    n_batches = config["batch"]["n"]
    n_tissues = config["tissue"]["n"]
    n_celltypes = config["celltype"]["n"]
    n_concepts = config["concept"]["n"]

    rng = np.random.default_rng(seed)
    # Sample parameters for batches, tissues, cell types, and concepts
    p_batch = rng.dirichlet(10 * np.ones(n_batches)) # probability of sampling each batch
    p_tissue = rng.dirichlet(10 * np.ones(n_tissues)) # probability of sampling each tissue
    p_celltype = rng.dirichlet(np.ones(n_celltypes), size=n_tissues) # probability of sampling each cell type within a tissue
    p_concept = rng.dirichlet(np.ones(n_concepts) / 5, size=n_celltypes) # probability of sampling each concept within a cell type
    # Sample standard deviation of library size for each batch
    std_libsize = rng.uniform(config["library"]["std_lower"], config["library"]["std_upper"], size=n_batches) # standard deviation of library size for each batch

    # Sample effect coefficients for batches, cell types, tissues, and concepts
    batch_effect = rng.normal(0, config["batch"]["std"], size=(n_batches, n_vars)) # batch effect coefficients for each batch and gene
    celltype_effect = rng.normal(0, config["celltype"]["std"], size=(n_celltypes, n_vars)) # cell type effect coefficients for each cell type and gene
    tissue_effect = rng.normal(0, config["tissue"]["std"], size=(n_tissues, n_vars)) # tissue effect coefficients for each tissue and gene
    concept_effect = rng.normal(0, config["concept"]["std"], size=(n_concepts, n_vars)) # concept effect coefficients for each concept and gene

    if config["concept"]["use_dependency"]:
        cov_prior_range = config["concept"]["cov_prior_range"]
        scale_matrix_elements = rng.uniform(
            -cov_prior_range, cov_prior_range, size=(n_concepts * (n_concepts + 1)) // 2
        )
        scale_matrix = np.zeros((n_concepts, n_concepts))
        tril_indices = np.tril_indices_from(scale_matrix)
        scale_matrix[tril_indices[0], tril_indices[1]] = scale_matrix_elements
        scale_matrix = scale_matrix @ scale_matrix.T
        scale_matrix += np.eye(n_concepts) * 1e-6
        covariance = wishart.rvs(df=n_concepts, scale=scale_matrix, random_state=rng)
        # SciPy returns a scalar for the one-concept case.
        if n_concepts == 1:
            covariance = np.array([[covariance]])
    else:
        covariance = np.zeros((n_concepts, n_concepts))

    # Baseline expression for each variable
    r, s = np.log(config["baseline"]["lower"]), np.log(config["baseline"]["upper"])
    baseline_expr = rng.uniform(r, s, size=n_vars)

    if config["concept"]["zero_inflate"]:
        # As in scCBGM, sparsity masks concept coefficients, not observed counts.
        probability = 1 - rng.beta(config["concept"]["beta_a"], config["concept"]["beta_b"], size=n_concepts)
        mask = np.vstack([rng.binomial(1, probability[i], size=n_vars) for i in range(n_concepts)])
        concept_effect = concept_effect * mask

    inputs = np.empty((n_obs, n_vars), dtype=np.float32)
    concepts = np.empty((n_obs, n_concepts + 3), dtype=np.float32)
    exog_noise = _prepare_exog_noise(exog_noise, n_obs, n_vars)
    sample_exog = exog_noise is None
    exog_mat = np.zeros((n_obs, n_vars)) if sample_exog else exog_noise
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
            baseline_expr
            + np.sum(concept_effect * concept_indicators[:, None], axis=0)
            + celltype_effect[celltype]
            + batch_effect[batch]
            + tissue_effect[tissue]
            + libsize
            + noise
        )
        if sample_exog:
            exog_mat[i, :] = rng.uniform(0, 1, size=n_vars)
        inputs[i] = poisson.ppf(exog_mat[i, :], np.exp(log_rate))
        # Context values are class indices, appended after the binary concepts.
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

    ``exog_noise`` optionally supplies Poisson quantiles of shape
    ``(cells.n, genes.n)`` strictly between 0 and 1. If omitted, they are
    sampled uniformly. They are used only during generation and are distinct
    from the Gaussian gene noise.

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
        exog_noise=None,  # optional Poisson quantiles, one per cell and gene
    ):
        # If root is not provided, use a local folder automatically
        self.root = os.fspath(root) if root is not None else os.path.join(os.getcwd(), "data", "single_cell")
        self.seed = seed
        self.label_descriptions = label_descriptions
        self.generation_config = _resolve_generation_config(generation_config)
        self.n_gen = self.generation_config["cells"]["n"]
        self._provided_exog_noise = _prepare_exog_noise(
            exog_noise, self.n_gen, self.generation_config["genes"]["n"]
        )

        # inputs is a torch tensor of gene counts (cells.n, genes.n)
        # concepts is a torch tensor of binary indicators and context class indices
        inputs, concepts, annotations, graph = self.load()

        # Initialize parent class
        super().__init__(
            input_data=inputs,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
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
        exog_key = (
            hashlib.sha256(self._provided_exog_noise.tobytes()).hexdigest()
            if self._provided_exog_noise is not None else None
        )
        settings = json.dumps({"generation_config": self.generation_config,
                               "exog_noise": exog_key}, sort_keys=True)
        key = hashlib.sha256(settings.encode()).hexdigest()[:16]
        suffix = f"N_{self.n_gen}_seed_{self.seed}_{key}"
        return [f"inputs_{suffix}.pt", f"concepts_{suffix}.pt",
                f"annotations_{suffix}.pt", f"graph_{suffix}.h5"]

    def download(self):
        """No download needed; data is generated locally."""
        pass

    def build(self):
        # Generate gene counts and concept values
        inputs, concepts = _generate_single_cell(
            generation_config=self.generation_config, seed=self.seed,
            exog_noise=self._provided_exog_noise,
        )

        # Create annotations in the same order as the generated concept columns.
        n_concepts = self.generation_config["concept"]["n"]
        context_sizes = [self.generation_config[quantity]["n"] for quantity in ("celltype", "tissue", "batch")]
        annotations = Annotations(
            labels=[f"concept_{i}" for i in range(n_concepts)] + ["celltype", "tissue", "batch"],
            cardinalities=[1] * n_concepts + context_sizes,
            types=["binary"] * n_concepts + ["categorical" if n > 1 else "binary" for n in context_sizes],
        )

        # No directed concept graph is defined by this generator.
        graph = pd.DataFrame()

        # Save each component separately, as in BnLearnDataset.
        logger.info(f"Saving dataset to {self.root_dir}")
        torch.save(inputs, self.processed_paths[0])
        torch.save(concepts, self.processed_paths[1])
        torch.save(annotations, self.processed_paths[2])
        graph.to_hdf(self.processed_paths[3], key="graph", mode="w")

    def load_raw(self):
        self.maybe_build()
        logger.info("Loading raw single cell data.")
        inputs = torch.load(self.processed_paths[0], weights_only=True)
        concepts = torch.load(self.processed_paths[1], weights_only=True)
        annotations = torch.load(self.processed_paths[2], weights_only=False)
        graph = pd.read_hdf(self.processed_paths[3], "graph")
        if graph.empty:
            graph = None
        return inputs, concepts, annotations, graph
