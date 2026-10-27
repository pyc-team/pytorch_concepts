"""Synthetic single-cell counts following scCBGM's OmicsDataGenerator.

Generation state is organized into three dictionaries:

* ``generation_config``: counts and distribution settings for cells, genes,
  batch, celltype, tissue, original_concept, baseline, library and noise.
  Defaults are defined below in ``_DEFAULT_GENERATION_CONFIG``.
* ``model_parameters``: global ``baseline_expr``, ``concept_effect``,
  ``celltype_effect``, ``tissue_effect``, ``batch_effect``, ``p_batch``,
  ``p_tissue``, ``p_celltype``, ``p_concept``, ``std_libsize`` and ``covariance``.
* ``cell_background``: per-cell ``cell_indices``, ``celltype``, ``tissue``, ``batch``,
  ``bernoulli_prob``, ``libsize``, ``noise`` and ``exog_noise``, held fixed
  during concept interventions. Binary concepts are stored separately.

Reference: https://github.com/almaan/scCBGM/blob/main/conceptlab/datagen/omics.py
"""

import hashlib
import json
import os
import logging
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real

import numpy as np
import pandas as pd
import torch
from scipy.stats import poisson, wishart

from ...annotations import Annotations
from ..base.dataset import ConceptDataset


logger = logging.getLogger(__name__)

# generation_config: supported quantities, settings and defaults before sampling.
_DEFAULT_GENERATION_CONFIG = {
    "cells": {"n": 10000},
    "genes": {"n": 1000},
    "batch": {"n": 3, "std": 0.08},
    "celltype": {"n": 4, "std": 0.08},
    "tissue": {"n": 2, "std": 0.07},
    "original_concept": {
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
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                    raise ValueError(f"Expected {quantity}.n to be a positive integer.")
                params[name] = int(value)
            elif name in ("zero_inflate", "use_dependency"):
                if not isinstance(value, (bool, np.bool_)):
                    raise TypeError(f"Expected {quantity}.{name} to be a boolean.")
                params[name] = bool(value)
            else:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                    raise TypeError(f"Expected {quantity}.{name} to be a real number.")
                if not np.isfinite(value):
                    raise ValueError(f"Expected {quantity}.{name} to be finite.")
                if name == "std" and value < 0:
                    raise ValueError("Noise and coefficient standard deviations must be nonnegative.")
                params[name] = float(value)
    if not 0 < config["baseline"]["lower"] <= config["baseline"]["upper"]:
        raise ValueError("Expected 0 < baseline.lower <= baseline.upper.")
    if not 0 <= config["library"]["std_lower"] <= config["library"]["std_upper"]:
        raise ValueError("Expected 0 <= library.std_lower <= library.std_upper.")
    if config["original_concept"]["beta_a"] <= 0 or config["original_concept"]["beta_b"] <= 0 or config["original_concept"]["cov_prior_range"] < 0:
        raise ValueError("Beta parameters must be positive and cov_prior_range nonnegative.")
    return config


def _prepare_exog_noise(exog_noise, n_cells, n_genes):
    """Normalize optional Poisson quantiles without modifying the caller's array."""
    if exog_noise is None:
        return None
    values = np.array(exog_noise, dtype=float, copy=True)
    if values.shape != (n_cells, n_genes):
        raise ValueError("exog_noise must have shape (cells.n, genes.n).")
    if not np.all(np.isfinite(values) & (values > 0) & (values < 1)):
        raise ValueError("exog_noise must contain finite quantiles strictly between 0 and 1.")
    return values


def _cache_key(config, seed, parameters=None, background=None, concepts=None,
               exog_noise=None):
    """Identify generation inputs before sampling, including any fixed state."""
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    tensors = {f"{group}.{name}": value
               for group, values in (("parameters", parameters), ("background", background))
               for name, value in (values or {}).items()}
    if exog_noise is not None:
        tensors["exog_noise"] = exog_noise
    if concepts is not None:
        tensors["concepts"] = concepts
    for name, value in sorted(tensors.items()):
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        array = np.ascontiguousarray(array)
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(memoryview(array).cast("B"))
    return f"seed_{seed}_{digest.hexdigest()[:16]}"


def _concept_names(config):
    return [f"concept_{i}" for i in range(config["original_concept"]["n"])]


def _cell_names(indices):
    return [f"cell_{i}" for i in indices.tolist()]


def _validate_concepts(concepts):
    """Require binary concept indicators."""
    if not np.isin(np.asarray(concepts), [0, 1]).all():
        raise ValueError("Original concepts must be binary (0 or 1).")


def _compute_model_parameters(config, rng=None) -> dict:
    """Sample global model parameters, using seed 42 when rng is omitted."""
    rng = np.random.default_rng(42) if rng is None else rng
    n_genes = config["genes"]["n"]
    n_batches = config["batch"]["n"]
    n_tissues = config["tissue"]["n"]
    n_celltypes = config["celltype"]["n"]
    n_original_concepts = config["original_concept"]["n"]
    # Sample probabilities for batches, tissues, cell types, and concepts
    p_batch = rng.dirichlet(10 * np.ones(n_batches)) # probability of sampling each batch
    p_tissue = rng.dirichlet(10 * np.ones(n_tissues)) # probability of sampling each tissue
    p_celltype = rng.dirichlet(np.ones(n_celltypes), size=n_tissues) # probability of sampling each cell type within a tissue
    p_concept = rng.dirichlet(np.ones(n_original_concepts) / 5, size=n_celltypes) # probability of sampling each concept within a cell type
    # Sample standard deviation of library size for each batch
    std_libsize = rng.uniform(config["library"]["std_lower"], config["library"]["std_upper"], size=n_batches)

    # Sample effect coefficients for batches, cell types, tissues, and concepts
    batch_effect = rng.normal(0, config["batch"]["std"], size=(n_batches, n_genes)) # batch effect coefficients for each batch and gene
    celltype_effect = rng.normal(0, config["celltype"]["std"], size=(n_celltypes, n_genes)) # cell type effect coefficients for each cell type and gene
    tissue_effect = rng.normal(0, config["tissue"]["std"], size=(n_tissues, n_genes)) # tissue effect coefficients for each tissue and gene
    concept_effect = rng.normal(0, config["original_concept"]["std"], size=(n_original_concepts, n_genes)) # concept effect coefficients for each concept and gene

    if config["original_concept"]["use_dependency"]:
        cov_prior_range = config["original_concept"]["cov_prior_range"]
        scale_matrix_elements = rng.uniform(
            -cov_prior_range, cov_prior_range, size=(n_original_concepts * (n_original_concepts + 1)) // 2
        )
        scale_matrix = np.zeros((n_original_concepts, n_original_concepts))
        tril_indices = np.tril_indices_from(scale_matrix)
        scale_matrix[tril_indices[0], tril_indices[1]] = scale_matrix_elements
        scale_matrix = scale_matrix @ scale_matrix.T
        scale_matrix += np.eye(n_original_concepts) * 1e-6
        covariance = wishart.rvs(df=n_original_concepts, scale=scale_matrix, random_state=rng)
        # SciPy returns a scalar for the one-concept case.
        if n_original_concepts == 1:
            covariance = np.array([[covariance]])
    else:
        covariance = np.zeros((n_original_concepts, n_original_concepts))

    # Baseline expression for each variable
    r, s = np.log(config["baseline"]["lower"]), np.log(config["baseline"]["upper"])
    baseline_expr = rng.uniform(r, s, size=n_genes)

    if config["original_concept"]["zero_inflate"]:
        # As in scCBGM, sparsity masks concept coefficients, not observed counts.
        probability = 1 - rng.beta(config["original_concept"]["beta_a"], config["original_concept"]["beta_b"], size=n_original_concepts)
        mask = np.vstack([rng.binomial(1, probability[i], size=n_genes) for i in range(n_original_concepts)])
        concept_effect = concept_effect * mask

    model_parameters = {
        "baseline_expr": baseline_expr,
        "concept_effect": concept_effect,
        "celltype_effect": celltype_effect,
        "tissue_effect": tissue_effect,
        "batch_effect": batch_effect,
        "p_batch": p_batch,
        "p_tissue": p_tissue,
        "p_celltype": p_celltype,
        "p_concept": p_concept,
        "std_libsize": std_libsize,
        "covariance": covariance,
    }
    return {name: torch.as_tensor(value, dtype=torch.float64) for name, value in model_parameters.items()}


def model(
    model_parameters: dict,
    concepts=None,
    *,
    n_cells=None,
    cell_background: dict | None = None,
    rng=None,
    std_noise=0.01,
    exog_noise=None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute concepts, background and counts in one loop over cells.

    Sample concepts and background when omitted; otherwise reuse supplied values.
    Concepts must be binary. Celltype, tissue and batch are sampled with the
    background and held fixed during interventions. Concept matrices are complete.
    Return ``(inputs, concepts, cell_background)``. Counts and concepts are
    float32; floating-point background retains float64 precision and context
    uses int64 class indices.
    """
    params = {name: value.numpy() for name, value in model_parameters.items()}
    if concepts is not None:
        if concepts.ndim != 2 or concepts.shape[1] != len(params["concept_effect"]):
            raise ValueError("concepts must contain all binary concept columns.")
        _validate_concepts(concepts)
    sample_concepts = concepts is None
    values = (np.empty((n_cells, len(params["concept_effect"])), dtype=np.float32)
              if concepts is None else concepts.numpy().astype(np.float32, copy=True))
    n_cells, n_genes = len(values), len(params["baseline_expr"])
    exog_noise = _prepare_exog_noise(exog_noise, n_cells, n_genes)
    sample_background = cell_background is None
    if sample_concepts or sample_background:
        rng = np.random.default_rng(42) if rng is None else rng
    if sample_background:
        cell_background = {
            "cell_indices": torch.arange(n_cells),
            "celltype": torch.empty(n_cells, dtype=torch.long),
            "tissue": torch.empty(n_cells, dtype=torch.long),
            "batch": torch.empty(n_cells, dtype=torch.long),
            "bernoulli_prob": torch.empty((n_cells, len(params["concept_effect"])), dtype=torch.float64),
            "libsize": torch.empty(n_cells, dtype=torch.float64),
            "noise": torch.empty((n_cells, n_genes), dtype=torch.float64),
            "exog_noise": (torch.empty((n_cells, n_genes), dtype=torch.float64)
                           if exog_noise is None else torch.from_numpy(exog_noise)),
        }
    background = {name: value.numpy() for name, value in cell_background.items()}
    if not sample_background:
        for name in ("celltype", "tissue", "batch"):
            indices = background[name]
            if indices.shape != (n_cells,) or not np.all(
                (indices == np.floor(indices)) & (indices >= 0)
                & (indices < len(params[f"{name}_effect"]))
            ):
                raise ValueError("Invalid celltype, tissue or batch in cell_background.")
    inputs = np.empty((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        if sample_background:
            background["batch"][i] = rng.multinomial(1, params["p_batch"]).argmax()
            background["tissue"][i] = rng.multinomial(1, params["p_tissue"]).argmax()
            background["celltype"][i] = rng.multinomial(1, params["p_celltype"][background["tissue"][i]]).argmax()
        celltype = int(background["celltype"][i])
        tissue = int(background["tissue"][i])
        batch = int(background["batch"][i])
        if sample_concepts or sample_background:
            probability = params["p_concept"][celltype]
            mean = np.log((probability + 1e-8) / (1 - probability + 1e-8))
            logits = np.clip(rng.multivariate_normal(mean, params["covariance"]), -8, 8)
            bernoulli_prob = 1 / (1 + np.exp(-logits))
            if sample_concepts:
                values[i] = rng.binomial(1, bernoulli_prob)
        if sample_background:
            background["bernoulli_prob"][i] = bernoulli_prob
            background["libsize"][i] = rng.normal(0, params["std_libsize"][batch])
            background["noise"][i] = rng.normal(0, std_noise, size=n_genes)
            if exog_noise is None:
                background["exog_noise"][i] = rng.uniform(0, 1, size=n_genes)
        log_lambda = (
            params["baseline_expr"]
            + np.sum(params["concept_effect"] * values[i, :, None], axis=0)
            + params["celltype_effect"][celltype]
            + params["batch_effect"][batch]
            + params["tissue_effect"][tissue]
            + background["libsize"][i]
            + background["noise"][i]
        )
        inputs[i] = poisson.ppf(background["exog_noise"][i], np.exp(log_lambda))
    return torch.from_numpy(inputs), torch.from_numpy(values).float(), cell_background


class SingleCellDataset(ConceptDataset):
    """Dataset class for synthetic single-cell data from scCBGM.

    This dataset models the relationship between binary concepts, cell type,
    tissue, batch effects, and gene expression as raw Poisson counts.
    Concept annotations contain only binary indicators. Celltype, tissue and
    batch are fixed cell background. No directed concept graph is assigned.

    Generation state uses three ordinary dictionaries: ``generation_config``,
    ``model_parameters`` and ``cell_background``. Binary ``concepts`` are stored
    separately as supervision and intervention targets. All binary columns are
    retained for interventions, including those omitted from ``concept_subset``.

    ``generation_config`` groups dimensions and distribution settings by quantity.
    ``n`` counts cells, genes, batches, cell types, tissues, or original binary
    concepts. ``original_concept`` configures the binary concept indicators.
    Effect contributions sum to the log Poisson rate of each gene. Omitted
    quantities and parameters use these defaults::

        {
            "cells": {"n": 10000},
            "genes": {"n": 1000},
            "batch": {"n": 3, "std": 0.08},
            "celltype": {"n": 4, "std": 0.08},
            "tissue": {"n": 2, "std": 0.07},
            "original_concept": {
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
    sampled uniformly and saved separately from the Gaussian gene noise.

    The other two dictionaries hold sampled values:

    * ``model_parameters``: ``baseline_expr`` (genes), ``concept_effect``
      (concepts, genes, after sparsity masking), ``celltype_effect``,
      ``tissue_effect``, ``batch_effect`` (respective classes, genes),
      ``p_batch``, ``p_tissue``, ``p_celltype`` (tissues, cell types),
      ``p_concept`` (cell types, concepts), ``std_libsize`` (batches), and
      ``covariance`` (concepts, concepts).
    * ``cell_background``: categorical ``celltype``, ``tissue`` and ``batch``
      indices (cells, int64), ``libsize`` log offsets (cells), Gaussian ``noise``
      and Poisson quantiles ``exog_noise`` (cells, genes), and
      ``bernoulli_prob`` (cells, original concepts), the probabilities used
      before sampling or overriding concept values. ``cell_indices`` (cells,
      int64) identifies the original rows. All these values are held fixed
      during concept interventions.

    The three properties return independent copies. Floating-point parameters
    and background use float64; counts and concepts use float32.

    ``generate_interventions`` changes only binary indicators
    while holding the cell background fixed. ``generate_from_dataset`` samples
    new cells with the same global parameters. Both preserve the source dataset.

    Construction generates or loads cached data. ``generation_state_path``
    stores the three dictionaries, including the seed in ``generation_config``;
    the seed is exposed separately as ``dataset.seed``. Inputs, concepts,
    annotations and graph are saved in separate files. All concept columns
    and background, including ``cell_background["cell_indices"]``, follow row
    selection. ``generation_config["cells"]["n"]`` records the original
    generated size; ``len(dataset)`` is the current size. ``load_raw()`` reads
    the complete cached data before selection.

    Settings and seed identify new dataset directories. Generated variants are
    saved in sibling directories; identical requests reuse them. Load any saved
    dataset with ``SingleCellDataset(root=dataset.root)``: its saved settings
    and seed take precedence over generation arguments. Incomplete directories
    raise ``RuntimeError``; repeat generation from the source to recreate a
    variant, or use a new cache directory for initial generation.
    Separate directories also isolate backbone embedding caches: their file
    names do not identify generation settings or concept interventions.

    The distributions and count equations follow scCBGM. All random draws
    use the seeded generator here; upstream also uses global ``np.random``
    for categorical draws and the covariance prior, so identical seeds do
    not produce identical datasets across implementations.

    Example::

        dataset = SingleCellDataset(
            seed=42,
            generation_config={"cells": {"n": 1000}, "batch": {"n": 2, "std": 0.15}},
        )
        edited = dataset.generate_interventions(concepts={"concept_0": 1})
        new_cells = dataset.generate_from_dataset(n_cells=500, seed=7)
    """

    def __init__(
        self,
        root: str | None = None,  # root directory to store/load the dataset
        seed: int = 42,  # seed for data generation
        concept_subset: list | None = None,  # subset of binary concept labels
        generation_config: dict | None = None,  # partial counts and distribution parameters grouped by quantity
        label_descriptions: dict | None = None,  # mapping from concept names to descriptions
        exog_noise=None,  # optional Poisson quantiles, one per cell and gene
    ):
        self.root = os.fspath(root) if root is not None else os.path.join(os.getcwd(), "data", "single_cell")
        self.seed = seed
        self.label_descriptions = label_descriptions
        # A saved dataset can be loaded by its directory alone. New requests
        # get a separate directory identified by their settings and seed.
        if not any(os.path.exists(path) for path in self.processed_paths):
            self._generation_config = _resolve_generation_config(generation_config)
            self._exog_noise = _prepare_exog_noise(
                exog_noise, self._generation_config["cells"]["n"],
                self._generation_config["genes"]["n"],
            )
            self.root = os.path.join(self.root, _cache_key(
                self._generation_config, seed, exog_noise=self._exog_noise,
            ))
        inputs, concepts, annotations, graph = self.load()
        self._full_concepts = concepts
        state = torch.load(self.generation_state_path, weights_only=True)
        self._generation_config = state["generation_config"].copy()
        self.seed = self._generation_config.pop("seed")
        self._model_parameters = state["model_parameters"]
        self._cell_background = state["cell_background"]

        super().__init__(
            input_data=inputs,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,  # subset of concept names
            name="SingleCellDataset",
        )

    @property
    def generation_config(self) -> dict:
        """Return an independent copy of the generation settings."""
        return deepcopy(self._generation_config)

    @property
    def model_parameters(self) -> dict:
        """Return independent parameters; apply changes via generate_from_dataset."""
        return deepcopy(self._model_parameters)

    @property
    def cell_background(self) -> dict:
        """Return an independent copy of the fixed cell background."""
        return deepcopy(self._cell_background)

    @property
    def raw_filenames(self) -> list[str]:
        """List of raw filenames that need to be present in the raw directory
        for the dataset to be considered present."""
        return []  # nothing to download, data is generated locally

    @property
    def processed_filenames(self) -> list[str]:
        """List of processed filenames that will be created during build step."""
        return ["inputs.pt", "concepts.pt", "annotations.pt", "graph.h5", "generation_state.pt"]

    @property
    def generation_state_path(self) -> str:
        """File containing generation_config, model_parameters and cell_background."""
        return self.processed_paths[4]

    def download(self):
        """No download needed; data is generated locally."""
        pass

    def build(self):
        """Generate and save an initial dataset from its settings and seed."""
        if any(os.path.exists(path) for path in self.processed_paths):
            if all(os.path.isfile(path) for path in self.processed_paths):
                return
            raise RuntimeError(f"Incomplete saved dataset at {self.root}. Regenerate it from its source.")
        config = self._generation_config
        rng = np.random.default_rng(self.seed)
        parameters = _compute_model_parameters(config, rng)
        inputs, concepts, background = model(
            parameters, n_cells=config["cells"]["n"], rng=rng, std_noise=config["noise"]["std"],
            exog_noise=self._exog_noise,
        )
        self._save_dataset(self.root, inputs, concepts, config, parameters, background,
                           self.seed)

    def _save_dataset(self, root, inputs, concepts, config, parameters, background, seed):
        """Write a complete dataset; initial generation and variants use the same format."""
        os.makedirs(root, exist_ok=True)
        paths = [os.path.join(root, name) for name in self.processed_filenames]
        n_original_concepts = config["original_concept"]["n"]
        annotations = Annotations(
            labels=_concept_names(config),
            cardinalities=[1] * n_original_concepts,
            types=["binary"] * n_original_concepts,
        )

        logger.info(f"Saving dataset to {root}")
        torch.save(inputs, paths[0])
        torch.save(concepts, paths[1])
        torch.save(annotations, paths[2])
        pd.DataFrame().to_hdf(paths[3], key="graph", mode="w")
        torch.save({"generation_config": {**config, "seed": seed},
                    "model_parameters": parameters, "cell_background": background},
                   paths[4])

    def generate_from_dataset(self, *, n_cells=None, model_parameters=None, seed=None):
        """Sample new cells with copied global parameters and optional overrides.

        Always samples new concepts and background, even when n_cells is unchanged
        or omitted. Defaults to the current number of cells and the source seed.
        Note: model_parameters accepts a partial dictionary, e.g.
        {"baseline_expr": new_baseline}. Every omitted parameter retains
        its value from this dataset.
        """
        if n_cells is None and model_parameters is None and seed is None:
            raise ValueError("Provide at least one of n_cells, model_parameters or seed.")
        config = _resolve_generation_config({
            **self.generation_config,
            "cells": {"n": len(self) if n_cells is None else n_cells},
        })
        parameters = self.model_parameters
        seed = self.seed if seed is None else seed
        for name, value in (model_parameters or {}).items():
            if name not in parameters:
                raise ValueError(f"Unknown model parameter: {name!r}.")
            value = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
            expected_shape = parameters[name].shape
            if value.shape != expected_shape or not value.isfinite().all():
                raise ValueError(f"Parameter {name!r} must be finite and have shape {tuple(expected_shape)}.")
            parameters[name] = value.clone()
        root = os.path.join(os.path.dirname(self.root), _cache_key(config, seed, parameters=parameters))
        if not all(os.path.isfile(os.path.join(root, name)) for name in self.processed_filenames):
            inputs, concepts, background = model(
                parameters, n_cells=config["cells"]["n"], rng=np.random.default_rng(seed),
                std_noise=config["noise"]["std"],
            )
            self._save_dataset(root, inputs, concepts, config, parameters, background,
                               seed)
        return type(self)(root=root, concept_subset=self.concept_names,
                          label_descriptions=deepcopy(self.label_descriptions))

    def generate_interventions(self, concepts):
        """Change concepts of the existing cells, preserving their background.

        Accepts named overrides, a DataFrame with any subset of cell and concept names,
        or a matrix in binary concept order. Concepts must be 0 or 1.
        Celltype, tissue and batch remain fixed in cell_background.
        Note: {"concept_0": 1} changes only concept_0. A partial DataFrame
        changes only its named cells and columns; all omitted values are retained.
        Parameters, untouched concepts and every background tensor stay fixed.
        """
        config = {**self.generation_config, "cells": {"n": len(self)}}
        names = _concept_names(self._generation_config)
        values = pd.DataFrame(
            self._full_concepts.numpy().astype(float),
            index=_cell_names(self._cell_background["cell_indices"]), columns=names,
        )
        if isinstance(concepts, pd.DataFrame):
            replacement = concepts.astype(float)
        elif isinstance(concepts, Mapping):
            replacement = pd.DataFrame(concepts, index=values.index, dtype=float)
        else:
            replacement = (concepts.detach().cpu().numpy()
                           if isinstance(concepts, torch.Tensor) else np.asarray(concepts))
            if replacement.shape != values.shape:
                raise ValueError("concepts must have shape (cells, binary concepts).")
            replacement = pd.DataFrame(replacement, index=values.index, columns=names)
        if not replacement.columns.is_unique or not set(replacement).issubset(names) or not np.isfinite(replacement.to_numpy()).all():
            raise ValueError("Invalid concepts: expected known names and finite values.")
        if not replacement.index.is_unique or not replacement.index.isin(values.index).all():
            raise ValueError("Invalid cells: expected unique existing cell names.")
        values.loc[replacement.index, replacement.columns] = replacement
        _validate_concepts(values)

        parameters, background = self._model_parameters, self._cell_background
        concepts = torch.tensor(values.to_numpy(), dtype=torch.float32)
        root = os.path.join(os.path.dirname(self.root), _cache_key(
            config, self.seed, parameters, background, concepts,
        ))
        if not all(os.path.isfile(os.path.join(root, name)) for name in self.processed_filenames):
            inputs, concepts, background = model(parameters, concepts, cell_background=background)
            self._save_dataset(root, inputs, concepts, config, parameters, background,
                               self.seed)
        return type(self)(root=root, concept_subset=self.concept_names,
                          label_descriptions=deepcopy(self.label_descriptions))

    def _subset_rows(self, indices) -> None:
        """Also subset full concepts and background, including cell identifiers."""
        row_indices = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        super()._subset_rows(row_indices)
        self._full_concepts = self._full_concepts[row_indices]
        self._cell_background = {
            name: values[row_indices]
            for name, values in self._cell_background.items()
        }

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
