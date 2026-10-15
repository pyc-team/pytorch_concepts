"""Synthetic single-cell counts following scCBGM's OmicsDataGenerator.

Reference: https://github.com/almaan/scCBGM/blob/main/conceptlab/datagen/omics.py
"""

import hashlib
import json
import os
import logging
from collections.abc import Mapping
from copy import copy, deepcopy
from numbers import Real

import numpy as np
import pandas as pd
import torch
from scipy.stats import poisson, wishart

from ...annotations import Annotations
from ..base.dataset import ConceptDataset


_COUNT_QUANTITIES = ("cells", "genes", "batch", "celltype", "tissue", "original_concept")
_CONTEXT_NAMES = ("celltype", "tissue", "batch")
logger = logging.getLogger(__name__)

# Supported quantities and their default counts and distribution parameters.
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
    if min(config[factor]["std"] for factor in ("batch", "celltype", "tissue", "original_concept", "noise")) < 0:
        raise ValueError("Noise and coefficient standard deviations must be nonnegative.")
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


def _cache_filenames(metadata, seed, exog_noise=None, concepts=None):
    """Identify generation inputs before sampling, including any fixed state."""
    config = metadata["generation_config"]
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    tensors = {f"{group}.{name}": value
               for group in ("model_parameters", "cell_background")
               for name, value in (metadata.get(group) or {}).items()}
    tensors["cell_indices"] = metadata["cell_indices"]
    if exog_noise is not None:
        tensors["exog_noise"] = exog_noise
    if concepts is not None:
        tensors["concepts"] = concepts
    for name, value in sorted(tensors.items()):
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        array = np.ascontiguousarray(array)
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(memoryview(array).cast("B"))
    suffix = f"v2_seed_{seed}_{digest.hexdigest()[:16]}"
    return [f"inputs_{suffix}.pt", f"concepts_{suffix}.pt",
            f"annotations_{suffix}.pt", f"graph_{suffix}.h5",
            f"generation_metadata_{suffix}.pt"]


def _concept_names(data):
    return [f"concept_{i}" for i in range(data["generation_config"]["original_concept"]["n"])]


def _cell_names(data):
    return [f"cell_{i}" for i in data["cell_indices"].tolist()]


def _validate_concepts(concepts):
    """Require binary concept indicators."""
    if not np.isin(np.asarray(concepts), [0, 1]).all():
        raise ValueError("Original concepts must be binary (0 or 1).")


def _compute_model_parameters(config, rng=None):
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


def model(model_parameters, concepts=None, *, n_cells=None, cell_background=None, rng=None, std_noise=0.01, exog_noise=None):
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
            **{name: torch.empty(n_cells, dtype=torch.long) for name in _CONTEXT_NAMES},
            "bernoulli_prob": torch.empty((n_cells, len(params["concept_effect"])), dtype=torch.float64),
            "libsize": torch.empty(n_cells, dtype=torch.float64),
            "noise": torch.empty((n_cells, n_genes), dtype=torch.float64),
            "exog_noise": (torch.empty((n_cells, n_genes), dtype=torch.float64)
                           if exog_noise is None else torch.from_numpy(exog_noise)),
        }
    background = {name: value.numpy() for name, value in cell_background.items()}
    if not sample_background:
        context = np.column_stack([background[name] for name in _CONTEXT_NAMES])
        sizes = [len(params[f"{name}_effect"]) for name in _CONTEXT_NAMES]
        if context.shape != (n_cells, 3) or not np.all(
            (context == np.floor(context)) & (context >= 0) & (context < sizes)
        ):
            raise ValueError("Invalid celltype, tissue or batch in cell_background.")
    inputs = np.empty((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        if sample_background:
            background["batch"][i] = rng.multinomial(1, params["p_batch"]).argmax()
            background["tissue"][i] = rng.multinomial(1, params["p_tissue"]).argmax()
            background["celltype"][i] = rng.multinomial(1, params["p_celltype"][background["tissue"][i]]).argmax()
        celltype, tissue, batch = (int(background[name][i]) for name in _CONTEXT_NAMES)
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

    ``generation_config`` groups counts and distribution parameters by quantity.
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

    ``build()`` saves inputs, concepts, annotations, graph and generation
    metadata in separate processed files, following BnLearnDataset.
    ``generation_metadata`` contains the model state and generation settings:

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
      before sampling or overriding concept values. These values are held
      fixed during concept interventions.
    * ``cell_indices``: original integer cell indices in the current row
      order. Cell identifiers (cell_0, ...) are reconstructed from these
      indices. Gene, concept and category names follow their configured
      counts and are not stored as additional metadata.
    * ``generation_config``: counts and distribution settings.

    ``generation_config``, ``model_parameters`` and ``cell_background`` are
    properties returning independent copies, so in-place changes do not alter
    the dataset. Floating-point parameters and background retain float64
    precision; counts and concepts use float32.

    ``generate_interventions`` changes only binary indicators
    while holding the cell background fixed. ``generate_from_dataset`` samples
    new cells with the same global parameters. Both preserve the source dataset.

    Row selection keeps all native concept columns, background and cell
    identifiers aligned, including concepts omitted from training supervision.
    ``load_raw()`` reads the complete cached data before row or concept selection.
    Missing files are rebuilt from the original generation inputs and seed.
    Cache filenames identify those inputs; identical requests reuse their files.

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
        *,
        parameters: dict | None = None,  # fixed global model parameters
        concepts=None,  # optional full concept matrix for interventions
    ):
        self.root = os.fspath(root) if root is not None else os.path.join(os.getcwd(), "data", "single_cell")
        self.seed = seed
        self.label_descriptions = label_descriptions
        self._generation_metadata = {
            "generation_config": _resolve_generation_config(generation_config),
            "model_parameters": deepcopy(parameters) if parameters else None,
            "cell_background": None,
        }
        if isinstance(concepts, pd.DataFrame):
            indices = torch.tensor([int(str(name).removeprefix("cell_")) for name in concepts.index])
            if indices.unique().numel() != len(indices):
                raise ValueError("Cell identifiers must be unique.")
            names = _concept_names(self._generation_metadata)
            if not concepts.columns.is_unique or set(concepts.columns) != set(names):
                raise ValueError("concepts must contain only the binary concept columns.")
            concepts = concepts[names].to_numpy()
        else:
            indices = torch.arange(self.n_gen)
        self._generation_metadata["cell_indices"] = indices
        self._generation_concepts = (None if concepts is None else
                                     torch.as_tensor(concepts, dtype=torch.float64).detach().cpu().clone())
        if self._generation_concepts is not None:
            shape = (self.n_gen, len(_concept_names(self._generation_metadata)))
            if self._generation_concepts.shape != shape or not self._generation_concepts.isfinite().all():
                raise ValueError(f"concepts must be finite and have shape {shape}.")
            _validate_concepts(self._generation_concepts)
            self._generation_concepts = self._generation_concepts.float()
        self._exog_noise = _prepare_exog_noise(
            exog_noise, self.n_gen, self.generation_config["genes"]["n"]
        )
        self._processed_filenames = _cache_filenames(
            self._generation_metadata, self.seed, self._exog_noise, self._generation_concepts
        )
        # Rebuild from generation inputs, independently of in-memory row selection.
        self._generation_inputs = (self._generation_metadata.copy(), self._generation_concepts)
        inputs, concepts, annotations, graph = self.load()
        # Keep the complete matrix for generation, including concepts omitted
        # from training supervision. Rows follow the in-memory dataset order.
        self._generation_concepts = concepts
        self._generation_metadata = torch.load(self.generation_metadata_path, weights_only=True)

        super().__init__(
            input_data=inputs,
            concepts=concepts,
            annotations=annotations,
            graph=graph,
            concept_names_subset=concept_subset,  # subset of concept names
            name="SingleCellDataset",
        )

    @property
    def generation_metadata(self) -> dict:
        """Return an independent copy of the generation state."""
        return deepcopy(self._generation_metadata)

    @property
    def generation_config(self) -> dict:
        """Return an independent copy of the generation settings."""
        return deepcopy(self._generation_metadata["generation_config"])

    @property
    def model_parameters(self) -> dict:
        """Return independent parameters; apply changes via generate_from_dataset."""
        return deepcopy(self._generation_metadata["model_parameters"])

    @property
    def cell_background(self) -> dict:
        """Return an independent copy of the fixed cell background."""
        return deepcopy(self._generation_metadata["cell_background"])

    @property
    def n_gen(self) -> int:
        """Number of cells originally generated, before row selection."""
        return self.generation_config["cells"]["n"]

    @property
    def raw_filenames(self) -> list[str]:
        """List of raw filenames that need to be present in the raw directory
        for the dataset to be considered present."""
        return []  # nothing to download, data is generated locally

    @property
    def processed_filenames(self) -> list[str]:
        """List of processed filenames that will be created during build step."""
        return self._processed_filenames

    @property
    def generation_metadata_path(self) -> str:
        """Path of the separate cache containing generation metadata."""
        return self.processed_paths[4]

    def download(self):
        """No download needed; data is generated locally."""
        pass

    def build(self):
        """Sample missing state, compute counts, and save the dataset components."""
        metadata, concepts = self._generation_inputs
        config = metadata["generation_config"]
        parameters, background = metadata["model_parameters"], metadata["cell_background"]
        rng = np.random.default_rng(self.seed) if background is None else None
        if parameters is None:
            parameters = _compute_model_parameters(config, rng)
        inputs, concepts, background = model(
            parameters, concepts, n_cells=config["cells"]["n"], rng=rng,
            cell_background=background, std_noise=config["noise"]["std"],
            exog_noise=self._exog_noise,
        )
        metadata = {**metadata, "model_parameters": parameters, "cell_background": background}

        # Create annotations in the same order as the generated concept columns.
        n_original_concepts = config["original_concept"]["n"]
        annotations = Annotations(
            labels=_concept_names(metadata),
            cardinalities=[1] * n_original_concepts,
            types=["binary"] * n_original_concepts,
        )

        # No directed concept graph is defined by this generator.
        graph = pd.DataFrame()

        # Save each component separately, as in BnLearnDataset.
        logger.info(f"Saving dataset to {self.root_dir}")
        torch.save(inputs, self.processed_paths[0])
        torch.save(concepts, self.processed_paths[1])
        torch.save(annotations, self.processed_paths[2])
        graph.to_hdf(self.processed_paths[3], key="graph", mode="w")
        torch.save(metadata, self.generation_metadata_path)

    def generate_from_dataset(self, *, n_cells=None, model_parameters=None, seed=None):
        """Sample new cells with copied global parameters and optional overrides.

        Always samples new concepts and background, even when n_cells is unchanged
        or omitted. Defaults to the current number of cells and the source seed.
        Supply at least one of n_cells, model_parameters or seed.
        Note: model_parameters accepts a partial dictionary, e.g.
        {"baseline_expr": new_baseline}. Every omitted parameter retains
        its value from this dataset.
        Parameter overrides must retain their shapes and contain finite values.
        The source dataset, concept selection and descriptions are preserved.
        Results are saved beside the source and identical requests reuse their files.
        """
        if n_cells is None and model_parameters is None and seed is None:
            raise ValueError("Provide at least one of n_cells, model_parameters or seed.")
        config = {
            **self.generation_config,
            "cells": {"n": len(self) if n_cells is None else n_cells},
        }
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
        return type(self)(
            root=self.root, seed=seed, concept_subset=self.concept_names,
            label_descriptions=deepcopy(self.label_descriptions),
            generation_config=config, parameters=parameters,
        )

    def generate_interventions(self, concepts):
        """Change concepts of the existing cells, preserving their background.

        Accepts named overrides, a DataFrame with any subset of cell and concept names,
        or a matrix in binary concept order. Concepts must be 0 or 1.
        Celltype, tissue and batch remain fixed in cell_background.
        Note: {"concept_0": 1} changes only concept_0. A partial DataFrame
        changes only its named cells and columns; all omitted values are retained.
        Parameters, untouched concepts and every background tensor stay fixed.
        Results are saved beside the source and identical requests reuse their files.
        """
        config = {**self.generation_config, "cells": {"n": len(self)}}
        names = _concept_names(self._generation_metadata)
        values = pd.DataFrame(
            self._generation_concepts.numpy().astype(float),
            index=_cell_names(self._generation_metadata), columns=names,
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

        dataset = copy(self)
        dataset.label_descriptions = deepcopy(self.label_descriptions)
        dataset._generation_metadata = {**self._generation_metadata, "generation_config": config}
        dataset._generation_concepts = torch.tensor(values.to_numpy(), dtype=torch.float32)
        dataset._exog_noise = None
        dataset._processed_filenames = _cache_filenames(
            dataset._generation_metadata, dataset.seed, concepts=dataset._generation_concepts,
        )
        dataset._generation_inputs = (dataset._generation_metadata.copy(), dataset._generation_concepts)
        inputs, concepts, annotations, graph = dataset.load()
        dataset._generation_concepts = concepts
        dataset._generation_metadata = torch.load(dataset.generation_metadata_path, weights_only=True)
        dataset.is_subset, dataset.subset_seed = False, None
        ConceptDataset.__init__(
            dataset, input_data=inputs, concepts=concepts, annotations=annotations,
            graph=graph, concept_names_subset=self.concept_names, name="SingleCellDataset",
        )
        return dataset

    def precompute_embeddings(self, backbone, batch_size=64, workers=0, cache=True,
                              cache_dir=None, force=False):
        """Cache embeddings per variant; an explicit cache_dir keeps its meaning."""
        if cache_dir is None:
            cache_dir = os.path.join(self.root_dir, "embeddings",
                                     os.path.splitext(self.processed_filenames[0])[0])
        return super().precompute_embeddings(
            backbone, batch_size, workers, cache, cache_dir, force,
        )

    def plot(self, *, color_by=None, **kwargs):
        """Plot UMAP as in scCBGM's synthetic-data notebook (requires scanpy).

        Normalize counts to 10,000 per cell, log-transform, run PCA and UMAP.
        By default, show all concepts and context labels. Keyword arguments
        are forwarded to ``scanpy.pl.umap``; figures are returned without showing.
        """
        if len(self) < 4 or self.input_data.shape[1] < 2:
            raise ValueError("UMAP plotting requires at least 4 cells and 2 genes.")
        import scanpy as sc

        names = _concept_names(self._generation_metadata)
        obs = pd.DataFrame(self._generation_concepts.numpy(), columns=names,
                           index=_cell_names(self._generation_metadata))
        for name in _CONTEXT_NAMES:
            values = self._generation_metadata["cell_background"][name].numpy().astype(str)
            obs[name] = pd.Categorical([f"{name}_{value}" for value in values])
        data = sc.AnnData(self.input_data.numpy().astype(float), obs=obs)
        sc.pp.normalize_total(data, target_sum=1e4)
        sc.pp.log1p(data)
        sc.pp.pca(data)
        sc.pp.neighbors(data, n_neighbors=50, metric="cosine")
        sc.tl.umap(data, min_dist=1.0, spread=0.4)
        kwargs.setdefault("show", False)
        return sc.pl.umap(data, color=names + list(_CONTEXT_NAMES) if color_by is None else color_by, **kwargs)

    def _subset_rows(self, indices) -> None:
        """Also subset full concepts and background, which the base class omits."""
        row_indices = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        super()._subset_rows(row_indices)
        self._generation_concepts = self._generation_concepts[row_indices]
        self._generation_metadata["cell_background"] = {
            name: values[row_indices]
            for name, values in self._generation_metadata["cell_background"].items()
        }
        self._generation_metadata["cell_indices"] = self._generation_metadata["cell_indices"][row_indices]

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
