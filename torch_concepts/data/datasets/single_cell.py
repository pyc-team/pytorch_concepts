"Synthetic dataset and corresponding interventions introduced in scCBGM (https://arxiv.org/abs/2606.07760)"

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

_DEFAULT_GENERATION_CONFIG = {
    "n_cells": 10000,
    "n_genes": 1000,
    "n_batch": 3,
    "n_celltype": 4,
    "n_tissue": 2,
    "n_concepts": 8,
    "std_dev_batch": 0.08,
    "std_dev_celltype": 0.08,
    "std_dev_tissue": 0.07,
    "std_dev_concept": 0.05,
    "zero_inflate": True,
    "beta_a": 1,
    "beta_b": 0.5,
    "use_concept_dependency": False,
    "cov_prior_range": 5,
    "baseline_lower": 1,
    "baseline_upper": 5,
    "std_dev_libsize_lower": 0.01,
    "std_dev_libsize_upper": 0.03,
    "std_dev_noise": 0.01,
}


def _resolve_generation_config(generation_config):
    """Merge new generation settings with defaults and validate them."""
    config = _DEFAULT_GENERATION_CONFIG.copy()
    if generation_config is not None:
        if not isinstance(generation_config, Mapping):
            raise TypeError("generation_config must be a mapping of parameter names to values.")
        for name in generation_config:
            if name not in config:
                raise ValueError(f"Unknown generation parameter: {name!r}.")
        config.update(generation_config)

    for name, value in config.items():
        if name.startswith("n_"):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"Expected {name} to be a positive integer.")
            config[name] = int(value)
        elif name in ("zero_inflate", "use_concept_dependency"):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"Expected {name} to be a boolean.")
            config[name] = bool(value)
        else:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"Expected {name} to be a real number.")
            if not np.isfinite(value):
                raise ValueError(f"Expected {name} to be finite.")
            if name.startswith("std_dev_") and value < 0:
                raise ValueError("Noise and coefficient standard deviations (std_dev_*) must be nonnegative.")
            config[name] = float(value)
    if not 0 < config["baseline_lower"] <= config["baseline_upper"]:
        raise ValueError("Expected 0 < baseline_lower <= baseline_upper.")
    if not 0 <= config["std_dev_libsize_lower"] <= config["std_dev_libsize_upper"]:
        raise ValueError("Expected 0 <= std_dev_libsize_lower <= std_dev_libsize_upper.")
    if config["beta_a"] <= 0 or config["beta_b"] <= 0 or config["cov_prior_range"] < 0:
        raise ValueError("Beta parameters must be positive and cov_prior_range nonnegative.")
    return config


def _prepare_exog_noise(exog_noise, n_cells, n_genes):
    """Validate optional random values used to generate gene expression counts."""
    if exog_noise is None:
        return None
    values = np.array(exog_noise, dtype=float, copy=True)
    if values.shape != (n_cells, n_genes):
        raise ValueError("exog_noise must have shape (n_cells, n_genes).")
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
    return [f"concept_{i}" for i in range(config["n_concepts"])]


def _cell_names(indices):
    return [f"cell_{i}" for i in indices.tolist()]


def _validate_concepts(concepts):
    """Require binary concept indicators."""
    if not np.isin(np.asarray(concepts), [0, 1]).all():
        raise ValueError("Original concepts must be binary (0 or 1).")


def _validate_model_parameters(parameters):
    """Check probability, covariance and scale constraints without changing values."""
    for name, value in parameters.items():
        if not value.isfinite().all():
            raise ValueError(f"Parameter {name!r} must be finite.")
    for name in ("p_batch", "p_tissue", "p_celltype", "p_concept"):
        value = parameters[name]
        if ((value < 0) | (value > 1)).any():
            raise ValueError(f"Parameter {name!r} must contain probabilities between 0 and 1.")
        # Concepts are independent Bernoulli probabilities, not a categorical distribution.
        if name != "p_concept":
            total = value.sum(dim=-1)
            if not torch.allclose(total, torch.ones_like(total), rtol=1e-7, atol=1e-8):
                raise ValueError(f"Parameter {name!r} must sum to 1 along its last dimension.")
    covariance = parameters["covariance"]
    if (not torch.allclose(covariance, covariance.T, rtol=1e-7, atol=1e-8)
            or torch.linalg.eigvalsh(covariance).min() < -1e-8):
        raise ValueError("Parameter 'covariance' must be symmetric and positive semidefinite.")
    if (parameters["std_dev_libsize"] < 0).any():
        raise ValueError("Parameter 'std_dev_libsize' must be nonnegative.")


def _compute_model_parameters(config, rng=None) -> dict:
    """Compute the parameters and coefficients of the model used to generate single-cell gene expression data (see "model"). """
    rng = np.random.default_rng(42) if rng is None else rng
    n_genes = config["n_genes"]
    n_batches = config["n_batch"]
    n_tissues = config["n_tissue"]
    n_celltypes = config["n_celltype"]
    n_original_concepts = config["n_concepts"]
    # Sample probabilities for batches, tissues, cell types, and concepts
    p_batch = rng.dirichlet(10 * np.ones(n_batches)) # probability of sampling each batch
    p_tissue = rng.dirichlet(10 * np.ones(n_tissues)) # probability of sampling each tissue
    p_celltype = rng.dirichlet(np.ones(n_celltypes), size=n_tissues) # probability of sampling each cell type within a tissue
    p_concept = rng.dirichlet(np.ones(n_original_concepts) / 5, size=n_celltypes) # probability of sampling each concept within a cell type
    # Sample standard deviation of library size for each batch
    std_dev_libsize = rng.uniform(config["std_dev_libsize_lower"], config["std_dev_libsize_upper"], size=n_batches)

    # Sample effect coefficients for batches, cell types, tissues, and concepts
    batch_effect = rng.normal(0, config["std_dev_batch"], size=(n_batches, n_genes))
    celltype_effect = rng.normal(0, config["std_dev_celltype"], size=(n_celltypes, n_genes))
    tissue_effect = rng.normal(0, config["std_dev_tissue"], size=(n_tissues, n_genes))
    concept_effect = rng.normal(0, config["std_dev_concept"], size=(n_original_concepts, n_genes))

    if config["use_concept_dependency"]:
        cov_prior_range = config["cov_prior_range"]
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
    r, s = np.log(config["baseline_lower"]), np.log(config["baseline_upper"])
    baseline_expr = rng.uniform(r, s, size=n_genes)

    if config["zero_inflate"]:
        # As in scCBGM, sparsity masks concept coefficients, not observed counts.
        probability = 1 - rng.beta(config["beta_a"], config["beta_b"], size=n_original_concepts)
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
        "std_dev_libsize": std_dev_libsize,
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
    std_dev_noise=0.01,
    exog_noise=None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Generate synthetic single-cell gene expression data using a
    hierarchical overdispersed Poisson model.

    Gene expression is determined by gene-specific baseline expression,
    cell-specific library size, batch, tissue, cell-type and concept effects,
    and residual noise.

    The generated data are separated into gene expression counts (inputs),
    binary concept activations (concepts), and additional cell-specific
    information (cell_background). This separation allows concept interventions
    to regenerate gene expression while preserving the other cell characteristics.
    A supplied cell_background includes the count-sampling quantiles and cannot
    be combined with exog_noise; doing so raises ValueError.

    Returns
    -------
    inputs : torch.Tensor
        Gene expression counts of shape (n_cells, n_genes).
    concepts : torch.Tensor
        Binary concept activations of shape (n_cells, n_concepts).
    cell_background : dict
        Cell-specific information, including batch, tissue, cell type,
        library size, concept activation probabilities, and noise.
    """
    if cell_background is not None and exog_noise is not None:
        raise ValueError("exog_noise cannot be provided together with cell_background.")
    _validate_model_parameters(model_parameters)
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
            background["libsize"][i] = rng.normal(0, params["std_dev_libsize"][batch])
            background["noise"][i] = rng.normal(0, std_dev_noise, size=n_genes)
            sampled_exog_noise = rng.uniform(0, 1, size=n_genes)
            if exog_noise is None:
                background["exog_noise"][i] = sampled_exog_noise
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
    """Dataset class for synthetic single-cell gene expression dataset from scCBGM.

    This class generates a synthetic single-cell dataset, where each cell is represented by gene expression counts across multiple genes 
    and a set of binary concept annotations.
    Gene expression counts are simulated using a hierarchical overdispersed Poisson model. 
    For each cell, gene expression depends on gene-specific baseline expression, library size, batch, tissue, cell type, 
    and additional biological factors called *concepts*, together with residual noise.
    Concepts represent interpretable biological factors, such as the activation of gene programs or signaling pathways. 
    Each concept can be active (1) or inactive (0) in a cell, with an activation probability depending on the cell type, 
    and influences gene expression through concept-specific coefficients.

    The class also supports interventions on concepts, allowing their activation states to be changed and the corresponding gene expression profiles 
    to be regenerated while keeping all other cell characteristics and model parameters fixed. 
    New cells can also be generated using the same model parameters.

    For more details, see Andersson et al. (2026), *scCBGM: Interpretable Single-Cell
    Counterfactual Editing*, for details: https://arxiv.org/abs/2606.07760.

    Args:
        root: Directory where generated data are stored in a subdirectory
            identified by the generation settings. Defaults to ./data/single_cell.
            A saved dataset directory is reloaded automatically when seed,
            generation_config and exog_noise are omitted.
            User paths (~) are expanded and relative paths made absolute.
        seed: Random seed for data generation. Defaults to 42.
        concept_subset: Which of these factors to return as concept labels,
            e.g. ["concept_0", "concept_2"]. Defaults to all concepts.
            This selection does not change the generated gene expression.
        generation_config: Flat dictionary with the settings for the generation
            described above: dataset size, baseline expression, batch, tissue,
            cell-type and concept effects, and noise. For example,
            {"n_cells": 1000, "n_concepts": 5} generates 1000 cells with 5
            concept factors. Omitted settings keep their defaults.
        label_descriptions: Optional mapping from concept names to descriptions.
        exog_noise: Optional values controlling count sampling, with shape
            (n_cells, n_genes) and entries strictly between 0 and 1.
        normalize: Normalize each cell to 10,000 total counts, then apply log1p.
            Defaults to False (raw counts). Generated variants retain this choice;
            load_raw() and saved files always contain raw counts.

    Example::

        dataset = SingleCellDataset(generation_config={"n_cells": 1000})
        edited = dataset.generate_interventions({"concept_0": 1})
        new_cells = dataset.generate_from_dataset(n_cells=500, seed=7)
        restored = SingleCellDataset(root=dataset.root)
    """

    def __init__(
        self,
        root: str | None = None,  # root directory to store/load the dataset
        seed: int | None = None,  # defaults to 42 for generation
        concept_subset: list | None = None,  # subset of binary concept labels
        generation_config: dict | None = None,  # flat generation settings
        label_descriptions: dict | None = None,  # mapping from concept names to descriptions
        exog_noise=None,  # optional Poisson quantiles, one per cell and gene
        normalize: bool = False,
    ):
        root = root if root is not None else os.path.join("data", "single_cell")
        self.root = os.fspath(root)
        self.root = os.path.abspath(self.root_dir)
        self.seed = 42 if seed is None else seed
        self.label_descriptions = label_descriptions
        self.normalize = normalize
        if (seed is not None or generation_config is not None or exog_noise is not None
                or not any(os.path.exists(path) for path in self.processed_paths)):
            self._generation_config = _resolve_generation_config(generation_config)
            self._exog_noise = _prepare_exog_noise(
                exog_noise, self._generation_config["n_cells"],
                self._generation_config["n_genes"],
            )
            self.root = os.path.join(self.root, _cache_key(
                self._generation_config, self.seed, exog_noise=self._exog_noise,
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
            parameters, n_cells=config["n_cells"], rng=rng, std_dev_noise=config["std_dev_noise"],
            exog_noise=self._exog_noise,
        )
        self._save_dataset(self.processed_paths, inputs, concepts, config, parameters, background,
                           self.seed)

    def _save_dataset(self, paths, inputs, concepts, config, parameters, background, seed):
        """Write the supplied files; initial generation and variants share the format."""
        root = os.path.dirname(paths[0])
        os.makedirs(root, exist_ok=True)
        n_original_concepts = config["n_concepts"]
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

        Note: model_parameters accepts a partial dictionary, e.g.
        {"baseline_expr": new_baseline}. Every omitted parameter retains
        its value from this dataset. Probabilities must be in [0, 1]; categorical
        distributions must sum to 1 along the last dimension. The covariance
        must be symmetric positive semidefinite and std_dev_libsize nonnegative.
        """
        if n_cells is None and model_parameters is None and seed is None:
            raise ValueError("Provide at least one of n_cells, model_parameters or seed.")
        config = _resolve_generation_config({
            **self.generation_config,
            "n_cells": len(self) if n_cells is None else n_cells,
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
        _validate_model_parameters(parameters)
        root = os.path.join(os.path.dirname(self.root), _cache_key(config, seed, parameters=parameters))
        paths = [os.path.join(root, name) for name in self.processed_filenames]
        if not all(os.path.isfile(path) for path in paths):
            inputs, concepts, background = model(
                parameters, n_cells=config["n_cells"], rng=np.random.default_rng(seed),
                std_dev_noise=config["std_dev_noise"],
            )
            self._save_dataset(paths, inputs, concepts, config, parameters, background,
                               seed)
        return type(self)(root=root, normalize=self.normalize, concept_subset=self.concept_names,
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
        config = {**self.generation_config, "n_cells": len(self)}
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
        paths = [os.path.join(root, name) for name in self.processed_filenames]
        if not all(os.path.isfile(path) for path in paths):
            inputs, concepts, background = model(parameters, concepts, cell_background=background)
            self._save_dataset(paths, inputs, concepts, config, parameters, background,
                               self.seed)
        return type(self)(root=root, normalize=self.normalize, concept_subset=self.concept_names,
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

    def load(self):
        """Load counts and optionally normalize gene expression for each cell."""
        inputs, concepts, annotations, graph = self.load_raw()
        if self.normalize:
            counts = inputs.float().numpy()
            totals = counts.sum(axis=1, keepdims=True)
            inputs = torch.from_numpy(
                np.log1p(counts / np.where(totals == 0, 1, totals) * 1e4)
            )
        return inputs, concepts, annotations, graph

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
