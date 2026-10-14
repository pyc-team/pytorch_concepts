"""Synthetic single-cell counts following scCBGM's OmicsDataGenerator.

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


def _cache_filenames(config, seed, exog_noise=None):
    """Use the same cache names when generating and loading a dataset."""
    exog_key = hashlib.sha256(exog_noise.tobytes()).hexdigest() if exog_noise is not None else None
    settings = json.dumps({"generation_config": config, "exog_noise": exog_key}, sort_keys=True)
    key = hashlib.sha256(settings.encode()).hexdigest()[:16]
    suffix = f"N_{config['cells']['n']}_seed_{seed}_{key}"
    return [f"inputs_{suffix}.pt", f"concepts_{suffix}.pt",
            f"annotations_{suffix}.pt", f"graph_{suffix}.h5",
            f"generation_data_{suffix}.pt"]


def _concept_names(data):
    return [f"concept_{i}" for i in range(data["generation_config"]["original_concept"]["n"])] + list(_CONTEXT_NAMES)


def _cell_names(data):
    return [f"cell_{i}" for i in data["cell_indices"].tolist()]


def _compute_model_parameters(config, rng):
    """Sample only the global model parameters."""
    n_genes = config["genes"]["n"]
    n_batches = config["batch"]["n"]
    n_tissues = config["tissue"]["n"]
    n_celltypes = config["celltype"]["n"]
    n_concepts = config["original_concept"]["n"]
    # Sample probabilities for batches, tissues, cell types, and concepts
    p_batch = rng.dirichlet(10 * np.ones(n_batches)) # probability of sampling each batch
    p_tissue = rng.dirichlet(10 * np.ones(n_tissues)) # probability of sampling each tissue
    p_celltype = rng.dirichlet(np.ones(n_celltypes), size=n_tissues) # probability of sampling each cell type within a tissue
    p_concept = rng.dirichlet(np.ones(n_concepts) / 5, size=n_celltypes) # probability of sampling each concept within a cell type
    # Sample standard deviation of library size for each batch
    std_libsize = rng.uniform(config["library"]["std_lower"], config["library"]["std_upper"], size=n_batches)

    # Sample effect coefficients for batches, cell types, tissues, and concepts
    batch_effect = rng.normal(0, config["batch"]["std"], size=(n_batches, n_genes)) # batch effect coefficients for each batch and gene
    celltype_effect = rng.normal(0, config["celltype"]["std"], size=(n_celltypes, n_genes)) # cell type effect coefficients for each cell type and gene
    tissue_effect = rng.normal(0, config["tissue"]["std"], size=(n_tissues, n_genes)) # tissue effect coefficients for each tissue and gene
    concept_effect = rng.normal(0, config["original_concept"]["std"], size=(n_concepts, n_genes)) # concept effect coefficients for each concept and gene

    if config["original_concept"]["use_dependency"]:
        cov_prior_range = config["original_concept"]["cov_prior_range"]
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
    baseline_expr = rng.uniform(r, s, size=n_genes)

    if config["original_concept"]["zero_inflate"]:
        # As in scCBGM, sparsity masks concept coefficients, not observed counts.
        probability = 1 - rng.beta(config["original_concept"]["beta_a"], config["original_concept"]["beta_b"], size=n_concepts)
        mask = np.vstack([rng.binomial(1, probability[i], size=n_genes) for i in range(n_concepts)])
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

    Sample missing concepts and background; otherwise reuse supplied values.
    Original concept columns accept real values; celltype, tissue and batch
    are categorical class indices. Missing entries are sampled.
    Return ``(inputs, concepts, cell_background)``. Counts and concepts are
    float32; background retains float64 precision.
    """
    params = {name: value.numpy() for name, value in model_parameters.items()}
    values = (np.full((n_cells, len(params["concept_effect"]) + 3), np.nan, dtype=np.float32)
              if concepts is None else concepts.numpy().astype(np.float32, copy=True))
    n_cells, n_genes = len(values), len(params["baseline_expr"])
    sample_concepts = np.isnan(values).any()
    sample_background = cell_background is None
    if sample_concepts or sample_background:
        rng = np.random.default_rng(42) if rng is None else rng
    if sample_background:
        cell_background = {
            "bernoulli_prob": torch.empty((n_cells, len(params["concept_effect"])), dtype=torch.float64),
            "libsize": torch.empty(n_cells, dtype=torch.float64),
            "noise": torch.empty((n_cells, n_genes), dtype=torch.float64),
            "exog_noise": (torch.empty((n_cells, n_genes), dtype=torch.float64)
                           if exog_noise is None else torch.from_numpy(exog_noise.copy())),
        }
    background = {name: value.numpy() for name, value in cell_background.items()}
    inputs = np.empty((n_cells, n_genes), dtype=np.float32)
    for i in range(n_cells):
        if sample_concepts:
            batch = rng.multinomial(1, params["p_batch"]).argmax()
            tissue = rng.multinomial(1, params["p_tissue"]).argmax()
            celltype = rng.multinomial(1, params["p_celltype"][tissue]).argmax()
            values[i, -3:] = np.where(np.isnan(values[i, -3:]), [celltype, tissue, batch], values[i, -3:])
        celltype, tissue, batch = values[i, -3:].astype(int)
        if sample_concepts or sample_background:
            probability = params["p_concept"][celltype]
            mean = np.log((probability + 1e-8) / (1 - probability + 1e-8))
            logits = np.clip(rng.multivariate_normal(mean, params["covariance"]), -8, 8)
            bernoulli_prob = 1 / (1 + np.exp(-logits))
            if sample_concepts:
                indicators = rng.binomial(1, bernoulli_prob)
                values[i, :-3] = np.where(np.isnan(values[i, :-3]), indicators, values[i, :-3])
        if sample_background:
            background["bernoulli_prob"][i] = bernoulli_prob
            background["libsize"][i] = rng.normal(0, params["std_libsize"][batch])
            background["noise"][i] = rng.normal(0, std_noise, size=n_genes)
            if exog_noise is None:
                background["exog_noise"][i] = rng.uniform(0, 1, size=n_genes)
        log_lambda = (
            params["baseline_expr"]
            + np.sum(params["concept_effect"] * values[i, :-3, None], axis=0)
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
    Concept annotations include binary indicators and class indices for
    celltype, tissue, and batch. No directed concept graph is assigned.

    ``generation_config`` groups counts and distribution parameters by quantity.
    ``n`` counts cells, genes, batches, cell types, tissues, or original binary
    concepts. ``original_concept`` configures these binary concepts only;
    the dataset's final concepts also include celltype, tissue and batch.
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

    ``generation_data`` is loaded from a separate ``generation_data_*.pt``
    cache file containing tensors and plain metadata:

    * ``model_parameters``: ``baseline_expr`` (genes), ``concept_effect``
      (concepts, genes, after sparsity masking), ``celltype_effect``,
      ``tissue_effect``, ``batch_effect`` (respective classes, genes),
      ``p_batch``, ``p_tissue``, ``p_celltype`` (tissues, cell types),
      ``p_concept`` (cell types, concepts), ``std_libsize`` (batches), and
      ``covariance`` (concepts, concepts).
    * ``cell_background``: ``libsize`` log offsets (cells), Gaussian ``noise``
      and Poisson quantiles ``exog_noise`` (cells, genes), and
      ``bernoulli_prob`` (cells, original concepts), the probabilities used
      before sampling or overriding concept values. These values are held
      fixed during concept interventions.
    * ``cell_indices``: original integer cell indices in the current row
      order. Cell identifiers (cell_0, ...) are reconstructed from these
      indices. Gene, concept and category names follow their configured
      counts and are not stored as additional metadata.
    * ``generation_config``: counts and distribution settings.

    Concepts are saved in their own cache file, and the seed is available as
    ``dataset.seed``. Neither is duplicated in ``generation_data``.
    ``dataset.concepts`` exposes the concepts selected for training.

    ``cell_background`` does not contain every cell-level quantity. The separation
    follows each quantity's role in the model: ``concepts`` contains the
    variables exposed for supervision and intervention, including celltype,
    tissue and batch; ``cell_background`` contains background variation reused
    when evaluating the same cells under changed concepts. Holding this
    variation fixed keeps the comparison from also reflecting new noise draws.
    Being cell-specific or continuous does not by itself determine which
    group a quantity belongs to.

    ``model_parameters['std_libsize']`` contains one distribution standard deviation
    per batch. ``cell_background['libsize']`` contains the realized log offset per cell,
    sampled from Normal(0, std_libsize[batch]). Adding it to every gene's log rate
    scales all expression rates for that cell by exp(libsize). It is not a total
    gene count. This model treats that scale variation as background to preserve
    during concept interventions. Treating library scale as an intervention
    target instead would require exposing it as a concept in the model.

    After real-valued interventions, original concept annotations still
    describe their binary sampling type. They do not assert that intervened
    values are 0 or 1; consumers selecting losses or metrics by type should
    account for the real-valued targets.

    Floating-point parameters, probabilities and noises retain float64
    precision to reconstruct the original counts or generate interventions
    using the same cell-level noise. Row subsampling also subsets
    ``concepts``, ``cell_background`` and ``cell_indices`` in memory; the cache file
    retains the complete generated dataset. Missing processed files trigger
    a full rebuild from configuration and seed; interventions are recreated
    with ``generate_from_dataset``. ``load_raw()`` still returns only inputs,
    concepts, annotations, and graph.
    Use ``generate_from_dataset(concepts=...)`` to change selected concepts
    while preserving the same cell-level noise.

    Example::

        dataset = SingleCellDataset(
            seed=42,
            generation_config={"cells": {"n": 1000}, "batch": {"n": 2, "std": 0.15}},
        )
        edited = dataset.generate_from_dataset(concepts={"concept_0": 1})
        new_cells = dataset.generate_from_dataset(n_cells=500, seed=7)
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
        # Keep the complete matrix for generation, including concepts omitted
        # from training supervision. Rows follow the in-memory dataset order.
        self._generation_concepts = concepts
        self.generation_data = self.load_generation_data()

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
        return _cache_filenames(self.generation_config, self.seed, self._provided_exog_noise)

    @property
    def generation_data_path(self) -> str:
        """Path of the separate cache containing generation metadata."""
        return self.processed_paths[4]

    def download(self):
        """No download needed; data is generated locally."""
        pass

    def build(self):
        """Generate the complete dataset from configuration and seed."""
        rng = np.random.default_rng(self.seed)
        parameters = _compute_model_parameters(self.generation_config, rng)
        inputs, concepts, background = model(
            parameters, n_cells=self.n_gen, rng=rng, std_noise=self.generation_config["noise"]["std"],
            exog_noise=self._provided_exog_noise,
        )
        generation_data = {
            "generation_config": self.generation_config,
            "model_parameters": parameters,
            "cell_background": background,
            "cell_indices": torch.arange(len(concepts)),
        }

        # Create annotations in the same order as the generated concept columns.
        n_concepts = self.generation_config["original_concept"]["n"]
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
        torch.save(generation_data, self.generation_data_path)

    def load_generation_data(self) -> dict:
        """Read generation metadata from the cache, before any row selection."""
        self.maybe_build()
        return torch.load(self.generation_data_path, weights_only=True)

    def get_generation_config(self) -> dict:
        """Copy the distribution settings for constructing another dataset."""
        config = deepcopy(self.generation_config)
        config["cells"]["n"] = len(self)
        return config

    def get_model_parameters(self) -> dict:
        """Copy the global coefficients and probabilities shared by all cells."""
        return deepcopy(self.generation_data["model_parameters"])

    def get_cell_background(self) -> dict:
        """Copy the background variation held fixed during concept interventions.

        Includes Bernoulli probabilities, library log offsets, Gaussian gene
        noise and Poisson quantiles. Bernoulli probabilities describe the
        original sampling; they are not recomputed for interventions and
        are not used to calculate the intervened counts.
        Selected concept supervision is available as ``dataset.concepts``.
        """
        return deepcopy(self.generation_data["cell_background"])

    def get_generation_data(self) -> dict:
        """Copy the complete current generation data, including row selection."""
        data = deepcopy(self.generation_data)
        data["generation_config"] = self.get_generation_config()
        return data

    def generate_from_dataset(self, *, concepts=None, model_parameters=None, n_cells=None, seed=None, root=None):
        """Create another SingleCellDataset using this dataset's configuration.

        Supply at least one of concepts, model_parameters, n_cells or seed.
        Original concepts accept finite real values, given as named overrides,
        a DataFrame aligned by cell and column names, or a matrix in raw column
        order (original concepts only, or original concepts plus context).
        Parameter overrides must keep the existing shapes; unknown names
        and nonfinite values are rejected.
        ``concepts`` and ``model_parameters`` override selected values;
        for example, ``dataset.generate_from_dataset(concepts={'concept_0': 1})``.

        Passing ``n_cells`` or ``seed`` draws new cells and states from the
        same global model parameters. Context is sampled before applying
        partial overrides: replacing tissue does not resample celltype from
        the new tissue's distribution. Bernoulli probabilities for new cells
        use the final celltype. With existing cells, the saved probabilities
        remain a record of their original sampling and do not affect counts.
        The original settings, concept selection and descriptions are copied.
        ``root`` defaults to this dataset's
        cache directory. Variants are stored in a subdirectory identified
        by their generated contents, so distinct interventions remain separate.
        """
        if n_cells is None and seed is None and not model_parameters and (concepts is None or len(concepts) == 0):
            raise ValueError("Provide at least one change: concepts, model_parameters, n_cells or seed.")
        new_generation_data = self.get_generation_data()
        config = new_generation_data["generation_config"]
        fresh = n_cells is not None or seed is not None
        if fresh:
            config["cells"]["n"] = len(self) if n_cells is None else n_cells
            config = new_generation_data["generation_config"] = _resolve_generation_config(config)
            new_generation_data["cell_indices"] = torch.arange(config["cells"]["n"])
            new_generation_data["cell_background"] = None

        names = _concept_names(new_generation_data)
        values = pd.DataFrame(
            np.nan if fresh else self._generation_concepts.numpy(),
            index=_cell_names(new_generation_data), columns=names,
        )
        if concepts is not None:
            if isinstance(concepts, (Mapping, pd.DataFrame)):
                replacement = pd.DataFrame(concepts, index=values.index, dtype=float)
            else:
                replacement = torch.as_tensor(concepts).detach().cpu().numpy()
                if replacement.shape not in (values.shape, (len(values), len(names) - 3)):
                    raise ValueError("concepts must have shape (cells, binary concepts) or (cells, all concepts).")
                replacement = pd.DataFrame(replacement, index=values.index, columns=names[:replacement.shape[1]])
            if not set(replacement).issubset(names) or not np.isfinite(replacement.to_numpy()).all():
                raise ValueError("Invalid concepts: expected known names and finite real values.")
            context = replacement.filter(items=_CONTEXT_NAMES)
            limits = pd.Series({name: config[name]["n"] for name in context})
            if not ((context == np.floor(context)) & (context >= 0) & (context < limits)).all().all():
                raise ValueError("Invalid class indices for celltype, tissue or batch.")
            values[replacement.columns] = replacement

        for name, value in (model_parameters or {}).items():
            if name not in new_generation_data["model_parameters"]:
                raise ValueError(f"Unknown model parameter: {name!r}.")
            value = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
            expected_shape = new_generation_data["model_parameters"][name].shape
            if value.shape != expected_shape or not value.isfinite().all():
                raise ValueError(f"Parameter {name!r} must be finite and have shape {tuple(expected_shape)}.")
            new_generation_data["model_parameters"][name] = value.clone()
        inputs, new_concepts, new_generation_data["cell_background"] = model(
            new_generation_data["model_parameters"], torch.from_numpy(values.to_numpy()),
            cell_background=new_generation_data["cell_background"],
            rng=np.random.default_rng(self.seed if seed is None else seed) if fresh else None,
            std_noise=config["noise"]["std"],
        )
        output_seed = self.seed if seed is None else seed
        digest = hashlib.sha256()
        values = {**new_generation_data["model_parameters"], **new_generation_data["cell_background"],
                  "concepts": new_concepts, "cell_indices": new_generation_data["cell_indices"]}
        for name, value in sorted(values.items()):
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.numpy().tobytes())
        cache_root = os.path.expanduser(os.path.normpath(self.root if root is None else os.fspath(root)))
        variant = f"generated_{digest.hexdigest()[:16]}"
        cache_root = cache_root if os.path.basename(cache_root) == variant else os.path.join(cache_root, variant)
        os.makedirs(cache_root, exist_ok=True)
        filenames = _cache_filenames(new_generation_data["generation_config"], output_seed)
        torch.save(inputs, os.path.join(cache_root, filenames[0]))
        torch.save(new_concepts, os.path.join(cache_root, filenames[1]))
        torch.save(torch.load(self.processed_paths[2], weights_only=False), os.path.join(cache_root, filenames[2]))
        pd.DataFrame().to_hdf(os.path.join(cache_root, filenames[3]), key="graph", mode="w")
        torch.save(new_generation_data, os.path.join(cache_root, filenames[4]))
        return type(self)(
            root=cache_root, seed=output_seed, generation_config=new_generation_data["generation_config"],
            concept_subset=self.concept_names,
            label_descriptions=deepcopy(self.label_descriptions),
        )

    def plot(self, *, color_by="concept_0", ax=None, reference=None, title=None):
        """Plot PCA of log1p counts, colored by a native concept.

        ``reference`` supplies the dataset on which PCA is fitted. Use the
        same reference and shared axes to compare original and edited cells
        in the same coordinate system. Returns the matplotlib Axes.
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        reference = self if reference is None else reference
        if self.generation_config["genes"]["n"] != reference.generation_config["genes"]["n"]:
            raise ValueError("The reference must have the same number of genes.")
        if min(reference.input_data.shape) < 2:
            raise ValueError("PCA plotting requires at least two cells and two genes.")
        concept_names = _concept_names(self.generation_data)
        if color_by not in concept_names:
            raise ValueError(f"Unknown concept: {color_by!r}.")
        groups = self._generation_concepts[:, concept_names.index(color_by)].numpy()
        pca = PCA(n_components=2, svd_solver="full").fit(np.log1p(reference.input_data.numpy()))
        points = pca.transform(np.log1p(self.input_data.numpy()))
        ax = plt.subplots(figsize=(6, 5), layout="constrained")[1] if ax is None else ax
        n_classes = self.generation_config[color_by]["n"] if color_by in _CONTEXT_NAMES else 2
        palette = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
        for value in range(n_classes):
            mask = groups == value
            if mask.any():
                ax.scatter(points[mask, 0], points[mask, 1], s=18, alpha=0.75,
                           color=palette(value % palette.N), edgecolors="none", label=str(value))
        ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
               ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
               title=title or f"Single-cell counts · {color_by}")
        ax.legend(title=color_by, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        return ax

    def _subset_rows(self, indices) -> None:
        """Keep realized cell background aligned with selected dataset rows."""
        row_indices = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        cell_background = {
            name: values[row_indices]
            for name, values in self.generation_data["cell_background"].items()
        }
        cell_indices = self.generation_data["cell_indices"][row_indices]
        concepts = self._generation_concepts[row_indices]
        super()._subset_rows(row_indices)
        self._generation_concepts = concepts
        self.generation_data["cell_background"] = cell_background
        self.generation_data["cell_indices"] = cell_indices

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
