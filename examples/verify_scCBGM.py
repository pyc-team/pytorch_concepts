"""Compare PyC synthetic single-cell data and counterfactuals with scCBGM.

Run: python examples/verify_scCBGM.py
Set SC_CBGM to the local clone. Requires the dependencies of both PyC and the
original conceptlab package, including its training and single-cell analysis
modules. The package is imported normally, without filtering or rewriting
its source code.

Choose seeds in SEEDS; main passes the same seed to both implementations.
PyC creates its RNG in SingleCellDataset.build(); scCBGM creates its RNG by
calling SeededRandom.default_rng(seed) below. These are separate RNGs,
initialized with the same seed and consumed in the same order.

Aligning the random draws requires one adaptation: scCBGM also uses global
NumPy draws that are independent of its seed. SeededRandom routes those draws
through its local RNG. The comparison therefore checks equivalence under
the same random draws. It checks raw counts, concepts, parameters, cell
background and counterfactuals, without normalization or training. The clone
is only read, and PyC caches are temporary.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from torch_concepts.data import SingleCellDataset


SC_CBGM = Path(__file__).resolve().parents[2] / "scCBGM"
CONFIG = {"n_cells": 32, "n_genes": 12, "n_concepts": 3}
SEEDS = (7, 42)
TARGET_CONCEPT = "concept_0"
CELL_TYPE = 2

# Our configuration/parameter names -> names in the original implementation.
CONFIG_NAMES = {
    "n_cells": "n_obs", "n_genes": "n_vars", "n_batch": "n_batches",
    "n_tissue": "n_tissues", "n_celltype": "n_celltypes",
    "std_dev_batch": "std_batch", "std_dev_celltype": "std_celltype",
    "std_dev_tissue": "std_tissue", "std_dev_concept": "std_concept",
    "std_dev_libsize_lower": "std_libsize_lower",
    "std_dev_libsize_upper": "std_libsize_upper", "std_dev_noise": "std_noise",
}
PARAMETER_NAMES = {
    "baseline_expr": "baseline", "concept_effect": "concept_coef",
    "celltype_effect": "celltype_coef", "tissue_effect": "tissue_coef",
    "batch_effect": "batch_coef", "p_batch": "p_batch", "p_tissue": "p_tissue",
    "p_celltype": "p_celltype_in_tissue", "p_concept": "p_concept_in_celltype",
    "std_dev_libsize": "std_libsize", "covariance": "concept_covariance",
}


def load_original():
    """Import the complete original package without modifying its code."""
    filename = SC_CBGM / "conceptlab/datagen/omics.py"
    if not filename.is_file():
        raise FileNotFoundError(f"Missing {filename}; set SC_CBGM to your clone.")

    # Make the clone importable. Disabling bytecode writes only prevents
    # __pycache__ files in the clone; all imports are still executed.
    sys.path.insert(0, str(SC_CBGM))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from conceptlab.datagen import omics
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = previous_bytecode_setting

    # Ensure we imported the requested clone rather than another installation.
    if Path(omics.__file__).resolve() != filename.resolve():
        raise RuntimeError(f"Imported {omics.__file__}, expected {filename}.")
    return omics


class SeededRandom:
    """Use one seeded RNG for all random draws in the original generator."""

    def default_rng(self, seed):
        # scCBGM.generate(seed=seed) calls this method with the actual seed.
        # Each generation starts with a fresh RNG, just like PyC.build().
        self.rng = np.random.default_rng(seed)
        return self.rng

    def multinomial(self, *args, **kwargs):
        # The original uses np.random.multinomial for batch, tissue and cell type.
        return self.rng.multinomial(*args, **kwargs)

    def uniform(self, *args, **kwargs):
        # The original uses np.random.uniform for the covariance prior.
        return self.rng.uniform(*args, **kwargs)


def check_generation(ours, original):
    """Compare values and row/column order before preprocessing."""
    np.testing.assert_array_equal(ours.input_data.numpy(), original["data"].values)
    np.testing.assert_array_equal(ours.native_concepts.tensor.numpy(), original["concepts"].values)
    assert ours.concept_names == original.concept.values.tolist()
    parameters = ours.model_parameters
    background = ours.cell_background
    for name, reference in PARAMETER_NAMES.items():
        np.testing.assert_array_equal(parameters[name].numpy(), original[reference].values,
                                      err_msg=name)
    for name, reference in {"celltype": "celltypes", "tissue": "tissues", "batch": "batches"}.items():
        expected = [f"{name}_{int(value)}" for value in background[name]]
        np.testing.assert_array_equal(expected, original[reference].values)
    for name, reference in {"libsize": "libsize", "noise": "noise",
                            "exog_noise": "exogenous_noise", "bernoulli_prob": "bernoulli_prob"}.items():
        np.testing.assert_array_equal(background[name].numpy(), original[reference].values,
                                      err_msg=name)


def check_counterfactual(ours, original, generator, changes, cell_type):
    """Compare interventions while keeping the same cells in the same order."""
    concepts = original["concepts"].to_pandas().copy()
    if cell_type is None:
        mask = np.ones(len(ours), dtype=bool)
    else:
        mask = original["celltypes"].values == f"celltype_{cell_type}"
    for name, value in changes.items():
        concepts.loc[mask, name] = value
    expected = generator.generate_intervention(original, concepts, seed=ours.seed)
    intervened = ours.generate_interventions(changes, cell_type=cell_type)
    actual = intervened.input_data.numpy()
    np.testing.assert_array_equal(intervened.native_concepts.tensor.numpy(), concepts.values)
    np.testing.assert_array_equal(actual, expected.values)
    unchanged = (concepts.values == original["concepts"].values).all(axis=1)
    np.testing.assert_array_equal(actual[unchanged], original["data"].values[unchanged])
    np.testing.assert_array_equal(intervened.cell_types.numpy(), ours.cell_types.numpy())
    mse = np.mean((actual - original["data"].values) ** 2)
    print(f"  {changes}, cell_type={cell_type}: matches scCBGM; "
          f"{int((~unchanged).sum())} concept rows changed; MSE vs factual={mse:.6g}")


def main():
    print(f"Reference: {SC_CBGM}")
    print("Same seed and aligned RNG draws; no files are written into scCBGM.")

    omics = load_original()
    generator = omics.OmicsDataGenerator
    print(f"Imported original module: {omics.__file__}")

    # All attributes refer to the real NumPy objects; only 'random' changes.
    # The original package has already been fully imported at this point.
    original_numpy = SimpleNamespace(**vars(np))
    original_numpy.random = SeededRandom()

    interventions = [
        ({}, None),
        ({TARGET_CONCEPT: 1}, None),
        ({TARGET_CONCEPT: 1}, CELL_TYPE),
        ({TARGET_CONCEPT: 0}, CELL_TYPE),
    ]
    with TemporaryDirectory(prefix="verify_scCBGM_") as cache:
        for seed, dependency, sparse in product(SEEDS, (False, True), (False, True)):
            # 1. PyC: build() creates np.random.default_rng(seed) and uses that
            #    RNG first for the parameters, then for the cells.
            ours = SingleCellDataset(
                root=cache, seed=seed, normalize=False,
                generation_config={**CONFIG, "use_concept_dependency": dependency,
                                   "zero_inflate": sparse},
            )
            # 2. Translate setting names while keeping their values unchanged.
            settings = {}
            for name, value in ours.generation_config.items():
                original_name = CONFIG_NAMES.get(name, name)
                settings[original_name] = value

            # 3. scCBGM: the SAME seed reaches SeededRandom.default_rng().
            #    A fresh RNG restarts the sequence from the beginning, independently
            #    of PyC's RNG. Multinomial and uniform also consume this sequence.
            # The context only changes the NumPy reference in omics and restores
            # it even if a check fails. The original functions remain unchanged.
            with patch.object(omics, "np", original_numpy):
                original = generator.generate(seed=seed, **settings)

                # 4. Compare values exactly, including parameters and noise,
                #    without tolerances. Counterfactuals reuse the saved noise.
                check_generation(ours, original)
                print(f"seed={seed}, dependency={dependency}, sparse={sparse}: generation matches scCBGM")
                for changes, cell_type in interventions:
                    check_counterfactual(ours, original, generator, changes, cell_type)
                np.testing.assert_array_equal(ours.input_data.numpy(), original["data"].values)
                np.testing.assert_array_equal(ours.native_concepts.tensor.numpy(), original["concepts"].values)
    print("OK: all dataset and counterfactual comparisons passed.")


if __name__ == "__main__":
    main()
