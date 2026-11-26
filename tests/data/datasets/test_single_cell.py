"""Tests for single-cell counts and flat generation settings."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import poisson

from torch_concepts.data import SingleCellDataset, SingleCellDataModule
from torch_concepts.data.base.datamodule import ConceptDataModule
from torch_concepts.data.splitters import FixedIndicesSplitter, RandomSplitter
import torch_concepts.data.datasets.single_cell as single_cell
from torch_concepts.data.datasets.single_cell import (
    cell_simulator,
    _resolve_generation_config,
)


def _state(dataset):
    """Collect independent attributes for numerical assertions only."""
    return {"generation_config": dataset.generation_config,
            "model_parameters": dataset.model_parameters,
            "cell_background": dataset.cell_background}


def _saved_state(dataset):
    state = torch.load(dataset.generation_state_path, weights_only=True)
    state["generation_config"].pop("seed")
    return state


def _concept_frame(dataset):
    """Expose all concepts in the current cell order."""
    return pd.DataFrame(dataset._full_concepts.numpy(),
                        index=[f"cell_{i}" for i in dataset.cell_background["cell_indices"].tolist()],
                        columns=single_cell._concept_names(dataset.generation_config))


def _generation_tensors(dataset):
    return dataset.input_data, dataset._full_concepts.clone(), deepcopy(_state(dataset))


@pytest.fixture
def small_config():
    return {
        "n_cells": 10, "n_genes": 6, "n_batch": 2,
        "n_tissue": 2, "n_celltype": 3, "n_concepts": 3,
    }


@pytest.fixture
def counterfactual_source(tmp_path, small_config):
    """A known population: type 2 has six concept-off and four concept-on cells."""
    source = SingleCellDataset(root=tmp_path, generation_config={**small_config, "n_cells": 16})
    background = source.cell_background
    background["celltype"] = torch.tensor([2] * 10 + [0] * 3 + [1] * 3)
    concepts = torch.stack((
        torch.tensor([0] * 6 + [1] * 4 + [0, 1, 0, 1, 0, 1]),
        torch.arange(16) % 2,
        torch.arange(16) // 2 % 2,
    ), dim=1).float()
    inputs, concepts, background = cell_simulator(source.model_parameters, concepts, cell_background=background)
    source._save_dataset(source.processed_paths, inputs, concepts, source.generation_config,
                         source.model_parameters, background, source.seed)
    return SingleCellDataset(root=source.root)


def test_default_counts_and_partial_distribution_settings():
    config = _resolve_generation_config({"std_dev_batch": 0.15})
    assert {key: value for key, value in config.items() if key.startswith("n_")} == {
        "n_cells": 10000, "n_genes": 1000, "n_batch": 3,
        "n_celltype": 4, "n_tissue": 2, "n_concepts": 8,
    }
    assert config["std_dev_batch"] == 0.15
    assert config["zero_inflate"] is True


@pytest.mark.parametrize("seed", [7, 42])
@pytest.mark.parametrize("dependency", [False, True])
@pytest.mark.parametrize("sparsity", [False, True])
def test_matches_upstream_reference(tmp_path, seed, dependency, sparsity):
    """Compare with frozen scCBGM generate() outputs, independently of our cell_simulator.

    Reference: scCBGM conceptlab/datagen/omics.py, commit
    292d2f86ef4e8cb3564dcc689a64883c536a650e. The fixture also records the source
    URL and checksum.
    Upstream global categorical and covariance-prior draws were routed through
    its seeded Generator to align the random streams. No network is needed.
    """
    dataset = SingleCellDataset(root=tmp_path, seed=seed, generation_config={
        "n_cells": 15, "n_genes": 9,
        "n_concepts": 3, "use_concept_dependency": dependency, "zero_inflate": sparsity,
    })
    case = f"{seed}_{int(dependency)}_{int(sparsity)}"
    path = Path(__file__).with_name("fixtures") / "single_cell_reference.npz"
    with np.load(path) as expected:
        np.testing.assert_array_equal(dataset.input_data.numpy(), expected[f"{case}.inputs"])
        np.testing.assert_array_equal(dataset._full_concepts.numpy(), expected[f"{case}.concepts"])
        for group, values in (("model_parameters", dataset.model_parameters),
                              ("cell_background", dataset.cell_background)):
            if group == "cell_background":
                np.testing.assert_array_equal(values.pop("cell_indices").numpy(), np.arange(15))
            for name, value in values.items():
                # The frozen upstream fixture keeps its original parameter name.
                reference_name = "std_libsize" if name == "std_dev_libsize" else name
                np.testing.assert_allclose(value.numpy(), expected[f"{case}.{group}.{reference_name}"],
                                           rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dependency", [False, True])
@pytest.mark.parametrize("input_kind", ["mapping", "tensor", "numpy", "dataframe", "list"])
def test_generator_applies_concepts_before_counts(tmp_path, small_config, dependency, input_kind):
    config = deepcopy(small_config)
    config["use_concept_dependency"] = dependency
    dataset = SingleCellDataset(root=tmp_path, generation_config=config, seed=7)
    original_inputs, original_concepts, data = _generation_tensors(dataset)
    replacement = original_concepts.clone()
    replacement[:, 0] = 1
    replacement[:, -1] = 1
    overrides = {
        "mapping": {"concept_0": 1, "concept_2": 1}, "tensor": replacement,
        "numpy": replacement.numpy(), "list": replacement.tolist(),
        "dataframe": pd.DataFrame(replacement.numpy(), index=_concept_frame(dataset).index,
                                  columns=_concept_frame(dataset).columns).iloc[::-1, ::-1],
    }[input_kind]
    edited = dataset.generate_interventions(overrides)
    torch.testing.assert_close(edited._full_concepts, replacement)
    torch.testing.assert_close(edited.input_data, _reconstruct_counts(data, replacement), rtol=0, atol=0)
    for name, value in data["cell_background"].items():
        torch.testing.assert_close(edited.cell_background[name], value, rtol=0, atol=0)
    torch.testing.assert_close(dataset.input_data, original_inputs)


def test_model_reuses_saved_data_without_sampling(tmp_path, small_config, monkeypatch):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    inputs, concepts, data = _generation_tensors(dataset)
    original_data = deepcopy(data)
    original_concepts = concepts.clone()

    def unexpected_sampling(*args, **kwargs):
        raise AssertionError("The cell_simulator must reuse the supplied state.")

    monkeypatch.setattr(np.random, "default_rng", unexpected_sampling)
    restored_inputs, _, _ = cell_simulator(data["model_parameters"], concepts, cell_background=data["cell_background"])
    torch.testing.assert_close(restored_inputs, inputs, rtol=0, atol=0)
    torch.testing.assert_close(concepts, original_concepts, rtol=0, atol=0)
    for group in ("model_parameters", "cell_background"):
        for name, value in original_data[group].items():
            torch.testing.assert_close(data[group][name], value, rtol=0, atol=0)


@pytest.mark.parametrize("prepared_concepts", [False, True])
def test_model_rejects_exog_noise_with_fixed_background(tmp_path, small_config, prepared_concepts):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    background = dataset.cell_background
    original = deepcopy(background)
    rng = np.random.default_rng(19)
    rng_state = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="exog_noise.*cell_background"):
        cell_simulator(dataset.model_parameters,
              dataset._full_concepts if prepared_concepts else None,
              n_cells=len(dataset), cell_background=background, rng=rng,
              exog_noise=np.full((10, 6), 0.5))
    torch.testing.assert_close(background, original, rtol=0, atol=0)
    assert rng.bit_generator.state == rng_state


def test_model_samples_background_from_prepared_concepts(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    parameters = deepcopy(dataset.model_parameters)
    concepts = dataset._full_concepts.clone()
    concepts[:, 0] = 1
    quantiles = np.full((10, 6), 0.5)
    inputs, _, background = cell_simulator(parameters, concepts, rng=np.random.default_rng(19),
                               std_dev_noise=0.3, exog_noise=quantiles)
    rng = np.random.default_rng(19)
    for i in range(10):
        batch = rng.multinomial(1, parameters["p_batch"].numpy()).argmax()
        tissue = rng.multinomial(1, parameters["p_tissue"].numpy()).argmax()
        celltype = rng.multinomial(1, parameters["p_celltype"][tissue].numpy()).argmax()
        for name, value in zip(("celltype", "tissue", "batch"), (celltype, tissue, batch)):
            assert background[name][i].item() == value
        probability = parameters["p_concept"][celltype].numpy()
        mean = np.log((probability + 1e-8) / (1 - probability + 1e-8))
        logits = np.clip(rng.multivariate_normal(mean, parameters["covariance"].numpy()), -8, 8)
        np.testing.assert_array_equal(background["bernoulli_prob"][i].numpy(), 1 / (1 + np.exp(-logits)))
        expected_library = rng.normal(0, parameters["std_dev_libsize"][batch].item())
        expected_noise = rng.normal(0, 0.3, size=6)
        rng.uniform(0, 1, size=6)
        assert background["libsize"][i].item() == expected_library
        np.testing.assert_array_equal(background["noise"][i].numpy(), expected_noise)
    np.testing.assert_array_equal(background["exog_noise"].numpy(), quantiles)
    expected = _reconstruct_counts({"model_parameters": parameters, "cell_background": background}, concepts)
    torch.testing.assert_close(inputs, expected, rtol=0, atol=0)
    assert not np.shares_memory(background["exog_noise"].numpy(), quantiles)


@pytest.mark.parametrize("dependency", [False, True])
@pytest.mark.parametrize("prepared_concepts", [False, True])
@pytest.mark.parametrize("reuse_quantiles", [False, True])
def test_model_external_exog_noise_preserves_other_draws(
    tmp_path, small_config, dependency, prepared_concepts, reuse_quantiles,
):
    config = dict(small_config, use_concept_dependency=dependency)
    dataset = SingleCellDataset(root=tmp_path, generation_config=config)
    concepts = dataset._full_concepts if prepared_concepts else None
    parameters = dataset.model_parameters
    baseline_rng = np.random.default_rng(19)
    baseline_inputs, baseline_concepts, baseline_background = cell_simulator(
        parameters, concepts, n_cells=10, rng=baseline_rng,
    )
    quantiles = (baseline_background["exog_noise"].numpy().copy()
                 if reuse_quantiles else np.full((10, 6), 0.5))
    override_rng = np.random.default_rng(19)
    inputs, actual_concepts, background = cell_simulator(
        parameters, concepts, n_cells=10, rng=override_rng, exog_noise=quantiles,
    )
    torch.testing.assert_close(actual_concepts, baseline_concepts, rtol=0, atol=0)
    for name, value in baseline_background.items():
        if name != "exog_noise":
            torch.testing.assert_close(background[name], value, rtol=0, atol=0)
    np.testing.assert_array_equal(background["exog_noise"].numpy(), quantiles)
    np.testing.assert_array_equal(override_rng.random(10), baseline_rng.random(10))
    expected = _reconstruct_counts(
        {"model_parameters": parameters, "cell_background": background}, actual_concepts,
    )
    torch.testing.assert_close(inputs, expected, rtol=0, atol=0)
    if reuse_quantiles:
        torch.testing.assert_close(inputs, baseline_inputs, rtol=0, atol=0)


def test_model_samples_concepts_background_and_counts_in_one_pass(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    parameters = deepcopy(dataset.model_parameters)
    short = cell_simulator(parameters, n_cells=3, rng=np.random.default_rng(19))
    complete = cell_simulator(parameters, n_cells=5, rng=np.random.default_rng(19))
    # Adding cells must not change the concepts, background or counts already drawn.
    for small, large in zip(short[:2], complete[:2]):
        torch.testing.assert_close(small, large[:3], rtol=0, atol=0)
    for name, background in short[2].items():
        torch.testing.assert_close(background, complete[2][name][:3], rtol=0, atol=0)
    expected = _reconstruct_counts({"model_parameters": parameters, "cell_background": short[2]}, short[1])
    torch.testing.assert_close(short[0], expected, rtol=0, atol=0)


def test_generator_reuses_supplied_parameters(tmp_path, small_config, monkeypatch):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    parameters = deepcopy(dataset.model_parameters)
    parameters["baseline_expr"] += 1
    expected = _generation_tensors(dataset.generate_new_cells(seed=7, model_parameters=parameters))

    def unexpected_parameters(*args, **kwargs):
        raise AssertionError("The cell_simulator must use the supplied global parameters.")

    monkeypatch.setattr(single_cell, "_compute_simulator_parameters", unexpected_parameters)
    actual = _generation_tensors(dataset.generate_new_cells(seed=7, model_parameters=parameters))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dependency", [False, True])
def test_flat_parameters_match_direct_generation(tmp_path, small_config, dependency):
    config = deepcopy(small_config)
    config.update({
        "baseline_lower": 2, "baseline_upper": 6,
        "std_dev_libsize_lower": 0.02, "std_dev_libsize_upper": 0.04,
        "n_concepts": 3, "std_dev_concept": 0.06, "zero_inflate": False,
        "beta_a": 2, "beta_b": 1.5, "use_concept_dependency": dependency,
        "cov_prior_range": 0.5,
        "std_dev_noise": 0.03,
    })
    config["std_dev_batch"] = 0.12
    config["std_dev_celltype"] = 0.11
    config["std_dev_tissue"] = 0.09
    dataset = SingleCellDataset(root=tmp_path, seed=7, generation_config=config)
    expected = _generation_tensors(dataset)
    for actual, wanted in zip(dataset.load_raw()[:2], expected[:2]):
        torch.testing.assert_close(actual, wanted)
    assert dataset.input_data.shape == (10, 6)
    assert len(dataset.concept_names) == 3
    assert dataset.annotations.cardinalities == [1, 1, 1]


def test_partial_config_preserves_defaults_and_caller_data(tmp_path, small_config):
    config = deepcopy(small_config)
    config["baseline_lower"] = 2
    config["std_dev_batch"] = 0.15
    original = deepcopy(config)
    dataset = SingleCellDataset(root=tmp_path, seed=7, generation_config=config)
    assert config == original
    assert dataset.generation_config["baseline_upper"] == 5
    config["std_dev_batch"] = 9
    assert dataset.generation_config["std_dev_batch"] == 0.15
    dataset.generation_config["baseline_upper"] = 20
    default = SingleCellDataset(root=tmp_path, generation_config=small_config)
    assert default.generation_config["baseline_upper"] == 5


def test_equivalent_configs_generate_identical_data(tmp_path, small_config):
    original = SingleCellDataset(root=tmp_path, generation_config=small_config)
    explicit = SingleCellDataset(root=tmp_path, generation_config=original.generation_config)
    assert explicit is not original
    torch.testing.assert_close(explicit.input_data, original.input_data, rtol=0, atol=0)
    assert all(Path(path).is_file() for path in original.processed_paths)


@pytest.mark.parametrize("kwargs", [{}, {"max_samples": None}])
def test_datamodule_forwards_config_and_generation_seed(tmp_path, small_config, kwargs):
    config = deepcopy(small_config)
    config["std_dev_batch"] = 0.15
    module = SingleCellDataModule(root=tmp_path, generation_seed=7, seed=12,
                                  generation_config=config, normalize=False, **kwargs)
    assert module.dataset.input_data.shape == (config["n_cells"], config["n_genes"])
    expected = _generation_tensors(module.dataset)
    for actual, wanted in zip(module.dataset.load_raw()[:2], expected[:2]):
        torch.testing.assert_close(actual, wanted)
    module.setup("fit")
    assert sum(len(part) for part in (module.trainset, module.valset, module.testset)) == 10


@pytest.mark.parametrize("max_samples", [0, 5, 10, 20, False])
def test_datamodule_rejects_max_samples_before_generation(tmp_path, small_config, monkeypatch, max_samples):
    def unexpected_dataset(*args, **kwargs):
        raise AssertionError("Invalid max_samples must be rejected before constructing the dataset.")

    monkeypatch.setattr(SingleCellDataset, "__init__", unexpected_dataset)
    with pytest.raises(ValueError, match=r'max_samples.*generation_config\["n_cells"\]'):
        SingleCellDataModule(root=tmp_path, generation_config=small_config,
                             max_samples=max_samples)


@pytest.mark.parametrize("target_sum", [None, 1_000, 1_000_000, 1234.5])
def test_datamodule_normalization_target_and_saved_counts(tmp_path, small_config, target_sum):
    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    counts = source.input_data.clone()
    counts[0] = torch.tensor([1, 3, 0, 0, 0, 0])
    counts[1] = 2 * counts[0]
    counts[2] = 0
    torch.save(counts, source.processed_paths[0])
    saved_files = {path: Path(path).read_bytes() for path in source.processed_paths}

    kwargs = {} if target_sum is None else {"target_sum": target_sum}
    expected_total = 10_000 if target_sum is None else target_sum
    module = SingleCellDataModule(root=source.root, batch_size=3, **kwargs)
    assert module.dataset.target_sum == expected_total
    assert module.dataset.root == source.root
    expression = module.dataset.input_data
    expected = torch.tensor([np.log1p(expected_total / 4), np.log1p(3 * expected_total / 4),
                             0, 0, 0, 0], dtype=torch.float32)
    torch.testing.assert_close(expression[0], expected)
    torch.testing.assert_close(expression[1], expression[0], rtol=0, atol=0)
    assert torch.count_nonzero(expression[2]) == 0
    assert expression.dtype == torch.float32
    assert expression.isfinite().all()
    totals = expression.expm1().sum(dim=1)
    torch.testing.assert_close(totals[counts.sum(dim=1) > 0],
                               torch.full((9,), expected_total, dtype=expression.dtype))
    torch.testing.assert_close(module.dataset.load_raw()[0], counts, rtol=0, atol=0)
    torch.testing.assert_close(module.dataset.concepts.tensor, source.concepts.tensor, rtol=0, atol=0)
    assert {path: Path(path).read_bytes() for path in saved_files} == saved_files

    module.setup()
    batch = next(iter(module.train_dataloader()))["inputs"]["x"]
    assert torch.all((batch == 0).all(dim=1) | torch.isclose(
        batch.expm1().sum(dim=1), torch.tensor(expected_total, dtype=batch.dtype)))
    restored = SingleCellDataModule(root=source.root, **kwargs)
    torch.testing.assert_close(restored.dataset.input_data, expression, rtol=0, atol=0)
    raw = SingleCellDataModule(root=source.root, normalize=False, **kwargs)
    torch.testing.assert_close(raw.dataset.input_data, counts, rtol=0, atol=0)


@pytest.mark.parametrize("entrypoint", [SingleCellDataset, SingleCellDataModule])
@pytest.mark.parametrize("target_sum", [0, -1, np.nan, np.inf, -np.inf, True, "10000", None])
def test_normalization_rejects_invalid_target_sum(tmp_path, entrypoint, target_sum):
    with pytest.raises(ValueError, match="target_sum must be a positive finite number"):
        entrypoint(root=tmp_path, target_sum=target_sum)


@pytest.mark.parametrize("target_sum", [10_000, 1234.5])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("kind", ["intervened", "generated"])
@pytest.mark.parametrize("entrypoint", ["dataset", "datamodule"])
def test_normalization_is_preserved_in_generated_variants(
    tmp_path, small_config, normalize, kind, entrypoint, target_sum,
):
    kwargs = {"root": tmp_path, "generation_config": small_config,
              "normalize": normalize, "target_sum": target_sum}
    source = (SingleCellDataset(**kwargs) if entrypoint == "dataset"
              else SingleCellDataModule(**kwargs).dataset)
    source._subset_rows([7, 2, 9])
    original = _generation_tensors(source)
    saved_files = {path: Path(path).read_bytes() for path in source.processed_paths}
    variant = (source.generate_interventions({"concept_0": 1}) if kind == "intervened"
               else source.generate_new_cells(n_cells=5, seed=7))
    counts = variant.load_raw()[0]
    if normalize:
        nonempty = counts.sum(dim=1) > 0
        totals = variant.input_data.expm1().sum(dim=1)
        torch.testing.assert_close(totals[nonempty], torch.full_like(totals[nonempty], target_sum))
    else:
        torch.testing.assert_close(variant.input_data, counts, rtol=0, atol=0)
    assert source.normalize == variant.normalize == normalize
    assert source.target_sum == variant.target_sum == target_sum
    torch.testing.assert_close(variant.load()[0], variant.input_data, rtol=0, atol=0)
    restored = SingleCellDataset(root=variant.root, normalize=normalize, target_sum=target_sum)
    torch.testing.assert_close(_generation_tensors(restored),
                               _generation_tensors(variant), rtol=0, atol=0)
    unchanged = variant.generate_interventions({})
    assert unchanged.normalize == normalize
    assert unchanged.target_sum == target_sum
    torch.testing.assert_close(unchanged.input_data, variant.input_data, rtol=0, atol=0)
    torch.testing.assert_close(_generation_tensors(source), original, rtol=0, atol=0)
    assert {path: Path(path).read_bytes() for path in saved_files} == saved_files


def test_numpy_counts_are_supported(tmp_path, small_config):
    config = deepcopy(small_config)
    config["n_cells"] = np.int64(10)
    config["std_dev_batch"] = np.float32(0.15)
    dataset = SingleCellDataset(root=tmp_path, generation_config=config)
    assert len(dataset) == 10


def test_annotations_describe_complete_and_selected_concepts(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config,
                                concept_subset=["concept_1"])
    inputs, concepts, annotations, graph = dataset.load_raw()
    assert graph is None
    assert inputs.shape == (10, 6)
    assert concepts.shape == (10, 3)
    assert annotations.labels == ["concept_0", "concept_1", "concept_2"]
    assert annotations.cardinalities == [1, 1, 1]
    assert dataset.concept_names == ["concept_1"]
    assert dataset.annotations.cardinalities == [1]


def _reconstruct_counts(generation_state, concepts):
    parameters = {name: values.numpy() for name, values in generation_state["model_parameters"].items()}
    state = {name: values.numpy() for name, values in generation_state["cell_background"].items()}
    concepts = concepts.to_numpy() if isinstance(concepts, pd.DataFrame) else concepts.numpy()
    celltype, tissue, batch = state["celltype"], state["tissue"], state["batch"]
    log_rate = (
        parameters["baseline_expr"]
        + np.sum(parameters["concept_effect"][None] * concepts[:, :, None], axis=1)
        + parameters["celltype_effect"][celltype]
        + parameters["batch_effect"][batch]
        + parameters["tissue_effect"][tissue]
        + state["libsize"][:, None]
        + state["noise"]
    )
    return torch.from_numpy(poisson.ppf(state["exog_noise"], np.exp(log_rate))).float()


@pytest.mark.parametrize("dependency,n_original_concepts", [(False, 3), (True, 3), (True, 1)])
@pytest.mark.parametrize("supplied_quantiles", [False, True])
def test_generation_state_reconstructs_counts_and_is_reproducible(
    tmp_path, small_config, monkeypatch, dependency, n_original_concepts, supplied_quantiles,
):
    config = deepcopy(small_config)
    config.update({
        "n_concepts": n_original_concepts, "use_concept_dependency": dependency,
        "zero_inflate": not dependency, "cov_prior_range": 0.5,
    })
    quantiles = np.linspace(0.1, 0.9, 60).reshape(10, 6) if supplied_quantiles else None
    dataset = SingleCellDataset(root=tmp_path, seed=7, generation_config=config, exog_noise=quantiles)
    saved = deepcopy(_state(dataset))
    assert set(saved) == {"generation_config", "model_parameters", "cell_background"}
    assert dataset.seed == 7
    assert saved["generation_config"] == dataset.generation_config
    torch.testing.assert_close(saved["cell_background"]["cell_indices"], torch.arange(10))
    expected_shapes = {
        "baseline_expr": (6,), "concept_effect": (n_original_concepts, 6),
        "celltype_effect": (3, 6), "tissue_effect": (2, 6), "batch_effect": (2, 6),
        "p_batch": (2,), "p_tissue": (2,), "p_celltype": (2, 3),
        "p_concept": (3, n_original_concepts), "std_dev_libsize": (2,), "covariance": (n_original_concepts, n_original_concepts),
    }
    assert {name: tuple(values.shape) for name, values in saved["model_parameters"].items()} == expected_shapes
    assert all(values.dtype == torch.float64 for values in saved["model_parameters"].values())
    state = saved["cell_background"]
    assert {name: tuple(values.shape) for name, values in state.items()} == {
        "cell_indices": (10,),
        "celltype": (10,), "tissue": (10,), "batch": (10,),
        "libsize": (10,),
        "noise": (10, 6), "exog_noise": (10, 6), "bernoulli_prob": (10, n_original_concepts),
    }
    for name, value in state.items():
        assert value.dtype == (torch.long if name in ("cell_indices", "celltype", "tissue", "batch") else torch.float64)
    raw_concepts = dataset.load_raw()[1]
    if supplied_quantiles:
        np.testing.assert_array_equal(state["exog_noise"].numpy(), quantiles)
    torch.testing.assert_close(_reconstruct_counts(saved, raw_concepts), dataset.input_data, rtol=0, atol=0)
    replayed, concepts, _ = _generation_tensors(dataset.generate_interventions(concepts=raw_concepts))
    torch.testing.assert_close(replayed, dataset.input_data, rtol=0, atol=0)
    torch.testing.assert_close(concepts, raw_concepts, rtol=0, atol=0)

    restored = SingleCellDataset(root=tmp_path, seed=7, generation_config=config,
                                 exog_noise=quantiles, concept_subset=["concept_0"])
    assert len(restored.load_raw()) == 4
    for group in ("model_parameters", "cell_background"):
        for name, values in saved[group].items():
            torch.testing.assert_close(_state(restored)[group][name], values, rtol=0, atol=0)
    torch.testing.assert_close(_reconstruct_counts(_state(restored), _concept_frame(restored)), restored.input_data, rtol=0, atol=0)


@pytest.mark.parametrize("normalize", [False, True])
def test_subsampling_keeps_inputs_native_full_concepts_and_background_aligned(tmp_path, small_config, normalize):
    complete = SingleCellDataset(root=tmp_path, generation_config=small_config, normalize=normalize)
    selected = SingleCellDataset(root=complete.root, concept_subset=["concept_1"], normalize=normalize)
    module = ConceptDataModule(dataset=selected, max_samples=5, seed=12)
    dataset = module.dataset
    indices = torch.randperm(10, generator=torch.Generator().manual_seed(12))[:5]
    torch.testing.assert_close(dataset.input_data, complete.input_data[indices], rtol=0, atol=0)
    torch.testing.assert_close(dataset.native_concepts.tensor, complete.native_concepts["concept_1"].tensor[indices], rtol=0, atol=0)
    torch.testing.assert_close(dataset._full_concepts, complete._full_concepts[indices], rtol=0, atol=0)
    torch.testing.assert_close(_state(dataset)["cell_background"]["cell_indices"], indices)
    for name, values in _state(complete)["cell_background"].items():
        torch.testing.assert_close(_state(dataset)["cell_background"][name], values[indices])
    torch.testing.assert_close(_reconstruct_counts(_state(dataset), _concept_frame(dataset)),
                               complete.load_raw()[0][indices], rtol=0, atol=0)
    # Background and identifiers follow the current in-memory selection.
    saved = deepcopy(_state(dataset))
    assert len(saved["cell_background"]["cell_indices"]) == 5
    assert _state(dataset)["cell_background"]["noise"].shape[0] == 5

    dataset._subset_rows([3, 1])
    indices = indices[[3, 1]]
    torch.testing.assert_close(dataset.input_data, complete.input_data[indices], rtol=0, atol=0)
    torch.testing.assert_close(dataset.native_concepts.tensor, complete.native_concepts["concept_1"].tensor[indices], rtol=0, atol=0)
    torch.testing.assert_close(dataset._full_concepts, complete._full_concepts[indices], rtol=0, atol=0)
    for name, values in _state(complete)["cell_background"].items():
        torch.testing.assert_close(_state(dataset)["cell_background"][name], values[indices])
    torch.testing.assert_close(_reconstruct_counts(_state(dataset), _concept_frame(dataset)),
                               complete.load_raw()[0][indices], rtol=0, atol=0)
    replayed, _, _ = _generation_tensors(dataset.generate_interventions(concepts=_concept_frame(dataset)))
    torch.testing.assert_close(replayed, dataset.input_data, rtol=0, atol=0)


@pytest.mark.parametrize("as_numpy", [False, True])
def test_public_generator_intervenes_with_fixed_noise_and_preserves_source(tmp_path, small_config, monkeypatch, as_numpy):
    config = deepcopy(small_config)
    config.update({"baseline_lower": 3, "baseline_upper": 3,
                   "std_dev_libsize_lower": 0, "std_dev_libsize_upper": 0, "std_dev_noise": 0})
    for name in ("std_dev_batch", "std_dev_celltype", "std_dev_tissue", "std_dev_concept"):
        config[name] = 0
    quantiles = np.linspace(0.1, 0.9, 60).reshape(10, 6)
    dataset = SingleCellDataset(root=tmp_path, generation_config=config, exog_noise=quantiles)
    # A known effect makes the intervened rate exactly twice the baseline rate.
    parameters = dataset.model_parameters
    parameters["concept_effect"][0] = np.log(2)
    dataset = dataset.generate_new_cells(model_parameters=parameters)
    quantiles = dataset.cell_background["exog_noise"].numpy()
    _, original_concepts, state = _generation_tensors(dataset)
    original_state = deepcopy(state)
    edited = original_concepts.clone()
    edited[:] = 0
    edited[:, 0] = 1

    def unexpected_sampling(*args, **kwargs):
        raise AssertionError("Reusing state must not draw fresh randomness.")

    monkeypatch.setattr(np.random, "default_rng", unexpected_sampling)
    inputs, concepts, updated = _generation_tensors(dataset.generate_interventions(concepts=edited.numpy() if as_numpy else edited))
    expected = torch.from_numpy(poisson.ppf(quantiles, np.exp(np.log(3) + np.log(2)))).float()
    torch.testing.assert_close(inputs, expected, rtol=0, atol=0)
    torch.testing.assert_close(concepts, edited)
    torch.testing.assert_close(updated["cell_background"], original_state["cell_background"], rtol=0, atol=0)
    for group in ("model_parameters", "cell_background"):
        for name, values in original_state[group].items():
            torch.testing.assert_close(state[group][name], values, rtol=0, atol=0)
    edited[:] = 0
    assert torch.all(concepts[:, 0] == 1)
    replayed, _, _ = _generation_tensors(dataset.generate_interventions(concepts=concepts))
    torch.testing.assert_close(replayed, inputs, rtol=0, atol=0)


@pytest.mark.parametrize("concepts", [torch.zeros(10, 7), torch.zeros(5, 3),
                                     torch.full((10, 3), float("inf")), torch.full((10, 3), float("nan"))])
def test_generator_rejects_invalid_interventions(tmp_path, small_config, concepts):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    with pytest.raises(ValueError, match="concepts|class indices"):
        _generation_tensors(dataset.generate_interventions(concepts=concepts))


def test_generation_state_can_be_copied_independently(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config,
                                concept_subset=["concept_1"])
    expected = _state(dataset)
    parameters = dataset.model_parameters
    parameters["baseline_expr"].zero_()
    background = dataset.cell_background
    background["noise"].zero_()
    copied = _state(dataset)
    copied["cell_background"]["cell_indices"].zero_()
    copied["model_parameters"]["baseline_expr"].zero_()
    torch.testing.assert_close(_state(dataset)["cell_background"]["cell_indices"], torch.arange(10))
    config = dataset.generation_config
    config["n_cells"] = 3
    assert dataset.generation_config["n_cells"] == 10
    torch.testing.assert_close(_state(dataset), expected, rtol=0, atol=0)
    with pytest.raises(AttributeError):
        dataset.generation_config = config
    torch.testing.assert_close(_reconstruct_counts(_state(dataset), _concept_frame(dataset)), dataset.input_data, rtol=0, atol=0)


@pytest.mark.parametrize("n_cells", [None, 10, 5])
@pytest.mark.parametrize("parameters_changed", [False, True])
def test_generate_new_cells_samples_new_cells(tmp_path, small_config, n_cells, parameters_changed):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config,
                                concept_subset=["concept_1"], label_descriptions={"concept_1": "Example"})
    original_metadata = deepcopy(_state(dataset))
    parameters = {"baseline_expr": dataset.model_parameters["baseline_expr"] + 1} if parameters_changed else None
    seed = dataset.seed if parameters_changed else 7
    fresh = dataset.generate_new_cells(n_cells=n_cells, model_parameters=parameters, seed=seed)
    assert fresh is not dataset
    assert len(fresh) == (10 if n_cells is None else n_cells)
    assert fresh.concept_names == dataset.concept_names
    assert fresh.label_descriptions == dataset.label_descriptions
    assert fresh.label_descriptions is not dataset.label_descriptions
    assert fresh.scalers is not dataset.scalers
    assert fresh.generated_concepts is not dataset.generated_concepts
    for name, value in dataset.model_parameters.items():
        expected = parameters[name] if parameters is not None and name in parameters else value
        torch.testing.assert_close(fresh.model_parameters[name], expected, rtol=0, atol=0)
    assert not torch.equal(fresh.cell_background["noise"], dataset.cell_background["noise"][:len(fresh)])
    torch.testing.assert_close(_state(dataset), original_metadata, rtol=0, atol=0)


@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("name", ["baseline_expr", "concept_effect", "celltype_effect", "tissue_effect",
                                  "batch_effect", "p_batch", "p_tissue", "p_celltype", "p_concept",
                                  "std_dev_libsize", "covariance"])
def test_parameter_overrides_reject_changed_dimensions(tmp_path, small_config, name, fresh, monkeypatch):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    parameters = deepcopy(dataset.model_parameters)
    shape = parameters[name].shape
    incompatible = torch.zeros((shape[0] + 1, *shape[1:]))

    def unexpected_model(*args, **kwargs):
        raise AssertionError("Invalid parameter shapes must be rejected before running the cell_simulator.")

    monkeypatch.setattr(single_cell, "cell_simulator", unexpected_model)
    with pytest.raises(ValueError, match="shape"):
        dataset.generate_new_cells(model_parameters={name: incompatible}, **({"n_cells": 5} if fresh else {}))
    torch.testing.assert_close(dataset.model_parameters[name], parameters[name], rtol=0, atol=0)


@pytest.mark.parametrize("parameters", [{"unknown": 1.0}, {"baseline_expr": torch.full((6,), float("nan"))},
                                      {"baseline_expr": torch.full((6,), float("inf"))}])
def test_parameter_overrides_reject_unknown_names_and_nonfinite_values(tmp_path, small_config, parameters):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    with pytest.raises(ValueError, match="parameter|Parameter"):
        dataset.generate_new_cells(model_parameters=parameters)


@pytest.mark.parametrize("entrypoint", ["cell_simulator", "generate_new_cells"])
@pytest.mark.parametrize("name, value", [
    ("p_batch", [-0.1, 1.1]),
    ("p_batch", [0.2, 0.2]),
    ("p_batch", [0.7, 0.7]),
    ("p_tissue", [1.1, -0.1]),
    ("p_tissue", [0.6, 0.6]),
    ("p_celltype", [[-0.1, 0.5, 0.6], [0.2, 0.3, 0.5]]),
    ("p_celltype", [[0.2, 0.3, 0.5], [0.2, 0.3, 0.4]]),
    ("p_concept", [[-0.1, 0.5, 0.6], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]),
    ("p_concept", [[1.1, 0.5, 0.6], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]),
    ("covariance", [[1., 0.5, 0.], [0., 1., 0.], [0., 0., 1.]]),
    ("covariance", [[1., 2., 0.], [2., 1., 0.], [0., 0., 1.]]),
    ("covariance", [[float("nan"), 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
    ("std_dev_libsize", [-0.01, 0.02]),
])
def test_model_parameters_reject_invalid_distributions(tmp_path, small_config, entrypoint, name, value):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    original = _state(dataset)
    with pytest.raises(ValueError, match=name):
        if entrypoint == "cell_simulator":
            parameters = dataset.model_parameters
            parameters[name] = torch.tensor(value, dtype=torch.float64)
            cell_simulator(parameters, n_cells=len(dataset))
        else:
            dataset.generate_new_cells(model_parameters={name: value})
    torch.testing.assert_close(_state(dataset), original, rtol=0, atol=0)
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("entrypoint", ["cell_simulator", "generate_new_cells"])
@pytest.mark.parametrize("covariance", [torch.zeros(3, 3), torch.ones(3, 3), torch.eye(3)])
def test_model_parameters_accept_valid_boundary_distributions(tmp_path, small_config, entrypoint, covariance):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    overrides = {
        "p_batch": [0., 1.], "p_tissue": [1., 0.],
        "p_celltype": [[1., 0., 0.], [0., 0.5, 0.5]],
        # Each concept has its own Bernoulli probability; rows need not sum to 1.
        "p_concept": [[0., 1., 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]],
        "std_dev_libsize": [0., 0.], "covariance": covariance,
    }
    parameters = {**dataset.model_parameters,
                  **{name: torch.as_tensor(value, dtype=torch.float64) for name, value in overrides.items()}}
    if entrypoint == "cell_simulator":
        inputs, _, background = cell_simulator(parameters, n_cells=len(dataset))
    else:
        generated = dataset.generate_new_cells(model_parameters=overrides)
        torch.testing.assert_close(generated.model_parameters, parameters, rtol=0, atol=0)
        inputs, background = generated.input_data, generated.cell_background
    assert inputs.isfinite().all()
    assert torch.all(background["batch"] == 1)
    assert torch.all(background["tissue"] == 0)
    assert torch.all(background["celltype"] == 0)
    assert torch.count_nonzero(background["libsize"]) == 0


def test_interventions_keep_sampled_context(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    fresh = dataset.generate_new_cells(
        n_cells=5, seed=19,
        model_parameters={"p_tissue": [1., 0.], "p_celltype": [[1., 0., 0.], [0., 1., 0.]]},
    )
    edited = fresh.generate_interventions({"concept_0": 1})
    assert torch.all(edited.cell_background["tissue"] == 0)
    assert torch.all(edited.cell_background["celltype"] == 0)
    for name, value in fresh.cell_background.items():
        torch.testing.assert_close(edited.cell_background[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["celltype", "tissue", "batch"])
@pytest.mark.parametrize("kind", ["mapping", "dataframe"])
def test_context_is_not_a_concept_or_intervention_target(tmp_path, small_config, name, kind):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    assert name not in dataset.concept_names
    assert name in dataset.cell_background
    changes = {name: 0}
    if kind == "dataframe":
        changes = pd.DataFrame(changes, index=_concept_frame(dataset).index)
    with pytest.raises(ValueError, match="concept"):
        dataset.generate_interventions(changes)


def test_legacy_context_concept_arrays_are_rejected(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    legacy = torch.cat([dataset._full_concepts,
                        torch.stack([dataset.cell_background[name]
                                     for name in ("celltype", "tissue", "batch")], dim=1)], dim=1)
    with pytest.raises(ValueError, match="shape"):
        dataset.generate_interventions(legacy)
    with pytest.raises(ValueError, match="binary concept columns"):
        cell_simulator(dataset.model_parameters, legacy, cell_background=dataset.cell_background)


@pytest.mark.parametrize("name", ["celltype", "tissue", "batch"])
@pytest.mark.parametrize("value", [-1, 0.5, float("nan"), 3])
def test_model_rejects_invalid_fixed_context(tmp_path, small_config, name, value):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    background = dataset.cell_background
    background[name] = background[name].double()
    background[name][0] = value
    with pytest.raises(ValueError, match="cell_background"):
        cell_simulator(dataset.model_parameters, dataset._full_concepts, cell_background=background)


@pytest.mark.parametrize("kind", ["mapping", "tensor", "numpy", "dataframe", "list"])
@pytest.mark.parametrize("value", [0.25, -0.25, 1.35, 1 - 1e-9])
def test_interventions_reject_nonbinary_concepts(tmp_path, small_config, kind, value):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    replacement = dataset._full_concepts.clone()
    replacement = replacement.double()
    replacement[:, 0] = value
    frame = _concept_frame(dataset)
    overrides = {
        "mapping": {"concept_0": value},
        "tensor": replacement[:, :3],
        "numpy": replacement[:, :3].numpy(), "list": replacement[:, :3].tolist(),
        "dataframe": pd.DataFrame(replacement.numpy(), index=frame.index,
                                 columns=frame.columns).iloc[::-1, ::-1],
    }[kind]
    with pytest.raises(ValueError, match="binary"):
        dataset.generate_interventions(concepts=overrides)
    with pytest.raises(ValueError, match="binary"):
        cell_simulator(dataset.model_parameters, replacement, cell_background=dataset.cell_background)


@pytest.mark.parametrize("normalize", [False, True])
def test_intervened_dataset_shares_explicit_indices_with_original_data(counterfactual_source, normalize):
    module = SingleCellDataModule(root=counterfactual_source.root, seed=12, normalize=normalize,
                                 concept_subset=["concept_1"], batch_size=2)
    source = module.dataset
    original = _generation_tensors(source)
    saved_files = {path: Path(path).read_bytes() for path in source.processed_paths}
    module.setup()
    default_splitter = module.splitter
    intervened = source.generate_interventions({"concept_0": 1}, cell_type=2)
    assert module.splitter is default_splitter
    assert not hasattr(intervened, "splitter")
    split = FixedIndicesSplitter(train_idxs=[3, 4, 5, *range(10, 16)],
                                 val_idxs=[], test_idxs=[0, 1, 2])
    assert split.fitted
    assert len(split.test_idxs) == 3
    assert split.val_idxs == []
    assert set(split.test_idxs).issubset(range(6))
    assert set(split.train_idxs).isdisjoint(split.test_idxs)
    assert set(split.train_idxs) | set(split.test_idxs) == set(range(16)) - set(range(6, 10))
    assert set(range(10, 16)).issubset(split.train_idxs)
    assert len(set(split.train_idxs) & set(range(6))) == 3
    assert len(intervened) == len(source)
    assert intervened.normalize == normalize
    assert intervened.concept_names == source.concept_names
    assert torch.all(intervened._full_concepts[:10, 0] == 1)
    torch.testing.assert_close(intervened._full_concepts[10:], original[1][10:], rtol=0, atol=0)
    torch.testing.assert_close(intervened.input_data[10:], original[0][10:], rtol=0, atol=0)
    torch.testing.assert_close(_generation_tensors(source), original, rtol=0, atol=0)
    torch.testing.assert_close(intervened.cell_background, source.cell_background, rtol=0, atol=0)
    assert {path: Path(path).read_bytes() for path in saved_files} == saved_files

    module.splitter = split
    module.setup()
    assert module.trainset.dataset is source
    assert module.testset.dataset is source
    assert module.valset is None
    assert module.trainset.indices == split.train_idxs
    assert module.testset.indices == split.test_idxs
    torch.testing.assert_close(module.trainset[0]["inputs"]["x"],
                               original[0][split.train_idxs[0]], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["mapping", "tensor", "numpy", "dataframe", "list"])
def test_cell_type_interventions_filter_by_current_cell_names(counterfactual_source, kind):
    source = counterfactual_source
    source._subset_rows([12, 1, 7, 3])
    original = _generation_tensors(source)
    replacement = source._full_concepts.clone()
    replacement[:, 0] = 1
    overrides = {
        "mapping": {"concept_0": 1}, "tensor": replacement,
        "numpy": replacement.numpy(), "list": replacement.tolist(),
        "dataframe": pd.DataFrame(replacement.numpy(), index=_concept_frame(source).index,
                                  columns=source.concept_names).iloc[::-1, ::-1],
    }
    intervened = source.generate_interventions(overrides[kind], cell_type=2)
    expected = original[1].clone()
    expected[1:, 0] = 1
    torch.testing.assert_close(intervened._full_concepts, expected, rtol=0, atol=0)
    torch.testing.assert_close(intervened.input_data, _reconstruct_counts(original[2], expected), rtol=0, atol=0)
    torch.testing.assert_close(intervened.cell_background["cell_indices"], torch.tensor([12, 1, 7, 3]))
    assert "split_mapping.h5" not in intervened.processed_filenames


def test_interventions_keep_default_datamodule_splitting(counterfactual_source):
    module = SingleCellDataModule(root=counterfactual_source.root, seed=12)
    default_splitter = module.splitter
    intervened = module.dataset.generate_interventions({"concept_0": 1}, cell_type=2)
    assert not hasattr(intervened, "splitter")
    assert "split_mapping.h5" not in intervened.processed_filenames
    assert module.splitter is default_splitter
    assert isinstance(module.splitter, RandomSplitter)
    module.setup()
    assert len(module.trainset) + len(module.valset) + len(module.testset) == 16


@pytest.mark.parametrize("cell_type", [-1, 3, 1.5, "2", True])
def test_cell_type_interventions_reject_invalid_ids(counterfactual_source, cell_type):
    with pytest.raises(ValueError, match="cell_type"):
        counterfactual_source.generate_interventions({"concept_0": 1}, cell_type=cell_type)


def test_interventions_preserve_source_and_return_independent_datasets(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config,
                                concept_subset=["concept_1"])
    source = deepcopy(_state(dataset))
    copied = dataset.generate_interventions(dataset._full_concepts)
    torch.testing.assert_close(copied.input_data, dataset.input_data, rtol=0, atol=0)
    assert copied.concept_names == dataset.concept_names
    edited = dataset.generate_interventions({"concept_0": 1, "concept_1": 0})
    frame = _concept_frame(edited)
    assert (frame["concept_0"] == 1).all()
    assert (frame["concept_1"] == 0).all()
    for name, value in source["cell_background"].items():
        torch.testing.assert_close(edited.cell_background[name], value, rtol=0, atol=0)
    _state(edited)["model_parameters"]["baseline_expr"].zero_()
    _state(edited)["cell_background"]["noise"].zero_()
    torch.testing.assert_close(_state(dataset), source, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["concepts", "parameters"])
def test_dataset_overrides_only_evaluate_final_model(tmp_path, small_config, monkeypatch, change):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    calls = []

    def unexpected_parameters(*args, **kwargs):
        raise AssertionError("Reuse the existing global parameters.")

    def evaluate(*args, **kwargs):
        calls.append(kwargs.get("cell_background"))
        return cell_simulator(*args, **kwargs)

    monkeypatch.setattr(single_cell, "_compute_simulator_parameters", unexpected_parameters)
    monkeypatch.setattr(single_cell, "cell_simulator", evaluate)
    if change == "concepts":
        edited = dataset.generate_interventions({"concept_0": 1})
        assert calls[0] is dataset._cell_background
    else:
        edited = dataset.generate_new_cells(model_parameters={"baseline_expr": dataset.model_parameters["baseline_expr"] + 1})
        assert calls[0] is None
    assert len(calls) == 1
    torch.testing.assert_close(edited.input_data, _reconstruct_counts(_state(edited), edited._full_concepts), rtol=0, atol=0)


def test_dataframe_interventions_align_by_name(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    concepts = _concept_frame(dataset)
    concepts.loc[concepts.index[:4], "concept_0"] = 1
    shuffled = concepts.iloc[::-1, ::-1]
    edited = dataset.generate_interventions(concepts=shuffled)
    pd.testing.assert_frame_equal(_concept_frame(edited), concepts)
    expected = dataset.generate_interventions(concepts={"concept_0": concepts["concept_0"].values})
    torch.testing.assert_close(edited.input_data, expected.input_data, rtol=0, atol=0)


def test_partial_dataframe_interventions_preserve_omitted_cells_and_concepts(tmp_path, small_config, monkeypatch):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config,
                                concept_subset=["concept_1"])
    dataset._subset_rows([7, 2, 9])
    source = deepcopy(_state(dataset))
    original_concepts = _concept_frame(dataset).copy()
    changes = pd.DataFrame({"concept_2": [1, 0], "concept_0": [0, 1]},
                           index=["cell_9", "cell_7"])
    expected = original_concepts.copy()
    expected.loc[changes.index, changes.columns] = changes

    def unexpected_sampling(*args, **kwargs):
        raise AssertionError("Omitted concepts must be retained, without resampling.")

    monkeypatch.setattr(np.random, "default_rng", unexpected_sampling)
    edited = dataset.generate_interventions(changes)
    pd.testing.assert_frame_equal(_concept_frame(edited), expected)
    pd.testing.assert_frame_equal(_concept_frame(dataset), original_concepts)
    torch.testing.assert_close(_state(dataset), source, rtol=0, atol=0)
    for group in ("model_parameters", "cell_background"):
        torch.testing.assert_close(_state(edited)[group], source[group], rtol=0, atol=0)
    torch.testing.assert_close(edited.input_data, _reconstruct_counts(source, expected), rtol=0, atol=0)
    complete = dataset.generate_interventions(expected)
    assert complete.processed_paths == edited.processed_paths


@pytest.mark.parametrize("cell_names", [["cell_unknown"], ["cell_0", "cell_0"]])
def test_partial_dataframe_interventions_reject_invalid_cells(tmp_path, small_config, cell_names):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    changes = pd.DataFrame({"concept_0": 1.0}, index=cell_names)
    with pytest.raises(ValueError, match="cells"):
        dataset.generate_interventions(changes)


def test_copy_of_intervened_dataset_preserves_existing_changes(tmp_path, small_config):
    original = SingleCellDataset(root=tmp_path, generation_config=small_config)
    baseline = original.model_parameters["baseline_expr"] + 1
    edited = original.generate_new_cells(
        model_parameters={"baseline_expr": baseline},
    ).generate_interventions({"concept_0": 1})
    copied = edited.generate_interventions(concepts=edited._full_concepts)
    torch.testing.assert_close(copied.input_data, edited.input_data, rtol=0, atol=0)
    pd.testing.assert_frame_equal(_concept_frame(copied), _concept_frame(edited))
    torch.testing.assert_close(copied.model_parameters["baseline_expr"], baseline)
    for name, value in edited.cell_background.items():
        torch.testing.assert_close(copied.cell_background[name], value, rtol=0, atol=0)
    chained = copied.generate_interventions(concepts={"concept_1": 0})
    assert (_concept_frame(chained)["concept_0"] == 1).all()
    assert (_concept_frame(chained)["concept_1"] == 0).all()
    assert copied.scalers is not edited.scalers
    assert copied.generated_concepts is not edited.generated_concepts
    repeated = copied.generate_interventions(concepts=copied._full_concepts)
    torch.testing.assert_close(repeated.input_data, copied.input_data, rtol=0, atol=0)


def test_copy_of_row_subset_keeps_selected_cells_and_raw_tensors(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config,
                                concept_subset=["concept_1"])
    module = ConceptDataModule(dataset=dataset, max_samples=5, seed=12)
    source = module.dataset
    copied = source.generate_interventions(concepts=source._full_concepts)
    assert copied.generation_config["n_cells"] == 5
    assert copied.concept_names == source.concept_names
    torch.testing.assert_close(_state(copied)["cell_background"]["cell_indices"], _state(source)["cell_background"]["cell_indices"])
    torch.testing.assert_close(copied.input_data, source.input_data, rtol=0, atol=0)
    copied._subset_rows([2, 0])
    # Raw tensors retain the full generated dataset before row selection.
    saved_inputs, _, _, _ = copied.load_raw()
    torch.testing.assert_close(saved_inputs, source.input_data, rtol=0, atol=0)
    assert _state(copied)["cell_background"]["noise"].shape[0] == 2
    assert copied.cell_background["noise"].shape[0] == 2
    assert source.cell_background["noise"].shape[0] == 5


def test_dataframe_intervention_on_selected_cells_preserves_identifiers(tmp_path, small_config):
    original = SingleCellDataset(root=tmp_path, generation_config=small_config)
    original._subset_rows([7, 2, 9])
    concepts = _concept_frame(original)
    assert concepts.index.tolist() == ["cell_7", "cell_2", "cell_9"]
    concepts.loc["cell_2", "concept_0"] = 1
    edited = original.generate_interventions(concepts=concepts.iloc[::-1, ::-1])
    pd.testing.assert_frame_equal(_concept_frame(edited), concepts)
    torch.testing.assert_close(_state(edited)["cell_background"]["cell_indices"], torch.tensor([7, 2, 9]))


def test_generate_new_cells_reuses_parameters(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, seed=7, generation_config=small_config)
    fresh = dataset.generate_new_cells(n_cells=5, seed=19)
    repeated = dataset.generate_new_cells(n_cells=5, seed=19)
    assert fresh.input_data.shape == (5, 6)
    assert fresh.cell_background["noise"].shape == (5, 6)
    assert fresh.seed == 19
    assert fresh.generation_config["n_cells"] == 5
    assert len(dataset) == 10
    for name, value in dataset.model_parameters.items():
        torch.testing.assert_close(fresh.model_parameters[name], value, rtol=0, atol=0)
    torch.testing.assert_close(fresh.input_data, repeated.input_data, rtol=0, atol=0)
    stronger = dataset.generate_new_cells(model_parameters={"baseline_expr": dataset.model_parameters["baseline_expr"] + 1})
    counts, _, background = cell_simulator(dataset.model_parameters, n_cells=len(dataset),
                                   rng=np.random.default_rng(dataset.seed))
    assert torch.all(stronger.input_data >= counts)
    for name, value in background.items():
        torch.testing.assert_close(stronger.cell_background[name], value, rtol=0, atol=0)


def test_new_cells_apply_modified_sampling_parameters_then_interventions(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    fresh = dataset.generate_new_cells(n_cells=5, seed=19,
        model_parameters={"p_tissue": [0, 1], "p_batch": [0, 1], "std_dev_libsize": [0, 0]})
    edited = fresh.generate_interventions({"concept_0": 1})
    inputs, concepts, data = _generation_tensors(edited)
    assert inputs.shape == (5, 6)
    assert torch.all(concepts[:, 0] == 1)
    assert torch.all(data["cell_background"]["tissue"] == 1)
    assert torch.all(data["cell_background"]["batch"] == 1)
    assert torch.all(data["cell_background"]["libsize"] == 0)
    torch.testing.assert_close(_reconstruct_counts(data, concepts), inputs, rtol=0, atol=0)
    assert len(dataset) == 10
    assert not torch.all(dataset.model_parameters["std_dev_libsize"] == 0)


@pytest.mark.parametrize("changes", [{"unknown": 1}, {"concept_0": float("nan")}, {"batch": 2},
                                    {"celltype": -1}, {"tissue": 0.5}, {"concept_0": [1, 0]}])
def test_named_intervention_validation(tmp_path, small_config, changes):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    with pytest.raises(ValueError):
        dataset.generate_interventions(concepts=changes)


def test_plot_uses_umap_with_numeric_concepts_and_categorical_context(tmp_path, small_config):
    pytest.importorskip("scanpy")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    original = dataset.input_data.clone()
    axes = dataset.plot(ncols=3)
    assert len(axes) == 6
    for ax, name in zip(axes, dataset.concept_names + ["celltype", "tissue", "batch"]):
        assert ax.get_title() == name
        assert ax.get_xlabel() == "UMAP1"
        assert sum(len(c.get_offsets()) for c in ax.collections) == len(dataset)
    assert axes[0].collections[0].get_array() is not None
    assert axes[-1].get_legend() is not None
    torch.testing.assert_close(dataset.input_data, original, rtol=0, atol=0)
    axes[0].figure.savefig(tmp_path / "umap.png")
    plt.close(axes[0].figure)


@pytest.mark.parametrize("n_cells,n_genes", [(1, 6), (3, 6), (10, 1)])
def test_plot_rejects_too_few_cells_or_genes(tmp_path, small_config, n_cells, n_genes):
    config = {**small_config, "n_cells": n_cells, "n_genes": n_genes}
    dataset = SingleCellDataset(root=tmp_path, generation_config=config)
    with pytest.raises(ValueError, match="cells|genes"):
        dataset.plot(color_by="concept_0")


def test_supplied_exog_noise_controls_poisson_counts(tmp_path, small_config):
    config = deepcopy(small_config)
    config.update({"baseline_lower": 3, "baseline_upper": 3,
                   "std_dev_libsize_lower": 0, "std_dev_libsize_upper": 0, "std_dev_noise": 0})
    for name in ("std_dev_batch", "std_dev_celltype", "std_dev_tissue", "std_dev_concept"):
        config[name] = 0
    quantiles = np.linspace(0.1, 0.9, 60).reshape(10, 6)
    original = quantiles.copy()
    dataset = SingleCellDataset(root=tmp_path, generation_config=config, exog_noise=quantiles)
    expected = torch.from_numpy(poisson.ppf(quantiles, np.exp(np.log(3)))).float()
    torch.testing.assert_close(dataset.input_data, expected)
    np.testing.assert_array_equal(quantiles, original)
    restored = SingleCellDataset(root=tmp_path, generation_config=config, exog_noise=quantiles)
    torch.testing.assert_close(restored.input_data, dataset.input_data)
    changed = SingleCellDataset(root=tmp_path, generation_config=config, exog_noise=quantiles / 2)
    assert not torch.equal(changed.input_data, dataset.input_data)
    quantiles[:] = 0.5
    torch.testing.assert_close(dataset.input_data, expected)


def test_datamodule_forwards_exog_noise(tmp_path, small_config):
    quantiles = np.full((10, 6), 0.5)
    module = SingleCellDataModule(root=tmp_path, generation_config=small_config, exog_noise=quantiles)
    expected, _, _ = _generation_tensors(module.dataset)
    torch.testing.assert_close(module.dataset.input_data, expected)


def test_dependency_handles_one_concept_reproducibly(tmp_path, small_config):
    config = deepcopy(small_config)
    config.update({"n_concepts": 1, "use_concept_dependency": True})
    dataset = SingleCellDataset(root=tmp_path, generation_config=config)
    restored = SingleCellDataset(root=tmp_path, generation_config=config)
    torch.testing.assert_close(restored.input_data, dataset.input_data)
    assert dataset.input_data.isfinite().all()


@pytest.mark.parametrize("quantiles", [np.full((2, 2), 0.5), np.full((10, 6), np.nan),
                                     np.full((10, 6), 0), np.full((10, 6), 1)])
def test_invalid_exog_noise(tmp_path, small_config, quantiles):
    with pytest.raises(ValueError, match="exog_noise"):
        SingleCellDataset(root=tmp_path, generation_config=small_config, exog_noise=quantiles)


@pytest.mark.parametrize("quantiles", [np.full((2, 2), 0.5), np.full((10, 6), np.nan),
                                     np.full((10, 6), 0), np.full((10, 6), 1)])
def test_model_rejects_invalid_exog_noise(tmp_path, small_config, quantiles):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    with pytest.raises(ValueError, match="exog_noise"):
        cell_simulator(dataset.model_parameters, n_cells=len(dataset), exog_noise=quantiles)


@pytest.mark.parametrize("config, error, message", [
    ([], TypeError, "must be a mapping"),
    ({"unknown": 1}, ValueError, "Unknown generation parameter"),
    ({"batch": {"std": 0.15}}, ValueError, "Unknown generation parameter"),
    ({"std_dev_batch": {}}, TypeError, "std_dev_batch.*real number"),
    ({"baseline_lower": 0}, ValueError, "baseline_lower"),
    ({"std_dev_libsize_lower": 1}, ValueError, "std_dev_libsize_lower"),
    ({"std_dev_noise": -1}, ValueError, "standard deviations"),
    ({"beta_a": 0}, ValueError, "Beta parameters"),
    ({"cov_prior_range": -1}, ValueError, "cov_prior_range"),
    ({"std_dev_batch": "large"}, TypeError, "std_dev_batch.*real number"),
    ({"std_dev_batch": float("nan")}, ValueError, "std_dev_batch.*finite"),
    ({"zero_inflate": "false"}, TypeError, "zero_inflate.*boolean"),
])
def test_invalid_generation_config(tmp_path, config, error, message):
    with pytest.raises(error, match=message):
        SingleCellDataset(root=tmp_path, generation_config=config)


@pytest.mark.parametrize("name", ["n_cells", "n_genes", "n_batch", "n_celltype", "n_tissue", "n_concepts"])
@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_counts(tmp_path, name, n):
    with pytest.raises(ValueError, match=rf"{name}.*positive integer"):
        SingleCellDataset(root=tmp_path, generation_config={name: n})


def test_compute_parameters_defaults_to_seed_42(small_config):
    config = _resolve_generation_config(small_config)
    actual = single_cell._compute_simulator_parameters(config)
    expected = single_cell._compute_simulator_parameters(config, np.random.default_rng(42))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["generated", "intervened"])
def test_variants_save_separate_files_beside_source(tmp_path, small_config, monkeypatch, kind):
    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    before = {path: Path(path).read_bytes() for path in source.processed_paths}
    save = torch.save

    def save_variant(value, path, *args, **kwargs):
        assert path not in before
        return save(value, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", save_variant)
    dataset = (source.generate_new_cells(seed=7) if kind == "generated"
               else source.generate_interventions({"concept_0": 0}))
    assert Path(dataset.root).parent == Path(source.root).parent
    assert set(dataset.processed_paths).isdisjoint(source.processed_paths)
    assert all(Path(path).is_file() for path in dataset.processed_paths)
    assert {str(path) for path in tmp_path.iterdir()} == {source.root, dataset.root}
    torch.testing.assert_close(dataset.load_raw()[0], dataset.input_data, rtol=0, atol=0)
    assert {path: Path(path).read_bytes() for path in before} == before


def test_generating_from_subset_samples_new_cell_identifiers(tmp_path, small_config):
    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    source._subset_rows([7, 2, 9])
    fresh = source.generate_new_cells(seed=7)
    assert len(fresh) == 3
    torch.testing.assert_close(_state(fresh)["cell_background"]["cell_indices"], torch.arange(3))
    assert not torch.equal(fresh.cell_background["noise"], source.cell_background["noise"])


def test_repeated_generation_reuses_cache_without_evaluating_model(tmp_path, small_config, monkeypatch):
    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    parameters = {"baseline_expr": source.model_parameters["baseline_expr"] + 1}
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(1)
        return cell_simulator(*args, **kwargs)

    monkeypatch.setattr(single_cell, "cell_simulator", evaluate)
    first = source.generate_new_cells(model_parameters=parameters, seed=7)
    second = source.generate_new_cells(model_parameters=parameters, seed=7)
    assert len(calls) == 1
    assert first.processed_paths == second.processed_paths
    torch.testing.assert_close(first.input_data, second.input_data, rtol=0, atol=0)
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("arguments", [
    {}, {"n_cells": None}, {"model_parameters": None}, {"seed": None},
    {"n_cells": None, "model_parameters": None, "seed": None},
])
def test_generate_new_cells_rejects_only_all_none_arguments(tmp_path, small_config, monkeypatch, arguments):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)

    def unexpected_model(*args, **kwargs):
        raise AssertionError("Reject missing generation arguments before sampling.")

    monkeypatch.setattr(single_cell, "cell_simulator", unexpected_model)
    with pytest.raises(ValueError, match="at least one"):
        dataset.generate_new_cells(**arguments)


@pytest.mark.parametrize("arguments", [
    {"model_parameters": {}}, {"n_cells": 10}, {"seed": 42},
    {"n_cells": 10, "seed": 42}, {"same_parameters": True},
    {"same_parameters": True, "n_cells": 10, "seed": 42},
])
def test_generate_new_cells_accepts_unchanged_arguments(tmp_path, small_config, arguments):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    original = deepcopy(_state(dataset))
    arguments = arguments.copy()
    if arguments.pop("same_parameters", False):
        arguments["model_parameters"] = dataset.model_parameters

    generated = dataset.generate_new_cells(**arguments)
    assert generated is not dataset
    assert len(generated) == len(dataset)
    assert generated.seed == dataset.seed
    torch.testing.assert_close(generated.model_parameters, dataset.model_parameters, rtol=0, atol=0)
    assert not torch.equal(generated.cell_background["noise"], dataset.cell_background["noise"])
    torch.testing.assert_close(_state(dataset), original, rtol=0, atol=0)


def test_generate_new_cells_accepts_mixed_equal_and_changed_parameters(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    parameters = deepcopy(dataset.model_parameters)
    parameters["baseline_expr"][0] += 0.01
    fresh = dataset.generate_new_cells(model_parameters=parameters, n_cells=len(dataset), seed=dataset.seed)
    torch.testing.assert_close(fresh.model_parameters, parameters, rtol=0, atol=0)
    assert len(fresh) == len(dataset)


@pytest.mark.parametrize("root", [
    None, "data/single_cell", Path("data/single_cell"),
    "~/data/single_cell", Path("~/data/single_cell"),
])
def test_root_normalization_keeps_saved_data_and_variants_consistent(
    tmp_path, small_config, monkeypatch, root,
):
    user_dir = tmp_path / "user"
    work_dir = tmp_path / "work"
    user_dir.mkdir()
    work_dir.mkdir()
    # Isolate expansion of ~ without writing to the real user's data directory.
    monkeypatch.setenv("HOME", str(user_dir))
    monkeypatch.chdir(work_dir)
    user_relative = root is not None and str(root).startswith("~/")
    base = (user_dir if user_relative else work_dir) / "data" / "single_cell"

    source = SingleCellDataset(root=root, generation_config=small_config)
    datasets = [source, source.generate_new_cells(n_cells=5, seed=7),
                source.generate_interventions({"concept_0": 1})]
    for dataset in datasets:
        assert Path(dataset.root).parent == base
        assert dataset.root == dataset.root_dir
        assert all(Path(path).is_file() for path in dataset.processed_paths)
        if user_relative:
            saved_root = "~/" + str(Path(dataset.root).relative_to(user_dir))
        else:
            saved_root = str(Path(dataset.root).relative_to(work_dir))
        if isinstance(root, Path):
            saved_root = Path(saved_root)
        restored = SingleCellDataset(root=saved_root)
        assert restored.root == dataset.root
        torch.testing.assert_close(_generation_tensors(restored),
                                   _generation_tensors(dataset), rtol=0, atol=0)

    # Existing instances and their variants must keep working after a chdir.
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)
    for dataset in datasets:
        torch.testing.assert_close(dataset.load_raw()[:2],
                                   (dataset.input_data, dataset._full_concepts), rtol=0, atol=0)
    assert source.generate_new_cells(n_cells=5, seed=7).root == datasets[1].root
    assert source.generate_interventions({"concept_0": 1}).root == datasets[2].root
    assert not (work_dir / "~").exists()
    assert not (other_dir / "data").exists()


def test_build_saves_components_and_constructor_reloads_them(tmp_path, small_config, monkeypatch):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    assert len(dataset.processed_paths) == 5
    assert all(Path(path).is_file() for path in dataset.processed_paths)
    inputs, concepts, annotations, graph = dataset.load_raw()
    torch.testing.assert_close(torch.load(dataset.processed_paths[0], weights_only=True), inputs)
    torch.testing.assert_close(torch.load(dataset.processed_paths[1], weights_only=True), concepts)
    assert torch.load(dataset.processed_paths[2], weights_only=False).labels == annotations.labels
    assert pd.read_hdf(dataset.processed_paths[3], "graph").empty
    assert graph is None
    metadata = _saved_state(dataset)
    assert len(torch.load(dataset.generation_state_path, weights_only=True)) == 3
    assert not hasattr(dataset, "generation_metadata")
    torch.testing.assert_close(metadata, _state(dataset), rtol=0, atol=0)

    def unexpected_build(*args, **kwargs):
        raise AssertionError("Reload existing processed files without regenerating.")

    monkeypatch.setattr(SingleCellDataset, "build", unexpected_build)
    restored = SingleCellDataset(root=tmp_path, generation_config=small_config,
                                 concept_subset=["concept_1"])
    torch.testing.assert_close(restored.input_data, dataset.input_data, rtol=0, atol=0)
    torch.testing.assert_close(_state(restored), metadata, rtol=0, atol=0)
    assert restored.concept_names == ["concept_1"]


@pytest.mark.parametrize("changed", ["generation_config", "seed", "exog_noise"])
def test_generation_respects_requests_when_root_contains_saved_data(tmp_path, small_config, changed):
    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    original_files = {path: Path(path).read_bytes() for path in source.processed_paths}
    arguments = {"generation_config": small_config, "seed": 42}
    if changed == "generation_config":
        arguments[changed] = {**small_config, "n_cells": 5}
    elif changed == "seed":
        arguments[changed] = 7
    else:
        arguments[changed] = np.full((10, 6), 0.5)
    generated = SingleCellDataset(root=source.root, **arguments)
    expected = SingleCellDataset(root=tmp_path / "reference", **arguments)
    assert Path(generated.root).parent == Path(source.root)
    assert generated.seed == arguments["seed"]
    assert generated.generation_config == expected.generation_config
    torch.testing.assert_close(_generation_tensors(generated), _generation_tensors(expected), rtol=0, atol=0)
    assert {path: Path(path).read_bytes() for path in original_files} == original_files


@pytest.mark.parametrize("arguments", [
    {"seed": 42}, {"seed": 7}, {"generation_config": {}},
    {"exog_noise": np.full((10, 6), 0.5)},
])
def test_generation_arguments_take_precedence_over_saved_data(
    tmp_path, small_config, monkeypatch, arguments,
):
    monkeypatch.setattr(single_cell, "_DEFAULT_GENERATION_CONFIG",
                        _resolve_generation_config(small_config))
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    before = {path: Path(path).read_bytes() for path in dataset.processed_paths}
    generated = SingleCellDataset(root=dataset.root, **arguments)
    expected = SingleCellDataset(root=tmp_path / "reference", **arguments)
    assert Path(generated.root).parent == Path(dataset.root)
    torch.testing.assert_close(_generation_tensors(generated),
                               _generation_tensors(expected), rtol=0, atol=0)
    assert {path: Path(path).read_bytes() for path in before} == before


def test_missing_root_generates_dataset_automatically(tmp_path, small_config):
    missing = tmp_path / "missing"
    dataset = SingleCellDataset(root=missing, generation_config=small_config)
    assert Path(dataset.root).parent == missing
    assert len(dataset) == small_config["n_cells"]
    assert all(Path(path).is_file() for path in dataset.processed_paths)


def test_datamodule_loads_saved_dataset_with_independent_split_seed(tmp_path, small_config, monkeypatch):
    source = SingleCellDataset(root=tmp_path, seed=7, generation_config=small_config)

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("Loading saved data must not run generation.")

    monkeypatch.setattr(single_cell, "cell_simulator", unexpected_generation)
    monkeypatch.setattr(single_cell, "_compute_simulator_parameters", unexpected_generation)
    module = SingleCellDataModule(root=source.root, seed=12,
                                 concept_subset=["concept_1"], normalize=False)
    assert module.dataset.seed == 7
    assert module.dataset.concept_names == ["concept_1"]
    torch.testing.assert_close(_generation_tensors(module.dataset), _generation_tensors(source), rtol=0, atol=0)
    module.setup()
    assert len(module.train_dataloader().dataset) == 7


@pytest.mark.parametrize("arguments", [
    {"generation_seed": 42}, {"generation_config": {}},
    {"exog_noise": np.full((10, 6), 0.5)},
])
def test_datamodule_generation_arguments_take_precedence_over_saved_data(
    tmp_path, small_config, monkeypatch, arguments,
):
    monkeypatch.setattr(single_cell, "_DEFAULT_GENERATION_CONFIG",
                        _resolve_generation_config(small_config))
    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    before = {path: Path(path).read_bytes() for path in source.processed_paths}
    module = SingleCellDataModule(root=source.root, seed=12, **arguments)
    reference = SingleCellDataModule(root=tmp_path / "reference", seed=12, **arguments)
    assert Path(module.dataset.root).parent == Path(source.root)
    torch.testing.assert_close(_generation_tensors(module.dataset),
                               _generation_tensors(reference.dataset), rtol=0, atol=0)
    assert {path: Path(path).read_bytes() for path in before} == before


def test_cache_does_not_load_legacy_context_concept_files(tmp_path, small_config):
    legacy_paths = [tmp_path / f"{name}_v3_seed_42.pt"
                    for name in ("inputs", "concepts", "annotations", "graph", "generation_state")]
    for path in legacy_paths:
        path.write_bytes(b"obsolete cache with context in concepts")
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    assert dataset.concepts.shape == (10, 3)
    assert {"celltype", "tissue", "batch"}.issubset(dataset.cell_background)
    assert set(map(str, legacy_paths)).isdisjoint(dataset.processed_paths)
    assert all(path.read_bytes() == b"obsolete cache with context in concepts" for path in legacy_paths)


@pytest.mark.parametrize("missing", range(5))
def test_incomplete_saved_dataset_is_reported_without_overwriting(tmp_path, small_config, missing):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    expected = dataset.input_data.clone()
    Path(dataset.processed_paths[missing]).unlink()
    remaining = {path: Path(path).read_bytes() for path in dataset.processed_paths if Path(path).exists()}
    with pytest.raises(RuntimeError, match="Incomplete saved dataset"):
        SingleCellDataset(root=dataset.root)
    assert {path: Path(path).read_bytes() for path in remaining} == remaining
    rebuilt = SingleCellDataset(root=tmp_path / "rebuilt", generation_config=small_config)
    torch.testing.assert_close(rebuilt.input_data, expected, rtol=0, atol=0)


def test_generation_uses_explicit_parameter_overrides(tmp_path, small_config):
    config = {**small_config, "std_dev_noise": 0}
    dataset = SingleCellDataset(root=tmp_path, generation_config=config)
    metadata = _state(dataset)
    parameters = dataset.model_parameters
    parameters["baseline_expr"] += 1
    generated = dataset.generate_new_cells(seed=7, model_parameters=parameters)
    assert generated.generation_config["std_dev_noise"] == 0
    assert torch.count_nonzero(generated.cell_background["noise"]) == 0
    torch.testing.assert_close(generated.model_parameters, parameters, rtol=0, atol=0)
    torch.testing.assert_close(_reconstruct_counts(_state(generated), _concept_frame(generated)),
                               generated.input_data, rtol=0, atol=0)
    torch.testing.assert_close(_state(dataset), metadata, rtol=0, atol=0)


def test_model_parameters_separate_cache_and_equivalent_requests_share_it(tmp_path, small_config, monkeypatch):
    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    baseline = source.model_parameters["baseline_expr"] + 1
    first = source.generate_new_cells(model_parameters={"baseline_expr": baseline})
    second = source.generate_new_cells(model_parameters={"baseline_expr": baseline + 1})
    assert set(first.processed_paths).isdisjoint(second.processed_paths)
    torch.testing.assert_close(first.model_parameters["baseline_expr"], baseline, rtol=0, atol=0)
    assert torch.all(second.input_data >= first.input_data)

    def unexpected_model(*args, **kwargs):
        raise AssertionError("Equivalent effective parameters must reuse their saved data.")

    monkeypatch.setattr(single_cell, "cell_simulator", unexpected_model)
    parameters = {**source.model_parameters, "baseline_expr": baseline}
    restored = source.generate_new_cells(model_parameters=dict(reversed(list(parameters.items()))))
    assert restored.processed_paths == first.processed_paths
    torch.testing.assert_close(restored.input_data, first.input_data, rtol=0, atol=0)


def test_intervention_values_and_background_separate_cache(tmp_path, small_config, monkeypatch):
    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    first = source.generate_interventions({"concept_0": 0})
    second = source.generate_interventions({"concept_0": 1})
    assert set(first.processed_paths).isdisjoint(second.processed_paths)
    new_cells = source.generate_new_cells(seed=7)
    third = new_cells.generate_interventions({"concept_0": 0})
    assert set(first.processed_paths).isdisjoint(third.processed_paths)
    torch.testing.assert_close(third.cell_background, new_cells.cell_background, rtol=0, atol=0)

    def unexpected_model(*args, **kwargs):
        raise AssertionError("Equivalent interventions must reuse their saved data.")

    monkeypatch.setattr(single_cell, "cell_simulator", unexpected_model)
    restored = new_cells.generate_interventions({"concept_0": 0})
    assert restored.processed_paths == third.processed_paths
    torch.testing.assert_close(restored.input_data, third.input_data, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["generated", "intervened"])
@pytest.mark.parametrize("missing", range(5))
def test_incomplete_variant_can_be_regenerated_from_source(tmp_path, small_config, kind, missing):
    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    before = {path: Path(path).read_bytes() for path in source.processed_paths}

    def generate():
        if kind == "generated":
            return source.generate_new_cells(seed=7)
        return source.generate_interventions({"concept_0": 0})

    variant = generate()
    expected = _generation_tensors(variant)
    variant._subset_rows([7, 2, 9])
    selected = _generation_tensors(variant)
    Path(variant.processed_paths[missing]).unlink()
    with pytest.raises(RuntimeError, match="Incomplete saved dataset"):
        variant.load_raw()
    torch.testing.assert_close(_generation_tensors(variant), selected, rtol=0, atol=0)
    restored = generate()
    assert restored.root == variant.root
    torch.testing.assert_close(_generation_tensors(restored), expected, rtol=0, atol=0)
    assert {path: Path(path).read_bytes() for path in before} == before


@pytest.mark.parametrize("kind", ["generated", "intervened"])
def test_saved_variant_loads_by_directory_without_generation(tmp_path, small_config, monkeypatch, kind):
    import shutil

    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    source._subset_rows([7, 2, 9])
    if kind == "generated":
        variant = source.generate_new_cells(n_cells=5, seed=7)
    else:
        variant = source.generate_interventions({"concept_0": 1})
    target = tmp_path / "copied_dataset"
    shutil.copytree(variant.root, target)

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("A saved dataset loads independently of the source and generation inputs.")

    monkeypatch.setattr(single_cell, "cell_simulator", unexpected_generation)
    monkeypatch.setattr(single_cell, "_compute_simulator_parameters", unexpected_generation)
    monkeypatch.setattr(SingleCellDataset, "build", unexpected_generation)
    restored = SingleCellDataset(root=target)
    assert restored.seed == variant.seed
    torch.testing.assert_close(_generation_tensors(restored), _generation_tensors(variant), rtol=0, atol=0)


def test_intervention_cache_reuses_saved_selected_cells(tmp_path, small_config, monkeypatch):
    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    source._subset_rows([7, 2, 9])
    first = source.generate_interventions({"concept_0": 0})

    def unexpected_model(*args, **kwargs):
        raise AssertionError("Saved intervention data must reload without evaluating the cell_simulator.")

    monkeypatch.setattr(single_cell, "cell_simulator", unexpected_model)
    restored = first.generate_interventions({})
    assert restored.processed_paths == first.processed_paths
    torch.testing.assert_close(restored.input_data, first.input_data, rtol=0, atol=0)
    torch.testing.assert_close(_state(restored)["cell_background"]["cell_indices"], torch.tensor([7, 2, 9]))


@pytest.mark.parametrize("name", ["parameters", "concepts", "cell_background"])
def test_constructor_accepts_settings_only(tmp_path, name):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        SingleCellDataset(root=tmp_path, **{name: {}})


@pytest.mark.parametrize("concepts", [torch.zeros(5, 3), torch.zeros(10, 6), torch.full((10, 3), float("nan"))])
def test_interventions_reject_incompatible_concept_matrices(tmp_path, small_config, concepts):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    with pytest.raises(ValueError, match="concepts"):
        dataset.generate_interventions(concepts)


def test_interventions_reject_duplicate_cell_identifiers(tmp_path, small_config):
    dataset = SingleCellDataset(root=tmp_path, generation_config=small_config)
    concepts = _concept_frame(dataset)
    concepts.index = ["cell_0"] * len(dataset)
    with pytest.raises(ValueError, match="unique|cells"):
        dataset.generate_interventions(concepts)


@pytest.mark.parametrize("kind", ["generated", "intervened", "constructor"])
def test_rebuild_is_independent_of_exposed_parameters(tmp_path, small_config, kind):
    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    parameters = source.model_parameters
    if kind == "generated":
        variant = source.generate_new_cells(seed=7)
    elif kind == "intervened":
        variant = source.generate_interventions({"concept_0": 1})
    else:
        variant = SingleCellDataset(root=source.root)
    expected = _generation_tensors(variant)
    parameters["baseline_expr"].add_(10)
    source.model_parameters["baseline_expr"].add_(10)
    _state(source)["model_parameters"]["baseline_expr"].add_(10)
    variant.build()
    torch.testing.assert_close(variant.load_raw()[:2], expected[:2], rtol=0, atol=0)
    torch.testing.assert_close(_saved_state(variant),
                               expected[2], rtol=0, atol=0)


def test_embedding_cache_is_separate_per_variant(tmp_path, small_config, monkeypatch):
    class Identity(torch.nn.Module):
        filename = "identity.pt"
        source = "local"

        def forward(self, values):
            return values

    source = SingleCellDataset(root=tmp_path, generation_config=small_config)
    parameters = {"baseline_expr": source.model_parameters["baseline_expr"] + 1}
    variant = source.generate_new_cells(seed=7, model_parameters=parameters)
    assert not torch.equal(source.input_data, variant.input_data)
    backbone = Identity()
    for dataset in (source, variant):
        expected = dataset.input_data.clone()
        dataset.precompute_embeddings(backbone)
        torch.testing.assert_close(dataset.input_data, expected, rtol=0, atol=0)
    assert len(list(tmp_path.glob("*/identity_n10.pt"))) == 2

    def unexpected_forward(*args, **kwargs):
        raise AssertionError("Reuse the variant's saved embeddings.")

    monkeypatch.setattr(backbone, "forward", unexpected_forward)
    restored = source.generate_new_cells(seed=7, model_parameters=parameters)
    restored.precompute_embeddings(backbone)
    torch.testing.assert_close(restored.input_data, variant.input_data, rtol=0, atol=0)
