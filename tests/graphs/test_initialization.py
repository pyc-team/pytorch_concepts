"""Tests for initialization."""

import pytest
import torch
from torch_concepts.graphs import GraphGeneratorLearnable
from types import SimpleNamespace
from ._helpers import _learnable
from torch_concepts.graphs import GraphGeneratorLearnableSpec
from torch_concepts.graphs import initialize_from_entropy


def test_initializers_validate_input_shapes():
    generator = _learnable(
        name="dagma_cgm",
        concept_names=["a", "b"],
    )
    with pytest.raises(ValueError, match="2D tensor"):
        initialize_from_entropy(torch.tensor([0.0, 1.0]))(generator.fc1.weight)
    with pytest.raises(ValueError, match="one column per graph node"):
        initialize_from_entropy(torch.zeros(3, 1))(generator.fc1.weight)
    with pytest.raises(ValueError, match="at least one row"):
        initialize_from_entropy(torch.empty(0, 2))(generator.fc1.weight)


def test_initializers_work_on_plain_weight_tensors():
    weights = torch.nn.Parameter(torch.zeros(2, 2))
    torch.manual_seed(0)
    assert weights.requires_grad
    initialize_from_entropy(torch.tensor([[0., 0.], [1., 1.], [1., 0.]]))(weights)
    assert torch.isfinite(weights).all()
    assert torch.count_nonzero(weights.diagonal()) == 0
    assert weights.requires_grad


def test_entropy_leaves_task_positions_to_source():
    observations = torch.tensor([[0., 0.], [1., 1.]])
    weights = torch.nn.Parameter(torch.zeros(2, 2))
    initialize_from_entropy(observations)(weights)
    assert weights[1, 0] > 0  # The final node is not automatically a target.
    generator = GraphGeneratorLearnable(
        "dagma_cgm", n_concepts=2, task_indices=[0],
        initialization=initialize_from_entropy(observations),
    )
    graph = generator(observations, ["target", "concept"])
    assert graph.get_successors("target") == []
    assert graph.has_edge("concept", "target")


def test_custom_initializer_requires_source_weights():
    class Custom(GraphGeneratorLearnable):
        _source_loaders = {}

    @Custom.register_source("plain", names=["plain"])
    def load(generator, name):
        return GraphGeneratorLearnableSpec(forward=lambda *args: torch.zeros(2, 2))

    with pytest.raises(ValueError, match="provide weights"):
        Custom("plain", initialization=torch.nn.init.zeros_)


def test_entropy_handles_zero_mean_and_dataset_input():
    values = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    weights = torch.nn.Parameter(torch.ones(2, 2))
    initialize_from_entropy(SimpleNamespace(concepts=SimpleNamespace(tensor=values)))(weights)
    assert not weights.count_nonzero() and torch.isfinite(weights).all()


@pytest.mark.parametrize("apply_mask", [False, True])
@pytest.mark.parametrize("add_noise", [False, True])
def test_entropy_mask_and_noise_flags(apply_mask, add_noise):
    values = torch.tensor([[0., 0., 0.], [0., 1., 1.], [1., 0., 0.], [1., 1., 1.]])
    mask = ~torch.eye(3, dtype=torch.bool)
    mask[1, 2] = False
    mask[2, :] = False
    weights = torch.nn.Parameter(torch.zeros(3, 3, dtype=torch.float64))
    initialize_from_entropy(
        values, apply_mask=apply_mask, add_noise=add_noise, mask=mask,
    )(weights)
    assert weights.dtype == torch.float64 and weights.requires_grad
    assert not weights.diagonal().count_nonzero()
    assert (weights[0, 1] > 0).item() == add_noise
    assert (weights[1, 2] == 0).item() == apply_mask
    if apply_mask:
        assert not weights[~mask].count_nonzero()
    if add_noise:
        assert 0 < weights[0, 1] <= .99 + 1e-7


def test_entropy_mask_validation_preserves_weights():
    values = torch.zeros(2, 2)
    weights = torch.nn.Parameter(torch.ones(2, 2))
    for mask in (torch.ones(2, 2), torch.ones(3, 3, dtype=torch.bool), [[True]]):
        with pytest.raises(ValueError, match="boolean tensor"):
            initialize_from_entropy(values, mask=mask)(weights)
        torch.testing.assert_close(weights, torch.ones(2, 2))
    with pytest.raises(ValueError, match="requires a mask"):
        initialize_from_entropy(values, apply_mask=True)(weights)
    torch.testing.assert_close(weights, torch.ones(2, 2))


def test_entropy_mask_is_applied_before_normalization():
    values = torch.tensor([[0., 0., 0.], [1., 0., 1.], [2., 0., 0.], [3., 0., 1.]])
    mask = ~torch.eye(3, dtype=torch.bool)
    mask[1, 0] = mask[1, 2] = mask[2, 0] = False
    weights = torch.zeros(3, 3)
    initialize_from_entropy(values, apply_mask=True, mask=mask)(weights)
    # The retained raw weight is -1: masking first gives mean -1/9.
    expected = torch.tensor([[0., .99, 0.], [0., 0., 0.], [0., 0., 0.]])
    torch.testing.assert_close(weights, expected)
