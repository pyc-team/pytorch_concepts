"""Tests for learning."""

import numpy as np
import pytest
import torch
from torch_concepts import ConceptGraph
from torch_concepts.graphs import GraphGeneratorLearnable
from torch_concepts.graphs import refine_llm
from ._helpers import Backend
from ._helpers import _materialize_graph
from torch_concepts.graphs import GraphGenerator
from ._helpers import _learnable
from ._helpers import _seed_weights
from ._helpers import _materialize
from torch_concepts.graphs import GraphGeneratorLearnableSpec
from torch_concepts.graphs import initialize_from_entropy
from torch import nn
from torch_concepts.graphs import remove_weakest_cycles
from pathlib import Path
from functools import partial


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_dagma_rejects_invalid_n_concepts(count):
    with pytest.raises(ValueError, match="n_concepts"):
        _learnable("dagma_cgm", n_concepts=count)


def test_training_graph_keeps_dense_gradients_and_exports_sparse_edges():
    values = torch.tensor([[0., .5], [0., 0.]], requires_grad=True)
    graph = ConceptGraph(values, ["a", "b"])
    assert graph.data is values
    torch.testing.assert_close(graph.data, values)
    indices, weights = graph.dense_to_sparse()
    assert indices.shape == (2, 1) and weights.shape == (1,)
    graph.data.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    np.testing.assert_array_equal(graph.to_pandas().values, values.detach().numpy())


def test_training_does_not_prepare_or_mutate_refinement_context(monkeypatch):
    refinement = refine_llm(Backend(), concept_descriptions={"a": "original"})
    generator = GraphGeneratorLearnable("dagma_cgm", n_concepts=2, refinement=refinement)

    def unexpected(*args, **kwargs):
        raise AssertionError("Training must not prepare refinement context")

    monkeypatch.setattr(generator, "_prepare_context", unexpected)
    generator(torch.zeros(2, 2), ["a", "b"], {"a": "new"})
    assert generator.refinement[0] is refinement
    assert refinement.keywords["concept_descriptions"] == {"a": "original"}


def test_custom_learnable_source_accepts_tensor_parameters():
    class Custom(GraphGeneratorLearnable):
        _source_loaders = {}

    @Custom.register_source("Custom", names=["custom"])
    def load(generator, name, initial):
        generator.n_concepts = 2
        generator.weights = nn.Parameter(initial.clone())
        return GraphGeneratorLearnableSpec(forward=lambda generator, values, names, descriptions: generator.weights)

    initial = torch.tensor([[0., .5], [0., 0.]])
    generator = Custom("custom", initial=initial)
    generator().data.sum().backward()
    assert generator.weights.grad is not None
    generator.eval()
    graph = _materialize(generator)
    assert not graph.data.requires_grad
    with torch.no_grad():
        generator.weights.mul_(2)
    convert = GraphGenerator._cache_parameter
    assert convert(initial) == convert(initial.clone())
    assert convert(initial) != convert(initial.double())
    assert convert(initial) != convert(initial.flatten())


def test_new_learnable_generators_use_their_threshold_and_dag_requirement():
    weights = torch.tensor([[0., .5], [0., 0.]])
    def build(threshold, require_dag=True):
        generator = _learnable(
            "dagma_cgm", concept_names=["a", "b"], threshold=threshold,
            require_dag=require_dag,
            initialization=_seed_weights(weights),
        )
        generator.eval()
        return generator

    original = build(.02)
    initial = _materialize(original)
    assert _materialize(build(1)).data.count_nonzero() == 0
    weights = torch.tensor([[0., .5], [.5, 0.]])
    assert not _materialize(build(0, require_dag=False)).is_dag()
    with pytest.raises(ValueError, match="after refinement"):
        _materialize(build(0))


def test_training_graph_preserves_straight_through_gradients_at_zero_edges():
    generator = _learnable(
        "dagma_cgm", concept_names=["a", "b"], threshold=.4,
        initialization=_seed_weights(torch.tensor([[0., .1], [.1, 0.]])),
    )
    raw = generator._spec.forward(generator, None, ["a", "b"], {})
    expected = torch.autograd.grad(raw.sum(), generator.fc1.weight)[0]
    adjacency = generator().data
    assert isinstance(adjacency, torch.Tensor)
    assert adjacency.count_nonzero() == 0
    adjacency.sum().backward()
    torch.testing.assert_close(generator.fc1.weight.grad, expected)
    assert expected[0, 1] != 0


def test_learnable_refinement_and_validation_only_run_in_eval():
    calls = []

    def refinement(graph):
        calls.append(graph)
        return remove_weakest_cycles(graph)

    generator = _learnable(
        "dagma_cgm", concept_names=["a", "b"], refinement=[refinement],
        initialization=_seed_weights(torch.tensor([[0., .5], [.5, 0.]])),
    )
    training_graph = generator()
    assert training_graph.data.requires_grad
    assert not training_graph.is_dag()
    assert len(calls) == 0
    generator.eval()
    graph = generator()
    assert not graph.data.requires_grad
    assert graph.is_dag()
    assert len(calls) == 1
    torch.testing.assert_close(generator().data, graph.data)
    assert len(calls) == 1

    without_refinement = _learnable(
        "dagma_cgm", concept_names=["a", "b"],
        initialization=_seed_weights(torch.tensor([[0., .5], [.5, 0.]])),
    )
    assert not without_refinement().is_dag()
    without_refinement.eval()
    with pytest.raises(ValueError, match="after refinement"):
        _materialize(without_refinement)


def test_learnable_generator_trains_as_model_submodule_and_exports(datamodule, tmp_path):
    class Model(nn.Module):
        def __init__(self, generator):
            super().__init__()
            self.concept_predictor = nn.Linear(2, 2)
            self.task_predictor = nn.Linear(2, 1)
            self.graph_generator = generator

        def forward(self, inputs):
            concepts = self.concept_predictor(inputs)
            nodes = torch.cat((concepts, self.task_predictor(concepts)), dim=1)
            graph = self.graph_generator(nodes, ["a", "b", "target"])
            concepts = concepts + concepts @ graph.data[:2, :2]
            task = self.task_predictor(concepts) + concepts @ graph.data[:2, 2:]
            return concepts, task, graph

    torch.manual_seed(42)
    generator = _learnable(
        "dagma_cgm", concept_names=["a", "b", "target"], n_tasks=1, threshold=0,
        refinement=[remove_weakest_cycles],
        initialization=_seed_weights(torch.tensor([[0., .4, .6], [.2, 0., .3], [0., 0., 0.]])),
    )
    model = Model(generator)
    assert "graph_generator.fc1.weight" in model.state_dict()
    before = generator.fc1.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    values = datamodule.dataset.concepts.tensor[:2]
    concepts, task, _ = model(values)
    loss = (concepts - values).square().mean() + task.square().mean()
    loss.backward()
    assert generator.fc1.weight.grad is not None
    optimizer.step()
    assert not torch.equal(before, generator.fc1.weight)
    training_graph = generator()
    assert isinstance(training_graph, ConceptGraph)
    assert training_graph.data.requires_grad
    assert model.training
    assert model.eval() is model
    assert not model.training and not generator.training
    graph = _materialize(generator)
    assert graph.is_dag()
    assert not graph.data.requires_grad
    concepts, task, result = model(values)
    node_values = torch.cat((concepts, model.task_predictor(concepts)), dim=1)
    expected = (node_values + node_values @ result.data)[:, 2:]
    assert result.node_names == ["a", "b", "target"]
    assert torch.count_nonzero(result.data[-1]) == 0
    torch.testing.assert_close(task, expected)
    assert isinstance(generator(), ConceptGraph)
    assert not generator().data.requires_grad
    model.train()
    optimizer.zero_grad()
    concepts, task, _ = model(values)
    ((concepts - values).square().mean() + task.square().mean()).backward()
    optimizer.step()
    model.eval()
    updated = _materialize(generator)
    assert updated is not graph
    assert not torch.equal(updated.data, graph.data)
    model(values)
    assert not list(tmp_path.glob("*.pt"))
    generator.to(dtype=torch.float64)
    assert _materialize(generator).data.dtype == torch.float64
    with torch.no_grad():
        generator.edge_mask.zero_()
    assert torch.count_nonzero(_materialize(generator).data) == 0
    with pytest.raises(TypeError, match="unexpected keyword"):
        _learnable("dagma_cgm", concept_names=["a", "b"], threshhold=.1)


@pytest.mark.parametrize("pairs", [[(0, 0)], [(0, 2)], [(-1, 0)], [(0.5, 1)], [(True, 1)], [(0,)], [0]])
def test_dagma_rejects_invalid_orientation_pairs(pairs):
    with pytest.raises(ValueError, match="valid, distinct node indices"):
        _learnable("dagma_cgm", concept_names=["a", "b"], edges_to_check=pairs)


def test_learnable_dagma_initialization_forward_and_materialization():
    data = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
    )
    generator = _learnable(
        name="dagma_cgm",
        concept_names=["a", "b", "task"],
        task_names=["task"],
        require_dag=False,
        initialization=initialize_from_entropy(data),
    )
    assert generator.fc1.weight.abs().sum() > 0
    adjacency = generator().data
    assert adjacency.requires_grad
    assert torch.all(adjacency.diagonal() == 0)
    assert torch.all(adjacency[2] == 0)
    graph = _materialize_graph(generator)
    assert graph.node_names == ["a", "b", "task"]
    assert generator.fitted


def test_learnable_dagma_edges_to_check_and_task_defaults():
    generator = _learnable(
        name="dagma_cgm",
        concept_names=["a", "b", "task"],
        n_tasks=1,
        edges_to_check=[(0, 1)],
        threshold=0.4,
        require_dag=False,
    )
    with torch.no_grad():
        generator.fc1.weight.zero_()
        generator.edge_matrix[0, 1] = 0.0

    adjacency = generator().data

    assert generator.task_indices == [2]
    assert adjacency[0, 1] == 0.5
    assert adjacency[1, 0] == 0.5
    assert torch.all(adjacency[2] == 0)


def test_learnable_dagma_loader_validation():
    with pytest.raises(ValueError, match="n_tasks"):
        _learnable(
            name="dagma_cgm",
            concept_names=["a"],
            n_tasks=2,
        )
    with pytest.raises(ValueError, match="task_indices"):
        _learnable(
            name="dagma_cgm",
            concept_names=["a"],
            task_indices=[1],
        )
    with pytest.raises(TypeError, match="initialization"):
        _learnable(
            name="dagma_cgm",
            concept_names=["a"],
            initialization=object(),
        )


def test_learnable_construct_graph_uses_current_weights():
    generator = _learnable(
        name="dagma_cgm",
        concept_names=["a", "b"],
        require_dag=False,
    )
    first = _materialize_graph(generator)
    second = _materialize_graph(generator)
    assert second is not first
    torch.testing.assert_close(second.data, first.data)
    with torch.no_grad():
        generator.fc1.weight.add_(1.0)
    third = _materialize_graph(generator)
    assert third is not second
    assert not torch.equal(third.data, second.data)
    assert generator.graph is third
    assert generator.fitted


def test_entropy_noise_allows_zero_edges_to_learn():
    torch.manual_seed(7)
    values = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    generator = GraphGeneratorLearnable(
        "dagma_cgm", n_concepts=2,
        initialization=initialize_from_entropy(values, add_noise=True), threshold=1.,
    )
    before = generator.fc1.weight.detach().clone()
    graph = generator(torch.zeros(3, 2), ["b", "a"])
    assert not graph.data.count_nonzero()
    (-graph.data.sum()).backward()
    assert generator.fc1.weight.grad[0, 1] != 0
    assert generator.fc1.weight.grad[1, 0] != 0
    assert not generator.fc1.weight.grad.diagonal().count_nonzero()
    optimizer = torch.optim.SGD(generator.parameters(), lr=1000)
    optimizer.step()
    assert not torch.equal(generator.fc1.weight, before)
    assert generator().data.count_nonzero() == 2


@pytest.mark.parametrize("initialization", [None, torch.nn.init.zeros_])
def test_custom_source_keeps_zero_weights(initialization):
    class Custom(GraphGeneratorLearnable):
        _source_loaders = {}

    @Custom.register_source("plain", names=["plain"])
    def load(generator, name):
        generator.n_concepts = 2
        generator.weights = torch.nn.Parameter(torch.zeros(2, 2))
        return GraphGeneratorLearnableSpec(
            forward=lambda generator, values, names, descriptions: generator.weights,
            weights=generator.weights,
        )

    generator = Custom("plain", initialization=initialization)
    assert not generator.weights.count_nonzero()
    generator().data.sum().backward()
    torch.testing.assert_close(generator.weights.grad, torch.ones(2, 2))
    generator.eval()
    assert not generator().data.count_nonzero()


def test_dagma_keeps_explicit_zero_initialization():
    generator = GraphGeneratorLearnable(
        "dagma_cgm", n_concepts=4, n_tasks=1, edges_to_check=[(0, 1)],
        initialization=torch.nn.init.zeros_,
    )
    assert not generator.fc1.weight.count_nonzero()
    graph = generator()
    assert not graph.data[-1].count_nonzero()
    assert not graph.data.diagonal().count_nonzero()
    assert graph.data[0, 1] == graph.data[1, 0] == .5


def test_custom_source_common_options_work_in_train_and_eval(tmp_path):
    class Custom(GraphGeneratorLearnable):
        _source_loaders = {}

    @Custom.register_source("options", names=["options"])
    def load(generator, name, dtype=torch.float64, device=torch.device("cpu"), path=Path("weights")):
        generator.n_concepts = 2
        generator.weights = torch.nn.Parameter(torch.tensor([[0., .5], [0., 0.]], dtype=dtype, device=device))
        return GraphGeneratorLearnableSpec(
            forward=lambda generator, values, names, descriptions: generator.weights,
        )

    generator = Custom("options", path=tmp_path)
    training = generator()
    assert training.data.dtype == torch.float64 and training.data.requires_grad
    generator.eval()
    final = generator()
    assert not final.data.requires_grad
    torch.testing.assert_close(final.data, training.data)
    assert generator._build_cache_key()["method_parameters"] == {
        "dtype": "torch.float64", "device": "cpu", "path": str(tmp_path),
    }
    generator.to(dtype=torch.float32)
    assert generator().data.dtype == torch.float32


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf")])
def test_dagma_rejects_nonfinite_threshold(threshold):
    with pytest.raises(ValueError, match="threshold must be finite"):
        GraphGeneratorLearnable("dagma_cgm", n_concepts=2, threshold=threshold)


@pytest.mark.parametrize("pairs", [[(0, 1), (0, 1)], [(0, 1), (1, 0)]])
def test_dagma_rejects_repeated_orientation_pairs(pairs):
    with pytest.raises(ValueError, match="must not repeat"):
        GraphGeneratorLearnable("dagma_cgm", n_concepts=2, edges_to_check=pairs)


def test_learnable_rejects_node_count_different_from_batch():
    generator = GraphGeneratorLearnable("dagma_cgm", n_concepts=2)
    with pytest.raises(ValueError, match="number of generator nodes"):
        generator(torch.zeros(4, 3), ["a", "b", "c"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_dagma_device_move_and_refinement_on_cuda():
    generator = GraphGeneratorLearnable(
        "dagma_cgm", n_concepts=3, refinement=remove_weakest_cycles,
        initialization=partial(torch.nn.init.constant_, val=.5),
    ).eval()
    generator()
    generator.cuda()
    graph = generator()
    assert graph.data.device.type == "cuda" and graph.is_dag()
    generator.train()
    generator().data.sum().backward()
    assert generator.fc1.weight.grad.device.type == "cuda"
