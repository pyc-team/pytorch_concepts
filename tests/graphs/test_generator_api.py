"""Tests for generator api."""

import numpy as np
import pytest
import torch
from torch_concepts import ConceptGraph
from torch_concepts.graphs import GraphGeneratorLearnable
from torch_concepts.graphs import GraphGeneratorStatic
from torch_concepts.graphs import refine_llm
from types import SimpleNamespace
import torch_concepts
from torch_concepts.graphs.generation.generators.static import causallearn as static_sources
from ._helpers import Backend
from ._helpers import _materialize_graph
from torch_concepts.graphs.generation.generators.static import causallearn
from ._helpers import _native_fixture
from torch_concepts.graphs import GraphGenerator
from torch_concepts.graphs import dfs_remove_cycles
import torch_concepts.graphs as graph_module
from ._helpers import _learnable
from ._helpers import _seed_weights


def test_direct_generators_accept_concept_observations(monkeypatch):
    from types import SimpleNamespace

    values = torch.tensor([[0., 1.], [1., 0.]])
    observed = []

    def pc(data, alpha, indep_test):
        observed.append(data)
        return SimpleNamespace(G=SimpleNamespace(graph=np.zeros((2, 2))))

    monkeypatch.setattr(causallearn, "_import_causallearn", lambda name: pc)
    static = GraphGeneratorStatic("pc")
    graph = static(values, concept_names=["a", "b"])
    np.testing.assert_array_equal(observed[0], values.numpy())
    assert graph.data.shape == (2, 2)
    assert isinstance(graph, ConceptGraph)
    assert static.graph is graph

    learned = _learnable("dagma_cgm", concept_names=["a", "b"], require_dag=False)
    for training in (True, False):
        learned.train(training)
        adjacency = learned(values)
        torch.testing.assert_close(adjacency.data, learned().data)
        assert adjacency.data.requires_grad is training
        assert (learned.graph is None) is training


@pytest.mark.parametrize("learnable", [False, True])
def test_direct_generators_share_names_and_descriptions(learnable):
    values = torch.zeros(2, 2)
    names = ["a", "b"]
    descriptions = {"a": "first concept", "b": "second concept"}
    backend = Backend("A->B")
    if learnable:
        generator = _learnable(
            "dagma_cgm", concept_names=names,
            initialization=_seed_weights(torch.tensor([[0., .5], [.5, 0.]])),
            refinement=refine_llm(backend),
        ).eval()
    else:
        generator = GraphGeneratorStatic("test-model", source="LLM", llm_backend=backend)
    graph = generator(values, names, descriptions)
    assert graph.node_names == names
    assert graph.has_edge("a", "b")
    assert "first concept" in backend._calls[0][0]
    assert "second concept" in backend._calls[0][0]
    assert descriptions == {"a": "first concept", "b": "second concept"}


@pytest.mark.parametrize("learnable", [False, True])
def test_common_input_validation(learnable):
    generator = GraphGeneratorLearnable("dagma_cgm", n_concepts=2) if learnable else GraphGeneratorStatic(
        "test-model", source="LLM", llm_backend=Backend(),
    )
    with pytest.raises(ValueError, match="unique strings"):
        generator(torch.zeros(2, 2), ["a", "a"], {})
    with pytest.raises(ValueError, match="dictionary of strings"):
        generator(torch.zeros(2, 2), ["a", "b"], {"a": 5})


@pytest.mark.parametrize("cache", [False, True])
def test_static_context_is_prepared_once_per_operation(datamodule, tmp_path, monkeypatch, cache):
    generator = GraphGeneratorStatic("test-model", source="LLM", llm_backend=Backend())
    calls = []
    prepare_context = generator._prepare_context

    def record_context(concept_names, concept_descriptions=None):
        calls.append((list(concept_names), concept_descriptions))
        prepare_context(concept_names, concept_descriptions)

    monkeypatch.setattr(generator, "_prepare_context", record_context)
    for operation, force in enumerate((False, False, True), start=1):
        datamodule.precompute_graph(generator, cache=cache, cache_dir=tmp_path, force=force)
        assert len(calls) == operation
        assert calls[-1][0] == list(datamodule.dataset.concept_names)


@pytest.mark.parametrize("pair", [(0, 1), (1, 0)])
@pytest.mark.parametrize("no_out_task", [False, True])
def test_task_mask_takes_precedence_over_orientation_pairs(pair, no_out_task):
    generator = _learnable(
        "dagma_cgm", concept_names=["a", "task"], n_tasks=1,
        no_out_task=no_out_task, edges_to_check=[pair], require_dag=False,
        initialization=_seed_weights(torch.zeros(2, 2)),
    )
    adjacency = generator().data
    assert adjacency[0, 1] == .5
    assert adjacency[1, 0] == (0. if no_out_task else .5)
    adjacency.sum().backward()
    assert generator.edge_matrix.grad is not None
    if no_out_task:
        assert generator.fc1.weight.grad[1].count_nonzero() == 0


def test_base_generator_is_abstract_and_registries_are_separate():
    with pytest.raises(TypeError):
        GraphGenerator(name="x", source="x")
    assert torch_concepts.graphs is graph_module
    assert "graphs" in torch_concepts.__all__
    assert "GroundTruth" not in GraphGeneratorStatic._source_loaders
    assert "DAGMA_CGM" in GraphGeneratorLearnable._source_loaders
    assert "GroundTruth" not in GraphGeneratorLearnable._source_loaders
    assert "DAGMA_CGM" not in GraphGeneratorStatic._source_loaders


@pytest.mark.parametrize("cls", [GraphGeneratorStatic, GraphGeneratorLearnable])
def test_unknown_source_reports_registered_sources(cls):
    with pytest.raises(ValueError, match="Unknown source.*registered sources"):
        cls(name="missing", source="Missing")


def test_source_resolution_reports_missing_and_ambiguous_names():
    with pytest.raises(ValueError, match="Cannot infer a source"):
        GraphGeneratorStatic(name="unregistered")

    @GraphGeneratorStatic.register_source("AmbiguousA", names=["ambiguous_test"])
    def load_a(generator, name):
        return graph_module.GraphGeneratorStaticSpec(
            compute=lambda _generator, values, names, descriptions: torch.zeros(len(names), len(names))
        )

    @GraphGeneratorStatic.register_source("AmbiguousB", names=["ambiguous_test"])
    def load_b(generator, name):
        return graph_module.GraphGeneratorStaticSpec(
            compute=lambda _generator, values, names, descriptions: torch.zeros(len(names), len(names))
        )

    with pytest.raises(ValueError, match="provided by multiple sources"):
        GraphGeneratorStatic(name="ambiguous_test")


@pytest.mark.parametrize("name,value", [
    ("name", "pc"), ("source", "LLM"), ("threshold", 1.),
    ("n_concepts", 1), ("no_out_task", False), ("require_dag", False),
    ("refinement", dfs_remove_cycles), ("initialization", None),
])
def test_generator_configuration_is_read_only(name, value):
    generator = _learnable("dagma_cgm", concept_names=["a", "b"])
    assert generator.initialization is None
    assert generator.refinement is None
    with pytest.raises(AttributeError):
        setattr(generator, name, value)
    with pytest.raises(AttributeError):
        delattr(generator, name)
    assert not hasattr(generator, "update")
    assert not hasattr(generator, "method_parameters")


def test_static_configuration_is_read_only():
    static = GraphGeneratorStatic(
        "fake", source="LLM", llm_backend=lambda *_a, **_k: "none",
    )
    assert not hasattr(static, "concept_descriptions")
    with pytest.raises(AttributeError, match="new generator"):
        static.domain = "another domain"


def test_dag_validation_rejects_cycles(dataset):
    dataset.graph_native = ConceptGraph(
        torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        ),
        node_names=dataset.concept_names,
    )
    generator = _native_fixture(dataset)
    with pytest.raises(ValueError, match="not a directed acyclic graph"):
        _materialize_graph(generator, dataset)


def test_static_callback_validates_dataset_concept_names(dataset):
    @GraphGeneratorStatic.register_source(
        "TensorNamesSource", names=["tensor_names_test"]
    )
    def load_source(generator, name):
        generator.concept_names = ["other", "names", "here"]
        return graph_module.GraphGeneratorStaticSpec(
            compute=lambda _generator, values, names, descriptions: ConceptGraph(
                torch.zeros(3, 3), node_names=generator.concept_names
            )
        )

    generator = GraphGeneratorStatic(name="tensor_names_test")
    with pytest.raises(ValueError, match="must match"):
        _materialize_graph(generator, dataset)


@pytest.mark.parametrize("output", [torch.zeros(2, 2), [[0, 1], [0, 0]], None])
def test_static_callback_must_return_graph(dataset, output):
    @GraphGeneratorStatic.register_source("BadReturnSource", names=["bad_return_test"])
    def load_source(generator, name):
        return graph_module.GraphGeneratorStaticSpec(
            compute=lambda _generator, values, names, descriptions: output
        )

    generator = GraphGeneratorStatic(name="bad_return_test")
    with pytest.raises((TypeError, ValueError), match="Source compute|match|names"):
        _materialize_graph(generator, dataset)


@pytest.mark.parametrize("method", ["pc", "ges"])
def test_partially_directed_output_is_controlled_by_require_dag(dataset, monkeypatch, method):
    adjacency = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., 0.]])
    causal_graph = SimpleNamespace(graph=adjacency)

    def algorithm(*args, **kwargs):
        return SimpleNamespace(G=causal_graph) if method == "pc" else {"G": causal_graph}

    monkeypatch.setattr(static_sources, "_import_causallearn", lambda name: algorithm)
    raw = _materialize_graph(GraphGeneratorStatic(method, require_dag=False), dataset)
    assert raw.data[0, 1] != 0 and raw.data[1, 0] != 0
    with pytest.raises(ValueError, match="after refinement"):
        _materialize_graph(GraphGeneratorStatic(method, require_dag=True), dataset)
    final = _materialize_graph(GraphGeneratorStatic(method, require_dag=True, refinement=[dfs_remove_cycles]), dataset)
    assert final.is_dag()


def test_descriptions_are_source_specific(dataset):
    with pytest.raises(TypeError, match="concept_descriptions"):
        _native_fixture(dataset, concept_descriptions={"rain": "rain"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("require_dag", [False, True])
@pytest.mark.parametrize("learnable", [False, True])
def test_nonfinite_adjacency_is_rejected_before_topology(dataset, monkeypatch, value, require_dag, learnable):
    adjacency = dataset.graph_native.data
    adjacency[0, 1] = value
    dataset.graph_native = ConceptGraph(adjacency, node_names=dataset.concept_names)
    generator = (
        _learnable(
            "dagma_cgm", concept_names=dataset.concept_names, require_dag=require_dag,
            initialization=_seed_weights(adjacency),
        ) if learnable else _native_fixture(dataset, require_dag=require_dag)
    )
    monkeypatch.setattr(ConceptGraph, "is_dag", lambda graph: pytest.fail("Topology checked before finiteness"))
    with pytest.raises(ValueError, match="finite values"):
        _materialize_graph(generator, dataset)


def test_ground_truth_method_is_not_registered():
    with pytest.raises(ValueError, match="Cannot infer a source"):
        GraphGeneratorStatic("ground_truth")
    with pytest.raises(ValueError, match="Unknown source"):
        GraphGeneratorStatic("ground_truth", source="GroundTruth")


@pytest.mark.parametrize("values,names", [
    ([0., 1.], ["a", "b"]),
    (torch.ones(2), ["a", "b"]),
    (torch.ones(3, 2), "ab"),
    (torch.ones(3, 2), ["a"]),
    (None, None),
])
def test_static_direct_call_rejects_invalid_observations_and_names(values, names):
    generator = GraphGeneratorStatic("test", source="LLM", llm_backend=lambda *args, **kwargs: "none")
    with pytest.raises(ValueError):
        generator(values, names)


@pytest.mark.parametrize("start", [-1, 2, "unknown"])
def test_dfs_rejects_invalid_start_node(start):
    graph = ConceptGraph(torch.tensor([[0., 1.], [1., 0.]]), ["a", "b"])
    with pytest.raises(ValueError):
        dfs_remove_cycles(graph, start_node=start)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_static_direct_graph_follows_input_device():
    values = torch.zeros(4, 2, device="cuda")
    graph = GraphGeneratorStatic("test", source="LLM", llm_backend=lambda *args, **kwargs: "A->B")(
        values, ["b", "a"],
    )
    assert graph.data.device == values.device
    assert graph.has_edge("b", "a")
