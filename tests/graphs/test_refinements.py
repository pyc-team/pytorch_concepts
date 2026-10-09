"""Tests for refinements."""

import pytest
import torch
from torch_concepts import ConceptGraph
from torch_concepts.graphs import GraphGeneratorStatic
from torch_concepts.graphs import refine_llm
import warnings
from ._helpers import Backend
from ._helpers import _materialize_graph
from torch_concepts.data.base.dataset import ConceptDataset
from ._helpers import _native_fixture
from torch_concepts.graphs import dfs_remove_cycles
from ._helpers import _learnable
from torch_concepts.graphs import remove_weakest_cycles
from functools import partial


@pytest.mark.parametrize("refinement", [dfs_remove_cycles, remove_weakest_cycles])
def test_cycle_refinement_preserves_dense_input_and_retained_gradients(refinement):
    values = torch.tensor([[0., .5], [.2, 0.]], requires_grad=True)
    graph = ConceptGraph(values, ["a", "b"])
    refined = refinement(graph)
    torch.testing.assert_close(graph.data, values)
    assert graph.data.count_nonzero() == 2
    assert refined.is_dag()
    refined.data.sum().backward()
    assert values.grad[0, 1] == 1
    assert values.grad[1, 0] == 0


def test_callable_refinement_instances_are_supported(datamodule, tmp_path):
    class Refinement:
        def __init__(self, weight):
            self.weight = weight

        def __call__(self, graph):
            return graph

    generator = GraphGeneratorStatic("test-model", source="LLM", 
                                     llm_backend=Backend(), refinement=Refinement(.5))
    datamodule.precompute_graph(generator, cache_dir=tmp_path)
    graph = datamodule.dataset.graph
    datamodule.precompute_graph(generator, cache_dir=tmp_path)
    assert generator._refinement_cache_key(partial(dfs_remove_cycles, start_node=0)) != generator._refinement_cache_key(partial(dfs_remove_cycles, start_node=1))


@pytest.mark.parametrize("refinement", [dfs_remove_cycles, remove_weakest_cycles])
def test_cycle_refinement_handles_disconnected_weighted_cycles_and_self_loops(refinement):
    adjacency = torch.tensor([[.9, .7, 0.], [.4, .8, 0.], [0., 0., .3]])
    graph = ConceptGraph(adjacency, node_names=["a", "b", "isolated"])
    result = refinement(graph)
    assert result.is_dag()
    assert result.data.dtype == adjacency.dtype
    assert torch.all((result.data == 0) | (result.data == adjacency))
    torch.testing.assert_close(graph.data, adjacency)
    acyclic = ConceptGraph(torch.tensor([[0., .7], [0., 0.]]))
    torch.testing.assert_close(refinement(acyclic).data, acyclic.data)


@pytest.mark.parametrize("start_node", [0, "a"])
def test_cycle_refinements_remove_multiple_cycles_sharing_start_node(start_node):
    adjacency = torch.tensor([[0., .8, .6], [.2, 0., 0.], [.1, 0., 0.]])
    graph = ConceptGraph(adjacency, node_names=["a", "b", "c"])
    dfs = dfs_remove_cycles(graph, start_node=start_node)
    weakest = remove_weakest_cycles(graph)
    torch.testing.assert_close(
        dfs.data, torch.tensor([[0., 0., 0.], [.2, 0., 0.], [.1, 0., 0.]])
    )
    torch.testing.assert_close(
        weakest.data, torch.tensor([[0., .8, .6], [0., 0., 0.], [0., 0., 0.]])
    )
    assert dfs.is_dag() and weakest.is_dag()
    torch.testing.assert_close(graph.data, adjacency)


@pytest.mark.parametrize("refinement", [dfs_remove_cycles, remove_weakest_cycles])
def test_cycle_refinements_preserve_edges_between_components(refinement):
    adjacency = torch.zeros(6, 6, dtype=torch.float64)
    adjacency[0, 1], adjacency[1, 0] = .8, .2
    adjacency[2, 3], adjacency[3, 2] = .6, -.1
    adjacency[4, 4] = .3
    # Bridges are much weaker than cycle edges, but do not belong to any cycle.
    adjacency[1, 2], adjacency[3, 4], adjacency[4, 5] = .001, -.002, .003
    graph = ConceptGraph(adjacency, node_names=list("abcdef"))
    if refinement is dfs_remove_cycles:
        result = refinement(graph, start_node="f")
    else:
        result = refinement(graph)
    assert result.is_dag()
    assert result.data.count_nonzero() == 5
    assert result.data[4, 4] == 0
    for edge in [(1, 2), (3, 4), (4, 5)]:
        assert result.data[edge] == adjacency[edge]
    assert result.data.dtype == adjacency.dtype
    assert result.node_names == graph.node_names
    assert torch.all((result.data == 0) | (result.data == adjacency))
    torch.testing.assert_close(graph.data, adjacency)
    torch.testing.assert_close(refinement(result).data, result.data)


@pytest.mark.parametrize("refinement", [dfs_remove_cycles, remove_weakest_cycles])
def test_cycle_refinements_do_not_confuse_converging_paths_with_cycles(refinement):
    adjacency = torch.tensor([
        [0., .8, .6, 0.], [0., 0., 0., .2],
        [0., 0., 0., .1], [0., 0., 0., 0.],
    ])
    graph = ConceptGraph(adjacency)
    torch.testing.assert_close(refinement(graph).data, adjacency)


def test_callable_refinement_runs_on_materialized_copy(dataset):
    calls = []

    def refine(graph):
        calls.append(True)
        data = graph.data
        data[1, 2] = 1
        return ConceptGraph(data, node_names=list(graph.node_names))

    generator = _native_fixture(dataset, refinement=refine)
    graph = _materialize_graph(generator, dataset)
    assert calls == [True]
    assert graph.data[1, 2] == 1
    assert dataset.graph_native.data[1, 2] == 0


@pytest.mark.parametrize("sequence", [list, tuple])
def test_refinement_sequence_and_validation(dataset, sequence):
    def a(graph):
        data = graph.data
        data[0, 2] = 1
        return ConceptGraph(data, node_names=list(graph.node_names))

    def b(graph):
        assert graph.data[0, 2] == 1
        data = graph.data
        data[2, 1] = 1
        return ConceptGraph(data, node_names=list(graph.node_names))

    refined = sequence([a, b])
    graph = _materialize_graph(_native_fixture(dataset, refinement=refined, require_dag=False), dataset)
    assert graph.data[0, 2] == 1
    assert graph.data[2, 1] == 1
    assert _materialize_graph(_native_fixture(dataset, refinement=[]), dataset) is not dataset.graph_native
    with pytest.raises(TypeError):
        _native_fixture(dataset, refinement=[a, object()])


def test_invalid_refinement_rejected(dataset):
    with pytest.raises(TypeError, match="must be callable"):
        _native_fixture(dataset, refinement={"bad": True})
    generator = _native_fixture(dataset, refinement=lambda graph: graph.data)
    with pytest.raises(TypeError, match="must return a ConceptGraph"):
        _materialize_graph(generator, dataset)


@pytest.mark.parametrize("wrap", [lambda step: step, lambda step: [step], lambda step: (step,)])
def test_refinement_context_syncs_dataset_descriptions(dataset, wrap):
    calls = []

    def backend(prompt, **kwargs):
        calls.append(prompt)
        return "A->B"

    refinement = refine_llm(
        llm_backend=backend,
        concept_descriptions={"rain": "explicit rain"},
    )
    dataset.graph_native = ConceptGraph(
        torch.ones(3, 3) - torch.eye(3), node_names=dataset.concept_names
    )

    _materialize_graph(_native_fixture(dataset, refinement=wrap(refinement), require_dag=False), dataset)

    assert "rain - whether it rains" in calls[0]
    assert "wet - wet grass" in calls[0]


def test_partial_refinement_context_is_updated_from_dataset(dataset):
    calls = []

    def refine_with_descriptions(graph, concept_descriptions=None):
        calls.append(dict(concept_descriptions or {}))
        return graph

    generator = _native_fixture(dataset, refinement=partial(refine_with_descriptions, concept_descriptions={}))
    _materialize_graph(generator, dataset)

    assert calls == [
        {
            "rain": "whether it rains",
            "wet": "wet grass",
            "traffic": "",
        }
    ]


@pytest.mark.parametrize("refinement", [dfs_remove_cycles, remove_weakest_cycles])
def test_cycle_refinements_warn_about_undirected_edges(refinement):
    graph = ConceptGraph(
        torch.tensor([[0., -1.], [-1., 0.]]), node_names=["a", "b"], edge_encoding="endpoints",
    )
    with pytest.warns(UserWarning, match="partially directed.*refine_llm") as caught:
        refined = refinement(graph)
    assert len(caught) == 1
    assert "two opposite directed edges" in str(caught[0].message)
    assert refined.is_dag()
    assert torch.count_nonzero(refined.data) == 1
    torch.testing.assert_close(graph.data, torch.tensor([[0., -1.], [-1., 0.]]))


@pytest.mark.parametrize("refinement", [dfs_remove_cycles, remove_weakest_cycles])
def test_cycle_refinements_do_not_warn_about_directed_reciprocal_edges(refinement):
    graph = ConceptGraph(
        torch.tensor([[0., 1.], [1., 0.]]), node_names=["a", "b"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        refined = refinement(graph)
    assert not caught
    assert refined.is_dag()


def test_remove_weakest_cycles_breaks_cycle():
    graph = ConceptGraph(
        torch.tensor(
            [[0.0, 0.2, 0.0], [0.0, 0.0, 0.3], [0.1, 0.0, 0.0]]
        ),
        node_names=["a", "b", "c"],
    )
    refined = remove_weakest_cycles(graph)
    assert refined.data[2, 0] == 0
    assert refined.data[0, 1] == 0.2
    assert refined.data[1, 2] == 0.3


def test_cycle_refinements_leave_acyclic_graphs_unchanged(capsys):
    graph = ConceptGraph(
        torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        node_names=["a", "b", "c"],
    )

    weakest = remove_weakest_cycles(graph)
    dfs = dfs_remove_cycles(graph)

    torch.testing.assert_close(weakest.data, graph.data)
    torch.testing.assert_close(dfs.data.to(graph.data.dtype), graph.data)
    assert dfs.data.dtype in (torch.int32, torch.int64)
    assert capsys.readouterr().out == ""


def test_dfs_remove_cycles_breaks_cycle_from_named_start(capsys):
    graph = ConceptGraph(
        torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        node_names=["a", "b", "c"],
    )

    refined = dfs_remove_cycles(graph, start_node="c")

    assert refined.is_dag()
    # Incoming traversal: c <- b <- a encounters c -> a first.
    torch.testing.assert_close(
        refined.data, torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    )
    assert refined.data.dtype == graph.data.dtype
    assert capsys.readouterr().out == ""


def test_dfs_remove_cycles_matches_original_when_original_terminates():
    def original_dfs(node, adjacency, visited, active):
        visited[node] = active[node] = True
        for parent in range(len(adjacency)):
            if adjacency[parent, node] != 1:
                continue
            if not visited[parent]:
                if original_dfs(parent, adjacency, visited, active):
                    return True
            elif active[parent]:
                adjacency[parent, node] = 0
                return True
        active[node] = False
        return False

    # All three-node binary graphs, including self-loops, for every start node.
    for mask in range(1 << 9):
        adjacency = torch.tensor([(mask >> bit) & 1 for bit in range(9)]).reshape(3, 3)
        for start in range(3):
            expected = adjacency.clone()
            reference_terminates = True
            while not ConceptGraph(expected).is_dag():
                if not original_dfs(start, expected, [False] * 3, [False] * 3):
                    reference_terminates = False
                    break  # The upstream while loop would repeat forever.
            result = dfs_remove_cycles(ConceptGraph(adjacency), start_node=start)
            assert result.is_dag()
            assert torch.all((result.data == 0) | (result.data == adjacency))
            if reference_terminates:
                torch.testing.assert_close(result.data, expected)


def test_dfs_remove_cycles_handles_cycle_unreachable_from_start():
    adjacency = torch.tensor([[1., 0., 0.], [0., 0., .4], [0., .7, 0.]])
    graph = ConceptGraph(adjacency, node_names=["isolated", "a", "task"])
    result = dfs_remove_cycles(graph, start_node="isolated")
    torch.testing.assert_close(
        result.data, torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., .7, 0.]])
    )
    assert result.is_dag()
    torch.testing.assert_close(graph.data, adjacency)


def test_sequence_rejects_intermediate_non_graph(dataset):
    calls = []
    generator = _native_fixture(dataset, refinement=[lambda graph: graph.data, lambda graph: calls.append(True)])
    with pytest.raises(TypeError, match="must return a ConceptGraph"):
        _materialize_graph(generator, dataset)
    assert calls == []


def test_sequence_validates_dag_after_all_steps(dataset):
    def introduce_cycle(graph):
        adjacency = graph.data
        adjacency[1, 0] = 1
        return ConceptGraph(adjacency, node_names=graph.node_names)

    graph = _materialize_graph(_native_fixture(dataset, refinement=[introduce_cycle, remove_weakest_cycles]), dataset)
    assert graph.is_dag()


def test_learnable_refinement_sequence():
    calls = []

    def record(graph):
        calls.append(graph.node_names)
        return graph

    generator = _learnable(
        "dagma_cgm", concept_names=["a", "b"],
        refinement=[record, remove_weakest_cycles],
    )
    assert _materialize_graph(generator).is_dag()
    assert calls == [["a", "b"]]


def test_descriptions_share_context(dataset):
    generation_prompts = []
    contexts = []

    def backend(prompt, **kwargs):
        generation_prompts.append(prompt)
        return "none"

    def record(graph, concept_descriptions):
        contexts.append(dict(concept_descriptions))
        return graph

    first = partial(record, concept_descriptions={"wet": "first wet"})
    second = partial(record, concept_descriptions={"rain": "generator rain"})
    generator = GraphGeneratorStatic(
        "fake", source="LLM", llm_backend=backend,
        refinement=[first, dfs_remove_cycles, second],
    )
    dataset.label_descriptions["rain"] = "generator rain"
    _materialize_graph(generator, dataset)
    assert "rain - generator rain" in generation_prompts[0]
    assert "wet - wet grass" in generation_prompts[0]
    assert contexts == [
        {"rain": "generator rain", "wet": "wet grass", "traffic": ""},
        {"rain": "generator rain", "wet": "wet grass", "traffic": ""},
    ]
    assert first.keywords["concept_descriptions"] == {"wet": "first wet"}
    dataset.label_descriptions["traffic"] = "updated traffic"
    _materialize_graph(generator, dataset)
    assert contexts[2]["traffic"] == "updated traffic"
    assert contexts[3]["traffic"] == "updated traffic"


@pytest.mark.parametrize("learnable", [False, True])
def test_refinement_preserves_concept_names(dataset, learnable):
    calls = []

    def rename(graph):
        return ConceptGraph(graph.data, node_names=list(reversed(graph.node_names)))

    def restore(graph):
        calls.append(graph)
        return rename(graph)

    generator = (
        _learnable("dagma_cgm", concept_names=dataset.concept_names, refinement=[rename, restore])
        if learnable else _native_fixture(dataset, refinement=[rename, restore])
    )
    with pytest.raises(ValueError, match="concept names and order"):
        _materialize_graph(generator, dataset)
    assert generator.graph is None
    assert not generator.fitted
    assert not calls


def test_source_node_order_is_checked_before_refinement(dataset):
    dataset.graph_native = ConceptGraph(
        dataset.graph_native.data, node_names=list(reversed(dataset.concept_names)),
    )
    calls = []
    generator = _native_fixture(dataset, refinement=lambda graph: calls.append(graph))
    with pytest.raises(ValueError, match="concept names and order"):
        _materialize_graph(generator, dataset)
    assert not calls


@pytest.mark.parametrize("description", ["different rain", ""])
@pytest.mark.parametrize("operation", ["_build_cache_key", "_construct_graph"])
def test_dataset_descriptions_replace_refinement_descriptions(dataset, description, operation):
    generator = GraphGeneratorStatic(
        "fake", source="LLM", llm_backend=lambda *args, **kwargs: "none",
        refinement=refine_llm(
            llm_backend=lambda *args, **kwargs: "none",
            concept_descriptions={"rain": description},
        ),
    )
    generator._prepare_context(dataset.concept_names, getattr(dataset, "label_descriptions", None))
    generator._build_cache_key(cache_metadata=ConceptDataset._graph_cache_metadata(dataset)) if operation == "_build_cache_key" else _materialize_graph(generator, dataset)
    assert generator._resolved_refinements[0].keywords["concept_descriptions"]["rain"] == dataset.label_descriptions["rain"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scc_removal_keeps_edges_outside_cycles_and_handles_self_loops(dtype):
    graph = ConceptGraph(torch.tensor([
        [.1, .8, 0.], [.2, 0., -.9], [0., 0., 0.],
    ], dtype=dtype), ["b", "a", "target"])
    result = remove_weakest_cycles(graph)
    assert result.is_dag()
    assert result.data.dtype == dtype
    assert result.has_edge("b", "a")
    assert result.has_edge("a", "target")  # Lower weight, but outside any cycle.
    assert not result.has_edge("a", "b")
    assert not result.has_edge("b", "b")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_cycle_removal_rejects_nonfinite_values_without_modifying_input(value):
    graph = ConceptGraph(torch.tensor([[0., value], [.5, 0.]]))
    original = graph.data.clone()
    with pytest.raises(ValueError, match="finite adjacency"):
        remove_weakest_cycles(graph)
    torch.testing.assert_close(graph.data, original, equal_nan=True)


def test_llm_provider_errors_propagate_and_invalid_refinement_preserves_graph():
    from torch_concepts.graphs import refine_llm

    graph = ConceptGraph(torch.tensor([[0., -1.], [-1., 0.]]), ["a", "b"], edge_encoding="endpoints")

    def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        GraphGeneratorStatic("test", source="LLM", llm_backend=fail)(concept_names=graph.node_names)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        refine_llm(fail)(graph)

    calls = []

    def invalid(*args, **kwargs):
        calls.append(args)
        return "invalid"

    with pytest.warns(UserWarning):
        result = refine_llm(invalid)(graph)
    assert len(calls) == 3
    torch.testing.assert_close(result.data, graph.data)
