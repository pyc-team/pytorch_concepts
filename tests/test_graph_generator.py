"""Tests for the current graph_generator API.

Run from the repository root with either:

    pytest -q tests/test_graph_generator.py
    python tests/test_graph_generator.py
"""

import warnings
from functools import partial
from types import SimpleNamespace

import matplotlib
import matplotlib.style as mpl_style
import numpy as np
import pytest
import torch

warnings.filterwarnings(
    "ignore",
    message=".*read_style_directory function was deprecated.*",
    category=matplotlib.MatplotlibDeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*update_nested_dict function was deprecated.*",
    category=matplotlib.MatplotlibDeprecationWarning,
)

if not hasattr(mpl_style, "core"):
    mpl_style.core = mpl_style

import torch_concepts
import torch_concepts.graph_generator as graph_module
from torch_concepts.concept_graph import ConceptGraph
from torch_concepts.graph_generator import (
    GraphGenerator,
    GraphGeneratorFixed,
    GraphGeneratorLearnable,
    compose_refinements,
    entropy_initialization,
    fixed_dagma_initialization,
    dfs_remove_cycles,
    random_initialization,
    refine_llm,
    remove_weakest_cycles,
)


@pytest.fixture
def dataset():
    names = ["rain", "wet", "traffic"]
    return SimpleNamespace(
        name="toy",
        concept_names=names,
        concepts=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
        ),
        graph_native=ConceptGraph(
            torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            node_names=names,
        ),
        label_descriptions={"rain": "whether it rains", "wet": "wet grass"},
        n_samples=3,
        seed=7,
    )


def test_base_generator_is_abstract_and_registries_are_separate():
    with pytest.raises(TypeError):
        GraphGenerator(name="x", source="x")
    assert torch_concepts.graph_generator is graph_module
    assert "graph_generator" in torch_concepts.__all__
    assert "GroundTruth" in GraphGeneratorFixed._sources
    assert "DAGMA_CGM" in GraphGeneratorLearnable._sources
    assert "GroundTruth" not in GraphGeneratorLearnable._sources
    assert "DAGMA_CGM" not in GraphGeneratorFixed._sources


@pytest.mark.parametrize("cls", [GraphGeneratorFixed, GraphGeneratorLearnable])
def test_unknown_source_reports_registered_sources(cls):
    with pytest.raises(ValueError, match="Unknown source.*registered sources"):
        cls(name="missing", source="Missing")


def test_source_resolution_reports_missing_and_ambiguous_names():
    with pytest.raises(ValueError, match="Cannot infer a source"):
        GraphGeneratorFixed(name="unregistered")

    @GraphGeneratorFixed.register_source("AmbiguousA", names=["ambiguous_test"])
    def load_a(generator, name):
        return graph_module.GraphGeneratorFixedSpec(
            compute=lambda _generator, dataset: dataset.graph_native
        )

    @GraphGeneratorFixed.register_source("AmbiguousB", names=["ambiguous_test"])
    def load_b(generator, name):
        return graph_module.GraphGeneratorFixedSpec(
            compute=lambda _generator, dataset: dataset.graph_native
        )

    with pytest.raises(ValueError, match="provided by multiple sources"):
        GraphGeneratorFixed(name="ambiguous_test")


def test_ground_truth_construct_graph_returns_native_and_caches(dataset):
    generator = GraphGeneratorFixed(name="ground_truth")
    graph = generator.construct_graph(dataset)
    assert graph is dataset.graph_native
    assert generator.graph is graph
    assert generator.fitted
    second = generator.construct_graph(dataset)
    assert second is graph


def test_ground_truth_requires_native_graph(dataset):
    dataset.graph_native = None
    generator = GraphGeneratorFixed(name="ground_truth")
    with pytest.raises(ValueError, match="graph_native"):
        generator.construct_graph(dataset)


def test_fixed_construct_graph_requires_dataset():
    generator = GraphGeneratorFixed(name="ground_truth")
    with pytest.raises(ValueError, match="requires a dataset"):
        generator.construct_graph()


def test_cache_key_uses_stable_dataset_metadata(dataset):
    generator = GraphGeneratorFixed(name="ground_truth")
    clone = SimpleNamespace(**dataset.__dict__)
    assert generator._cache_key(dataset) == generator._cache_key(clone)
    clone.seed = 8
    assert generator._cache_key(dataset) != generator._cache_key(clone)
    clone.seed = dataset.seed
    clone.concept_names = ["wet", "rain", "traffic"]
    assert generator._cache_key(dataset) != generator._cache_key(clone)


def test_refinement_cache_key_normalizes_llm_backend_metadata():
    class Backend:
        model = "fake-model"

        def __call__(self, prompt, **kwargs):
            return "none"

    refinement = refine_llm(
        llm_backend=Backend(),
        domain="weather",
        concept_descriptions={"rain": "whether it rains"},
        repeats=2,
    )

    cache_key = GraphGeneratorFixed._refinement_cache_key(refinement)
    assert cache_key["function"].endswith("refine_llm")
    assert cache_key["keywords"]["domain"] == "weather"
    assert cache_key["keywords"]["repeats"] == 2
    assert cache_key["keywords"]["llm_backend"]["model"] == "fake-model"


def test_callable_refinement_runs_on_materialized_copy(dataset):
    calls = []

    def refine(graph):
        calls.append(True)
        data = graph.data.clone()
        data[1, 2] = 1
        return ConceptGraph(data, node_names=list(graph.node_names))

    generator = GraphGeneratorFixed(name="ground_truth", refinement=refine)
    graph = generator.construct_graph(dataset)
    assert calls == [True]
    assert graph.data[1, 2] == 1
    assert dataset.graph_native.data[1, 2] == 0


def test_compose_refinements_and_validation(dataset):
    def a(graph):
        data = graph.data.clone()
        data[0, 2] = 1
        return ConceptGraph(data, node_names=list(graph.node_names))

    def b(graph):
        data = graph.data.clone()
        data[2, 1] = 1
        return ConceptGraph(data, node_names=list(graph.node_names))

    refined = compose_refinements(a, b)
    graph = GraphGeneratorFixed(
        name="ground_truth", refinement=refined, require_dag=False
    ).construct_graph(dataset)
    assert graph.data[0, 2] == 1
    assert graph.data[2, 1] == 1
    with pytest.raises(ValueError):
        compose_refinements()
    with pytest.raises(TypeError):
        compose_refinements(a, object())


def test_invalid_refinement_rejected(dataset):
    with pytest.raises(TypeError, match="must be callable"):
        GraphGeneratorFixed(name="ground_truth", refinement={"bad": True})
    generator = GraphGeneratorFixed(
        name="ground_truth", refinement=lambda graph: graph.data
    )
    with pytest.raises(TypeError, match="must return a ConceptGraph"):
        generator.construct_graph(dataset)


def test_dag_validation_rejects_cycles(dataset):
    dataset.graph_native = ConceptGraph(
        torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        ),
        node_names=dataset.concept_names,
    )
    generator = GraphGeneratorFixed(name="ground_truth")
    with pytest.raises(ValueError, match="not a directed acyclic graph"):
        generator.construct_graph(dataset)


def test_tensor_callback_validates_dataset_concept_names(dataset):
    @GraphGeneratorFixed.register_source(
        "TensorNamesSource", names=["tensor_names_test"]
    )
    def load_source(generator, name):
        generator.concept_names = ["other", "names", "here"]
        return graph_module.GraphGeneratorFixedSpec(
            compute=lambda _generator, _dataset: torch.zeros(3, 3)
        )

    generator = GraphGeneratorFixed(name="tensor_names_test")
    with pytest.raises(ValueError, match="must match"):
        generator.construct_graph(dataset)


def test_callback_must_return_graph_or_tensor(dataset):
    @GraphGeneratorFixed.register_source("BadReturnSource", names=["bad_return_test"])
    def load_source(generator, name):
        return graph_module.GraphGeneratorFixedSpec(
            compute=lambda _generator, _dataset: [[0, 1], [0, 0]]
        )

    generator = GraphGeneratorFixed(name="bad_return_test")
    with pytest.raises(TypeError, match="ConceptGraph or Tensor"):
        generator.construct_graph(dataset)


class _CausalLearnGraph:
    def __init__(self, adjacency):
        self.graph = np.asarray(adjacency)


def test_causallearn_pc_adapter(monkeypatch, dataset):
    calls = []

    def pc(data, alpha, indep_test):
        calls.append((data, alpha, indep_test))
        return SimpleNamespace(
            G=_CausalLearnGraph([[0, -1, 0], [1, 0, -1], [0, -1, 0]])
        )

    monkeypatch.setattr(graph_module, "_import_causallearn", lambda name: pc)
    graph = GraphGeneratorFixed(
        name="pc",
        source="Causallearn",
        alpha=0.2,
        indep_test="fisherz",
        require_dag=False,
    ).construct_graph(dataset)
    np.testing.assert_array_equal(calls[0][0], dataset.concepts.numpy())
    assert calls[0][1:] == (0.2, "fisherz")
    torch.testing.assert_close(
        graph.data,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]),
    )


def test_causallearn_ges_adapter(monkeypatch, dataset):
    calls = []

    def ges(data, score_func):
        calls.append((data, score_func))
        return {"G": _CausalLearnGraph([[0, -1, 0], [1, 0, 0], [0, 0, 0]])}

    monkeypatch.setattr(graph_module, "_import_causallearn", lambda name: ges)
    graph = GraphGeneratorFixed(
        name="ges", source="Causallearn", score_func="custom"
    ).construct_graph(dataset)
    assert calls[0][1] == "custom"
    torch.testing.assert_close(
        graph.data,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_causallearn_pc_accepts_tuple_result(monkeypatch, dataset):
    def pc(data, alpha, indep_test):
        return (_CausalLearnGraph([[0, -1, 0], [1, 0, 0], [0, 0, 0]]),)

    monkeypatch.setattr(graph_module, "_import_causallearn", lambda name: pc)
    graph = GraphGeneratorFixed(name="pc", source="Causallearn").construct_graph(
        dataset
    )
    torch.testing.assert_close(
        graph.data,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1])
def test_pc_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        GraphGeneratorFixed(name="pc", source="Causallearn", alpha=alpha)


def test_llm_generation_and_refinement(dataset):
    prompts = []
    responses = iter(["A->B\nA->B\nnone", "B->A", "none"])

    def backend(prompt, repeats=1, **kwargs):
        prompts.append((prompt, repeats))
        return next(responses)

    graph = GraphGeneratorFixed(
        name="fake",
        source="LLM",
        llm_backend=backend,
        repeats=3,
        domain="weather",
    ).construct_graph(dataset)
    torch.testing.assert_close(
        graph.data,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert len(prompts) == 3
    assert "weather" in prompts[0][0]

    calls = []

    def orient(prompt, **kwargs):
        calls.append(prompt)
        return "B->A"

    reciprocal = ConceptGraph(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]), node_names=["a", "b"]
    )
    refined = refine_llm(llm_backend=orient)(reciprocal)
    torch.testing.assert_close(refined.data, torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    assert len(calls) == 1


def test_llm_query_retries_invalid_responses_and_uses_temperature(dataset):
    calls = []

    class Backend:
        completion_kwargs = {"temperature": 0}

        def __call__(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return "invalid" if len(calls) == 1 else "A->B"

    with pytest.warns(UserWarning, match="retrying"):
        graph = GraphGeneratorFixed(
            name="fake",
            source="LLM",
            llm_backend=Backend(),
        ).construct_graph(dataset)

    assert "previous response was invalid" in calls[1][0]
    assert calls[0][1] == {"repeats": 1}
    assert calls[1][1]["temperature"] == 0.1
    assert graph.data[0, 1] == 1


def test_llm_query_warns_and_skips_pair_after_invalid_retries(dataset):
    def backend(prompt, **kwargs):
        return "still invalid"

    with pytest.warns(UserWarning) as warnings:
        graph = GraphGeneratorFixed(
            name="fake",
            source="LLM",
            llm_backend=backend,
        ).construct_graph(dataset)

    messages = [str(warning.message) for warning in warnings]
    assert any("retrying" in message for message in messages)
    assert any("Falling back to None" in message for message in messages)
    assert torch.count_nonzero(graph.data) == 0


def test_llm_loader_validates_options():
    with pytest.raises(NotImplementedError, match="RAG"):
        GraphGeneratorFixed(
            name="fake",
            source="LLM",
            llm_backend=lambda *_a, **_k: "none",
            documents=["doc"],
        )
    with pytest.raises(ValueError, match="n_retrieved"):
        GraphGeneratorFixed(
            name="fake",
            source="LLM",
            llm_backend=lambda *_a, **_k: "none",
            n_retrieved=0,
        )
    with pytest.raises(TypeError, match="llm_backend"):
        GraphGeneratorFixed(name="fake", source="LLM", llm_backend=object())
    with pytest.raises(TypeError, match="embedding_backend"):
        GraphGeneratorFixed(
            name="fake",
            source="LLM",
            llm_backend=lambda *_a, **_k: "none",
            embedding_backend=object(),
        )


@pytest.mark.parametrize("repeats", [0, -1, 1.2, True])
def test_llm_repeats_validation(repeats):
    with pytest.raises(ValueError, match="positive integer"):
        GraphGeneratorFixed(
            name="fake",
            source="LLM",
            llm_backend=lambda *_a, **_k: "none",
            repeats=repeats,
        )


def test_refine_llm_validates_backend_and_repeats():
    with pytest.raises(TypeError, match="llm_backend"):
        refine_llm(llm_backend=object())
    with pytest.raises(ValueError, match="positive integer"):
        refine_llm(llm_backend=lambda *_a, **_k: "none", repeats=True)


def test_refinement_context_syncs_dataset_descriptions(dataset):
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

    GraphGeneratorFixed(
        name="ground_truth", refinement=refinement, require_dag=False
    ).construct_graph(dataset)

    assert "rain - explicit rain" in calls[0]
    assert "wet - wet grass" in calls[0]


def test_partial_refinement_context_is_updated_from_dataset(dataset):
    calls = []

    def refine_with_descriptions(graph, concept_descriptions=None):
        calls.append(dict(concept_descriptions or {}))
        return graph

    generator = GraphGeneratorFixed(
        name="ground_truth",
        refinement=partial(refine_with_descriptions, concept_descriptions={}),
    )
    generator.construct_graph(dataset)

    assert calls == [
        {
            "rain": "whether it rains",
            "wet": "wet grass",
            "traffic": "",
        }
    ]


def test_learnable_dagma_initialization_forward_and_materialization():
    data = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
    )
    generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=["a", "b", "task"],
        task_names=["task"],
        require_dag=False,
        initialization=entropy_initialization(data),
    )
    assert generator.fc1.weight.abs().sum() > 0
    assert torch.all(generator.fc1.weight[2] == 0)
    adjacency = generator()
    assert adjacency.requires_grad
    assert torch.all(adjacency.diagonal() == 0)
    assert torch.all(adjacency[2] == 0)
    graph = generator.construct_graph()
    assert graph.node_names == ["a", "b", "task"]
    assert generator.fitted


def test_learnable_dagma_edges_to_check_and_task_defaults():
    generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=["a", "b", "task"],
        n_tasks=1,
        edges_to_check=[(0, 1)],
        threshold=0.4,
        require_dag=False,
        initialization=random_initialization,
    )
    with torch.no_grad():
        generator.fc1.weight.zero_()
        generator.edge_matrix[0, 1] = 0.0

    adjacency = generator()

    assert generator.task_names == ["task"]
    assert adjacency[0, 1] == 0.5
    assert adjacency[1, 0] == 0.5
    assert torch.all(adjacency[2] == 0)


def test_learnable_dagma_loader_validation():
    with pytest.raises(ValueError, match="n_tasks"):
        GraphGeneratorLearnable(
            name="dagma_cgm",
            concept_names=["a"],
            n_tasks=2,
        )
    with pytest.raises(ValueError, match="task_names"):
        GraphGeneratorLearnable(
            name="dagma_cgm",
            concept_names=["a"],
            task_names=["missing"],
        )
    with pytest.raises(TypeError, match="initialization"):
        GraphGeneratorLearnable(
            name="dagma_cgm",
            concept_names=["a"],
            initialization=object(),
        )


def test_learnable_construct_graph_tracks_parameter_versions():
    generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=["a", "b"],
        require_dag=False,
        initialization=random_initialization,
    )
    first = generator.construct_graph()
    second = generator.construct_graph()
    assert second is not first
    torch.testing.assert_close(second.data, first.data)
    with torch.no_grad():
        generator.fc1.weight.add_(1.0)
    third = generator.construct_graph()
    assert third is not second
    assert not torch.equal(third.data, second.data)
    assert generator.graph is third
    assert generator.fitted


def test_learnable_construct_graph_warns_when_dataset_context_changes(dataset):
    generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=dataset.concept_names,
        require_dag=False,
        initialization=random_initialization,
    )
    first = generator.construct_graph(dataset)
    other = SimpleNamespace(**dataset.__dict__)
    other.name = "other"

    with pytest.warns(UserWarning, match="changed"):
        second = generator.construct_graph(other)

    assert second is not first


def test_fixed_dagma_initialization_freezes_weights():
    adjacency = torch.tensor([[0.0, 0.5], [0.0, 0.0]])
    generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=["a", "b"],
        initialization=fixed_dagma_initialization(adjacency),
    )
    torch.testing.assert_close(generator.fc1.weight, adjacency)
    assert not generator.fc1.weight.requires_grad
    assert torch.all(generator.edge_matrix == 0)


def test_initializers_validate_input_shapes():
    generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=["a", "b"],
        initialization=random_initialization,
    )
    with pytest.raises(ValueError, match="2D tensor"):
        entropy_initialization(torch.tensor([0.0, 1.0]))(generator)
    with pytest.raises(ValueError, match="one column per graph node"):
        entropy_initialization(torch.zeros(3, 1))(generator)
    with pytest.raises(ValueError, match="must match"):
        fixed_dagma_initialization(torch.zeros(3, 3))(generator)


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
    assert "no cycles" in capsys.readouterr().out


def test_dfs_remove_cycles_breaks_cycle_from_named_start(capsys):
    graph = ConceptGraph(
        torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        node_names=["a", "b", "c"],
    )

    refined = dfs_remove_cycles(graph, start_node="c")

    assert refined.is_directed_acyclic()
    assert refined.data.dtype == torch.int
    assert "cycle has been broken" in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
