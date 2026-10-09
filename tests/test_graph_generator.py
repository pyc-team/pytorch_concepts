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
import torch_concepts.graphs as graph_module
from torch_concepts.graphs.generation.generators.static import causallearn as static_sources
from torch_concepts.concept_graph import ConceptGraph
from torch_concepts.data.base.dataset import ConceptDataset
from torch_concepts.graphs import (
    GraphGenerator,
    GraphGeneratorStatic,
    GraphGeneratorLearnable,
    initialize_from_entropy,
    dfs_remove_cycles,
    refine_llm,
    remove_weakest_cycles,
)



def _seed_weights(adjacency):
    """Test fixture: start from a known topology."""
    @torch.no_grad()
    def initialize(weights):
        values = adjacency.data if isinstance(adjacency, ConceptGraph) else adjacency
        weights.copy_(values.to(weights))
    return initialize


def _learnable(name="dagma_cgm", *, concept_names=None, task_names=None, **kwargs):
    # Keep test labels separate from method construction.
    if concept_names is not None:
        kwargs["n_concepts"] = len(concept_names)
    if task_names is not None:
        kwargs["task_indices"] = [concept_names.index(n) for n in task_names]
    generator = GraphGeneratorLearnable(name, **kwargs)
    generator._test_names = concept_names
    return generator

def _materialize_graph(generator, dataset=None):
    values = dataset.concepts if dataset is not None else None
    names = list(dataset.concept_names) if dataset is not None else getattr(generator, "_test_names", None)
    if names is None:
        names = [str(i) for i in range(getattr(generator, "n_concepts", 0))]
    return generator._construct_graph(
        values, names, getattr(dataset, "label_descriptions", None),
    )


def _native_fixture(dataset, **kwargs):
    # Test-only source for shared lifecycle checks on a known graph.
    @GraphGeneratorStatic.register_source("NativeFixture", names=["native_fixture"])
    def load(generator, name):
        return graph_module.GraphGeneratorStaticSpec(
            compute=lambda generator, values, names, descriptions: dataset.graph_native.clone(),
        )
    return GraphGeneratorStatic("native_fixture", **kwargs)


def _prepared_cache_key(generator, dataset):
    generator._prepare_context(dataset.concept_names, getattr(dataset, "label_descriptions", None))
    return generator._build_cache_key(cache_metadata=ConceptDataset._graph_cache_metadata(dataset))


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


@pytest.mark.parametrize("pairs", [[(0, 0)], [(0, 2)], [(-1, 0)], [(0.5, 1)], [(True, 1)], [(0,)], [0]])
def test_dagma_rejects_invalid_orientation_pairs(pairs):
    with pytest.raises(ValueError, match="valid, distinct node indices"):
        _learnable("dagma_cgm", concept_names=["a", "b"], edges_to_check=pairs)


def test_cache_key_uses_stable_dataset_metadata(dataset):
    generator = _native_fixture(dataset)
    clone = SimpleNamespace(**dataset.__dict__)
    assert _prepared_cache_key(generator, dataset) == _prepared_cache_key(generator, clone)
    clone.seed = 8
    assert _prepared_cache_key(generator, dataset) != _prepared_cache_key(generator, clone)
    clone.seed = dataset.seed
    clone.concept_names = ["wet", "rain", "traffic"]
    assert _prepared_cache_key(generator, dataset) != _prepared_cache_key(generator, clone)


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

    cache_key = GraphGeneratorStatic._refinement_cache_key(refinement)
    assert cache_key["function"].endswith("refine_llm")
    assert cache_key["keywords"]["domain"] == "weather"
    assert cache_key["keywords"]["repeats"] == 2
    assert cache_key["keywords"]["llm_backend"]["model"] == "fake-model"


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

    monkeypatch.setattr(static_sources, "_import_causallearn", lambda name: pc)
    graph = _materialize_graph(GraphGeneratorStatic(name='pc', source='Causallearn', alpha=0.2, indep_test='fisherz', require_dag=False), dataset)
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

    monkeypatch.setattr(static_sources, "_import_causallearn", lambda name: ges)
    graph = _materialize_graph(GraphGeneratorStatic(name='ges', source='Causallearn', score_func='custom'), dataset)
    assert calls[0][1] == "custom"
    torch.testing.assert_close(
        graph.data,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_causallearn_pc_accepts_tuple_result(monkeypatch, dataset):
    def pc(data, alpha, indep_test):
        return (_CausalLearnGraph([[0, -1, 0], [1, 0, 0], [0, 0, 0]]),)

    monkeypatch.setattr(static_sources, "_import_causallearn", lambda name: pc)
    graph = _materialize_graph(GraphGeneratorStatic(name='pc', source='Causallearn'), dataset)
    torch.testing.assert_close(
        graph.data,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1])
def test_pc_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        GraphGeneratorStatic(name="pc", source="Causallearn", alpha=alpha)


def test_llm_generation_and_refinement(dataset):
    prompts = []
    responses = iter(["A->B\nA->B\nnone", "B->A", "none"])

    def backend(prompt, repeats=1, **kwargs):
        prompts.append((prompt, repeats))
        return next(responses)

    graph = _materialize_graph(GraphGeneratorStatic(name='fake', source='LLM', llm_backend=backend, repeats=3, domain='weather'), dataset)
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
        graph = _materialize_graph(GraphGeneratorStatic(name='fake', source='LLM', llm_backend=Backend()), dataset)

    assert "previous response was invalid" in calls[1][0]
    assert calls[0][1] == {"repeats": 1}
    assert calls[1][1]["temperature"] == 0.1
    assert graph.data[0, 1] == 1


def test_llm_query_warns_and_skips_pair_after_invalid_retries(dataset):
    def backend(prompt, **kwargs):
        return "still invalid"

    with pytest.warns(UserWarning) as warnings:
        graph = _materialize_graph(GraphGeneratorStatic(name='fake', source='LLM', llm_backend=backend), dataset)

    messages = [str(warning.message) for warning in warnings]
    assert any("retrying" in message for message in messages)
    assert any("Falling back to None" in message for message in messages)
    assert torch.count_nonzero(graph.data) == 0


def test_llm_loader_validates_options():
    with pytest.raises(TypeError, match="llm_backend"):
        GraphGeneratorStatic(name="fake", source="LLM", llm_backend=object())
    with pytest.raises(TypeError, match="unexpected keyword"):
        GraphGeneratorStatic(
            name="fake", source="LLM",
            llm_backend=lambda *_a, **_k: "none", use_rag=True,
        )


@pytest.mark.parametrize("repeats", [0, -1, 1.2, True])
def test_llm_repeats_validation(repeats):
    with pytest.raises(ValueError, match="positive integer"):
        GraphGeneratorStatic(
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


@pytest.mark.parametrize("method", ["ges", "pc"])
@pytest.mark.parametrize("refinement", [None, dfs_remove_cycles, remove_weakest_cycles])
def test_cache_tracks_all_dataset_descriptions(dataset, method, refinement):
    generator = GraphGeneratorStatic(method, refinement=refinement)
    before = _prepared_cache_key(generator, dataset)
    dataset.label_descriptions["rain"] = "changed rain description"
    assert _prepared_cache_key(generator, dataset) != before
    assert "method_concept_descriptions" not in before


def test_cache_tracks_descriptions_used_by_refinement(dataset):
    generator = GraphGeneratorStatic(
        "ges", refinement=refine_llm(llm_backend=lambda *_a, **_k: "A->B"),
    )
    before = _prepared_cache_key(generator, dataset)
    dataset.label_descriptions["rain"] = "changed rain description"
    after = _prepared_cache_key(generator, dataset)
    assert before != after
    assert "method_concept_descriptions" not in after
    assert after["refinement"]["refinements"][0]["keywords"]["concept_descriptions"]["rain"] == "changed rain description"


def test_cache_tracks_resolved_method_descriptions(dataset):
    generator = GraphGeneratorStatic(
        "fake", source="LLM", llm_backend=lambda *_a, **_k: "none",
    )
    before = _prepared_cache_key(generator, dataset)
    dataset.label_descriptions["rain"] = "changed rain description"
    after = _prepared_cache_key(generator, dataset)
    assert before != after
    assert "method_concept_descriptions" not in after
    assert after["concept_descriptions"]["rain"] == "changed rain description"
    assert "concept_descriptions" not in generator._method_parameters
    assert not hasattr(generator, "concept_descriptions")


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


@pytest.mark.parametrize("refinement", [dfs_remove_cycles, remove_weakest_cycles])
def test_cycle_refinements_warn_about_undirected_edges(refinement):
    graph = ConceptGraph(
        torch.tensor([[0., -1.], [-1., 0.]]), node_names=["a", "b"],
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


def test_is_dag_matches_networkx_without_modifying_graph():
    import networkx as nx
    from torch_concepts.graphs.utils import contains_cycle

    # Includes DAGs with converging paths, disconnected cycles and self-loops.
    for mask in range(1 << 9):
        adjacency = torch.tensor([(mask >> bit) & 1 for bit in range(9)]).reshape(3, 3)
        for weights in (adjacency, adjacency.float() * -.4):
            before = weights.clone()
            network = nx.from_numpy_array(weights.numpy(), create_using=nx.DiGraph)
            assert ConceptGraph(weights).is_dag() == nx.is_directed_acyclic_graph(network)
            assert contains_cycle(weights) == (not nx.is_directed_acyclic_graph(network))
            torch.testing.assert_close(weights, before)
    assert ConceptGraph(torch.empty(0, 0)).is_dag()
    assert not contains_cycle(torch.empty(0, 0))


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))


@pytest.mark.parametrize("sequence", [list, tuple])
def test_sequence_context_and_cache(dataset, sequence):
    calls = []

    def with_context(graph, concept_descriptions):
        calls.append(concept_descriptions)
        return graph

    steps = sequence([partial(with_context, concept_descriptions={}), dfs_remove_cycles])
    generator = _native_fixture(dataset, refinement=steps)
    _materialize_graph(generator, dataset)
    assert calls[0]["rain"] == dataset.label_descriptions["rain"]
    key = generator._refinement_cache_key(generator._spec.refinement)
    assert len(key["refinements"]) == 2
    assert key != generator._refinement_cache_key(tuple(reversed(generator._spec.refinement)))


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


def test_descriptions_are_source_specific():
    with pytest.raises(TypeError, match="concept_descriptions"):
        _native_fixture(dataset, concept_descriptions={"rain": "rain"})


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


def test_ground_truth_method_is_not_registered():
    with pytest.raises(ValueError, match="Cannot infer a source"):
        GraphGeneratorStatic("ground_truth")
    with pytest.raises(ValueError, match="Unknown source"):
        GraphGeneratorStatic("ground_truth", source="GroundTruth")


def test_initializers_work_on_plain_weight_tensors():
    weights = torch.nn.Parameter(torch.zeros(2, 2))
    torch.manual_seed(0)
    assert weights.requires_grad
    initialize_from_entropy(torch.tensor([[0., 0.], [1., 1.], [1., 0.]]))(weights)
    assert torch.isfinite(weights).all()
    assert torch.count_nonzero(weights.diagonal()) == 0
    assert weights.requires_grad
