"""Integration tests for graph data selection, caching and PyTorch training."""

import json
from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from torch_concepts import Annotations, ConceptGraph
from torch_concepts.data.base import ConceptDataModule
from torch_concepts.data.base.dataset import ConceptDataset
from torch_concepts.data.splitters.fixed import FixedIndicesSplitter
from torch_concepts.graphs import (
    GraphGenerator, GraphGeneratorLearnable, GraphGeneratorLearnableSpec,
    GraphGeneratorStatic, GraphGeneratorStaticSpec,
    dfs_remove_cycles, refine_llm, remove_weakest_cycles,
)
from torch_concepts.graphs.generation.generators.static import causallearn
from torch_concepts.llm_backends import LiteLLMBackend


def _native_fixture(dataset, **kwargs):
    # Test-only source for shared lifecycle checks on a known graph.
    @GraphGeneratorStatic.register_source("NativeFixture", names=["native_fixture"])
    def load(generator, name):
        return GraphGeneratorStaticSpec(
            compute=lambda generator, values, names, descriptions: dataset.graph_native.clone(),
        )
    return GraphGeneratorStatic("native_fixture", **kwargs)


def _prepared_cache_key(generator, dataset):
    generator._prepare_context(dataset.concept_names, getattr(dataset, "label_descriptions", None))
    return generator._build_cache_key(cache_metadata=ConceptDataset._graph_cache_metadata(dataset))




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

def _materialize(generator):
    with torch.no_grad():
        return generator._construct_graph(None, getattr(generator, "_test_names", None) or [str(i) for i in range(generator.n_concepts)], {})

@pytest.fixture
def datamodule():
    values = torch.tensor([[0., 1.], [1., 0.], [99., 99.], [88., 88.]])
    dataset = ConceptDataset(
        values.clone(), values.clone(),
        annotations=Annotations(labels=["a", "b"], cardinalities=[1, 1]),
    )
    dataset.label_descriptions = {"a": "description a", "b": "description b"}
    dm = ConceptDataModule(dataset, splitter=FixedIndicesSplitter(
        train_idxs=[0, 1], val_idxs=[2], test_idxs=[3],
    ))
    dm.setup('fit')
    return dm


class Backend:
    model = "test-model"

    def __init__(self, answer="none"):
        self.answer = answer
        self._calls = []

    def __call__(self, prompt, **kwargs):
        self._calls.append((prompt, kwargs))
        return self.answer


def test_static_direct_compute_and_precomputation(datamodule):
    calls = []

    @GraphGeneratorStatic.register_source("direct_api_test", names=["direct_api_test"])
    def loader(generator, name):
        def compute(generator, values, names, descriptions):
            calls.append("compute")
            return ConceptGraph(torch.zeros(2, 2), node_names=names)
        return GraphGeneratorStaticSpec(compute=compute)

    def refine(graph):
        calls.append("refine")
        return graph

    generator = GraphGeneratorStatic("direct_api_test", refinement=refine)
    values = torch.zeros(2, 2)
    assert isinstance(generator(values), ConceptGraph)
    generator(values)
    assert calls == ["compute", "refine", "compute", "refine"]
    assert generator.graph is not None
    datamodule.precompute_graph(generator, cache=False)
    assert calls == ["compute", "refine", "compute", "refine", "compute", "refine"]
    assert generator.graph is datamodule.dataset.graph
    assert not hasattr(generator, "to_graph")


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


def test_learnable_n_concepts_and_eval_cache():
    calls = []

    def refine(graph):
        calls.append(graph)
        return remove_weakest_cycles(graph)

    generator = _learnable(
        "dagma_cgm", n_concepts=2, refinement=refine,
        initialization=_seed_weights(torch.tensor([[0., .5], [.5, 0.]])),
    )
    values = torch.zeros(3, 2)
    names = ["a", "b"]
    assert list(generator.parameters())
    graph = generator(values, names, {})
    assert graph.node_names == names
    assert graph.data.requires_grad
    assert not graph.is_dag()
    assert not calls
    generator.eval()
    final = generator(values, names, {})
    assert final.is_dag() and not final.data.requires_grad
    torch.testing.assert_close(generator(values + 1, names, {}).data, final.data)
    assert len(calls) == 2
    with torch.no_grad():
        generator.fc1.weight.mul_(2)
    updated = generator(values, names, {})
    assert updated is not final and len(calls) == 3
    renamed = generator(values, ["x", "y"], {})
    assert renamed.node_names == ["x", "y"] and len(calls) == 4
    generator.train()
    assert generator(values, names, {}).data.requires_grad
    assert len(calls) == 4


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_dagma_rejects_invalid_n_concepts(count):
    with pytest.raises(ValueError, match="n_concepts"):
        _learnable("dagma_cgm", n_concepts=count)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("cache", [True, False])
def test_precompute_rejects_learnable_generators(datamodule, tmp_path, training, cache):
    generator = GraphGeneratorLearnable("dagma_cgm", n_concepts=2)
    generator.train(training)
    for target in (datamodule, datamodule.dataset):
        with pytest.raises(TypeError, match="only static"):
            target.precompute_graph(generator, cache=cache, cache_dir=tmp_path)
        assert generator.training == training
    assert not list(tmp_path.glob("*.pt"))


def test_eval_cache_depends_on_observations(datamodule):
    class Custom(GraphGeneratorLearnable):
        _source_loaders = {}

    @Custom.register_source("observations", names=["observations"])
    def load(generator, name):
        generator.n_concepts = 2
        generator.weight = nn.Parameter(torch.tensor([[0., .5], [0., 0.]]))
        return GraphGeneratorLearnableSpec(
            forward=lambda generator, values, names, descriptions: generator.weight * values.mean(),
        )

    generator = Custom("observations").eval()
    values = datamodule.dataset.concepts.tensor[datamodule.trainset.indices]
    direct = generator(values, ["a", "b"], {})
    torch.testing.assert_close(generator(values, ["a", "b"], {}).data, direct.data)
    changed = generator(values * 2, ["a", "b"], {})
    torch.testing.assert_close(changed.data, direct.data * 2)


def test_eval_cache_is_protected_from_result_mutation():
    generator = GraphGeneratorLearnable(
        "dagma_cgm", n_concepts=2,
        initialization=_seed_weights(torch.tensor([[0., .5], [0., 0.]])),
    ).eval()
    graph = generator(None, ["a", "b"], {})
    graph.edge_weight.zero_()
    graph.node_names[0] = "changed"
    generator.graph.edge_weight.zero_()
    result = generator(None, ["a", "b"], {})
    assert result.node_names == ["a", "b"]
    assert result.data[0, 1] == .5


@pytest.mark.parametrize("learnable", [False, True])
def test_common_input_validation(learnable):
    generator = GraphGeneratorLearnable("dagma_cgm", n_concepts=2) if learnable else GraphGeneratorStatic(
        "test-model", source="LLM", llm_backend=Backend(),
    )
    with pytest.raises(ValueError, match="unique strings"):
        generator(torch.zeros(2, 2), ["a", "a"], {})
    with pytest.raises(ValueError, match="dictionary of strings"):
        generator(torch.zeros(2, 2), ["a", "b"], {"a": 5})


def test_training_graph_keeps_dense_gradients_and_exports_sparse_edges():
    values = torch.tensor([[0., .5], [0., 0.]], requires_grad=True)
    graph = ConceptGraph(values, ["a", "b"])
    assert graph.data is values
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


def test_llm_direct_call_needs_only_names_and_descriptions():
    generator = GraphGeneratorStatic("test-model", source="LLM", llm_backend=Backend("A->B"))
    assert generator(None, ["a", "b"], {"a": "first"}).has_edge("a", "b")


def test_precomputed_graph_is_separate_from_native_graph():
    names = ["a", "b"]
    adjacency = pd.DataFrame([[0., 1.], [0., 0.]], index=names, columns=names)
    dataset = ConceptDataset(
        torch.zeros(4, 2), torch.zeros(4, 2),
        annotations=Annotations(labels=names, cardinalities=[1, 1]), graph=adjacency,
    )
    datamodule = ConceptDataModule(dataset)
    datamodule.setup('fit')
    native = dataset.graph_native
    assert dataset.graph is native
    assert native.data[0, 1] == 1

    datamodule.precompute_graph(GraphGeneratorStatic(
        "test-model", source="LLM", llm_backend=Backend(),
    ), cache=False)
    assert dataset.graph.data.count_nonzero() == 0
    assert dataset.graph_native is native
    assert native.data[0, 1] == 1

    datamodule.precompute_graph(_native_fixture(dataset), cache=False)
    assert dataset.graph is not native
    torch.testing.assert_close(dataset.graph.data, native.data)
    assert dataset.graph_native is native


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("through_datamodule", [False, True])
def test_native_graph_survives_method_and_refinement_updates(
    tmp_path, cache, through_datamodule,
):
    names = ["a", "b"]
    original = torch.tensor([[0., .7], [0., 0.]])
    dataset = ConceptDataset(
        torch.zeros(4, 2), torch.zeros(4, 2),
        annotations=Annotations(labels=names, cardinalities=[1, 1]),
        graph=pd.DataFrame(original.numpy(), index=names, columns=names),
    )
    dm = ConceptDataModule(dataset, splitter=FixedIndicesSplitter(
        train_idxs=[0, 1], val_idxs=[2], test_idxs=[3],
    ))
    dm.setup("fit")
    native = dataset.graph_native

    def assert_native_unchanged():
        assert dataset.graph_native is native
        torch.testing.assert_close(native.data, original)
        assert native.node_names == names

    def update(generator, expected, *, force=False):
        if through_datamodule:
            dm.precompute_graph(generator, cache=cache, cache_dir=tmp_path, force=force)
        else:
            dataset.precompute_graph(
                generator, cache=cache, cache_dir=tmp_path, force=force,
                training_indices=[0, 1],
            )
        assert_native_unchanged()
        assert dataset.graph is not native
        assert dataset.graph is generator.graph
        assert dataset.graph_generator is generator
        torch.testing.assert_close(dataset.graph.data, expected)

    def mutate_in_place(graph):
        graph.edge_index = torch.tensor([[0], [1]])
        graph.edge_weight = torch.tensor([.2])
        return graph

    update(_native_fixture(dataset), original)
    update(_native_fixture(dataset, refinement=mutate_in_place),
           torch.tensor([[0., .2], [0., 0.]]))
    update(_native_fixture(dataset), original)

    backend = Backend()

    def llm(refinement=None):
        return GraphGeneratorStatic(
            "test-model", source="LLM", llm_backend=backend, refinement=refinement,
        )

    update(llm(), torch.zeros(2, 2))
    calls = len(backend._calls)
    update(llm(), torch.zeros(2, 2))
    if cache:
        assert len(backend._calls) == calls
    else:
        assert len(backend._calls) > calls
    update(llm(mutate_in_place), torch.tensor([[0., .2], [0., 0.]]))
    update(llm(mutate_in_place), torch.tensor([[0., .2], [0., 0.]]), force=True)

    for refinement in (dfs_remove_cycles, remove_weakest_cycles):
        update(_native_fixture(dataset, refinement=refinement), original)

    def introduce_cycle(graph):
        graph.edge_index = torch.tensor([[0, 1], [1, 0]])
        graph.edge_weight = torch.tensor([.7, 1.])
        return graph

    previous = dataset.graph
    with pytest.raises(ValueError, match="after refinement"):
        update(_native_fixture(dataset, refinement=introduce_cycle), original)
    assert_native_unchanged()
    assert dataset.graph is previous
    update(_native_fixture(dataset), original)

    # Editing the active graph and its names must not affect the native graph.
    dataset.graph.edge_weight.zero_()
    dataset.graph.node_names[0] = "renamed"
    assert_native_unchanged()
    update(_native_fixture(dataset), original)


def test_precompute_uses_only_training_rows_without_mutating_dataset(datamodule, monkeypatch):
    seen = []

    def pc(data, alpha, indep_test):
        from types import SimpleNamespace
        seen.append(data.copy())
        return SimpleNamespace(G=SimpleNamespace(graph=np.zeros((2, 2))))

    monkeypatch.setattr(causallearn, "_import_causallearn", lambda name: pc)
    before = datamodule.dataset.concepts.tensor.clone()
    generator = GraphGeneratorStatic("pc")
    training_datasets = []
    construct_graph = generator._construct_graph

    def capture_training_dataset(values, names, descriptions=None, **kwargs):
        training_datasets.append(values)
        return construct_graph(values, names, descriptions, **kwargs)

    monkeypatch.setattr(generator, '_construct_graph', capture_training_dataset)
    result = datamodule.precompute_graph(generator, cache=False)
    assert result is None
    np.testing.assert_array_equal(seen[0], before[datamodule.trainset.indices].numpy())
    torch.testing.assert_close(datamodule.dataset.concepts.tensor, before)
    assert datamodule.dataset.graph is not None
    assert datamodule.dataset.graph is datamodule.dataset.graph_generator.graph
    torch.testing.assert_close(training_datasets[0], before[datamodule.trainset.indices])
    assert datamodule.splitter.train_idxs == [0, 1]
    datamodule.setup("fit")
    assert datamodule.trainset.indices == [0, 1]


def test_precompute_requires_setup(datamodule):
    dm = ConceptDataModule(datamodule.dataset)
    with pytest.raises(RuntimeError, match="setup"):
        dm.precompute_graph(GraphGeneratorStatic('pc'), cache=False)


def test_graph_cache_distinguishes_training_indices(datamodule, monkeypatch, tmp_path):
    seen = []

    def pc(data, alpha, indep_test):
        from types import SimpleNamespace
        seen.append(data.copy())
        return SimpleNamespace(G=SimpleNamespace(graph=np.zeros((2, 2))))

    monkeypatch.setattr(causallearn, '_import_causallearn', lambda name: pc)
    datamodule.precompute_graph(GraphGeneratorStatic('pc'), cache_dir=tmp_path)
    datamodule.trainset = [2, 3]
    datamodule.precompute_graph(GraphGeneratorStatic('pc'), cache_dir=tmp_path)
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[1], datamodule.dataset.concepts.tensor[[2, 3]].numpy())


def test_cache_tracks_method_parameters_and_can_be_forced_for_new_data(datamodule, tmp_path, monkeypatch, caplog):
    calls = []

    def pc(data, alpha, indep_test):
        from types import SimpleNamespace
        calls.append((alpha, indep_test))
        return SimpleNamespace(G=SimpleNamespace(graph=np.zeros((2, 2))))

    monkeypatch.setattr(causallearn, "_import_causallearn", lambda name: pc)
    for alpha in [.01, .5]:
        datamodule.precompute_graph(GraphGeneratorStatic("pc", alpha=alpha), cache_dir=tmp_path)
    with caplog.at_level('INFO', logger='torch_concepts.data.base.dataset'):
        datamodule.precompute_graph(GraphGeneratorStatic("pc", alpha=.5), cache_dir=tmp_path)
    assert "pre-existing graph cache" in caplog.text
    assert "method='pc'" in caplog.text
    assert "source='Causallearn'" in caplog.text
    assert "refinement=" in caplog.text
    assert "force=True" in caplog.text
    assert calls == [(.01, "chisq"), (.5, "chisq")]
    datamodule.dataset.concepts.tensor[0, 0] = .25
    datamodule.precompute_graph(GraphGeneratorStatic("pc", alpha=.5), cache_dir=tmp_path, force=True)
    assert len(calls) == 3
    datamodule.dataset.concepts.tensor[2, 0] = .5
    datamodule.precompute_graph(GraphGeneratorStatic("pc", alpha=.5), cache_dir=tmp_path, force=True)
    assert len(calls) == 4


def test_llm_cache_and_kwargs_survive_reload(datamodule, tmp_path):
    backend = Backend("A->B")

    def generator(domain="one", repeats=1, temperature=.5):
        return GraphGeneratorStatic(
            "test-model", source="LLM", llm_backend=backend,
            domain=domain, repeats=repeats, completion_kwargs={"temperature": temperature},
            api_key="private-test-key",
        )

    datamodule.precompute_graph(generator(), cache_dir=tmp_path)
    first = datamodule.dataset.graph
    assert backend._calls[0][1]["temperature"] == .5
    datamodule.precompute_graph(generator(), cache_dir=tmp_path)
    second = datamodule.dataset.graph
    assert len(backend._calls) == 1
    for options in [{"domain": "two"}, {"repeats": 2}, {"temperature": .9}]:
        datamodule.precompute_graph(generator(**options), cache_dir=tmp_path)
    datamodule.dataset.label_descriptions["a"] = "changed a"
    datamodule.precompute_graph(generator(), cache_dir=tmp_path)
    assert len(backend._calls) == 5
    for path in tmp_path.glob("*.pt"):
        payload = torch.load(path, weights_only=True)
        assert set(payload) == {"adjacency", "node_names"}
        torch.testing.assert_close(ConceptGraph.load(path).data, payload["adjacency"])
    assert list(second.to_networkx().edges) == [("a", "b")]


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



def test_cache_tracks_method_defaults_and_explicit_options(datamodule):
    dataset = datamodule.dataset
    assert _prepared_cache_key(GraphGeneratorStatic('pc'), dataset) == _prepared_cache_key(GraphGeneratorStatic('pc', alpha=0.05, indep_test='chisq'), dataset)
    assert _prepared_cache_key(GraphGeneratorStatic('ges'), dataset) != _prepared_cache_key(GraphGeneratorStatic('ges', score_func='local_score_BIC'), dataset)
    assert _prepared_cache_key(GraphGeneratorStatic('ges'), dataset) == _prepared_cache_key(GraphGeneratorStatic('ges', alpha=0.9, indep_test='ignored'), dataset)
    assert _prepared_cache_key(GraphGeneratorStatic('pc'), dataset) == _prepared_cache_key(GraphGeneratorStatic('pc', score_func='ignored'), dataset)

    class Custom(GraphGeneratorStatic):
        _source_loaders = {}

    @Custom.register_source("Custom", names=["custom"])
    def load(generator, name, scale=1):
        return GraphGeneratorStaticSpec(compute=lambda _, values, names, descriptions: ConceptGraph(
            torch.zeros(2, 2), node_names=names,
        ))

    assert Custom("custom")._method_parameters == {"scale": 1}
    assert Custom("custom", scale=1)._method_parameters == {"scale": 1}
    assert _prepared_cache_key(Custom('custom'), dataset) == _prepared_cache_key(Custom('custom', scale=1), dataset)
    assert _prepared_cache_key(Custom('custom'), dataset) != _prepared_cache_key(Custom('custom', scale=2), dataset)


def test_failed_precomputation_preserves_graph_and_generator(datamodule):
    first = GraphGeneratorStatic("first", source="LLM", llm_backend=Backend())
    datamodule.precompute_graph(first, cache=False)
    graph = datamodule.dataset.graph

    def failing_backend(*args, **kwargs):
        raise RuntimeError("provider failed")

    second = GraphGeneratorStatic("second", source="LLM", llm_backend=failing_backend)
    with pytest.raises(RuntimeError, match="provider failed"):
        datamodule.precompute_graph(second, cache=False)
    assert datamodule.dataset.graph is graph
    assert datamodule.dataset.graph_generator is first
    assert second.graph is None


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


def test_backend_cache_identity_is_shared_and_tracks_prompt_and_options(datamodule):
    dataset = datamodule.dataset

    def keys(prompt, temperature=.1, api_key="private"):
        backend = LiteLLMBackend(
            "test-model", system_prompt=prompt, temperature=temperature, api_key=api_key,
        )
        generator = GraphGeneratorStatic("test-model", source="LLM", llm_backend=backend)
        source_key = _prepared_cache_key(generator, dataset)
        refinement_key = generator._refinement_cache_key(refine_llm(llm_backend=backend))
        assert source_key["method_parameters"]["llm_backend"] == refinement_key["keywords"]["llm_backend"]
        assert api_key not in json.dumps(source_key)
        assert api_key not in json.dumps(refinement_key)
        return source_key, refinement_key

    assert keys("first") == keys("first", api_key="other-private")
    assert keys("first") != keys("second")
    assert keys("first") != keys("first", temperature=.9)


def test_partial_cache_tracks_positional_arguments_and_tensors():
    def scale(weight, graph):
        return ConceptGraph(graph.data * weight, graph.node_names)

    key = GraphGenerator._refinement_cache_key
    assert key(partial(scale, 2)) != key(partial(scale, 3))
    assert key(partial(scale, torch.tensor(2.))) == key(partial(scale, torch.tensor(2.)))
    assert key(partial(scale, torch.tensor(2.))) != key(partial(scale, torch.tensor(3.)))



def test_load_exported_graph_preserves_values_and_node_order(tmp_path):
    generator = _learnable(
        "dagma_cgm", concept_names=["b", "a", "isolated"],
        initialization=_seed_weights(torch.tensor([
            [0., .5, 0.], [0., 0., 0.], [0., 0., 0.],
        ])),
    ).double().eval()
    path = tmp_path / "graph.pt"
    _materialize(generator).save(path)
    graph = ConceptGraph.load(path, map_location=torch.device("cpu"))
    assert graph.node_names == ["b", "a", "isolated"]
    assert graph.data.device.type == "cpu"
    assert graph.data.dtype == torch.float64
    assert not graph.data.requires_grad
    torch.testing.assert_close(graph.data, _materialize(generator).data)
    assert graph.has_edge("b", "a")
    assert graph.is_dag()


@pytest.mark.parametrize("payload", [
    {}, torch.zeros(2, 2),
    {"adjacency": [[0]], "node_names": ["a"]},
    {"adjacency": torch.zeros(1, 1), "node_names": None},
    {"adjacency": torch.zeros(1, 1), "node_names": [0]},
    {"adjacency": torch.zeros(2, 3), "node_names": ["a", "b"]},
    {"adjacency": torch.zeros(2, 2), "node_names": ["a"]},
])
def test_load_graph_rejects_invalid_payloads(tmp_path, payload):
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        ConceptGraph.load(path)


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


def test_sparse_constructor_preserves_gradients_for_zero_weights():
    weights = torch.tensor([0., .5], requires_grad=True)
    indices = torch.tensor([[0, 1], [1, 0]])
    graph = ConceptGraph.from_sparse(indices, weights, 2)
    assert graph.edge_index is indices
    assert graph.edge_weight is weights
    graph.data.sum().backward()
    torch.testing.assert_close(weights.grad, torch.ones_like(weights))
    with torch.no_grad():
        graph.edge_weight[0] = 1
    torch.testing.assert_close(graph.edge_weight, torch.tensor([1., .5]))
    assert graph.data[0, 1] == 1
    # Dense access reconstructs a tensor; editing it does not alter sparse data.
    with torch.no_grad():
        graph.data[0, 1] = 2
    assert graph.edge_weight[0] == 1


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


def test_llm_none_removes_ambiguous_edges():
    graph = ConceptGraph(torch.tensor([[0., 1.], [1., 0.]]))
    result = refine_llm(llm_backend=Backend())(graph)
    assert result.data.count_nonzero() == 0


def test_learnable_generator_trains_as_model_submodule_and_exports(datamodule, tmp_path):
    import runpy
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples/utilization/2_model/13_learnable_graph.py"
    Model = runpy.run_path(str(example))["GraphConceptBottleneck"]
    generator = _learnable(
        "dagma_cgm", concept_names=["a", "b", "target"], n_tasks=1, threshold=0,
        refinement=[remove_weakest_cycles],
    )
    model = Model(2, Annotations(labels=["a", "b", "target"], cardinalities=[1, 1, 1]),
                  ["target"], generator)
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


@pytest.mark.parametrize("cache", [False, True])
def test_dataset_precompute_requires_training_indices(datamodule, cache):
    dataset = datamodule.dataset
    generator = GraphGeneratorStatic("fake", source="LLM", llm_backend=Backend())
    original_graph = dataset.graph
    with pytest.raises(ValueError, match=r'datamodule.setup\("fit"\)'):
        dataset.precompute_graph(generator, cache=cache)
    assert dataset.graph is original_graph
    assert dataset.graph_generator is None
    assert not generator.fitted


def test_dataset_precompute_rejects_empty_training_indices(datamodule):
    with pytest.raises(ValueError, match="non-empty training indices"):
        datamodule.dataset.precompute_graph(
            _native_fixture(datamodule.dataset), cache=False, training_indices=[],
        )


@pytest.mark.parametrize("direct_dataset", [False, True])
def test_precompute_description_overrides(datamodule, direct_dataset):
    seen = []
    dataset = datamodule.dataset
    dataset.label_descriptions = {"a": "Dataset a", "b": "Dataset b"}

    def backend(prompt, **kwargs):
        seen.append(prompt)
        return "A->B"

    generator = GraphGeneratorStatic("fake", source="LLM", llm_backend=backend)
    target = dataset if direct_dataset else datamodule
    options = {"training_indices": datamodule.trainset.indices} if direct_dataset else {}
    target.precompute_graph(
        generator, cache=False, concept_descriptions={"a": "Override a"}, **options,
    )
    assert "Override a" in seen[0] and "Dataset b" in seen[0]
    assert dataset.label_descriptions == {"a": "Dataset a", "b": "Dataset b"}


def test_learnable_eval_description_change_invalidates_refinement_cache():
    seen = []

    def record(graph, concept_descriptions):
        seen.append(dict(concept_descriptions))
        return graph

    generator = GraphGeneratorLearnable(
        "dagma_cgm", n_concepts=2,
        initialization=_seed_weights(torch.tensor([[0., .5], [0., 0.]])),
        refinement=partial(record, concept_descriptions={}),
    )
    generator(None, ["a", "b"], {"a": "Training"})
    assert not seen
    generator.eval()
    generator(None, ["a", "b"], {"a": "First"})
    generator(None, ["a", "b"], {"a": "First"})
    generator(None, ["a", "b"], {"a": "Changed"})
    assert len(seen) == 2
    assert seen[0]["a"] == "First" and seen[1]["a"] == "Changed"
