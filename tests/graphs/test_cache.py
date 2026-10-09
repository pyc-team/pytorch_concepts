"""Tests for cache."""

import numpy as np
import pytest
import torch
from torch_concepts import ConceptGraph
from torch_concepts.graphs import GraphGeneratorLearnable
from torch_concepts.graphs import GraphGeneratorStatic
from torch_concepts.graphs import refine_llm
from types import SimpleNamespace
from ._helpers import Backend
from ._helpers import _materialize_graph
from torch_concepts.graphs import GraphGeneratorStaticSpec
from torch_concepts.graphs.generation.generators.static import causallearn
from ._helpers import _native_fixture
from torch_concepts.graphs import GraphGenerator
from torch_concepts.graphs import dfs_remove_cycles
from ._helpers import _learnable
from ._helpers import _seed_weights
from torch_concepts.graphs import GraphGeneratorLearnableSpec
from torch import nn
from torch_concepts.graphs import remove_weakest_cycles
from functools import partial
import json
from torch_concepts.llm_backends import LiteLLMBackend
from ._helpers import _prepared_cache_key


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
    assert len(calls) == 1
    with torch.no_grad():
        generator.fc1.weight.mul_(2)
    updated = generator(values, names, {})
    assert updated is not final and len(calls) == 2
    renamed = generator(values, ["x", "y"], {})
    assert renamed.node_names == ["x", "y"] and len(calls) == 3
    generator.train()
    assert generator(values, names, {}).data.requires_grad
    assert len(calls) == 3


def test_eval_cache_excludes_observations(datamodule):
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
    torch.testing.assert_close(changed.data, direct.data)
    # An observation-dependent custom source needs explicit invalidation.
    generator._invalidate_cache()
    torch.testing.assert_close(generator(values * 2, ["a", "b"], {}).data, direct.data * 2)


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
        assert set(payload) == {"adjacency", "node_names", "edge_encoding"}
        torch.testing.assert_close(ConceptGraph.load(path).data, payload["adjacency"])
    assert list(second.to_networkx().edges) == [("a", "b")]


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


def test_static_disk_cache_distinguishes_partial_backend_arguments(datamodule, tmp_path):
    calls = []

    def backend(prompt, *, direction, **kwargs):
        calls.append(direction)
        return direction

    first = GraphGeneratorStatic("test", source="LLM", llm_backend=partial(backend, direction="A->B"))
    second = GraphGeneratorStatic("test", source="LLM", llm_backend=partial(backend, direction="B->A"))
    datamodule.precompute_graph(first, cache_dir=tmp_path)
    assert datamodule.dataset.graph.has_edge("a", "b")
    datamodule.precompute_graph(second, cache_dir=tmp_path)
    assert datamodule.dataset.graph.has_edge("b", "a")
    datamodule.precompute_graph(first, cache_dir=tmp_path)
    assert datamodule.dataset.graph.has_edge("a", "b")
    assert calls == ["A->B", "B->A"]
    assert len(list(tmp_path.glob("graph_*.pt"))) == 2


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


def test_backend_partial_cache_tracks_function_arguments_and_backend_options():
    def scale(value, factor=1):
        return value * factor

    key = GraphGeneratorStatic._cache_parameter
    assert key(partial(scale, 2)) != key(partial(scale, 3))
    assert key(partial(scale, factor=2)) != key(partial(scale, factor=3))
    assert key(partial(scale, torch.tensor(2.))) == key(partial(scale, torch.tensor(2.)))

    class Backend:
        def __init__(self, model):
            self.model = model
            self.completion_kwargs = {"temperature": 0., "api_key": "private"}

        def __call__(self, prompt, **kwargs):
            return "A->B"

    first = key(partial(Backend("first"), domain="test", api_key="private"))
    second = key(partial(Backend("second"), domain="test", api_key="private"))
    assert first != second
    assert "private" not in str(first)


def test_unsupported_cache_options_fail_clearly():
    with pytest.raises(TypeError, match="Cannot represent method parameter"):
        GraphGeneratorStatic._cache_parameter(object())
