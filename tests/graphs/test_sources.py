"""Tests for sources."""

import numpy as np
import pytest
import torch
from torch_concepts import ConceptGraph
from torch_concepts.graphs import GraphGeneratorLearnable
from torch_concepts.graphs import GraphGeneratorStatic
from torch_concepts.graphs import refine_llm
import warnings
from types import SimpleNamespace
import torch_concepts
from torch_concepts.graphs.generation.generators.static import causallearn as static_sources
import sys
import types
from ._helpers import Backend
from ._helpers import _CausalLearnGraph
from ._helpers import _materialize_graph


def test_llm_direct_call_needs_only_names_and_descriptions():
    generator = GraphGeneratorStatic("test-model", source="LLM", llm_backend=Backend("A->B"))
    assert generator(None, ["a", "b"], {"a": "first"}).has_edge("a", "b")


def test_llm_none_removes_ambiguous_edges():
    graph = ConceptGraph(torch.tensor([[0., 1.], [1., 0.]]))
    result = refine_llm(llm_backend=Backend())(graph)
    assert result.data.count_nonzero() == 0


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


@pytest.mark.parametrize("method,module,attribute", [
    ("pc", "causallearn.search.ConstraintBased.PC", "pc"),
    ("ges", "causallearn.search.ScoreBased.GES", "ges"),
])
def test_causallearn_lazy_import_and_missing_dependency(monkeypatch, method, module, attribute):
    from torch_concepts.graphs.generation.generators.static.causallearn import _import_causallearn

    dependency = types.ModuleType(module)
    algorithm = lambda *args, **kwargs: None
    setattr(dependency, attribute, algorithm)
    monkeypatch.setitem(sys.modules, module, dependency)
    assert _import_causallearn(method) is algorithm
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match="pip install causal-learn"):
        _import_causallearn(method)


def test_causallearn_requires_values_and_rejects_unknown_methods():
    from torch_concepts.graphs.generation.generators.static.causallearn import _import_causallearn

    with pytest.raises(ValueError, match="Unknown causallearn method"):
        _import_causallearn("unknown")
    with pytest.raises(ValueError, match="Unknown CausalLearn name"):
        GraphGeneratorStatic("unknown", source="Causallearn")
    with pytest.raises(ValueError, match="requires concept_values"):
        GraphGeneratorStatic("pc")(concept_names=["a", "b"])
    with pytest.raises(ValueError, match="supports only"):
        GraphGeneratorLearnable("unknown", source="DAGMA_CGM", n_concepts=2)


@pytest.mark.parametrize("method", ["pc", "ges"])
def test_causallearn_real_adapter_if_installed(method):
    pytest.importorskip("causallearn")
    values = torch.randint(0, 2, (100, 3), generator=torch.Generator().manual_seed(4)).float()
    graph = GraphGeneratorStatic(method, require_dag=False)(values, ["b", "a", "target"])
    assert graph.node_names == ["b", "a", "target"]
    assert graph.edge_encoding == "endpoints"
    assert graph.data.shape == (3, 3) and torch.isfinite(graph.data).all()


def test_default_llm_backend_receives_options_without_network(monkeypatch):
    import torch_concepts.llm_backends

    options = {}

    class Backend:
        def __init__(self, **kwargs):
            options.update(kwargs)

        def __call__(self, prompt, **kwargs):
            return "none"

    monkeypatch.setattr(torch_concepts.llm_backends, "LiteLLMBackend", Backend)
    graph = GraphGeneratorStatic(
        "test/model", source="LLM", api_key="private",
        completion_kwargs={"temperature": .2, "max_tokens": 50},
    )(concept_names=["a", "b"])
    assert not graph.data.count_nonzero()
    assert options["model"] == "test/model" and options["api_key"] == "private"
    assert options["temperature"] == .2 and options["max_tokens"] == 50
    assert options["retry_on_rate_limit"] is True
