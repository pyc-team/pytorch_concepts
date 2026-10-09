"""Tests for precomputation."""

import numpy as np
import pytest
import torch
from torch_concepts import ConceptGraph
from torch_concepts.graphs import GraphGeneratorLearnable
from torch_concepts.graphs import GraphGeneratorStatic
from types import SimpleNamespace
from ._helpers import Backend
import pandas as pd
from torch_concepts import Annotations
from torch_concepts.data.base import ConceptDataModule
from torch_concepts.data.base.dataset import ConceptDataset
from torch_concepts.graphs import GraphGeneratorStaticSpec
from torch_concepts.graphs.generation.generators.static import causallearn
from ._helpers import _native_fixture
from torch_concepts.graphs import dfs_remove_cycles
from torch_concepts.graphs import remove_weakest_cycles
from torch_concepts.data.splitters.fixed import FixedIndicesSplitter


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

    def replace_edges(graph):
        return ConceptGraph.from_sparse(
            torch.tensor([[0], [1]]), torch.tensor([.2]), graph.n_nodes,
            graph.node_names, edge_encoding=graph.edge_encoding,
        )

    update(_native_fixture(dataset), original)
    update(_native_fixture(dataset, refinement=replace_edges),
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
    update(llm(replace_edges), torch.tensor([[0., .2], [0., 0.]]))
    update(llm(replace_edges), torch.tensor([[0., .2], [0., 0.]]), force=True)

    for refinement in (dfs_remove_cycles, remove_weakest_cycles):
        update(_native_fixture(dataset, refinement=refinement), original)

    def introduce_cycle(graph):
        return ConceptGraph.from_sparse(
            torch.tensor([[0, 1], [1, 0]]), torch.tensor([.7, 1.]),
            graph.n_nodes, graph.node_names, edge_encoding=graph.edge_encoding,
        )

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
