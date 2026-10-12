"""Shared graph test factories and backend doubles."""

import numpy as np
import torch
from torch_concepts import ConceptGraph
from torch_concepts.data.base.dataset import ConceptDataset
from torch_concepts.graphs import GraphGeneratorLearnable
from torch_concepts.graphs import GraphGeneratorStatic
from torch_concepts.graphs import GraphGeneratorStaticSpec


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


class Backend:
    model = "test-model"

    def __init__(self, answer="none"):
        self.answer = answer
        self._calls = []

    def __call__(self, prompt, **kwargs):
        self._calls.append((prompt, kwargs))
        return self.answer


def _materialize_graph(generator, dataset=None):
    values = dataset.concepts if dataset is not None else None
    names = list(dataset.concept_names) if dataset is not None else getattr(generator, "_test_names", None)
    if names is None:
        names = [str(i) for i in range(getattr(generator, "n_concepts", 0))]
    return generator._construct_graph(
        values, names, getattr(dataset, "label_descriptions", None),
    )


class _CausalLearnGraph:
    def __init__(self, adjacency):
        self.graph = np.asarray(adjacency)
