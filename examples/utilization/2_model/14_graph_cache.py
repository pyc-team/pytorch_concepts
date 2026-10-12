"""Check static disk caching and learnable evaluation caching.

Run: PYTHONPATH=. python examples/utilization/2_model/14_graph_cache.py
The local LLM backend returns A->B; no API key is needed.
"""

from tempfile import TemporaryDirectory

import torch

from torch_concepts import Annotations
from torch_concepts.data.base import ConceptDataModule
from torch_concepts.data.base.dataset import ConceptDataset
from torch_concepts.graphs import (
    GraphGeneratorLearnable, GraphGeneratorStatic, remove_weakest_cycles,
)


static_calls = 0
refinement_calls = 0


def backend(prompt, **kwargs):
    global static_calls
    static_calls += 1
    return "A->B"


def refine(graph):
    global refinement_calls
    refinement_calls += 1
    return remove_weakest_cycles(graph)


def main():
    torch.manual_seed(42)
    names = ["a", "b"]
    values = torch.tensor([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])
    dataset = ConceptDataset(
        values, values, annotations=Annotations(labels=names, cardinalities=[1, 1]),
    )
    dm = ConceptDataModule(dataset, val_size=0, test_size=0, seed=42)
    dm.setup("fit")

    # Static: precompute_graph saves to disk and reuses matching results.
    with TemporaryDirectory() as cache_dir:
        static = GraphGeneratorStatic("demo", source="LLM", llm_backend=backend)
        dm.precompute_graph(static, cache_dir=cache_dir)
        first = dm.dataset.graph.data.clone()
        assert static_calls == 1
        print("Static: first request computes the graph")

        # Even a new generator with the same options reads the disk cache.
        static = GraphGeneratorStatic("demo", source="LLM", llm_backend=backend)
        dm.precompute_graph(static, cache_dir=cache_dir)
        assert static_calls == 1
        torch.testing.assert_close(dm.dataset.graph.data, first)
        print("Static: same options reuse the disk cache")

        dm.precompute_graph(static, cache_dir=cache_dir, force=True)
        assert static_calls == 2
        print("Static: force=True recomputes and replaces the cache")

        dm.precompute_graph(static, cache_dir=cache_dir, cache=False)
        assert static_calls == 3
        print("Static: cache=False computes without reading or writing the cache")

        dm.precompute_graph(static, cache_dir=cache_dir, concept_descriptions={"a": "Updated"})
        assert static_calls == 4
        print("Static: changed descriptions require a new computation")

    # Learnable: training computes afresh; eval caches the refined graph.
    learned = GraphGeneratorLearnable("dagma_cgm", n_concepts=2, refinement=refine)
    first = learned(values, names)
    second = learned(values, names)
    assert first.data.requires_grad and second.data.requires_grad
    assert first is not second and refinement_calls == 0
    print("Learnable: training uses fresh graphs, without refinement")

    learned.eval()
    first = learned(values, names)
    second = learned(values + 1, names)
    assert refinement_calls == 1
    torch.testing.assert_close(first.data, second.data)
    assert not second.data.requires_grad
    print("Learnable: eval reuses the graph, even with a different batch")

    with torch.no_grad():
        learned.fc1.weight.add_(0.1)
    learned(values, names)
    assert refinement_calls == 2
    print("Learnable: changed weights invalidate the eval cache")

    learned(values, names, {"a": "Updated"})
    assert refinement_calls == 3
    print("Learnable: changed descriptions invalidate the eval cache")

    learned.train()
    learned(values, names, {"a": "Updated"})
    learned.eval()
    learned(values, names, {"a": "Updated"})
    assert refinement_calls == 4
    print("Learnable: a training call clears the eval cache")
    print("All cache checks passed.")


if __name__ == "__main__":
    main()
