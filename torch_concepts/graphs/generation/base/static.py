"""Static graph generators with the common observations-and-metadata contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


from torch_concepts.concept_graph import ConceptGraph
from .base import GraphGenerator, GraphGeneratorSpec



class GraphGeneratorStatic(GraphGenerator):
    """Static graph generation with optional dataset disk caching.

    Call datamodule.setup("fit") and then datamodule.precompute_graph(generator).
    Discovery sees training rows and stores the result on datamodule.dataset.graph.
    Direct dataset.precompute_graph requires explicit non-empty training_indices;
    missing indices raise an error directing callers to datamodule.setup("fit").
    The source callback returns a ConceptGraph in dataset concept order.

    generator(concept_values, concept_names=...) computes, refines and validates
    a ConceptGraph without caching.

    Parameters
    ----------
    name : str
        Method identifier: pc, ges, or a model name for source="LLM".
    source : str, optional
        Causallearn, LLM, or a registered custom source. Inferred
        only for a uniquely registered method name.
    refinement : callable or list/tuple of callable, optional
        Graph-to-graph operations applied in order before validation.
    require_dag : bool, default True
        Require the final graph to be acyclic. PC/GES and pairwise LLM outputs
        may need orientation and cycle removal to satisfy this condition.
    **kwargs
        Arguments for the selected source, such as alpha/indep_test for PC,
        score_func for GES, or llm_backend/domain/repeats for LLM generation.

    Notes
    -----
    precompute_graph(cache=True) reuses disk results using method, refinement,
    description and dataset metadata.
    The cache fingerprints values and descriptions, but does not detect changes
    to generation/refinement code. If you change code while keeping the
    same configuration and dataset metadata, it may reuse an outdated graph.
    Pass force=True to recompute the graph and update the disk cache, or
    cache=False to recompute without reading or writing the disk cache.
    """

    trainable = False
    _source_loaders: dict[str, Callable] = {}

    def __call__(
        self, concept_values=None, concept_names=None, concept_descriptions=None,
    ) -> ConceptGraph:
        """Compute, refine and validate a graph from observations and node metadata."""
        return self._construct_graph(concept_values, concept_names, concept_descriptions)



@dataclass(frozen=True, kw_only=True)
class GraphGeneratorStaticSpec(GraphGeneratorSpec):
    """Static source: compute(generator, values, names, descriptions)."""

    compute: Callable
