"""Static graph generators and their dataset-based compute contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Sequence, TYPE_CHECKING

from torch_concepts.concept_graph import ConceptGraph
from .base import GraphGenerator, GraphGeneratorSpec

if TYPE_CHECKING:
    from torch_concepts.data.base.dataset import ConceptDataset



class GraphGeneratorStatic(GraphGenerator):
    """Dataset-based graph generation with optional disk caching by the caller.

    Call datamodule.setup("fit") and then datamodule.precompute_graph(generator).
    Discovery sees training rows and stores the result on datamodule.dataset.graph.
    Direct dataset.precompute_graph requires explicit non-empty training_indices;
    missing indices raise an error directing callers to datamodule.setup("fit").
    The source callback returns a ConceptGraph in dataset concept order.

    Parameters
    ----------
    name : str
        Method identifier: ground_truth, pc, ges, or a model name for source="LLM".
    source : str, optional
        GroundTruth, Causallearn, LLM, or a registered custom source. Inferred
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
    description and dataset metadata. GroundTruth skips disk persistence.
    The cache does not detect changes to dataset values or to the code of
    generation/refinement functions. If you change either while keeping the
    same configuration and dataset metadata, it may reuse an outdated graph.
    Pass force=True to recompute the graph and update the disk cache, or
    cache=False to recompute without reading or writing the disk cache.
    """

    trainable = False
    _sources: dict[str, Callable] = {}
    _name_sources: dict[str, set[str]] = {}

    def __init__(
        self,
        name: str,
        source: Optional[str] = None,
        refinement: Optional[Callable[[ConceptGraph], ConceptGraph] | Sequence[Callable[[ConceptGraph], ConceptGraph]]] = None,
        require_dag: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            source=source,
            refinement=refinement,
            require_dag=require_dag,
            **kwargs,
        )



@dataclass(frozen=True, kw_only=True)
class GraphGeneratorStaticSpec(GraphGeneratorSpec):
    """Callback contract returned by a registered static source loader.

    Attributes
    ----------
    compute : Callable[[GraphGeneratorStatic, ConceptDataset], ConceptGraph]
        Callback with signature ``compute(generator, dataset) -> ConceptGraph``.
        ``generator`` is the configured GraphGeneratorStatic instance, including
        options attached by the source loader. ``dataset`` is a ConceptDataset
        restricted to the training rows selected for precomputation.
        The returned ConceptGraph must have node names and order matching
        ``dataset.concept_names``; adjacency entry [i, j] represents i -> j.
        Return the graph before refinement and DAG validation: the generator
        applies those steps afterwards. 
    refinement : Callable[[ConceptGraph], ConceptGraph] or list/tuple of these, optional
        Inherited from GraphGeneratorSpec. Each step receives a ConceptGraph
        and must return a ConceptGraph passed to the next step. Steps run in
        order before DAG validation. The generator constructor supplies this
        field; None or an empty list/tuple skips refinement.
    """
    compute: Callable[[GraphGeneratorStatic, "ConceptDataset"], ConceptGraph]
