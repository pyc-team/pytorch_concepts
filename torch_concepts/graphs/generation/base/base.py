"""
Concept graph generation utilities.

- :class:`GraphGenerator` -- abstract base holding all source-independent
  logic: source registration, method-name lookup, refinement handling, concept
  descriptions, validation, and graph materialization. Not instantiable
  directly.
- :class:`GraphGeneratorFixed` -- fixed graph generation. A source provides a
  ``compute(generator, dataset)`` callback. Built-in sources are
  ``'GroundTruth'``, ``'Causallearn'``, and ``'LLM'``.
- :class:`GraphGeneratorLearnable` -- differentiable graph generation as a
  :class:`torch.nn.Module`. A source provides ``forward(generator)`` and may
  provide an initialization strategy. Built-in sources include
  ``'DAGMA_CGM'``.

Both concrete APIs return a :class:`ConceptGraph`, which owns graph inspection
and plotting. Optional refinement accepts any ``ConceptGraph -> ConceptGraph``
callable. Refinement always runs before DAG validation. Therefore a DAG
validation error from :meth:`construct_graph` means the final graph, after all
refinements, is still cyclic.

Caching
-------
Only fixed graphs are cacheable on disk through dataset ``precompute_graph``.
Learnable generators are trained end-to-end instead. After training,
``construct_graph`` materializes the latest valid ``forward`` result, then
applies refinement and validation.

Concept descriptions
--------------------
LLM generation resolves concept descriptions from ``concept_descriptions`` and
then from ``dataset.label_descriptions``. LLM refinement callables produced by
:func:`refine_llm` are kept synchronized with the generator's current
description mapping.

Extensibility:

- **per-name** (``'ges'`` vs ``'pc'``, or one LLM model vs another): pass a
  different ``name`` to an existing source; no source code changes are needed::

      generator = GraphGeneratorFixed(name="ges", source="Causallearn")

- **refinement**: pass a graph-to-graph callable through ``refinement``::

      from torch_concepts.graph_generator import refine_llm

      generator = GraphGeneratorFixed(
          name="pc",
          refinement=refine_llm(llm_backend=backend, domain="weather"),
      )

- **per-fixed-source**: register a source initializer that returns
  :class:`GraphGeneratorFixedSpec` with a ``compute`` callback::

      @GraphGeneratorFixed.register_source("mylab", names=["my_method"])
      def _load_mylab(generator, name, **kwargs):
          return GraphGeneratorFixedSpec(compute=_compute_mylab)

- **per-learnable-source**: register a source initializer that attaches any
  trainable state to the generator and returns
  :class:`GraphGeneratorLearnableSpec` with ``forward``::

      @GraphGeneratorLearnable.register_source("mylab", names=["my_method"])
      def _load_mylab_learnable(generator, name, **kwargs):
          generator.weight = nn.Parameter(torch.randn(1))
          return GraphGeneratorLearnableSpec(forward=_mylab_forward)

Use a new ``name`` for another method within an existing source. Use
``register_source`` for a new implementation family.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn

from torch_concepts import ConceptGraph

class GraphGenerator:
    """Abstract base class shared by both graph-generator APIs.

    The base class owns the source registry and common generator state.
    Subclasses keep separate source registries and implement their fixed or learnable contract
    according to their fixed or learnable semantics.

    Dataset identity is determined by stable dataset metadata: class, name,
    ordered concept names, sample count, subset information and ``seed`` when
    present.
    Learnable generators also include parameter versions in their own in-memory
    cache key to detect weight updates. Without a dataset, the dataset identity
    is ``None``.

    Parameters
    ----------
    name : str
        Method or model name understood by ``source``.
    source : str, optional
        Registered implementation family. It is inferred when ``name`` maps
        to exactly one registered source; otherwise it must be provided.
    refinement : callable, optional
        A ``ConceptGraph -> ConceptGraph`` callable applied after generation
        and before validation. Use ``refine_llm(llm_backend=backend)`` for LLM
        edge orientation.
        ``None`` disables refinement.
    require_dag : bool, default True
        Validate the final graph after generation and refinement, raising an
        error unless it is a directed acyclic graph.
    concept_descriptions : dict, optional
        Explicit descriptions keyed by concept name. Missing entries are
        filled from ``dataset.label_descriptions`` during ``construct_graph``.

    Attributes
    ----------
    name : str
        Configured method or model name.
    source : str
        Configured source family.
    graph : ConceptGraph or None
        Most recently materialized graph, or ``None`` before generation.
    fitted : bool
        Whether ``construct_graph`` has materialized and stored at least one
        graph snapshot. This is state information, especially useful for
        learnable generators; dataset ground-truth assignment does not rely
        on this flag.
    trainable : bool
        Class-level flag distinguishing fixed and learnable generators.
    """

    trainable: bool
    _sources: dict[str, Callable] = {}
    _name_sources: dict[str, set[str]] = {}

    def __init__(
        self,
        name: str,
        source: Optional[str] = None,
        refinement: Optional[Callable[[ConceptGraph], ConceptGraph]] = None,
        require_dag: bool = True,
        concept_descriptions: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ):
        if type(self) is GraphGenerator:
            raise TypeError(
                "GraphGenerator is abstract; instantiate "
                "GraphGeneratorFixed or GraphGeneratorLearnable."
            )
        super().__init__()
        self.name = name
        self.source = self.resolve_source(name, source)
        self.require_dag = bool(require_dag)
        self.graph: Optional[ConceptGraph] = None
        self.fitted = False
        self._concept_descriptions = dict(concept_descriptions or {})
        if self.source not in self._sources:
            raise ValueError(
                f"Unknown source {self.source!r} for {type(self).__name__}; "
                f"registered sources: {sorted(self._sources)}. Register new "
                f"ones with @{type(self).__name__}.register_source(...)."
            )
        spec = self._sources[self.source](self, self.name, **kwargs)
        if refinement is not None and not callable(refinement):
            raise TypeError("`refinement` must be callable or None.")
        self._spec = replace(spec, refinement=refinement)
        refinement_descriptions = self.extract_refinement_context_descriptions(
            refinement
        )
        if refinement_descriptions:
            self._concept_descriptions.update(refinement_descriptions)
            self._sync_refinement_context()

    @classmethod
    def register_source(
        cls, source: str, names: Optional[Sequence[str]] = None,
    ) -> Callable:
        """Register a source initializer and the method names it provides.

        The decorated function receives ``(generator, name, **kwargs)`` and
        returns the source-specific spec. ``names`` enables automatic source
        inference when a method name is unique across registered sources.
        """
        def decorator(fn: Callable) -> Callable:
            cls._sources[source] = fn
            for name in names or ():
                cls._name_sources.setdefault(name, set()).add(source)
            return fn
        return decorator

    @classmethod
    def resolve_source(cls, name: str, source: Optional[str] = None) -> str:
        """Resolve the implementation family for a method name.

        If ``source`` is provided, it is returned directly. Otherwise, the method
        name must be registered by exactly one source for this generator class.
        Raises a ValueError if the source cannot be inferred or if multiple sources
        match.
        """
        if source is not None:
            return source
        matches = sorted(cls._name_sources.get(name, set()))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"Cannot infer a source for method {name!r}; specify `source`."
            )
        raise ValueError(
            f"Method {name!r} is provided by multiple sources {matches}; "
            "specify `source`."
        )

    def _resolve_context(self, dataset: Optional[ConceptDataset]) -> None:
        """Fill missing concept descriptions from the current dataset."""
        if dataset is None:
            return
        dataset_descriptions = getattr(dataset, "label_descriptions", None) or {}
        self._concept_descriptions = {
            name: str(
                self._concept_descriptions.get(
                    name, dataset_descriptions.get(name, "")
                )
            )
            for name in dataset.concept_names
        }
        self._sync_refinement_context()


    @staticmethod
    def extract_refinement_context_descriptions(refinement) -> dict[str, str]:
        """Extract concept descriptions embedded in refinement callables."""
        if hasattr(refinement, "_refinements"):
            descriptions = {}
            for nested in refinement._refinements:
                descriptions.update(
                    GraphGenerator.extract_refinement_context_descriptions(
                        nested
                    )
                )
            return descriptions
        descriptions = getattr(
            refinement, "_refinement_context_descriptions", None,
        )
        if descriptions is not None:
            return dict(descriptions)
        if not isinstance(refinement, partial):
            return {}
        descriptions = (refinement.keywords or {}).get("concept_descriptions")
        return dict(descriptions or {})

    def extract_current_refinement_context_descriptions(self, refinement):
        """Return ``refinement`` with current descriptions when it supports them."""
        replace_context = getattr(refinement, "_replace_refinement_context", None)
        if replace_context is not None:
            return replace_context(dict(self._concept_descriptions))
        if not isinstance(refinement, partial):
            return refinement
        if "concept_descriptions" not in (refinement.keywords or {}):
            return refinement

        keywords = dict(refinement.keywords or {})
        keywords["concept_descriptions"] = dict(self._concept_descriptions)
        return partial(refinement.func, *refinement.args, **keywords)

    def _sync_refinement_context(self) -> None:
        """Update refinement callables with the latest concept descriptions."""
        refinement = self._spec.refinement
        if refinement is None:
            return

        if hasattr(refinement, "_refinements"):
            refinements = [
                self.extract_current_refinement_context_descriptions(nested)
                for nested in refinement._refinements
            ]
            self._spec = replace(
                self._spec,
                refinement=compose_refinements(*refinements),
            )
            return

        updated = self.extract_current_refinement_context_descriptions(refinement)
        if updated is not refinement:
            self._spec = replace(self._spec, refinement=updated)

    def _cache_key(self, dataset) -> Any:
        """Identify a graph by configuration, dataset and refinement metadata and weight versions."""
        key = {
            "source": self.source,
            "name": self.name,
            "dataset": self._dataset_cache_key(dataset),
            "refinement": self._refinement_cache_key(self._spec.refinement),
        }
        if self.trainable:
            key["parameter_versions"] = self._parameter_versions()
        return key

    @staticmethod
    def _dataset_cache_key(dataset) -> Optional[dict[str, Any]]:
        """Return stable dataset metadata used for graph cache identity."""
        if dataset is None:
            return None
        return {
            "class": type(dataset).__qualname__,
            "name": getattr(dataset, "name", None),
            "concept_names": list(dataset.concept_names),
            "n_samples": dataset.n_samples,
            "is_subset": getattr(dataset, "is_subset", False),
            "subset_seed": getattr(dataset, "subset_seed", None),
            "seed": getattr(dataset, "seed", None),
        }

    @staticmethod
    def _refinement_cache_key(refinement) -> Optional[dict[str, Any]]:
        """Describe a refinement callable in a cache-friendly structure."""
        if refinement is None:
            return None
        if hasattr(refinement, "_refinements"):
            return {
                "function": "compose_refinements",
                "refinements": [
                    GraphGenerator._refinement_cache_key(nested)
                    for nested in refinement._refinements
                ],
            }
        if isinstance(refinement, partial):
            function = refinement.func
            keywords = dict(refinement.keywords or {})
        else:
            function = refinement
            keywords = dict(
                getattr(refinement, "_refinement_cache_keywords", {})
            )
        llm_backend = keywords.pop("llm_backend", None)
        if llm_backend is not None:
            keywords["llm_backend"] = {
                "class": f"{type(llm_backend).__module__}.{type(llm_backend).__qualname__}",
                "model": getattr(llm_backend, "model", None),
            }
        return {
            "function": f"{function.__module__}.{function.__qualname__}",
            "keywords": keywords,
        }


    def invalidate_cache(self) -> None:
        """Discard the materialized graph without changing generator parameters."""
        self.graph = None
        self.fitted = False


    def _validate_graph(self, graph: ConceptGraph) -> None:
        """Validate the final graph according to ``require_dag``."""
        if self.require_dag and not graph.is_directed_acyclic():
            raise ValueError(
                f"Graph method {self.name!r} produced a graph that is not a "
                "directed acyclic graph (DAG) after refinement. DAG validation "
                "is enabled by default. Add or adjust a cycle-removal "
                "refinement, for example "
                "`refinement=compose_refinements(refine_llm(...), "
                "dfs_remove_cycles)`, or pass `require_dag=False` only when a "
                "non-DAG is intentional."
            )

    def construct_graph(
        self,
        dataset: Optional[ConceptDataset] = None,
    ) -> ConceptGraph:
        """Construct, refine, validate, and retain the resulting graph.

        Disk persistence is handled by dataset ``precompute_graph``.
        """
        self._resolve_context(dataset)
        cache_key = self._cache_key(dataset)
        # A learnable generator's parameters are tied to the training context.
        context = (self.source, self.name, cache_key["dataset"])
        previous_context = getattr(self, "_materialization_context", None)
        if self.trainable and previous_context not in (None, context):
            warnings.warn(
                "The dataset, source, or method of a learnable graph generator "
                "changed. Rebuilding the graph does not retrain its parameters; "
                "rerun the complete training pipeline for the new context.",
                UserWarning, stacklevel=2,
            )

        if self.trainable:
            # Learnable sources expose the current adjacency via forward().
            with torch.no_grad():
                generated = self()
            cache_key = self._cache_key(dataset)
        else:
            # Fixed sources need data at construction time.
            if dataset is None:
                raise ValueError("Fixed graph construction requires a dataset.")
            generated = self._spec.compute(self, dataset)
        if isinstance(generated, ConceptGraph):
            graph = generated
        elif isinstance(generated, torch.Tensor):
            concept_names = list(
                dataset.concept_names
                if dataset is not None
                else self.concept_names
            )
            expected = list(getattr(self, "concept_names", concept_names))
            if concept_names != expected:
                raise ValueError(
                    "Dataset concept names must match the generator concepts."
                )
            graph = ConceptGraph(generated.detach(), node_names=concept_names)
        else:
            raise TypeError(
                "Graph generator callbacks must return ConceptGraph or Tensor."
            )

        if self._spec.refinement is not None:
            # Refinement is always graph-to-graph and runs before validation.
            graph = self._spec.refinement(graph)
            if not isinstance(graph, ConceptGraph):
                raise TypeError("Refinement must return a ConceptGraph.")
        self._validate_graph(graph)
        self.graph = graph
        self.fitted = True
        self._materialization_context = context
        return graph

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"source={self.source!r}, trainable={self.trainable})"
        )



@dataclass(frozen=True, kw_only=True)
class GraphGeneratorSpec:
    """Options shared by fixed and learnable graph sources.

    Attributes
    ----------
    refinement : callable, optional
        Optional graph-to-graph post-processing function.
    """

    refinement: Optional[Callable[[ConceptGraph], ConceptGraph]] = None




