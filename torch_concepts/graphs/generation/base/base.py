"""Shared graph generation lifecycle and source registration.

Static sources provide ``compute(generator, dataset) -> ConceptGraph``;
learnable sources provide ``forward(generator) -> Tensor`` and optionally an
initializer. Source loaders attach options/state and return the concrete spec.
Configuration is chosen at construction and is read-only afterwards.

Materialization resolves descriptions, computes source output, applies ordered
``ConceptGraph -> ConceptGraph`` refinements, then validates node order and,
when require_dag=True, acyclicity. Generation and refinements share descriptions:
generator descriptions override dataset defaults.

Static disk caching belongs to dataset/datamodule.precompute_graph. Learnable
final graphs are cached in memory by to_graph() in eval mode; ordinary forward()
never uses that cache. Cache keys include options and descriptions where used,
but do not fingerprint dataset values or custom callable code. Use force=True
when data or external callable behavior changes.

Register custom loaders with GraphGeneratorStatic.register_source or
GraphGeneratorLearnable.register_source. Static specs require compute; learnable
specs require forward. The common spec holds refinement options only.
See doc/guides/graph_generation.rst for usage and extension examples.
"""

from __future__ import annotations

import hashlib
from inspect import Parameter, signature
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Optional, Sequence, TYPE_CHECKING

import torch

from torch_concepts.concept_graph import ConceptGraph

if TYPE_CHECKING:
    from torch_concepts.data.base.dataset import ConceptDataset

class GraphGenerator:
    """Common state and lifecycle for static and learnable graph generators.

    Concrete subclasses own separate source registries. This class resolves a
    source, calls its loader, stores its callback spec, resolves refinement
    fallbacks and constructs/validates final graphs. It does not optimize weights
    or manage cache files.

    Configuration is read-only after construction. Create a new generator to
    change any option; learned weights and buffers remain normal PyTorch state.

    Parameters
    ----------
    name : str
        Method or model name understood by the selected source.
    source : str, optional
        Source family. Inferred only when name belongs to exactly one registered
        source for the concrete generator class.
    refinement : callable or list/tuple of callable, optional
        Graph-to-graph operations applied in order before DAG validation.
        None or an empty sequence disables refinement.
    require_dag : bool, default True
        Raise ValueError if the final refined graph is not a DAG.
    **kwargs
        Source-specific arguments forwarded to the registered loader. Both
        explicit arguments and loader defaults contribute to cache identity.

    Attributes
    ----------
    name : str
        Configured method/model identifier.
    source : str
        Resolved implementation family.
    graph : ConceptGraph or None
        Most recent materialized graph; forward() on a learnable generator does
        not set this attribute.
    fitted : bool
        True after successful materialization or static cache loading; this is
        not evidence that a learnable source has been optimized.
    trainable : bool
        Class flag distinguishing static and learnable generators.
    """

    trainable: bool
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
        if type(self) is GraphGenerator:
            raise TypeError(
                "GraphGenerator is abstract; instantiate "
                "GraphGeneratorStatic or GraphGeneratorLearnable."
            )
        super().__init__()
        self.name = name
        self.source = self.resolve_source(name, source)
        self.graph: Optional[ConceptGraph] = None
        self.fitted = False
        self._concept_descriptions = {}
        self.require_dag = require_dag
        loader = self._sources[self.source]
        defaults = {
            name: parameter.default
            for name, parameter in signature(loader).parameters.items()
            if parameter.default is not Parameter.empty
        }
        spec = loader(self, self.name, **kwargs)
        self._method_parameters = {
            name: getattr(self, name, value)
            for name, value in (defaults | kwargs).items()
        }
        if refinement is not None and not callable(refinement) and not isinstance(refinement, (list, tuple)):
            raise TypeError("`refinement` must be callable, a list/tuple of callables, or None.")
        steps = (refinement,) if callable(refinement) else tuple(refinement or ())
        if not all(callable(step) for step in steps):
            raise TypeError("Each refinement must be callable.")
        self._spec = replace(spec, refinement=steps or None)
        if not self.trainable:
            self._freeze_configuration()

    def _freeze_configuration(self):
        self._configuration_fields = frozenset(
            {name for name in vars(self) if not name.startswith("_")}
            | self._method_parameters.keys() | {"trainable"}
        ) - {"graph", "fitted", "training"}

    def __setattr__(self, name, value):
        if name in object.__getattribute__(self, "__dict__").get("_configuration_fields", ()):
            raise AttributeError(f"{name} is read-only; create a new generator to change configuration.")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in object.__getattribute__(self, "__dict__").get("_configuration_fields", ()):
            raise AttributeError(f"{name} is read-only; create a new generator to change configuration.")
        super().__delattr__(name)

    @property
    def refinement(self):
        """Read-only refinement callable(s), or None."""
        return self._spec.refinement

    @classmethod
    def register_source(
        cls, source: str, names: Optional[Sequence[str]] = None,
    ) -> Callable:
        """Register a source initializer and the method names it provides.
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
        """
        if source is not None:
            if source not in cls._sources:
                raise ValueError(
                    f"Unknown source {source!r} for {cls.__name__}; "
                    f"registered sources: {sorted(cls._sources)}. Register new "
                    f"ones with @{cls.__name__}.register_source(...)."
                )
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

    def _prepare_context(self, dataset=None) -> None:
        """Resolve one shared description context for generation and refinements.

        Generator descriptions override dataset defaults.
        Stored refinements receive the resolved context;
        caller-owned dictionaries and callables stay intact.
        """
        steps = self._spec.refinement or ()
        explicit = dict(getattr(self, "concept_descriptions", None) or {})
        descriptions = {**(getattr(dataset, "label_descriptions", None) or {}), **explicit}
        names = getattr(dataset, "concept_names", None)
        if names is not None:
            descriptions = {name: descriptions.get(name, "") for name in names}
        self._concept_descriptions = descriptions
        if "concept_descriptions" in self._method_parameters:
            self._method_parameters["concept_descriptions"] = dict(descriptions)
        refinements = tuple(
            partial(step, concept_descriptions=descriptions)
            if isinstance(step, partial) and "concept_descriptions" in step.keywords
            else step
            for step in steps
        )
        self._spec = replace(self._spec, refinement=refinements or None)

    def _cache_key(self, dataset) -> Any:
        """Build static cache metadata, including descriptions only where consumed.

        Dataset contents and callable implementation code are not fingerprinted.
        Learnable callers use the same tensor-state tracking as to_graph.
        The caller must resolve descriptions with _prepare_context first.
        """
        key = {
            "source": self.source,
            "name": self.name,
            "method_parameters": self._cache_parameter(self._method_parameters),
            "require_dag": self.require_dag,
            "dataset": self._dataset_cache_key(dataset),
            "refinement": self._refinement_cache_key(self._spec.refinement),
        }
        if self.trainable:
            key["tensor_state"] = self._tensor_state()
        return key

    @staticmethod
    def _cache_parameter(value):
        """Convert supported cache options to JSON-compatible, credential-filtered values.
        """
        convert = GraphGenerator._cache_parameter
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                key: convert(item) for key, item in value.items()
                if key.lower() not in {"api_key", "token", "password", "secret"}
                and not key.lower().endswith("_api_key")
            }
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            raw_bytes = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            }
        if callable(value):
            return GraphGenerator._backend_cache_key(value)
        raise TypeError(
            f"Cannot represent method parameter of type {type(value).__name__} "
            "in a cache key. Use ordinary values or tensors; static graph "
            "precomputation also supports cache=False."
        )

    @staticmethod
    def _backend_cache_key(backend) -> dict:
        """Describe a backend by callable/class name, model, prompt and completion options.
        """
        cls = type(backend)
        module = getattr(backend, "__module__", cls.__module__)
        name = getattr(backend, "__qualname__", cls.__qualname__)
        return {
            "name": f"{module}.{name}",
            "model": getattr(backend, "model", None),
            "system_prompt": getattr(backend, "system_prompt", None),
            "completion_kwargs": GraphGenerator._cache_parameter(
                getattr(backend, "completion_kwargs", {}),
            ),
        }

    @staticmethod
    def _dataset_cache_key(dataset) -> Optional[dict[str, Any]]:
        """Identify a dataset by metadata and optional ordered training indices.
        """
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
            "training_indices": getattr(dataset, "graph_training_indices", None),
        }

    @staticmethod
    def _refinement_cache_key(refinement) -> Optional[dict[str, Any]]:
        """Represent refinement identity, partial arguments and declared cache keywords.
        """
        if refinement is None:
            return None
        if isinstance(refinement, (list, tuple)):
            return {
                "function": "refinement_sequence",
                "refinements": [
                    GraphGenerator._refinement_cache_key(nested)
                    for nested in refinement
                ],
            }
        if isinstance(refinement, partial):
            function = refinement.func
            args = refinement.args
            keywords = dict(refinement.keywords or {})
        else:
            function = refinement
            args = ()
            keywords = dict(
                getattr(refinement, "_refinement_cache_keywords", {})
            )
        return {
            "function": f"{function.__module__}.{getattr(function, '__qualname__', type(function).__qualname__)}",
            "args": GraphGenerator._cache_parameter(args),
            "keywords": GraphGenerator._cache_parameter(keywords),
        }


    def _invalidate_cache(self) -> None:
        """Discard the materialized graph without changing generator parameters."""
        self.graph = None
        self.fitted = False


    def _validate_graph(self, graph: ConceptGraph) -> None:
        """Require finite adjacency and, when requested, an acyclic graph."""
        if not torch.isfinite(graph.data).all():
            raise ValueError("Graph adjacency must contain only finite values.")
        if self.require_dag and not graph.is_dag():
            raise ValueError(
                f"Graph method {self.name!r} produced a graph that is not a "
                "directed acyclic graph (DAG) after refinement. DAG validation "
                "is enabled by default. Add or adjust a cycle-removal "
                "refinement, for example "
                "`refinement=[refine_llm(...), "
                "dfs_remove_cycles]`, or pass `require_dag=False` only when a "
                "non-DAG is intentional."
            )

    def _construct_graph(
        self,
        dataset: Optional[ConceptDataset] = None,
    ) -> ConceptGraph:
        """Materialize source output, refine it, validate it and retain the graph.

        Static compute receives a required dataset and returns ConceptGraph.
        The caller must resolve descriptions with _prepare_context first.
        """

        if self.trainable:
            with torch.no_grad():
                adjacency = self._spec.forward(self)
            if not isinstance(adjacency, torch.Tensor):
                raise TypeError("Learnable forward must return an adjacency Tensor.")
            concept_names = list(self.concept_names)
            if dataset is not None and list(dataset.concept_names) != concept_names:
                raise ValueError("Dataset concept names must match the generator concepts.")
            graph = ConceptGraph(adjacency.detach(), node_names=concept_names)
        else:
            if dataset is None:
                raise ValueError("Static graph construction requires a dataset.")
            graph = self._spec.compute(self, dataset)
            if not isinstance(graph, ConceptGraph):
                raise TypeError("Static compute must return a ConceptGraph.")
            concept_names = list(dataset.concept_names)

        if list(graph.node_names) != concept_names:
            self._invalidate_cache()
            raise ValueError("Graph nodes must match the concept names and order.")
        if self._spec.refinement is not None:
            # Refinement is always graph-to-graph and runs before validation.
            for refinement in self._spec.refinement:
                graph = refinement(graph)
                if not isinstance(graph, ConceptGraph):
                    raise TypeError("Refinement must return a ConceptGraph.")
                if list(graph.node_names) != concept_names:
                    self._invalidate_cache()
                    raise ValueError("Refinement nodes must match the concept names and order.")
        self._validate_graph(graph)
        self.graph = graph
        self.fitted = True
        return graph

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"source={self.source!r}, "
            f"trainable={self.trainable})"
        )



@dataclass(frozen=True, kw_only=True)
class GraphGeneratorSpec:
    """Options shared by static and learnable graph sources.

    Generation callbacks are required by the concrete specs:
    GraphGeneratorStaticSpec.compute(generator, dataset) returns a ConceptGraph;
    GraphGeneratorLearnableSpec.forward(generator) returns an adjacency Tensor.

    Attributes
    ----------
    refinement : Callable[[ConceptGraph], ConceptGraph] or list/tuple of these, optional
        Each callback has signature ``refinement(graph) -> ConceptGraph``.
        Input is the generated graph for the first step, then the graph returned
        by the preceding step. Output is the refined ConceptGraph.
        Steps run in order before final DAG validation; a non-ConceptGraph
        output raises TypeError. None or an empty list/tuple skips refinement.
        The generator constructor supplies this field.
    """

    refinement: Optional[Callable[[ConceptGraph], ConceptGraph] | Sequence[Callable[[ConceptGraph], ConceptGraph]]] = None
