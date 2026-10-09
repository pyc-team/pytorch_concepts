"""Common construction, refinement, validation and cache identity for graph generators.

Static sources provide compute; learnable sources provide forward. Both receive
values, names and descriptions and return adjacency or a ConceptGraph.
Concrete subclasses control when finalization and caching occur.
"""

from __future__ import annotations

import hashlib
from inspect import Parameter, signature
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Optional, Sequence

import torch

from torch_concepts.concept_graph import ConceptGraph

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
        Most recent finalized graph; training calls do not set this attribute.
    fitted : bool
        True after successful materialization or static cache loading; this is
        not evidence that a learnable source has been optimized.
    trainable : bool
        Class flag distinguishing static and learnable generators.
    """

    trainable: bool
    _source_loaders: dict[str, Callable] = {}

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
        self._cache_key = None
        self._trained_cached_graph = None
        self._concept_descriptions = {}
        self.require_dag = require_dag
        loader = self._source_loaders[self.source]
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
            loader = partial(fn)
            loader._method_names = tuple(names or ())
            cls._source_loaders[source] = loader
            return fn
        return decorator

    @classmethod
    def _find_sources(cls, name: str) -> list[str]:
        """Find registered loaders supporting the method name."""
        return sorted(
            source for source, loader in cls._source_loaders.items()
            if name in loader._method_names
        )

    @classmethod
    def resolve_source(cls, name: str, source: Optional[str] = None) -> str:
        """Resolve the implementation family for a method name.
        """
        if source is not None:
            if source not in cls._source_loaders:
                raise ValueError(
                    f"Unknown source {source!r} for {cls.__name__}; "
                    f"registered sources: {sorted(cls._source_loaders)}. Register new "
                    f"ones with @{cls.__name__}.register_source(...)."
                )
            return source
        matches = cls._find_sources(name)
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

    def _prepare_context(self, concept_names, concept_descriptions=None):
        """Bind call descriptions to refinements without changing the originals."""
        descriptions = {
            name: (concept_descriptions or {}).get(name, "")
            for name in concept_names
        }
        self._concept_descriptions = descriptions
        self._context_names = list(concept_names)
        self._resolved_refinements = tuple(
            partial(step, concept_descriptions=descriptions)
            if isinstance(step, partial) and "concept_descriptions" in step.keywords
            else step for step in self._spec.refinement or ()
        )

    def _validate_inputs(self, values, names, descriptions):
        if values is not None and (not isinstance(values, torch.Tensor) or values.ndim != 2):
            raise ValueError("concept_values must be a Tensor of shape (samples, concepts).")
        count = values.shape[1] if values is not None else getattr(self, "n_concepts", None)
        if names is None:
            if count is None:
                raise ValueError("Provide concept_names when concept_values is omitted.")
            names = [str(i) for i in range(count)]
        if isinstance(names, (str, bytes)):
            raise ValueError("concept_names must be a sequence of unique strings.")
        names = list(names)
        if not all(isinstance(n, str) for n in names) or len(set(names)) != len(names):
            raise ValueError("concept_names must contain unique strings.")
        if count is not None and len(names) != count:
            raise ValueError("concept_names must match the concept columns.")
        if self.trainable and getattr(self, "n_concepts", len(names)) != len(names):
            raise ValueError("concept_names must match the number of generator nodes.")
        descriptions = {} if descriptions is None else descriptions
        if not isinstance(descriptions, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in descriptions.items()
        ):
            raise ValueError("concept_descriptions must be a dictionary of strings.")
        return names, descriptions

    def _build_cache_key(self, *, concept_values=None, cache_metadata=None) -> Any:
        """Build cache identity from all supplied inputs and generator configuration.

        Optional cache_metadata is supplied by the caller, independently of context.
        Input values are fingerprinted; callable implementation code is not.
        Learnable cache keys also track parameter/buffer state.
        The caller must resolve descriptions with _prepare_context first.
        """
        key = {
            "concept_names": list(getattr(self, "_context_names", ()) or ()),
            "source": self.source,
            "name": self.name,
            "method_parameters": self._cache_parameter(self._method_parameters),
            "require_dag": self.require_dag,
            "dataset": self._cache_parameter(cache_metadata),
            "refinement": self._refinement_cache_key(getattr(self, "_resolved_refinements", self._spec.refinement)),
        }
        if self.trainable:
            key["tensor_state"] = self._tensor_state()
        key["concept_descriptions"] = dict(self._concept_descriptions)
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
        self._cache_key = None
        self._trained_cached_graph = None

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
        self, concept_values, concept_names, concept_descriptions=None, *,
        finalize=True,
    ) -> ConceptGraph:
        """Compute through the common source contract, then optionally finalize."""
        names, descriptions = self._validate_inputs(concept_values, concept_names, concept_descriptions)
        compute = self._spec.forward if self.trainable else self._spec.compute
        output = compute(self, concept_values, names, descriptions)
        graph = ConceptGraph(output, node_names=names) if isinstance(output, torch.Tensor) else output
        if not isinstance(graph, ConceptGraph):
            raise TypeError("Source compute must return a Tensor or ConceptGraph.")
        if not torch.is_grad_enabled() and graph.data.requires_grad:
            graph = ConceptGraph(graph.data.detach(), node_names=names)
        if graph.node_names != names:
            raise ValueError("Graph nodes must match the concept names and order.")
        if not finalize:
            return graph
        resolved = {name: descriptions.get(name, "") for name in names}
        if getattr(self, "_context_names", None) != names or self._concept_descriptions != resolved:
            self._prepare_context(concept_names=names, concept_descriptions=descriptions)
        for refinement in self._resolved_refinements:
            graph = refinement(graph)
            if not isinstance(graph, ConceptGraph):
                raise TypeError("Refinement must return a ConceptGraph.")
            if graph.node_names != names:
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

    Static and learnable specs define their own source callbacks.

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
