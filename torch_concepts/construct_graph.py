"""
Concept graph generation utilities.

- :class:`GraphGenerator` — abstract base class defining source registration,
  generator state, and the common ``generate(dataset)`` contract. The output 
  is stored in ``graph`` and reports whether a
  graph has been materialized through ``fitted``.
- :class:`GraphGeneratorFixed` — fixed generation. Its source provides
  ``generate``. Built-in sources:
  ``'GroundTruth'``, ``'Causallearn'``, and ``'LLM'``.
- :class:`GraphGeneratorLearnable` — differentiable generation as a
  :class:`torch.nn.Module`. Its source provides ``forward``; the class provides
  the shared ``generate`` implementation. ``train`` is inherited from
  :class:`torch.nn.Module`. Built-in source: ``'WANDA'``.

Both concrete APIs inherit from :class:`GraphGenerator`, support optional
LLM refinement, and return a :class:`ConceptGraph`, which owns inspection and
plotting. Refinement currently requires ``source="LLM"`` because LLM is the
only registered fixed source that provides edge orientation. For a learnable
generator, refinement is applied when ``generate`` materializes the current
graph, typically after training.

Extensibility:

- **refinement**: refine a fixed or learned graph by passing an LLM generator
  configuration through ``refinement``. Currently, refinement supports only
  ``source="LLM"``; other sources do not provide the required ``orient_edges``
  callback. ``generate`` returns the refined graph directly::

      generator = GraphGeneratorFixed(
          name="ges",
          source="Causallearn",
          refinement={
              "name": "groq/openai/gpt-oss-20b",
              "source": "LLM",
              "api_key": api_key,
          },
      )

- **per-name** (``'ges'`` vs ``'pc'``, or one LLM model vs another): pass the
  new ``name`` to an existing source; no new callbacks are needed::

      generator = GraphGeneratorFixed(name="ges", source="Causallearn")

- **per-fixed-source**: define ``generate`` and register an initializer that
  returns :class:`GraphGeneratorFixedSpec`::

      @GraphGeneratorFixed.register_source("mylab")
      def _init_mylab(generator, name, **kwargs):
          return GraphGeneratorFixedSpec(generate=_generate_mylab)

- **per-learnable-source**: create the trainable parameters, define only
  ``forward``, and return :class:`GraphGeneratorLearnableSpec`. The inherited
  ``generate`` materializes the current adjacency returned by ``forward``::

      @GraphGeneratorLearnable.register_source("mylab")
      def _init_mylab_learnable(generator, name, **kwargs):
          generator.weight = nn.Parameter(torch.zeros(1))
          return GraphGeneratorLearnableSpec(forward=_mylab_forward)

Use a new ``name`` for another model or method within an existing source; use
``register_source`` only for a new implementation family.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn

from torch_concepts.concept_graph import ConceptGraph
from torch_concepts.data.concept_generator import llm_backends

if TYPE_CHECKING:
    from torch_concepts.data.base.dataset import ConceptDataset


class _LiteLLMBackendWithMillisecondRetry(llm_backends.LiteLLMBackend):
    @staticmethod
    def _rate_limit_wait_seconds(error: Exception) -> float | None:
        wait_seconds = llm_backends.LiteLLMBackend._rate_limit_wait_seconds(error)
        if wait_seconds is not None:
            return wait_seconds

        match = re.search(
            r"Please try again in (?P<delay>\d+(?:\.\d+)?)(?P<unit>ms|s)",
            str(error),
            flags=re.IGNORECASE,
        )
        if match:
            delay = float(match.group("delay"))
            if match.group("unit").lower() == "ms":
                delay /= 1000.0
            return delay + 1.0
        return None


class GraphGenerator(ABC):
    """Abstract base class shared by both graph-generator APIs.

    The base class owns the source registry and common generator state.
    Subclasses keep separate source registries and implement :meth:`generate`
    according to their fixed or learnable semantics.

    Parameters
    ----------
    name : str
        Method or model name understood by ``source``.
    source : str
        Registered implementation family used to initialize the generator.

    Attributes
    ----------
    name : str
        Configured method or model name.
    source : str
        Configured source family.
    graph : ConceptGraph or None
        Most recently materialized graph, or ``None`` before generation.
    fitted : bool
        Whether :meth:`generate` has materialized and stored at least one
        graph snapshot. This is state information, especially useful for
        learnable generators; dataset ground-truth assignment does not rely
        on this flag.
    trainable : bool
        Class-level flag distinguishing fixed and learnable generators.
    """

    trainable: bool
    _sources: dict[str, Callable] = {}

    def __init__(self, name: str, source: str):
        super().__init__()
        self.name = name
        self.source = source
        self.graph: Optional[ConceptGraph] = None
        self.fitted = False

    @classmethod
    def register_source(cls, source: str) -> Callable:
        """Register a source initializer on this graph-generator family."""
        def decorator(fn: Callable) -> Callable:
            cls._sources[source] = fn
            return fn
        return decorator

    def _initialize_source(self, **kwargs: Any) -> Any:
        if self.source not in self._sources:
            raise ValueError(
                f"Unknown source {self.source!r} for {type(self).__name__}; "
                f"registered sources: {sorted(self._sources)}. Register new "
                f"ones with @{type(self).__name__}.register_source(...)."
            )
        return self._sources[self.source](self, self.name, **kwargs)

    def _initialize_refinement(
        self,
        refinement: Optional[dict[str, Any]],
    ) -> None:
        self._refiner = GraphGeneratorFixed(**refinement) if refinement else None
        if self._refiner is not None and self._refiner._spec.orient_edges is None:
            raise TypeError(
                f"Refinement source {self._refiner.source!r} does not support "
                "edge orientation."
            )

    def _refine(
        self,
        graph: ConceptGraph,
        dataset: ConceptDataset,
    ) -> ConceptGraph:
        if self._refiner is not None and not graph.is_fully_directed():
            return self._refiner._spec.orient_edges(
                self._refiner,
                graph,
                dataset,
            )
        return graph

    @abstractmethod
    def generate(self, dataset: ConceptDataset) -> ConceptGraph:
        """Generate, store, and return a concept graph from ``dataset``.

        Concrete implementations set :attr:`graph` to the materialized graph
        and mark :attr:`fitted` as true.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"source={self.source!r}, trainable={self.trainable})"
        )



class GraphGeneratorLearnableSpec(NamedTuple):
    """Forward callback supplied by a learnable source."""

    forward: Callable

class GraphGeneratorLearnable(GraphGenerator, nn.Module):
    """Differentiable graph generator.

    A registered learnable source supplies ``forward``. The abstract contract
    is implemented by :meth:`generate`, which materializes a detached
    :class:`ConceptGraph` snapshot and records it in the common generator
    state. Optional refinement is currently supported only through an LLM
    refiner.
    """

    trainable = True
    _sources: dict[str, Callable] = {}

    def __init__(
        self,
        name: str,
        source: str,
        refinement: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, source=source)
        self._spec = self._initialize_source(**kwargs)
        self._initialize_refinement(refinement)

    def forward(self) -> torch.Tensor:
        """Return the current differentiable adjacency matrix."""
        return self._spec.forward(self)

    def generate(
        self,
        dataset: ConceptDataset,
    ) -> ConceptGraph:
        """Materialize a snapshot of the current learned adjacency."""
        concept_names = list(dataset.concept_names)
        if concept_names != self.concept_names:
            raise ValueError(
                "Dataset concept names must match the generator concepts."
            )
        graph = ConceptGraph(self().detach(), node_names=concept_names)
        self.graph = self._refine(graph, dataset)
        self.fitted = True
        return self.graph

class GraphGeneratorFixedSpec(NamedTuple):
    """Callbacks supplied by a registered fixed-generator source."""

    generate: Callable
    orient_edges: Optional[Callable] = None

class GraphGeneratorFixed(GraphGenerator):
    """Fixed graph generator.

    A registered fixed source supplies the generation callback and may also
    provide edge orientation for optional refinement. Currently only the LLM
    source provides that callback. Calling :meth:`generate` records the
    resulting graph in the common generator state.
    """

    trainable = False
    _sources: dict[str, Callable] = {}

    def __init__(
        self,
        name: str,
        source: str,
        refinement: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, source=source)
        self._spec = self._initialize_source(**kwargs)
        self._initialize_refinement(refinement)

    def generate(self, dataset: ConceptDataset) -> ConceptGraph:
        graph = self._spec.generate(self, dataset)
        self.graph = self._refine(graph, dataset)
        self.fitted = True
        return self.graph

# ------------------------------------------------------------------
# WANDA: differentiable graph generation
# ------------------------------------------------------------------
def _wanda_forward(self: GraphGeneratorLearnable) -> torch.Tensor:
    differences = self.np_params.T - self.np_params
    identity = torch.eye(self.n_concepts, device=differences.device)
    adjacency = differences * (1 - identity)

    if not self.hard_threshold:
        return adjacency

    hard_adjacency = (differences > self.threshold).float()
    hard_adjacency = torch.where(
        hard_adjacency.abs() < self.eps,
        torch.zeros_like(adjacency),
        hard_adjacency,
    )
    return adjacency + (hard_adjacency - adjacency).detach()


@GraphGeneratorLearnable.register_source("WANDA")
def _init_wanda_source(
    generator: GraphGeneratorLearnable,
    name: str,
    concept_names: List[str],
    priority_var: float = 1.0,
    hard_threshold: bool = True,
    threshold_init: float = 0.0,
    eps: float = 1e-12,
    ) -> GraphGeneratorLearnableSpec:
    if name != "wanda":
        raise ValueError("The WANDA source supports only name='wanda'.")
    if threshold_init < 0:
        raise ValueError("threshold_init must be non-negative.")
    generator.concept_names = list(concept_names)
    generator.n_concepts = len(generator.concept_names)
    generator.np_params = nn.Parameter(
        torch.zeros(generator.n_concepts, 1)
    )
    generator.priority_var = priority_var / math.sqrt(2)
    generator.register_buffer(
        "threshold",
        torch.full((generator.n_concepts,), threshold_init),
    )
    generator.hard_threshold = hard_threshold
    generator.eps = eps
    nn.init.normal_(generator.np_params, std=generator.priority_var)
    return GraphGeneratorLearnableSpec(forward=_wanda_forward)

# ------------------------------------------------------------------
# Ground-truth graph generator
# ------------------------------------------------------------------
def _generate_ground_truth(
    self: GraphGeneratorFixed,
    dataset: ConceptDataset,
) -> ConceptGraph:
    if dataset.graph_native is None:
        raise ValueError("The GroundTruth source requires `dataset.graph_native`.")
    return dataset.graph_native


@GraphGeneratorFixed.register_source("GroundTruth")
def _init_ground_truth_source(
    generator: GraphGeneratorFixed,
    name: str,
) -> GraphGeneratorFixedSpec:
    if name != "ground_truth":
        raise ValueError(
            "The GroundTruth source supports only name='ground_truth'."
        )
    return GraphGeneratorFixedSpec(generate=_generate_ground_truth)


# ------------------------------------------------------------------
# CausalLearn graph generator
# ------------------------------------------------------------------
_CONSTRAINT_BASED = {"pc"}
_SCORE_BASED = {"ges"}


def _import_causallearn(method: str):
    """Lazily import and return the requested CausalLearn algorithm.

    Args:
        method: One of ``'pc'``,``'ges'``.

    Raises:
        ValueError: If ``method`` is not supported.
        ImportError: If ``causallearn`` is not installed.
    """
    try:
        if method == "pc":
            from causallearn.search.ConstraintBased.PC import pc
            return pc
        elif method == "ges":
            from causallearn.search.ScoreBased.GES import ges
            return ges
        else:
            raise ValueError(
                f"Unknown causallearn method '{method}'. "
                f"Supported: {sorted(_CONSTRAINT_BASED | _SCORE_BASED)}."
            )
    except ImportError as exc:
        raise ImportError(
            "CausalLearn-based graph generator requires the `causallearn` package. "
            "Install it with: pip install causal-learn"
        ) from exc


def _cl_graph_to_adj(cl_graph: Any) -> torch.Tensor:
    """Convert CausalLearn endpoints without dropping ambiguous edges."""
    adj_np = np.array(cl_graph.graph, dtype=np.float32, copy=True)
    diff = adj_np - adj_np.T
    adj_np[diff == -2] = 1.0
    adj_np[diff == 2] = 0.0
    return torch.from_numpy(adj_np)


def _generate_causallearn(
    self: GraphGeneratorFixed,
    dataset: ConceptDataset,
) -> ConceptGraph:
    algorithm = _import_causallearn(self.name)
    data = dataset.concepts.detach().cpu().numpy()

    if self.name in _CONSTRAINT_BASED:
        result = algorithm(data, self.alpha, self.indep_test)
        cl_graph = result[0] if isinstance(result, tuple) else result.G
    else:
        cl_graph = algorithm(data, score_func=self.score_func)["G"]

    concept_names = list(dataset.concept_names)
    return ConceptGraph(
        _cl_graph_to_adj(cl_graph),
        node_names=concept_names,
    )


@GraphGeneratorFixed.register_source("Causallearn")
def _init_causallearn_source(
    generator: GraphGeneratorFixed,
    name: str,
    alpha: float = 0.05,
    indep_test: str = "chisq",
    score_func: str = "local_score_BDeu",
) -> GraphGeneratorFixedSpec:
    supported = _CONSTRAINT_BASED | _SCORE_BASED
    if name not in supported:
        raise ValueError(
            f"Unknown CausalLearn name {name!r}. "
            f"Supported names: {sorted(supported)}."
        )
    if name in _CONSTRAINT_BASED and not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")
    generator.alpha = alpha
    generator.indep_test = indep_test
    generator.score_func = score_func
    return GraphGeneratorFixedSpec(generate=_generate_causallearn)

# ------------------------------------------------------------------
# LLM graph generator
# ------------------------------------------------------------------

# Allowed response tokens
_EDGE_TOKENS = ("A->B", "B->A", "none")

# Prompt template
_PROMPT_TEMPLATE = (
    "You are a causal-inference expert {domain_clause}.\n"
    "Assess the direct causal relationship between:\n"
    "A: {concept_1_details}\n"
    "B: {concept_2_details}\n"
    "{context_section}\n"
    "Choose one answer, accounting for confounding, indirect effects, and "
    "mere association:\n"
    "A->B: A directly causes B\n"
    "B->A: B directly causes A\n"
    "none: no direct causal relationship\n\n"
    "Reason internally. Output exactly A->B, B->A, or none. "
    "No other response is allowed."
)


@GraphGeneratorFixed.register_source("LLM")
def _init_llm_source(
    self: GraphGeneratorFixed,
    name: str,
    api_key: Optional[str] = None,
    llm_backend: Optional[Callable[..., str]] = None,
    completion_kwargs: Optional[dict[str, Any]] = None,
    repeats: int = 1,
    domain: str = "",
    use_rag: Optional[bool] = None,
    rag: Optional[Any] = None,
    documents: Optional[List[str]] = None,
    n_retrieved: int = 3,
    embedding_model: str = "openai/text-embedding-3-small",
    embedding_backend: Optional[Callable[..., Any]] = None,
    embedding_kwargs: Optional[dict[str, Any]] = None,
) -> GraphGeneratorFixedSpec:
    if n_retrieved < 1:
        raise ValueError("n_retrieved must be at least 1.")
    if (
        not isinstance(repeats, int)
        or isinstance(repeats, bool)
        or repeats < 1
    ):
        raise ValueError("repeats must be a positive integer.")
    self.model = name
    self.api_key = api_key
    self.domain = domain
    self.repeats = repeats

    llm_options = {
        "temperature": 0,
        "max_tokens": 200,
        "retry_on_rate_limit": True,
        "max_rate_limit_wait": 120.0,
        **(completion_kwargs or {}),
    }
    if api_key is not None:
        llm_options["api_key"] = api_key
    self.llm_backend = llm_backend or _LiteLLMBackendWithMillisecondRetry(
        model=name,
        **llm_options,
    )
    if not callable(self.llm_backend):
        raise TypeError("`llm_backend` must be callable.")
    self.rag = rag
    self.documents: List[str] = list(documents or [])
    self.use_rag = (
        bool(self.rag is not None or self.documents)
        if use_rag is None
        else use_rag
    )
    self.n_retrieved = n_retrieved
    self.embedding_model = embedding_model
    embedding_options = dict(embedding_kwargs or {})
    if api_key is not None:
        embedding_options["api_key"] = api_key
    self.embedding_backend = embedding_backend
    if (
        self.use_rag
        and self.rag is None
        and self.documents
        and self.embedding_backend is None
    ):
        backend_type = getattr(
            llm_backends,
            "LiteLLMEmbeddingBackend",
            None,
        )
        if backend_type is None:
            raise ImportError(
                "Document RAG requires either an `embedding_backend` or "
                "`llm_backends.LiteLLMEmbeddingBackend`."
            )
        self.embedding_backend = backend_type(
            model=embedding_model,
            **embedding_options,
        )
    if self.embedding_backend is not None and not callable(
        self.embedding_backend
    ):
        raise TypeError("`embedding_backend` must be callable.")
    self._doc_embeddings: Optional[np.ndarray] = None

    if self.use_rag and self.rag is None and not self.documents:
        raise ValueError(
            "RAG is enabled, but neither `rag` nor `documents` was provided."
        )
    if self.rag is not None and not (
        callable(self.rag) or callable(getattr(self.rag, "retrieve", None))
    ):
        raise TypeError(
            "`rag` must be callable or expose a callable `retrieve(query, k)`."
        )
    return GraphGeneratorFixedSpec(
        generate=_generate_llm,
        orient_edges=_orient_llm_edges,
    )


def _generate_llm(
    self: GraphGeneratorFixedSpec,
    dataset: ConceptDataset,
) -> ConceptGraph:
    """Query the LLM for every concept pair."""
    concept_names = list(dataset.concept_names)
    concept_descriptions = _concept_descriptions(dataset, concept_names)
    adjacency = _build_llm_adjacency(
        self,
        concept_names,
        concept_descriptions,
    )
    return ConceptGraph(adjacency, node_names=concept_names)


def _orient_llm_edges(
    self: GraphGeneratorFixedSpec,
    graph: ConceptGraph,
    dataset: ConceptDataset,
) -> ConceptGraph:
    """Use the LLM to orient each ambiguous edge pair."""
    concept_names = list(graph.node_names)
    descriptions = _concept_descriptions(dataset, concept_names)
    adjacency = graph.data.clone()
    for i in range(adjacency.shape[0]):
        for j in range(i + 1, adjacency.shape[0]):
            if adjacency[i, j] == 0 or adjacency[j, i] == 0:
                continue
            concept_a, concept_b = concept_names[i], concept_names[j]
            response = _query_pair(
                self,
                concept_a,
                descriptions[concept_a],
                concept_b,
                descriptions[concept_b],
            )
            if response == "A->B":
                adjacency[i, j] = 1.0
                adjacency[j, i] = 0.0
            elif response == "B->A":
                adjacency[i, j] = 0.0
                adjacency[j, i] = 1.0
    return ConceptGraph(adjacency, node_names=concept_names)


def _concept_descriptions(
    dataset: ConceptDataset,
    concept_names: List[str],
) -> dict[str, str]:
    descriptions = getattr(dataset, "label_descriptions", None) or {}
    return {
        name: str(descriptions.get(name, "")) for name in concept_names
    }


def _build_llm_adjacency(
    self: GraphGeneratorFixedSpec,
    concept_names: List[str],
    concept_descriptions: dict[str, str],
) -> torch.Tensor:
    adjacency = torch.zeros(len(concept_names), len(concept_names))
    for i in range(len(concept_names)):
        for j in range(i + 1, len(concept_names)):
            concept_a = concept_names[i]
            concept_b = concept_names[j]
            response = _query_pair(
                self,
                concept_a,
                concept_descriptions[concept_a],
                concept_b,
                concept_descriptions[concept_b],
            )
            if response == "A->B":
                adjacency[i, j] = 1.0
            elif response == "B->A":
                adjacency[j, i] = 1.0
    return adjacency


def _query_pair(
    self: GraphGeneratorFixedSpec,
    concept_a: str,
    concept_a_description: str,
    concept_b: str,
    concept_b_description: str,
) -> str:
    domain_clause = f"in the domain of {self.domain}" if self.domain else ""
    concept_a_details = _concept_details(concept_a, concept_a_description)
    concept_b_details = _concept_details(concept_b, concept_b_description)

    context = ""

    prompt = _PROMPT_TEMPLATE.format(
        domain_clause=domain_clause,
        concept_1_details=concept_a_details,
        concept_2_details=concept_b_details,
        context_section=(
            f"\nRelevant context:\n{context}\n" if context else ""
        ),
    )
    response = self.llm_backend(prompt, repeats=self.repeats)
    return _most_frequent_token(response)


def _concept_details(name: str, description: str) -> str:
    return f"{name} — {description}" if description else name


def _most_frequent_token(response: Any) -> str:
    """Return the majority token, using ``none`` for invalid votes or ties."""
    raw_tokens = [
        line.strip()
        for line in str(response).splitlines()
        if line.strip()
    ]
    counts = {token: 0 for token in _EDGE_TOKENS}
    for token in raw_tokens:
        vote = token if token in _EDGE_TOKENS else "none"
        counts[vote] += 1

    highest_count = max(counts.values())
    winners = [token for token, count in counts.items() if count == highest_count]
    return winners[0] if len(winners) == 1 else "none"


__all__ = [
    "GraphGenerator",
    "GraphGeneratorFixedSpec",
    "GraphGeneratorLearnableSpec",
    "GraphGeneratorLearnable",
    "GraphGeneratorFixed",
]
