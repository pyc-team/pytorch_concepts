"""Static pairwise LLM discovery through a provider-independent backend.

Each unordered concept pair is queried for A->B, B->A or none. Directed
answers create unit-weight edges; none or exhausted invalid-answer retries
leave the pair absent. Pairwise answers need not form a DAG, so optional
refinements run before the generator's final DAG validation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional, TYPE_CHECKING

import torch

from torch_concepts.concept_graph import ConceptGraph
from ...base.static import GraphGeneratorStatic, GraphGeneratorStaticSpec
from ....utils import _query_pair

if TYPE_CHECKING:
    from torch_concepts.data.base.dataset import ConceptDataset


def _compute_llm(
    self: GraphGeneratorStatic,
    dataset: ConceptDataset,
) -> ConceptGraph:
    """Query every unordered node pair and return directed unit-weight adjacency.

    Use resolved concept descriptions and the configured domain/backend options.
    A->B and B->A select one direction; none and exhausted invalid-answer retries
    create no edge. Provider errors propagate. Global acyclicity is not enforced
    here; refinements and validation belong to the generator lifecycle.
    """
    concept_names = list(dataset.concept_names)
    adjacency = torch.zeros(len(concept_names), len(concept_names))
    for i in range(len(concept_names)):
        for j in range(i + 1, len(concept_names)):
            concept_a, concept_b = concept_names[i], concept_names[j]
            response = _query_pair(
                self.llm_backend,
                concept_a,
                self._concept_descriptions.get(concept_a, ""),
                concept_b,
                self._concept_descriptions.get(concept_b, ""),
                domain=self.domain, repeats=self.repeats,
                completion_kwargs=self.completion_kwargs,
            )
            if response == "A->B":
                adjacency[i, j] = 1.0
            elif response == "B->A":
                adjacency[j, i] = 1.0
    return ConceptGraph(adjacency, node_names=concept_names)


@GraphGeneratorStatic.register_source(
    "LLM",
)
def _load_llm_source(
    self: GraphGeneratorStatic,
    name: str,
    api_key: Optional[str] = None,
    llm_backend: Optional[Callable[..., str]] = None,
    completion_kwargs: Optional[dict[str, Any]] = None,
    repeats: int = 1,
    domain: str = "",
    concept_descriptions: Optional[dict[str, str]] = None,
) -> GraphGeneratorStaticSpec:
    """Configure direct pairwise generation under source="LLM".

    Parameters
    ----------
    self : GraphGeneratorStatic
        Generator receiving the backend and pairwise query configuration.
    name : str
        Provider-prefixed LiteLLM model identifier when using the default backend.
        Pass source="LLM" explicitly; arbitrary model names are not registered.
    api_key : str, optional
        Credential forwarded when constructing the default LiteLLMBackend.
        With a supplied backend, credentials belong to that backend instead.
    llm_backend : callable, optional
        Backend accepting prompt/completion keywords and returning text. If absent,
        create LiteLLMBackend with name, temperature=0, max_tokens=200, short
        rate-limit retry enabled and max_rate_limit_wait=120 seconds.
    completion_kwargs : dict, optional
        Overrides for the default backend and options forwarded to custom backends
        at query time. repeats and invalid-answer retry temperature are managed
        by the shared pairwise query helper.
    repeats : int, default 1
        Positive number of backend completions to aggregate by token vote.
    domain : str, default ""
        Optional domain included in each pairwise prompt.

    concept_descriptions : dict[str, str], optional
        Descriptions by concept name; dataset.label_descriptions fills missing
        entries. Refinement descriptions do not affect generation.

    Returns
    -------
    GraphGeneratorStaticSpec
        Callback that queries all unordered concept pairs. Descriptions are
        resolved by the generator from its configuration and dataset context.

    Notes
    -----
    This module does not read .env files. Set provider credentials in the caller
    or configure an authenticated custom backend before precomputation.
    """
    if (
        not isinstance(repeats, int)
        or isinstance(repeats, bool)
        or repeats < 1
    ):
        raise ValueError("repeats must be a positive integer.")
    self.concept_descriptions = dict(concept_descriptions or {})
    self.model = name
    self.api_key = api_key
    self.domain = domain
    self.repeats = repeats
    self.completion_kwargs = dict(completion_kwargs or {})

    llm_options = {
        "temperature": 0,
        "max_tokens": 200,
        "retry_on_rate_limit": True,
        "max_rate_limit_wait": 120.0,
        **(completion_kwargs or {}),
    }
    if api_key is not None:
        llm_options["api_key"] = api_key
    if llm_backend is None:
        from torch_concepts.llm_backends import LiteLLMBackend

        llm_backend = LiteLLMBackend(model=name, **llm_options)
    self.llm_backend = llm_backend
    if not callable(self.llm_backend):
        raise TypeError("`llm_backend` must be callable.")
    return GraphGeneratorStaticSpec(
        compute=_compute_llm,
    )
