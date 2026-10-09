"""Copy-based edge orientation and cycle removal for ConceptGraph.

Each public refinement returns a new graph with the same node names and
order. LLM orientation handles reciprocal edges only; cycle removal acts
on all nonzero edges. These are export/precomputation operations, not
training objectives. Generator validation runs after all refinements.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Optional
import warnings

import networkx as nx
import torch

from torch_concepts.concept_graph import ConceptGraph
from ...utils import _dfs, _query_pair, contains_cycle


def _warn_undirected_edges(adjacency: torch.Tensor, operation: str, policy: str) -> None:
    undirected = (adjacency == -1) & (adjacency.T == -1)
    undirected.fill_diagonal_(False)
    if undirected.any():
        warnings.warn(
            f"{operation}: the graph appears partially directed: reciprocal "
            "(-1, -1) entries are interpreted as undirected edges in PC/GES "
            "encoding. Each such edge will be treated as two opposite directed "
            f"edges forming a cycle; {policy}. To orient undirected edges first, "
            "apply an orientation refinement such as refine_llm before this "
            "operation and check that no undirected edges remain.",
            UserWarning,
            stacklevel=3,
        )


def refine_llm(
    llm_backend: Callable[..., str],
    *,
    domain: str = "",
    concept_descriptions: Optional[dict[str, str]] = None,
    repeats: int = 1,
) -> Callable[[ConceptGraph], ConceptGraph]:
    """Build a graph-to-graph LLM refinement for reciprocal nonzero edges.

    Parameters
    ----------
    llm_backend : callable
        Configured text backend accepting a prompt and repeats keyword.
        Authentication and provider error handling belong to the backend.
    domain : str, default ""
        Optional domain for pairwise prompts.
    concept_descriptions : dict[str, str], optional
        Descriptions used when calling this refinement directly. When attached
        to a generator, replaced by its resolved description context: generator
        descriptions override dataset defaults. Refinement descriptions are not
        merged into that context.
    repeats : int, default 1
        Positive number of completions to aggregate by valid-token vote.

    Returns
    -------
    callable
        refinement(graph) clones adjacency and queries only pairs where both
        directions are nonzero. A->B/B->A replaces the chosen direction with
        weight 1 and clears the reverse; none clears both. Exhausted invalid
        answers leave the original reciprocal pair unchanged. Directed and
        absent pairs are untouched. Node names and order are retained.

    Notes
    -----
    Attach this callable to refinement, optionally followed by
    remove_weakest_cycles or dfs_remove_cycles. LLM orientation alone does not
    guarantee a DAG. Provider exceptions propagate; invalid-answer retries are
    handled by the shared query helper. Cache metadata records backend identity,
    domain, descriptions and repeats without known credential fields.
    """
    if not callable(llm_backend):
        raise TypeError("`llm_backend` must be callable.")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    def refinement(
        graph: ConceptGraph, *, llm_backend: Callable[..., str], domain: str,
        concept_descriptions: Optional[dict[str, str]] = None, repeats: int,
    ) -> ConceptGraph:
        descriptions = concept_descriptions or {}
        concept_names = list(graph.node_names)
        adjacency = graph.data
        for i in range(len(concept_names)):
            for j in range(i + 1, len(concept_names)):
                if adjacency[i, j] == 0 or adjacency[j, i] == 0:
                    continue
                concept_a, concept_b = concept_names[i], concept_names[j]
                response = _query_pair(
                    llm_backend,
                    concept_a,
                    descriptions.get(concept_a, ""),
                    concept_b,
                    descriptions.get(concept_b, ""),
                    domain=domain, repeats=repeats,
                )
                if response == "A->B":
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 0.0
                elif response == "B->A":
                    adjacency[i, j] = 0.0
                    adjacency[j, i] = 1.0
                elif response == "none":
                    adjacency[i, j] = adjacency[j, i] = 0
        return ConceptGraph(adjacency, node_names=concept_names)

    refinement.__name__ = "refine_llm"
    refinement.__qualname__ = "refine_llm"
    return partial(
        refinement,
        llm_backend=llm_backend,
        domain=domain,
        concept_descriptions=dict(concept_descriptions or {}),
        repeats=repeats,
    )


def remove_weakest_cycles(graph: ConceptGraph) -> ConceptGraph:
    """Remove minimum-absolute-weight cyclic edges until the graph is a DAG.

    Parameters
    ----------
    graph : ConceptGraph
        Directed weighted adjacency; every nonzero entry is an edge.

    Returns
    -------
    ConceptGraph
        A detached copy with the same node names/order, device and dtype.
        All surviving weights are preserved.

    Notes
    -----
    At each iteration, find strongly connected components and remove the
    weakest edge whose endpoints share a component, including self-loops.
    The choice considers all cyclic components, not a single selected cycle.
    Ties follow NetworkX edge iteration order. No input graph mutation occurs.
    Reciprocal (-1, -1) entries trigger a warning: PC/GES undirected edges
    are treated as opposite directed edges. Apply refine_llm first to orient them.
    """
    adjacency = graph.data.detach()
    _warn_undirected_edges(
        adjacency, "remove_weakest_cycles",
        "directions are removed by minimum absolute weight, with ties resolved "
        "by NetworkX edge iteration order, without causal orientation",
    )
    while True:
        network = nx.from_numpy_array(adjacency.cpu().numpy(), create_using=nx.DiGraph)
        components = {
            node: index
            for index, nodes in enumerate(nx.strongly_connected_components(network))
            for node in nodes
        }
        cyclic_edges = [
            edge for edge in network.edges
            if components[edge[0]] == components[edge[1]]
        ]
        if not cyclic_edges:
            break
        weakest = min(cyclic_edges, key=lambda edge: abs(float(adjacency[edge])))
        adjacency[weakest] = 0
    return ConceptGraph(adjacency, node_names=list(graph.node_names))


def dfs_remove_cycles(
    graph: ConceptGraph,
    start_node: int | str | None = None,
) -> ConceptGraph:
    """Remove one DFS back-edge at a time, traversing incoming edges.

    Parameters
    ----------
    graph : ConceptGraph
        Directed adjacency with a nonzero entry for every edge.
    start_node : int or str, optional
        Index or name of the first node visited. Remaining components are still
        traversed. By default visit the final node first, then preceding nodes
        in index order. Parents are visited in increasing node index order.

    Returns
    -------
    ConceptGraph
        Detached DAG copy retaining node names/order, device, dtype and surviving
        edge weights. Self-loops are removed as back-edges.

    Notes
    -----
    Matches the parent-first traversal and restart-after-removal strategy of
    https://github.com/gdefe/causally-reliable-cbm/blob/main/src/utils.py.
    Unlike that implementation, all components are visited, all nonzero weights
    count as edges, and surviving weights are preserved. Each pass removes an
    edge or finishes, so at most E removals occur. The recursive DFS retains
    the upstream structure and is subject to Python's recursion depth limit.

    Removal is determined by traversal order, not edge strength. Use
    remove_weakest_cycles when weights should determine which edges are removed.
    Reciprocal (-1, -1) entries trigger a warning: PC/GES undirected edges
    are treated as opposite directed edges. Apply refine_llm first to orient them.
    """
    adjacency = graph.data.detach()
    _warn_undirected_edges(
        adjacency, "dfs_remove_cycles",
        "back-edge directions are removed according to DFS traversal order, "
        "without causal orientation",
    )
    if start_node is None:
        start_node = len(adjacency) - 1
    else:
        start_node = (
            graph.node_names.index(start_node)
            if isinstance(start_node, str) else int(start_node)
        )
        if not 0 <= start_node < len(adjacency):
            raise ValueError("start_node must identify an existing graph node.")

    while contains_cycle(adjacency):
        visited = [False] * len(adjacency)
        stack = [False] * len(adjacency)
        if not _dfs(start_node, adjacency, visited, stack, True):
            # Upstream would repeat forever if a cycle lies outside this visit.
            # Continue into the remaining components and remove one cycle edge.
            for node in range(len(adjacency)):
                if not visited[node]:
                    stack = [False] * len(adjacency)
                    if _dfs(node, adjacency, visited, stack, True):
                        break
    return ConceptGraph(adjacency, node_names=list(graph.node_names))
