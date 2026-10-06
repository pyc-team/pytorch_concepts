def refine_llm(
    llm_backend: Callable[..., str],
    *,
    domain: str = "",
    concept_descriptions: Optional[dict[str, str]] = None,
    repeats: int = 1,
) -> Callable[[ConceptGraph], ConceptGraph]:
    """Return an LLM refinement that orients reciprocal edges.

    Reciprocal edges represent ambiguity, for example both ``A -> B`` and
    ``B -> A`` are present. Directed and absent pairs are left unchanged by the
    returned graph-to-graph callable.

    This refinement does not guarantee acyclicity by itself. If the downstream
    generator has ``require_dag=True`` and the source may produce cycles,
    compose it with a cycle-removal refinement such as
    :func:`remove_weakest_cycles` or :func:`dfs_remove_cycles`.
    """
    if not callable(llm_backend):
        raise TypeError("`llm_backend` must be callable.")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    descriptions = concept_descriptions or {}

    def refinement(graph: ConceptGraph) -> ConceptGraph:
        concept_names = list(graph.node_names)
        adjacency = graph.data.clone()
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
        return ConceptGraph(adjacency, node_names=concept_names)

    refinement.__name__ = "refine_llm"
    refinement.__qualname__ = "refine_llm"
    refinement._refinement_context_descriptions = dict(descriptions)
    refinement._refinement_cache_keywords = {
        "llm_backend": llm_backend,
        "domain": domain,
        "concept_descriptions": dict(descriptions),
        "repeats": repeats,
    }
    refinement._replace_refinement_context = lambda context: refine_llm(
        llm_backend=llm_backend,
        domain=domain,
        concept_descriptions=context,
        repeats=repeats,
    )
    return refinement


def remove_weakest_cycles(graph: ConceptGraph) -> ConceptGraph:
    """Project an adjacency to a DAG as in the original CausalCGM.

    Cycles are removed by repeatedly deleting the weakest edge inside a
    strongly connected component.
    """
    node_names = list(graph.node_names)
    adjacency = graph.data.detach().clone()
    while True:
        graph = nx.from_numpy_array(
            adjacency.cpu().numpy(), create_using=nx.DiGraph
        )
        try:
            list(nx.topological_sort(graph))
            return ConceptGraph(adjacency, node_names=node_names)
        except nx.NetworkXUnfeasible:
            cyclic_edges = set()
            for component in nx.strongly_connected_components(graph):
                if len(component) <= 1:
                    continue
                for source in component:
                    for target in graph.successors(source):
                        if target in component:
                            cyclic_edges.add((source, target))
            candidates = adjacency.clone()
            candidates[candidates == 0] = 100
            mask = torch.ones_like(candidates, dtype=torch.bool)
            for edge in cyclic_edges:
                mask[edge] = False
            candidates[mask] = 100
            weakest = torch.unravel_index(
                candidates.argmin(), candidates.shape
            )
            adjacency[weakest] = 0

def dfs_remove_cycles(
    graph: ConceptGraph,
    start_node: int | str | None = None,
) -> ConceptGraph:
    """Remove cycles by deleting the DFS back-edge that closes each cycle.

    This mirrors the lightweight DFS post-processing used by the older graph
    examples. It is deterministic for a fixed adjacency and start node, but it
    is a heuristic; :func:`remove_weakest_cycles` is the closer match to the
    CausalCGM projection rule for weighted learned graphs.
    """
    node_names = list(graph.node_names)
    adjacency = graph.data.detach().clone()
    if start_node is None:
        start_index = len(node_names) - 1
    elif isinstance(start_node, str):
        start_index = node_names.index(start_node)
    else:
        start_index = int(start_node)

    def dfs(node: int, visited: list[bool], stack: list[bool], remove: bool) -> bool:
        visited[node] = True
        stack[node] = True
        for neighbor in range(len(adjacency)):
            if adjacency[neighbor][node] == 1:
                if not visited[neighbor]:
                    if dfs(neighbor, visited, stack, remove):
                        return True
                elif stack[neighbor]:
                    if remove:
                        adjacency[neighbor][node] = 0
                        print(
                            "The cycle has been broken by removing the edge: "
                            f"{neighbor} -> {node}"
                        )
                    return True
        stack[node] = False
        return False

    def contains_cycle() -> bool:
        visited = [False] * len(adjacency)
        stack = [False] * len(adjacency)
        for node in range(len(adjacency)):
            if not visited[node] and dfs(node, visited, stack, False):
                return True
        return False

    if contains_cycle():
        while contains_cycle():
            visited = [False] * len(adjacency)
            stack = [False] * len(adjacency)
            dfs(start_index, visited, stack, True)
    else:
        print("there are no cycles in the graph, therefore the graph is left untouched")
    return ConceptGraph(adjacency.to(dtype=torch.int), node_names=node_names)

