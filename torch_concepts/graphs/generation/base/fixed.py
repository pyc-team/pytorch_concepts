
class GraphGeneratorFixed(GraphGenerator):
    """Fixed graph generator.

    A registered fixed source supplies the generation callback. Calling
    :meth:`construct_graph` records the resulting graph in the common generator
    state.

    Parameters
    ----------
    name : str
        Fixed method name, such as ``'ground_truth'``, ``'pc'`` or ``'ges'``.
    source : str, optional
        Registered implementation family. Inferred from ``name`` when unique.
    refinement : callable, optional
        Optional graph-to-graph refinement.
    require_dag : bool, default True
        Whether the generated graph must be a DAG.
    concept_descriptions : dict, optional
        Descriptions used by LLM-based generation or refinement.
    """

    trainable = False
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
        super().__init__(
            name=name,
            source=source,
            refinement=refinement,
            require_dag=require_dag,
            concept_descriptions=concept_descriptions,
            **kwargs,
        )



@dataclass(frozen=True, kw_only=True)
class GraphGeneratorFixedSpec(GraphGeneratorSpec):
    """Implementation contract for a fixed graph source.

    Attributes
    ----------
    compute : callable
        Callback ``compute(generator, dataset)`` returning a
        :class:`ConceptGraph` or adjacency tensor.
    """
    compute: Callable


