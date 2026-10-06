class GraphGeneratorLearnable(GraphGenerator, nn.Module):
    """Differentiable graph generator.

    A registered learnable source supplies ``forward`` and may provide an
    initialization factory. ``initialization`` accepts a
    ``GraphGeneratorLearnable -> None`` callable. :meth:`construct_graph`
    materializes a detached :class:`ConceptGraph` snapshot and records it in
    the shared generator state. Optional refinement accepts a graph callable.

    Parameters
    ----------
    name : str
        Learnable method name, such as ``'dagma_cgm'``.
    source : str, optional
        Registered implementation family. Inferred from ``name`` when unique.
    refinement : callable, optional
        Optional graph-to-graph refinement.
    require_dag : bool, default True
        Whether the materialized graph must be a DAG.
    initialization : callable, optional
        Custom initializer. Receives this generator instance.
    concept_descriptions : dict, optional
        Descriptions used by LLM-based refinements.
    """

    trainable = True
    _sources: dict[str, Callable] = {}
    _name_sources: dict[str, set[str]] = {}
    def __init__(
        self,
        name: str,
        source: Optional[str] = None,
        refinement: Optional[Callable[[ConceptGraph], ConceptGraph]] = None,
        require_dag: bool = True,
        initialization: Optional[Callable[["GraphGeneratorLearnable"], None]] = None,
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
        if initialization is None:
            initialization = self._spec.initialization
        self._spec = replace(self._spec, initialization=initialization)
        if self._spec.initialization is not None and not callable(self._spec.initialization):
            raise TypeError("`initialization` must be callable or None.")
        if self._spec.initialization is not None:
            self._spec.initialization(self)

    def forward(self) -> torch.Tensor:
        """Return the current differentiable adjacency matrix."""
        adjacency = self._spec.forward(self)
        # The materialized graph no longer reflects the latest parameters.
        self.invalidate_cache()
        return adjacency

    def _parameter_versions(self) -> tuple[int, ...]:
        """Return PyTorch parameter versions for in-memory cache invalidation."""
        return tuple(parameter._version for parameter in self.parameters())



@dataclass(frozen=True, kw_only=True)
class GraphGeneratorLearnableSpec(GraphGeneratorSpec):
    """Implementation contract for a learnable graph source.

    Attributes
    ----------
    forward : callable
        Callback ``forward(generator)`` returning the current differentiable
        adjacency tensor.
    initialization : callable, optional
        Callable receiving the generator and initializing its trainable state.
    """

    forward: Callable
    initialization: Optional[Callable[[Any], None]] = None