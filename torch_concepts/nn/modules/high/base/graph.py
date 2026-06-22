"""Graph-aware base classes for concept models.

This module hosts the abstract layer of the high-level hierarchy that owns
*everything related to the graph*: the :class:`~torch_concepts.ConceptGraph`
that wires concepts together, its topological ordering, and the split between
root concepts (encoded from the input) and internal concepts (predicted from
their parents).

The hierarchy is::

    BaseModel
    └── GraphModel (abstract)            # owns the graph + topology
        ├── DirectedGraphModel (abstract)    # graph is a DAG -> Bayesian network
        └── UndirectedGraphModel (abstract)  # reserved for MRF / factor graphs

Only the *directed* branch is implemented today (all current models are Bayesian
networks). The undirected branch is an explicit placeholder so that future
Markov-random-field / factor-graph models have an obvious home.

The actual graph -> ``BayesianNetwork`` assembly lives one level down, in
:mod:`torch_concepts.nn.modules.high.base.homogen`, because it depends on the
"homogeneous parametrization" assumption. ``GraphModel`` itself only stores and
exposes the graph structure.
"""
from abc import ABC
from typing import List, Optional

from .....annotations import AxisAnnotation
from .....concept_graph import ConceptGraph
from .model import BaseModel


class GraphModel(BaseModel, ABC):
    """Abstract base for concept models backed by an explicit concept graph.

    Owns the graph and derives the structural information every graph-based model
    needs: a topological ordering of the nodes, the set of *root* nodes (no
    parents, encoded from the input) and *internal* nodes (predicted from their
    parents). Subclasses are responsible for turning this structure into a
    concrete probabilistic model.

    The graph is resolved in ``__init__`` via :meth:`_resolve_graph`, which runs
    after the cooperative ``super().__init__()`` chain has set up annotations and
    ``task_names``. Subclasses that *derive* their graph (e.g. the bipartite models
    build it from ``task_names``) override ``_resolve_graph``; subclasses given an
    explicit graph simply return it.

    Attributes
    ----------
    graph : ConceptGraph
        The concept dependency graph (a DAG for the directed branch).
    graph_order : List[str]
        Node names in topological order.
    root_nodes : List[str]
        Nodes with no parents (encoded directly from the input).
    internal_nodes : List[str]
        Nodes with at least one parent (predicted from other concepts).
    """

    def __init__(self, *args, graph: Optional[ConceptGraph] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # The graph a subclass was *given* (may be None for models that derive it).
        self._given_graph = graph
        self.graph: Optional[ConceptGraph] = None
        self.graph_order: List[str] = []
        self.root_nodes: List[str] = []
        self.internal_nodes: List[str] = []
        # Resolve and store the graph immediately: concept_names and task_names
        # are already set by the cooperative super().__init__() chain above.
        self._set_graph(self._resolve_graph())

    # ------------------------------------------------------------------
    # Graph resolution + topology
    # ------------------------------------------------------------------
    def _resolve_graph(self) -> ConceptGraph:
        """Return the concept graph for this model.

        Default: the explicit graph passed at construction. Subclasses that
        derive their graph (e.g. bipartite models) override this.
        """
        if self._given_graph is None:
            raise ValueError(
                f"{type(self).__name__} requires a `graph` (a ConceptGraph). "
                "Pass one explicitly or use a subclass that derives it."
            )
        return self._given_graph

    def _set_graph(self, graph: ConceptGraph) -> None:
        """Store ``graph`` and compute the topological structure from it."""
        self._validate_graph(graph)
        self.graph = graph
        self.graph_order = list(graph.topological_sort())
        self.root_nodes = [n for n in graph.get_root_nodes()]
        self.internal_nodes = [n for n in self.graph_order if n not in self.root_nodes]

    def _validate_graph(self, graph: ConceptGraph) -> None:
        """Validate the graph against this model's assumptions.

        The base class checks that node names match the concept annotations.
        The directed branch additionally enforces acyclicity.
        """
        assert list(graph.node_names) == list(self.concept_names), (
            "ConceptGraph node names must match the concept annotation labels.\n"
            f"  graph: {list(graph.node_names)}\n"
            f"  annotations: {list(self.concept_names)}"
        )


class DirectedGraphModel(GraphModel, ABC):
    """Abstract base for *directed* graph models (Bayesian networks).

    The concept graph must be a DAG; edges encode parent → child conditional
    dependencies, and the assembled probabilistic model is a
    :class:`~torch_concepts.nn.BayesianNetwork`. This is the only branch of the
    hierarchy that is implemented today.

    Concrete models build ``self.model`` (or ``self.pgm``) in their own
    ``__init__``, then call :meth:`_assemble` to wire inference. Two optional
    building hooks are provided for subclasses that want a plate vs individual
    split:

    * :meth:`_build_plate_model` — plate variables, one per homogeneous level.
    * :meth:`_build_individual_model` — one variable per concept.

    Whether to use them, and how to choose between them, is left entirely to
    the concrete model.
    """

    def __init__(self, *args, graph: Optional[ConceptGraph] = None, **kwargs):
        super().__init__(*args, graph=graph, **kwargs)
        self.plate = self.plate_compatible_levels(self.concept_annotations, self.graph)


    def _validate_graph(self, graph: ConceptGraph) -> None:
        super()._validate_graph(graph)
        assert graph.is_directed_acyclic(), (
            "DirectedGraphModel requires a directed acyclic graph (DAG)."
        )

    # ------------------------------------------------------------------
    # Model-building hooks (implement one or both in concrete subclasses)
    # ------------------------------------------------------------------

    def _build_plate_model(self):
        """Build using plate variables (one per homogeneous concept level).

        Override this when the model represents homogeneous levels as a single
        plate :class:`~torch_concepts.nn.ConceptVariable`. Concrete subclasses
        may declare any keyword arguments they need and pass them from
        ``__init__``: ``self._build_plate_model(param=value)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement `_build_plate_model`."
        )

    def _build_individual_model(self):
        """Build using one variable per concept.

        Override this as the flat (non-plate) building path. Concrete subclasses
        may declare any keyword arguments they need and pass them from
        ``__init__``: ``self._build_individual_model(param=value)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement `_build_individual_model`."
        )
    
    #: Distribution parameter used for discrete variables — ``"logits"`` or
    #: ``"probs"``. Concrete models may override; defaults to ``"logits"`` so the
    #: layer output is fed raw and activated by the distribution downstream.
    param_for_discrete_var: str = "logits"

    def _flexible_parametrization(self, variable, first, second=None):
        """Build a ``ParametricCPD`` parametrization dict from ``variable``'s distribution.

        The dict's keys are the distribution's parameter names — taken from
        :data:`~torch_concepts.nn.modules.mid.models.variable.PARAM_DIM` and exposed
        per-variable as ``variable.param_sizes``:

        * **Discrete** families (Bernoulli / Categorical and their relaxed variants)
          use a single parameter, ``"probs"`` or ``"logits"`` as set by
          :attr:`param_for_discrete_var`, parametrized by ``first``.
        * **Delta** uses the single ``"value"`` parameter, parametrized by ``first``.
        * **Continuous** families (Normal, MultivariateNormal) need two parameters:
          the location (``"loc"``) from ``first`` and a scale parameter (``"scale"``
          or ``"scale_tril"``) whose output size depends on univariate vs
          multivariate — read from ``variable.param_sizes``. ``second`` is a
          partially-initialized layer (a callable missing only its output size) to
          be completed with that size.

        Parameters
        ----------
        variable : Variable
            The child variable whose CPD parametrization is being built.
        first : nn.Module
            Layer producing the primary parameter (logits / probs / value / loc).
        second : callable, optional
            Partially-initialized layer for a continuous variable's scale parameter,
            completed with the scale output size. Unused for discrete / Delta.

        Raises
        ------
        NotImplementedError
            For continuous variables — the variance/scale layer is not chosen yet.
        ValueError
            If the variable's distribution is unsupported.
        """
        param_sizes = variable.param_sizes  # {param_name: output_size}, from PARAM_DIM
        names = set(param_sizes)

        if names == {"value"}:
            return {"value": first}
        if names == {"probs", "logits"}:
            return {self.param_for_discrete_var: first}
        if "loc" in names:
            # Continuous: location from ``first``; the scale parameter
            # (``scale`` for Normal, ``scale_tril`` for MultivariateNormal) needs a
            # layer whose output size comes from PARAM_DIM via ``param_sizes``.
            scale_param = (names - {"loc"}).pop()
            scale_size = param_sizes[scale_param]
            raise NotImplementedError(
                f"_flexible_parametrization: continuous variable {variable.name!r} "
                f"({variable.distribution.__name__}) needs a '{scale_param}' layer of "
                f"output size {scale_size}; the variance/scale layer is not chosen "
                f"yet. Once decided, complete `second` to that output size and return "
                f"{{'loc': first, '{scale_param}': <completed second>}}."
            )
        raise ValueError(
            f"_flexible_parametrization: unsupported distribution "
            f"{variable.distribution.__name__} for variable {variable.name!r}."
        )

    @staticmethod
    def plate_compatible_levels(
        axis_annotation: AxisAnnotation,
        graph: ConceptGraph,
    ) -> List[bool]:
        """Flag, per graph level, whether its concepts can share a plate.

        Returns one boolean per level (in the order of
        :meth:`~torch_concepts.ConceptGraph.get_levels`): ``True`` when every
        concept at that level has the **same type and size** (cardinality), so the
        level could be represented by a single plate
        :class:`~torch_concepts.nn.ConceptVariable` with one member per concept;
        ``False`` otherwise. A level with a single concept is trivially ``True``.

        Whether to *actually* build a plate (vs. independent variables) is left to
        the child model — this only reports compatibility.

        Parameters
        ----------
        axis_annotation : AxisAnnotation
            Concept annotations carrying per-concept ``cardinalities`` and types.
        graph : ConceptGraph
            A directed acyclic concept graph whose node names are concept labels.

        Returns
        -------
        List[bool]
            One flag per graph level (roots → leaves).
        """
        def type_and_size(name: str):
            idx = axis_annotation.get_index(name)
            size = int(axis_annotation.cardinalities[idx])
            # Prefer the first-class ``types`` field, fall back to metadata['type'].
            if axis_annotation.types is not None:
                concept_type = axis_annotation.types[idx]
            else:
                concept_type = (axis_annotation.metadata.get(name, {}) or {}).get("type")
            return (concept_type, size)

        return [
            len({type_and_size(name) for name in level}) == 1
            for level in graph.get_levels()
        ]


class UndirectedGraphModel(GraphModel, ABC):
    """Placeholder for *undirected* graph models (Markov random fields / factor graphs).

    Reserved for future use: undirected models would assemble a factor graph of
    ``ParametricPotential`` factors rather than a directed Bayesian network of
    ``ParametricCPD`` factors. No concrete model extends this branch yet.
    """

    def _build_probabilistic_model(self):  # pragma: no cover - not implemented
        raise NotImplementedError(
            "Undirected graph models (Markov random fields / factor graphs) are "
            "reserved for future use and are not implemented yet."
        )
