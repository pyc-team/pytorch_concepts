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
import copy
from abc import ABC
from typing import List, Optional

import torch.nn as nn

from .....annotations import Annotations
from .....concept_graph import ConceptGraph
from ...low.lazy import LazyConstructor
from ...low.scales import TrilActivation
from ...low.sequential import Sequential
from ...mid.distributions import spec_for
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

    Concrete models build ``self.pgm`` in their own ``__init__`` (via a
    ``_build_model`` method) and then call :meth:`setup_inference` to wire
    inference. How the graph becomes variables and CPDs is left to the concrete
    model: the bipartite models group each level into the minimum number of plates,
    while the homogeneous graph assembler walks the DAG node-by-node (one variable
    per node).
    """

    def __init__(self, *args, graph: Optional[ConceptGraph] = None, **kwargs):
        super().__init__(*args, graph=graph, **kwargs)
        self.plate = self.plate_compatible_levels(self.concept_annotations, self.graph)


    def _validate_graph(self, graph: ConceptGraph) -> None:
        super()._validate_graph(graph)
        assert graph.is_directed_acyclic(), (
            "DirectedGraphModel requires a directed acyclic graph (DAG)."
        )
    
    #: Distribution parameter used for discrete variables — ``"logits"`` or
    #: ``"probs"``. Concrete models may override; defaults to ``"logits"`` so the
    #: layer output is fed raw and activated by the distribution downstream.
    param_for_discrete_var: str = "logits"

    def _flexible_parametrization(self, variable, first, second=None):
        """Build a ``ParametricCPD`` parametrization dict from ``variable``'s distribution.

        The dict's keys are the distribution's parameter names — taken from
        :class:`~torch_concepts.nn.modules.mid.distributions.DistributionSpec` and exposed
        per-variable as ``variable.param_sizes``:

        * **Discrete** families (Bernoulli / Categorical and their relaxed variants)
          use a single parameter, ``"probs"`` or ``"logits"`` as set by
          :attr:`param_for_discrete_var`, parametrized by ``first``.
        * **Delta** uses the single ``"value"`` parameter, parametrized by ``first``.
        * **Continuous** families (Normal, MultivariateNormal) need two parameters:
          the location (``"loc"``) from ``first`` and a scale parameter (``"scale"``
          or ``"scale_tril"``) built by :meth:`_scale_parametrization` from
          ``second``, or from a copy of ``first`` when ``second`` is omitted.

        Parameters
        ----------
        variable : Variable
            The child variable whose CPD parametrization is being built.
        first : nn.Module
            Layer producing the primary parameter (logits / probs / value / loc).
        second : nn.Module or ``'auto'``, optional
            The continuous variable's raw scale head: a layer (including an unbuilt
            :class:`~torch_concepts.nn.LazyConstructor`, sized by the CPD from the
            parents just like ``first``), or ``'auto'`` to use an independent copy
            of ``first``. Unused for discrete / Delta variables, so a caller whose
            variables may be of any type can pass ``'auto'`` unconditionally.

        Raises
        ------
        ValueError
            If the variable's distribution is unsupported, or a continuous
            variable's scale head cannot be derived from ``first`` and no
            ``second`` was given.
        """
        param_sizes = variable.param_sizes  # {param_name: output_size}, from the DistributionSpec
        names = set(param_sizes)

        if names == {"value"}:
            return {"value": first}
        if names == {"probs", "logits"}:
            return {self.param_for_discrete_var: first}
        if "loc" in names:
            # Normal, MultivariateNormal, etc., with a location and a scale parameter
            scale_param = (names - {"loc"}).pop() # either ``scale`` or ``scale_tril``
            return {
                "loc": first,
                scale_param: self._scale_parametrization(
                    variable, scale_param, first, second
                ),
            }
        raise ValueError(
            f"_flexible_parametrization: unsupported distribution "
            f"{variable.distribution.__name__} for variable {variable.name!r}."
        )

    def _scale_parametrization(self, variable, scale_param, first, second):
        """Build the module producing a continuous variable's scale parameter.

        A CPD applies no activation, so the result is a raw head followed by the
        activation that makes its output valid (see :meth:`_scale_activation`).

        The raw head comes from ``second``, one of:

        * a :class:`~torch_concepts.nn.LazyConstructor` — sized from this CPD's
          parents and this parameter's width when the CPD builds it, exactly like
          ``first`` (see :meth:`ParametricCPD._instantiate_lazy`);
        * a concrete layer — used as is;
        * ``'auto'`` — a copy of ``first``. Copying needs ``first`` to already emit
          the right number of values, so ``'auto'`` is rejected for a deferred
          ``first`` (no fixed width yet) and for a family that is not
          one-scalar-per-element (``scale_tril`` needs ``size * (size + 1) // 2``
          outputs, not ``size``); pass a ``LazyConstructor`` in those cases.
        """
        spec = spec_for(variable.distribution)
        scale_size = variable.param_sizes[scale_param]

        if second == "auto":
            deferred = isinstance(first, LazyConstructor) and first.module is None
            if deferred or not spec.is_per_element:
                why = ("is a LazyConstructor with no fixed output size until the "
                       "CPD builds it" if deferred else f"emits {variable.size}")
                raise ValueError(
                    f"_flexible_parametrization: {variable.name!r} "
                    f"({variable.distribution.__name__}) cannot copy `first` into a "
                    f"{scale_param!r} head of {scale_size} outputs: it {why}. Pass "
                    "`second` — that layer, or a LazyConstructor for it."
                )
            head = copy.deepcopy(first)
        elif second is None:
            raise ValueError(
                f"_flexible_parametrization: {variable.name!r} "
                f"({variable.distribution.__name__}) needs a {scale_param!r} head of "
                f"{scale_size} outputs. Pass `second` — a layer, a LazyConstructor "
                "for it, or 'auto' to copy `first`."
            )
        else:
            # A LazyConstructor (built later, from the parents) or a concrete layer.
            head = second

        return Sequential(head, self._scale_activation(variable))

    def _scale_activation(self, variable) -> nn.Module:
        """The activation mapping a raw head's output into the scale's domain.

        ``softplus`` for a per-element ``scale`` (a ``Normal``), the Cholesky
        assembly for a matrix-valued ``scale_tril`` (a ``MultivariateNormal``).
        Override to use a different one, e.g. an exponential.
        """
        if spec_for(variable.distribution).is_per_element:
            return nn.Softplus()
        return TrilActivation(variable.size)

    @staticmethod
    def plate_compatible_levels(
        axis_annotation: Annotations,
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
        axis_annotation : Annotations
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
            return (axis_annotation.types[idx], size)

        return [
            len({type_and_size(name) for name in level}) == 1
            for level in graph.get_levels()
        ]


class UndirectedGraphModel(GraphModel, ABC):
    """Placeholder for *undirected* graph models (Markov random fields).

    Reserved for future use: undirected models would assemble a factor graph of
    ``ParametricPotential`` factors rather than a directed Bayesian network of
    ``ParametricCPD`` factors. No concrete model extends this branch yet.
    """

    def _build_probabilistic_model(self):  # pragma: no cover - not implemented
        raise NotImplementedError(
            "Undirected graph models (Markov random fields) are "
            "reserved for future use and are not implemented yet."
        )
