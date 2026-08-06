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

import torch.nn as nn

from .....annotations import Annotations
from .....concept_graph import ConceptGraph
from ...low.sequential import Sequential
from ...mid.activations import DefaultActivation
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
          or ``"scale_tril"``) from ``second``.

        **Pass raw heads.** Every head is composed with the activation that lands
        its output in its own parameter's domain (see :meth:`_activate`), so a
        head that squashes its own output would be activated twice. This is
        uniform across parameters — there is no head that is left alone and no
        head that is wrapped, which is the whole point: the rule is the same
        wherever you look. Note it applies to *this* helper only; a
        :class:`ParametricCPD` built by hand still applies no activation of its
        own, and its modules must already emit a valid parameter.

        ``first`` and ``second`` are two **independent** heads. Whatever they
        share belongs in the CPD's ``trunk`` (see :class:`ParametricCPD`), which
        runs once and feeds both — so a shared feature extractor costs one
        forward pass, not two. Do not put the *whole* head in the trunk and leave
        the parameters bare: two bare heads over one trunk make the scale a fixed
        function of the location.

        Parameters
        ----------
        variable : Variable
            The child variable whose CPD parametrization is being built.
        first : nn.Module
            Raw layer producing the primary parameter (logits / probs / value / loc).
        second : nn.Module, optional
            The continuous variable's raw scale head: a layer, or an unbuilt
            :class:`~torch_concepts.nn.LazyConstructor` sized by the CPD from the
            parents just like ``first``. Ignored for discrete / Delta variables,
            which have no second parameter — so a caller whose variables may be
            of any type can pass one unconditionally and it is simply unused.

        Raises
        ------
        ValueError
            If the variable's distribution is unsupported, or a continuous
            variable is given no ``second``.
        """
        param_sizes = variable.param_sizes  # {param_name: output_size}, from the DistributionSpec
        names = set(param_sizes)

        if names == {"value"}:
            return {"value": self._activate(variable, "value", first)}
        if names == {"probs", "logits"}:
            param = self.param_for_discrete_var
            return {param: self._activate(variable, param, first)}
        if "loc" in names:
            # Normal, MultivariateNormal, etc., with a location and a scale parameter
            scale_param = (names - {"loc"}).pop() # either ``scale`` or ``scale_tril``
            if second is None:
                raise ValueError(
                    f"_flexible_parametrization: {variable.name!r} "
                    f"({variable.distribution.__name__}) needs a {scale_param!r} head "
                    f"of {param_sizes[scale_param]} outputs. Pass `second` — a raw "
                    "layer or a LazyConstructor for it. Anything it shares with "
                    "`first` belongs in the CPD's `trunk`, which runs once for both."
                )
            return {
                "loc": self._activate(variable, "loc", first),
                scale_param: self._activate(variable, scale_param, second),
            }
        raise ValueError(
            f"_flexible_parametrization: unsupported distribution "
            f"{variable.distribution.__name__} for variable {variable.name!r}."
        )

    def _activate(self, variable, param, head) -> nn.Module:
        """Compose ``head`` with the activation for ``param``'s domain.

        ``head`` may be an unbuilt :class:`~torch_concepts.nn.LazyConstructor`;
        the CPD sizes it from the parents and this parameter's width when it
        builds the ``Sequential`` (see :meth:`ParametricCPD._instantiate_lazy`),
        which is how a ``scale_tril`` head gets its ``size * (size + 1) // 2``
        outputs rather than ``size``.
        """
        activation = self._param_activation(variable, param)
        # An unconstrained parameter (`logits`, `loc`, a Delta's `value`) resolves
        # to the identity. Return the head untouched there rather than wrapping it
        # in a Sequential that computes nothing and shifts its state_dict keys.
        inner = getattr(activation, "activation", activation)
        if isinstance(inner, nn.Identity):
            return head
        return Sequential(head, activation)

    def _param_activation(self, variable, param) -> nn.Module:
        """The activation mapping a raw head's output into ``param``'s domain.

        The family's standard choice, read off its
        :class:`~torch_concepts.nn.modules.mid.distributions.DistributionSpec`:
        a sigmoid for a ``Bernoulli``'s ``probs``, a per-member softmax for a
        categorical's, ``softplus`` for a per-element ``scale`` (a ``Normal``),
        the Cholesky assembly for a matrix-valued ``scale_tril`` (a
        ``MultivariateNormal``), and the identity for anything unconstrained.
        Override to use a different one, e.g. an exponential for the scale.
        """
        return DefaultActivation.for_variable(variable, param)

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
