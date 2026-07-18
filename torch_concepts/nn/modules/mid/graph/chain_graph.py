"""
ChainGraph: a mixed (partially directed) probabilistic graphical model.

.. warning::
    **Not implemented yet — this is a placeholder.** Constructing a
    ``ChainGraph`` raises :class:`NotImplementedError`.

The third concrete specialization of :class:`ProbabilisticModel`, alongside
:class:`BayesianNetwork` (all directed) and :class:`MarkovNetwork` (all
undirected). A chain graph mixes both factor kinds: some variables are given by
a directed conditional ``p(child | parents)``, others are coupled by an
undirected potential ``exp(-E(scope))``. The joint is the product of the two
families, normalised over the undirected part.

Nothing new is needed at the *inference* level: every factor already exposes the
same ``scope`` / ``log_potential`` interface, so
:class:`~torch_concepts.nn.BeliefPropagation` consumes a mixed graph exactly as
it consumes a directed or undirected one. What is missing here is the
*structural* validation this class owes, which is why it is a stub rather than a
bare ``pass``:

TODO — to implement:
  * validate that every variable is covered consistently: a variable must not be
    both the child of a CPD and a free member of a potential's scope, or its
    conditional would be specified twice;
  * identify the chain components (the connected components left after removing
    directed edges) and check that the graph of components is acyclic — the
    defining condition of a chain graph;
  * decide the normalisation contract: whether the undirected blocks are
    normalised per chain component (the standard LWF semantics) or globally,
    since that determines what ``log_potential`` must return for scoring;
  * extend :attr:`ProbabilisticModel.is_mixed` coverage and the topological
    helpers (``levels`` / ``sorted_variables``) to the component ordering, so
    the forward engines can traverse a chain graph too.

Until then, use :class:`BayesianNetwork` or :class:`MarkovNetwork`.
"""

from __future__ import annotations

from typing import List, Optional

from ..factors.factor import ParametricFactor
from .probabilistic_model import ProbabilisticModel
from ..variable import Variable


class ChainGraph(ProbabilisticModel):
    """Mixed (partially directed) graphical model. **Not implemented yet.**

    Placeholder for the concrete model type holding both
    :class:`ParametricCPD` and :class:`ParametricPotential` factors. See the
    module docstring for the design notes and the outstanding work.

    Parameters
    ----------
    variables : list of Variable
        All random variables in the model.
    factors : list of ParametricFactor, optional
        A mix of directed CPDs and undirected potentials.

    Raises
    ------
    NotImplementedError
        Always — this class is a stub.
    """

    def __init__(
        self,
        variables: List[Variable],
        factors: Optional[List[ParametricFactor]] = None,
    ) -> None:
        raise NotImplementedError(
            "ChainGraph (mixed / partially directed models) is not implemented yet. "
            "Use BayesianNetwork for a fully directed model or MarkovNetwork for a "
            "fully undirected one."
        )
