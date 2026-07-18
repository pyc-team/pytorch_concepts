"""
MarkovNetwork: an undirected probabilistic graphical model (Markov random field).

The undirected special case of :class:`ProbabilisticModel`: a list of
:class:`Variable`s wired to a list of undirected :class:`ParametricPotential`
factors. The joint is ``p(x) ∝ exp(-Σ_f E_f(scope_f ; conditioning_f))``. There
is no topological order and no per-variable "one factor" constraint — a variable
may appear in any number of potentials.

Use :class:`BayesianNetwork` for directed models and a plain
:class:`ProbabilisticModel` for mixed (partially-directed / chain) graphs.
Inference is via :class:`BeliefPropagation`.
"""

from __future__ import annotations

from typing import List

from ..factors.potential import ParametricPotential
from .probabilistic_model import ProbabilisticModel
from ..variable import Variable


class MarkovNetwork(ProbabilisticModel):
    """Undirected graphical model (Markov random field) over energy-based potentials.

    Parameters
    ----------
    variables : list of Variable
        All random variables in the model. Names must be unique.
    factors : list of ParametricPotential
        The undirected potentials. Every variable in a potential's ``scope`` must
        be one of ``variables`` (the same object). Conditioning inputs (e.g. an
        embedding) are observed and are *not* part of any scope.
    """

    def __init__(
        self,
        variables: List[Variable],
        factors: List[ParametricPotential],
    ):
        factors = list(factors)
        for f in factors:
            if not isinstance(f, ParametricPotential):
                raise TypeError(
                    "MarkovNetwork factors must be ParametricPotential instances "
                    f"(got {type(f).__name__}). Use a BayesianNetwork for directed "
                    "models, or a plain ProbabilisticModel for mixed (chain) graphs."
                )
        # ProbabilisticModel registers factors ({potential name: potential}),
        # validates scopes, and builds the bipartite adjacency. The undirected
        # scope validation is exactly the base one, so nothing is overridden.
        super().__init__(variables, factors)
