"""ChainGraph: a mixed (partially directed) probabilistic graphical model."""

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
