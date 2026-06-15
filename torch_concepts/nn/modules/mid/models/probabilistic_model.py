"""
Abstract base class for probabilistic graphical models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import torch.nn as nn

from .variable import Variable


class ProbabilisticModel(nn.Module, ABC):
    """Abstract base class for probabilistic graphical models.

    Provides the minimal shared contract for all PGM implementations.
    Subclasses are responsible for defining the graph structure 
    and must implement :meth:`forward`.

    Parameters
    ----------
    variables : list of Variable
        All random variables in the model. Names must be unique. Stored as the
        ``variables`` mapping ``{name: Variable}`` (the name is taken from each
        ``Variable.name``, so no name needs to be passed separately).
    """

    def __init__(self, variables: List[Variable]) -> None:
        super().__init__()

        variables = list(variables)
        self.variables: Dict[str, Variable] = {v.name: v for v in variables}
        if len(self.variables) != len(variables):
            raise ValueError("Duplicate variable names in `variables`.")

        # Guide modules are stored here so pgm.parameters() includes them.
        # The latent/conditioning contract lives on the inference engine.
        self.guides: nn.ModuleDict = nn.ModuleDict()

    @property
    @abstractmethod
    def factors(self) -> nn.ModuleDict:
        """Mapping ``{name: factor module}``, enabling ``model.factors[name]``.

        The subclass defines the name->factor correspondence. For a
        :class:`BayesianNetwork` the key is a child variable's name and the
        value is the :class:`ParametricCPD` parametrizing that child.
        """
        raise NotImplementedError

    @property
    def has_guides(self) -> bool:
        """Whether any guide modules have been registered on this PGM."""
        return len(self.guides) > 0