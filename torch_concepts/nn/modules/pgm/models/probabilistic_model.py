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
        All random variables in the model.  Names must be unique.
    """

    def __init__(self, variables: List[Variable]) -> None:
        super().__init__()

        self.variables = list(variables)
        self._name_to_variable: Dict[str, Variable] = {
            v.name: v for v in self.variables
        }
        if len(self._name_to_variable) != len(self.variables):
            raise ValueError("Duplicate variable names in `variables`.")

        # Guide modules are stored here so pgm.parameters() includes them.
        # The latent/conditioning contract lives on the inference engine.
        self.guides: nn.ModuleDict = nn.ModuleDict()

    def name_to_variable(self, name: str) -> Variable:
        """Mapping from variable name to Variable instance."""
        return self._name_to_variable[name]
    
    @abstractmethod
    def name_to_factor(self, name: str) -> nn.Module:
        """Return the factor module associated with the variable name."""
        return NotImplementedError

    @property
    def has_guides(self) -> bool:
        """Whether any guide modules have been registered on this PGM."""
        return len(self.guides) > 0