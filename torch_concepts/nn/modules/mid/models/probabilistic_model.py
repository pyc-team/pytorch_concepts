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

        # Index every addressable name (variables and plate members).
        self._index_members()
        
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
    
    # --------------------------------------------------------- addressing
    def _index_members(self) -> None:
        """Map every queryable name to its owning variable.

        ``_addressable[name]`` -> the variable whose CPD produces ``name``. A plate
        contributes its own name plus one entry per member (all pointing to the
        plate); an ordinary variable contributes just itself. Column selection
        lives on the CPD (``ParametricCPD.select``).
        """
        self._addressable: Dict[str, Variable] = {}
        self.members_to_idx: Dict[str, Dict[str, int]] = {}
        for var in self.variables.values():
            self._addressable[var.name] = var
            if var.is_plate:
                self.members_to_idx[var.name] = dict()
                for var_idx, member in enumerate(var.members):
                    self._addressable[member] = var
                    self.members_to_idx[var.name][member] = var_idx

        # Cached once: validation looks this up on every query, and a plate can
        # contribute many member names, so we avoid rebuilding the set each call.
        self._queryable_names: frozenset = frozenset(self._addressable)

    @property
    def queryable_names(self) -> frozenset:
        """Names accepted by ``query``/``evidence``: variables plus plate members."""
        return self._queryable_names

    def resolve(self, name: str) -> Variable:
        """The owning variable for ``name`` (the plate itself for a member name).

        Column selection lives on the CPD (``ParametricCPD.select``), so this only
        maps a name to the variable whose CPD produces it.
        """
        return self._addressable[name]
