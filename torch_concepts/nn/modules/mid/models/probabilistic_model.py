"""
Probabilistic graphical model, structured as a factor graph.

``ProbabilisticModel`` is the concrete, general graph: a list of
:class:`Variable`s wired to a heterogeneous list of :class:`ParametricFactor`s
(directed :class:`ParametricCPD`s, undirected :class:`ParametricPotential`s, or
any mix). Because every factor exposes the same ``scope`` / ``log_potential``
interface, one inference engine (e.g. :class:`BeliefPropagation`) consumes
directed, undirected, and mixed (chain) graphs with no special-casing — a
partially-directed model is *just* a ``ProbabilisticModel`` holding both factor
kinds, needing no new class.

:class:`BayesianNetwork` (directed) and :class:`MarkovNetwork` (undirected) are
the two specializations that add structure-specific validation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .cpd import ParametricCPD
from .factor import ParametricFactor
from .variable import Variable


class ProbabilisticModel(nn.Module):
    """A probabilistic graphical model represented as a factor graph.

    Parameters
    ----------
    variables : list of Variable
        All random variables in the model. Names must be unique. Stored as the
        ``variables`` mapping ``{name: Variable}`` (the name is taken from each
        ``Variable.name``, so no name needs to be passed separately).
    factors : list of ParametricFactor, optional
        Any mix of :class:`ParametricCPD` and :class:`ParametricPotential`.
        Registered under ``factor.name`` (unique); every variable in a factor's
        ``scope`` must be one of ``variables`` (the same object).
    """

    def __init__(
        self,
        variables: List[Variable],
        factors: Optional[List[ParametricFactor]] = None,
    ) -> None:
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

        # Factor graph: register factors, validate structure, build adjacency.
        self._register_factors(list(factors) if factors is not None else [])
        self._validate_factors()
        self._build_adjacency()

    # ------------------------------------------------------------------ setup
    def _register_factors(self, factors: List[ParametricFactor]) -> None:
        """Populate ``self._factors`` as ``{factor.name: factor}`` (unique keys)."""
        self._factors: nn.ModuleDict = nn.ModuleDict()
        for f in factors:
            if not isinstance(f, ParametricFactor):
                raise TypeError(
                    f"{type(self).__name__}: every factor must be a ParametricFactor, "
                    f"got {type(f).__name__}."
                )
            if f.name in self._factors:
                raise ValueError(
                    f"{type(self).__name__}: duplicate factor name {f.name!r}."
                )
            self._factors[f.name] = f

    def _validate_factors(self) -> None:
        """Every scope variable must be a registered variable (the same object).

        Overridable: :class:`BayesianNetwork` replaces this with its DAG-specific
        parent validation (which also dedups parents).
        """
        for fname, f in self._factors.items():
            for v in f.scope:
                self._check_registered(fname, v)

    def _check_registered(self, fname: str, v: Variable) -> None:
        """Validate that scope variable ``v`` (or its owning plate) is registered."""
        if not isinstance(v, Variable):
            raise TypeError(
                f"Factor {fname!r}: scope entry must be a Variable, "
                f"got {type(v).__name__}."
            )
        plate = getattr(v, "_plate", None)
        if plate is not None:
            # Member handle: the plate must be the registered variable and own it.
            if self.variables.get(plate.name) is not plate:
                raise ValueError(
                    f"Factor {fname!r}: scope variable {v.name!r} is a member of "
                    f"plate {plate.name!r}, which is not the registered variable."
                )
            if v.name not in plate.members:
                raise ValueError(
                    f"Factor {fname!r}: {plate.name!r} has no member {v.name!r}."
                )
            return
        if v.name not in self.variables:
            raise ValueError(
                f"Factor {fname!r}: scope variable {v.name!r} not in variables list."
            )
        if self.variables[v.name] is not v:
            raise ValueError(
                f"Factor {fname!r}: scope variable {v.name!r} is a different Variable "
                "instance than the one registered in `variables`. Pass the same object."
            )

    def _build_adjacency(self) -> None:
        """Bipartite adjacency: ``self._var_factors[var_name] -> [factor name, ...]``."""
        self._var_factors: Dict[str, List[str]] = {name: [] for name in self.variables}
        for fname, f in self._factors.items():
            for v in f.scope:
                owner = v.plate.name  # member handle -> owning plate; else the var
                self._var_factors[owner].append(fname)

    # --------------------------------------------------------------- contract
    @property
    def factors(self) -> nn.ModuleDict:
        """Mapping ``{factor name: factor}`` (directed CPDs and/or potentials).

        For a :class:`BayesianNetwork` the key is a child variable's name and the
        value is the :class:`ParametricCPD` parametrizing that child.
        """
        return self._factors

    @property
    def has_guides(self) -> bool:
        """Whether any guide modules have been registered on this PGM."""
        return len(self.guides) > 0

    # ----------------------------------------------------------- graph queries
    def factor_names_of(self, var: Union[str, Variable]) -> List[str]:
        """Names of the factors whose ``scope`` includes ``var`` (or ``var``'s plate).

        The name-level view of :meth:`factors_of`, for callers (e.g. belief
        propagation) that key their own tables by factor name and would
        otherwise reach into the private adjacency map.
        """
        name = var.name if isinstance(var, Variable) else var
        return list(self._var_factors[name])

    def factors_of(self, var: Union[str, Variable]) -> List[ParametricFactor]:
        """All factors whose ``scope`` includes ``var`` (or ``var``'s plate)."""
        return [self._factors[fn] for fn in self.factor_names_of(var)]

    def neighbors(self, var: Union[str, Variable]) -> List[Variable]:
        """Variables that share at least one factor with ``var`` (excluding itself)."""
        name = var.name if isinstance(var, Variable) else var
        out: Dict[str, Variable] = {}
        for fn in self._var_factors[name]:
            for v in self._factors[fn].scope:
                owner = v.plate.name
                if owner != name:
                    out[owner] = self.variables[owner]
        return list(out.values())

    @property
    def is_directed(self) -> bool:
        """Whether every factor is a directed :class:`ParametricCPD`.

        Reflects factor *types* only; a :class:`BayesianNetwork` additionally
        guarantees acyclicity (and exposes ``levels`` / ``sorted_variables``).
        """
        return all(isinstance(f, ParametricCPD) for f in self._factors.values())

    @property
    def is_undirected(self) -> bool:
        """Whether no factor is a directed :class:`ParametricCPD`."""
        return all(not isinstance(f, ParametricCPD) for f in self._factors.values())

    @property
    def is_mixed(self) -> bool:
        """Whether the graph mixes directed CPDs and undirected potentials (a chain graph)."""
        return not self.is_directed and not self.is_undirected

    # --------------------------------------------------------- addressing
    def _index_members(self) -> None:
        """Map every queryable name to its owning variable.

        ``_addressable[name]`` -> the variable that produces ``name``. A plate
        contributes its own name plus one entry per member (all pointing to the
        plate); an ordinary variable contributes just itself. Column selection
        lives on the variable (``Variable.select``).
        """
        self._addressable: Dict[str, Variable] = {}
        for var in self.variables.values():
            self._addressable[var.name] = var
            if var.is_plate:
                for member in var.members:
                    self._addressable[member] = var

        # Cached once: validation looks this up on every query, and a plate can
        # contribute many member names, so we avoid rebuilding the set each call.
        self._queryable_names: frozenset = frozenset(self._addressable)

    @property
    def queryable_names(self) -> frozenset:
        """Names accepted by ``query``/``evidence``: variables plus plate members."""
        return self._queryable_names

    def resolve(self, name: str) -> Variable:
        """The owning variable for ``name`` (the plate itself for a member name).

        Column selection lives on the variable (``Variable.select``), so this only
        maps a name to the variable that produces it.
        """
        return self._addressable[name]

    def extract(self, name: str, values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Value for any queryable ``name`` from whole-variable ``values``.

        The whole tensor for a variable name, a column slice (a view) for a
        plate-member name. Delegates to the owning variable's ``select_value``,
        so it is independent of how many factors reference the variable (works
        for directed and undirected models alike).
        """
        var = self.resolve(name)
        return var.select_value(values[var.name], name)

    def extract_params(
        self, name: str, params: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Same as :meth:`extract` for per-parameter dicts (``select``)."""
        var = self.resolve(name)
        return var.select(params[var.name], name)
