"""ParametricPotential — undirected, energy-based factor over a clique.

A potential contributes ``exp(-E(scope ; conditioning))`` to the joint. Its
``scope`` is a *symmetric* clique of variables (no child/parent asymmetry); an
optional list of ``conditioning`` variables (e.g. an embedding) feeds the neural
energy but is **not** part of the scope — those are observed inputs, so they
create no undirected edges and are not marginalized. This is the conditional
random field (CRF) setting used by concept models.

The energy is domain-agnostic (continuous or discrete assignments). The concrete
:class:`TabularPotential` realizes the discrete log-linear case that belief
propagation consumes: a neural function of the conditioning inputs emits a
log-potential *table* over the discrete joint states of the scope.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.distributions as dist

from .factor import ParametricFactor
from .variable import Variable


# ---------------------------------------------------------------------------
# Discrete-state cardinality (shared by TabularPotential and the BP engine).
# ---------------------------------------------------------------------------
_BINARY_FAMILIES = (dist.Bernoulli, dist.RelaxedBernoulli)
_CATEGORICAL_FAMILIES = (
    dist.Categorical,
    dist.OneHotCategorical,
    dist.RelaxedOneHotCategorical,
)


def enumerable_cardinality(variable: Variable) -> int:
    """Number of discrete states of ``variable`` (for enumeration/BP).

    - Bernoulli-family with ``size == 1`` -> ``2`` (states ``0`` and ``1``).
    - Categorical/OneHot-family -> ``variable.size`` (one state per class).

    Raises ``ValueError`` for any other case (e.g. a size>1 Bernoulli, a Normal,
    or a Delta), which cannot be enumerated by belief propagation.
    """
    D = variable.distribution
    if any(issubclass(D, b) for b in _BINARY_FAMILIES):
        if variable.size != 1:
            raise ValueError(
                f"Variable {variable.name!r}: a size>1 {D.__name__} is a set of "
                "independent bits, not a single enumerable variable. Model each bit "
                "as its own binary variable, or use a Categorical/OneHotCategorical."
            )
        return 2
    if any(issubclass(D, c) for c in _CATEGORICAL_FAMILIES):
        return variable.size
    raise ValueError(
        f"Variable {variable.name!r}: distribution {D.__name__} is not discretely "
        "enumerable, so it cannot be a free (queried/latent) variable under belief "
        "propagation. Observe it as evidence, or use a discrete distribution."
    )


class ParametricPotential(ParametricFactor):
    """Undirected, energy-based factor over a symmetric ``scope``.

    Parameters
    ----------
    scope : list of Variable
        The clique of variables the energy ranges over (>= 1). These are the
        factor-graph edges of this potential.
    parametrization : dict[str, nn.Module]
        Maps energy-parameter names to modules that produce them from the
        conditioning inputs (or, when ``conditioning`` is empty, from no inputs —
        e.g. a :class:`~torch_concepts.nn.LearnablePrior`). Subclasses fix the
        parameter names they expect (e.g. :class:`TabularPotential` uses
        ``"table"``).
    conditioning : list of Variable, optional
        Observed input variables the energy depends on (CRF conditioning). Not
        part of ``scope``. Default: none (a plain MRF potential).
    name : str, optional
        Factor-graph key. Defaults to ``"phi(<scope names>)"``.
    aggregate : callable or dict, optional
        As in :class:`ParametricFactor`; aggregates the conditioning inputs.
    """

    def __init__(
        self,
        scope: List[Variable],
        parametrization: Union[nn.Module, Dict[str, nn.Module]],
        conditioning: Optional[List[Variable]] = None,
        name: Optional[str] = None,
        aggregate=None,
    ) -> None:
        if not isinstance(scope, (list, tuple)) or not scope:
            raise ValueError("ParametricPotential: `scope` must be a non-empty list of Variables.")
        for v in scope:
            if not isinstance(v, Variable):
                raise TypeError(
                    f"ParametricPotential: every scope entry must be a Variable, "
                    f"got {type(v).__name__}."
                )
        self._scope: List[Variable] = list(scope)
        self._conditioning: List[Variable] = list(conditioning) if conditioning else []
        # The conditioning inputs are exactly the aggregation inputs, so expose
        # them as ``parents`` to reuse ParametricFactor's aggregation machinery.
        self.parents: List[Variable] = list(self._conditioning)
        self._name: str = name if name is not None else self._default_name()

        parametrization = self._normalize_parametrization(parametrization)
        super().__init__(parametrization=parametrization, aggregate=aggregate)

    def _default_name(self) -> str:
        return f"phi({','.join(v.name for v in self._scope)})"

    def _normalize_parametrization(
        self, parametrization: Union[nn.Module, Dict[str, nn.Module]]
    ) -> Dict[str, nn.Module]:
        """Subclasses may override to wrap a bare ``nn.Module`` under their key."""
        if isinstance(parametrization, dict):
            return parametrization
        raise TypeError(
            f"{type(self).__name__}: `parametrization` must be a dict mapping "
            "energy-parameter names to nn.Module instances."
        )

    # ---- unified factor-graph interface (see ParametricFactor) --------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def scope(self) -> List[Variable]:
        return list(self._scope)

    @property
    def conditioning(self) -> List[Variable]:
        """Observed conditioning inputs (not part of :attr:`scope`)."""
        return list(self._conditioning)

    def _resolve_input(
        self, v: Variable, values: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Value for conditioning input ``v`` (exact name, or plate-member slice)."""
        val = values.get(v.name)
        if val is not None:
            return val
        owner = v.plate
        val = values.get(owner.name)
        if val is not None:
            return val[..., owner.column_of(v.name)]
        raise KeyError(
            f"{type(self).__name__}({self.name!r}): no value for conditioning input "
            f"{v.name!r} in keys {sorted(values)}."
        )

    def forward(
        self,
        inputs: Optional[Mapping[str, torch.Tensor]] = None,
        **layer_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Produce the energy parameters from the conditioning inputs.

        With no ``conditioning``, each module is called with no arguments (a
        learnable/fixed prior). Otherwise the conditioning values are aggregated
        (as for a CPD's parents) and fed through each parameter module.
        """
        if not self._conditioning:
            return {pname: mod() for pname, mod in self.parametrization.items()}

        values = inputs or {}
        cond = {v: self._resolve_input(v, values) for v in self._conditioning}
        result: Dict[str, torch.Tensor] = {}
        for pname, mod in self.parametrization.items():
            cat = self._aggregators[pname](cond)
            if isinstance(cat, dict):
                out = mod(**cat, **layer_kwargs)
            else:
                out = mod(cat, **layer_kwargs)
            result[pname] = out
        return result

    @abstractmethod
    def energy(
        self,
        scope_values: Dict[Variable, torch.Tensor],
        conditioning: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Scalar energy ``E(scope ; conditioning)`` of shape ``(batch,)``."""

    def log_potential(
        self,
        assignment: Mapping[Variable, torch.Tensor],
        conditioning: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """``-E(scope ; conditioning)`` at the given assignment (shape ``(batch,)``)."""
        scope_values: Dict[Variable, torch.Tensor] = {}
        for v in self._scope:
            if v in assignment:
                scope_values[v] = assignment[v]
            elif conditioning is not None and v.name in conditioning:
                scope_values[v] = conditioning[v.name]
            else:
                raise KeyError(
                    f"{type(self).__name__}({self.name!r}).log_potential: no value for "
                    f"scope variable {v.name!r}."
                )
        return -self.energy(scope_values, conditioning)


class TabularPotential(ParametricPotential):
    """Discrete log-linear potential: a log-potential *table* over the scope.

    The parametrization (keyed ``"table"``) emits, from the conditioning inputs,
    a tensor of ``prod(cardinalities)`` entries — one log-potential per joint
    discrete assignment of the scope — reshaped to ``(batch, k_1, ..., k_d)``.
    With no conditioning, pass a prior producing that flat vector (e.g.
    ``{"table": LearnablePrior(prod_of_cardinalities)}``). This is the family
    :class:`BeliefPropagation` consumes directly (no per-cell enumeration).

    Every scope variable must be discretely enumerable (see
    :func:`enumerable_cardinality`).
    """

    #: Guard against constructing an astronomically large table by accident.
    MAX_TABLE_SIZE: int = 1_000_000

    def __init__(
        self,
        scope: List[Variable],
        parametrization: Union[nn.Module, Dict[str, nn.Module]],
        conditioning: Optional[List[Variable]] = None,
        name: Optional[str] = None,
        aggregate=None,
    ) -> None:
        self._cardinalities: List[int] = [enumerable_cardinality(v) for v in scope]
        self._table_size: int = int(math.prod(self._cardinalities))
        if self._table_size > self.MAX_TABLE_SIZE:
            raise ValueError(
                f"TabularPotential({name or 'phi'}): table has {self._table_size} entries "
                f"(cardinalities {self._cardinalities}), exceeding MAX_TABLE_SIZE="
                f"{self.MAX_TABLE_SIZE}. Split the clique or use a smaller scope."
            )
        super().__init__(scope, parametrization, conditioning, name, aggregate)

    def _normalize_parametrization(self, parametrization):
        if isinstance(parametrization, nn.Module):
            return {"table": parametrization}
        if isinstance(parametrization, dict):
            if set(parametrization) != {"table"}:
                raise ValueError(
                    "TabularPotential: parametrization dict must have exactly the key "
                    f"'table', got {sorted(parametrization)}."
                )
            return parametrization
        raise TypeError(
            "TabularPotential: `parametrization` must be an nn.Module or a "
            "{'table': nn.Module} dict."
        )

    @property
    def cardinalities(self) -> List[int]:
        """Per-scope-variable number of discrete states (table axis sizes)."""
        return list(self._cardinalities)

    def log_potential_table(
        self,
        conditioning: Optional[Mapping[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """The log-potential table ``(batch, k_1, ..., k_d)`` (axis order == scope order).

        Consumed directly by :class:`BeliefPropagation` (the fast path — no
        per-assignment enumeration).
        """
        params = self.forward(conditioning)
        table = params["table"]
        if table.dim() == 1:
            # Unconditional prior produced a batch-less (table_size,) vector.
            B = batch_size if batch_size is not None else 1
            table = table.unsqueeze(0).expand(B, -1)
        return table.reshape(table.shape[0], *self._cardinalities)

    def _state_index(self, v: Variable, value: torch.Tensor) -> torch.Tensor:
        """Decode a scope variable's value to its discrete state index ``(batch,)``."""
        card = enumerable_cardinality(v)
        if card == 2 and v.size == 1:
            return value.reshape(value.shape[0]).round().long()
        return value.reshape(value.shape[0], v.size).argmax(dim=-1)

    def energy(
        self,
        scope_values: Dict[Variable, torch.Tensor],
        conditioning: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """``-`` the table entry at the assignment (so ``log_potential`` = table entry)."""
        table = self.log_potential_table(conditioning)
        B = table.shape[0]
        idx = [torch.arange(B, device=table.device)]
        idx += [self._state_index(v, scope_values[v]) for v in self._scope]
        gathered = table[tuple(idx)]
        return -gathered
