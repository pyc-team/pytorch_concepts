"""ParametricPotential — undirected, energy-based factor over a clique.

A potential contributes ``exp(-E(scope ; conditioning))`` to the joint. Its
``scope`` is a *symmetric* clique of variables (no child/parent asymmetry); an
optional list of ``conditioning`` variables (e.g. an embedding) also feeds the
energy but is observed — those inputs create no undirected edges and are not
marginalized. This is the conditional random field (CRF) setting used by
concept models.

The energy is a **plain scalar function** of the assignment: the parametrization
aggregates the scope values (and any conditioning values) exactly as a
:class:`ParametricCPD` aggregates its parents, and maps them to a single scalar
``E`` per batch element. ``log_potential = -E``. Belief propagation builds the
factor tables it needs by enumerating discrete assignments and evaluating this
energy — the same enumeration path it already uses for CPDs.

Note the modelling consequence: a linear energy over the concatenated inputs is
*additive* (no interaction between scope variables). Use an interaction-capable
module (e.g. an MLP) when you need a genuine coupling between concepts.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.distributions as dist

from .factor import ParametricFactor
from .variable import Variable


# ---------------------------------------------------------------------------
# Discrete-state cardinality (used by the BP engine to enumerate assignments).
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
    """Undirected, energy-based factor over a ``scope``.

    Parameters
    ----------
    scope : list of Variable
        The clique of variables the energy ranges over (>= 1). These are the
        factor-graph edges of this potential.
    parametrization : nn.Module or dict[str, nn.Module]
        The energy module. It receives the aggregated inputs — the scope values
        (and any conditioning values), concatenated along the last dim in
        ``scope`` then ``conditioning`` order — and must return **one scalar per
        batch element** (shape ``(batch,)`` or ``(batch, 1)``). Its input width is
        therefore ``sum(v.size for v in scope) + sum(v.size for v in
        conditioning)``. A bare ``nn.Module`` is wrapped as ``{"energy": module}``.
    conditioning : list of Variable, optional
        Observed input variables the energy also depends on (CRF conditioning).
        Not part of ``scope``. Default: none (a plain MRF potential).
    name : str, optional
        Factor-graph key. Defaults to ``"phi(<scope names>)"``.
    aggregate : callable or dict, optional
        As in :class:`ParametricFactor`; aggregates the inputs before the energy
        module (default: concatenate along the last dim).
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
        # Scope + conditioning are exactly the aggregation inputs, so expose them
        # as ``parents`` to reuse ParametricFactor's aggregation machinery.
        self.parents: List[Variable] = list(self._scope) + list(self._conditioning)
        self._name: str = name if name is not None else self._default_name()

        super().__init__(
            parametrization=self._normalize_parametrization(parametrization),
            aggregate=aggregate,
        )

    def _default_name(self) -> str:
        return f"phi({','.join(v.name for v in self._scope)})"

    def _normalize_parametrization(
        self, parametrization: Union[nn.Module, Dict[str, nn.Module]]
    ) -> Dict[str, nn.Module]:
        if isinstance(parametrization, nn.Module):
            return {"energy": parametrization}
        if isinstance(parametrization, dict):
            if set(parametrization) != {"energy"}:
                raise ValueError(
                    "ParametricPotential: parametrization dict must have exactly the key "
                    f"'energy', got {sorted(parametrization)}."
                )
            return parametrization
        raise TypeError(
            "ParametricPotential: `parametrization` must be an nn.Module or a "
            "{'energy': nn.Module} dict."
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
        """Value for input ``v`` (exact name, or a plate-member column slice)."""
        val = values.get(v.name)
        if val is not None:
            return val
        owner = v.plate
        val = values.get(owner.name)
        if val is not None:
            return val[..., owner.column_of(v.name)]
        raise KeyError(
            f"ParametricPotential({self.name!r}): no value for input {v.name!r} "
            f"in keys {sorted(values)}."
        )

    def energy(
        self,
        scope_values: Mapping[Variable, torch.Tensor],
        conditioning: Optional[Mapping[str, torch.Tensor]] = None,
        **layer_kwargs,
    ) -> torch.Tensor:
        """Scalar energy ``E(scope ; conditioning)`` of shape ``(batch,)``.

        Aggregates the scope values (from ``scope_values``) and conditioning
        values (resolved by name from ``conditioning``) and applies the energy
        module.
        """
        conditioning = conditioning or {}
        inputs: Dict[Variable, torch.Tensor] = {v: scope_values[v] for v in self._scope}
        for v in self._conditioning:
            inputs[v] = self._resolve_input(v, conditioning)

        mod = self.parametrization["energy"]
        cat = self._aggregators["energy"](inputs)
        out = mod(**cat, **layer_kwargs) if isinstance(cat, dict) else mod(cat, **layer_kwargs)
        return out.reshape(out.shape[0])

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
                    f"ParametricPotential({self.name!r}).log_potential: no value for "
                    f"scope variable {v.name!r}."
                )
        return -self.energy(scope_values, conditioning)

    def forward(
        self,
        inputs: Optional[Mapping[str, torch.Tensor]] = None,
        **layer_kwargs,
    ) -> torch.Tensor:
        """Energy for a name-keyed ``inputs`` dict holding all scope (and
        conditioning) values. Returns ``(batch,)``."""
        inputs = inputs or {}
        scope_values = {v: self._resolve_input(v, inputs) for v in self._scope}
        return self.energy(scope_values, inputs, **layer_kwargs)
