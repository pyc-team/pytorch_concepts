"""ParametricPotential — undirected, energy-based factor over a clique."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn

from ..distributions import spec_for
from .factor import ParametricFactor
from ..variable import Variable


def enumerable_cardinality(variable: Variable) -> int:
    """Number of discrete states of ``variable`` (for enumeration/BP).

    - Bernoulli-family with ``size == 1`` -> ``2`` (states ``0`` and ``1``).
    - Categorical/OneHot-family -> ``variable.size`` (one state per class).

    Raises ``ValueError`` for any other case (e.g. a size>1 Bernoulli, a Normal,
    or a Delta), which cannot be enumerated by belief propagation. The per-family
    answer comes from ``DistributionSpec.state_count``.
    """
    D = variable.distribution
    spec = spec_for(D, f"Variable {variable.name!r}")
    if spec.is_enumerable:
        card = spec.state_count(variable.size)
        if card is not None:
            return card
        raise ValueError(
            f"Variable {variable.name!r}: a size>1 {D.__name__} is a set of "
            "independent bits, not a single enumerable variable. Model each bit "
            "as its own binary variable, or use a Categorical/OneHotCategorical."
        )
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
            inputs[v] = self.resolve_value(v, conditioning)

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
        scope_values = {v: self.resolve_value(v, inputs) for v in self._scope}
        return self.energy(scope_values, inputs, **layer_kwargs)
