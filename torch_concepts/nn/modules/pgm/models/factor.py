"""
Abstract class for PGM factors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, List

import torch
import torch.nn as nn
from .variable import Variable


# Known PyC parameter-name combinations
_PYC_PARAM_SETS = [
    {'concepts'},
    {'embeddings'},
    {'concepts', 'embeddings'},
]


def _cat_parents(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Aggregate parent values by flattening each and concatenating."""
    flat = []
    for v in inputs.values():
        v = v.float() if not v.is_floating_point() else v
        # Ensure at least 2 dims: (batch, event).
        while v.dim() < 2:
            v = v.unsqueeze(-1)
        flat.append(v.flatten(start_dim=1))
    return torch.cat(flat, dim=-1)

def _default_aggregate(input: Dict[Variable, torch.Tensor]) -> torch.Tensor:
    """Default parent-value aggregation: flatten each parent's event dims, concatenate.

    Each value in ``parent_values`` has shape ``(*batch, *event_shape)``.  All
    event dimensions (dim ≥ 1) are flattened to produce shape
    ``(*batch, sum_of_flat_event_sizes)``.
    """
    return _cat_parents(input)


class ParametricFactor(nn.Module, ABC):
    """Abstract class for factors parameterised by torch.nn.Module.

    Concrete factor types (directed: :class:`ParametricCPD`; undirected:
    ``ParametricPotential``) must subclass this and implement :meth:`forward`.

    Subclasses call ``super().__init__(parametrization, aggregate)`` to store:

    - ``self.parametrization`` — an ``nn.ModuleDict`` mapping parameter names
      to ``nn.Module`` instances.
    - ``self.aggregate`` — a callable that collapses a
      ``Dict[str, Tensor]`` of parent values into a single input tensor.
      Defaults to :func:`_default_aggregate` (flatten + concatenate).
    """

    def __init__(
        self,
        parametrization: nn.ModuleDict,
        aggregate: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
    ):
        super().__init__()
        self.parametrization = parametrization
        self.aggregate: Callable[[Dict[str, torch.Tensor]], torch.Tensor] = (
            aggregate if aggregate is not None else _default_aggregate
        )

    @abstractmethod
    def forward(
        self,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the factor output given its input variable values.

        Subclasses define the precise signature and semantics:

        - :class:`ParametricCPD` accepts ``parent_values`` and returns a
          named distribution-parameter dict (e.g. ``{"probs": ...}``).
        - A future ``ParametricPotential`` will accept clique variable values
          and return a log-potential tensor.
        """