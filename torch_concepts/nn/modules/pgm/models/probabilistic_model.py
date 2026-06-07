"""Abstract base class for probabilistic graphical models."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import torch
from pyro.nn import PyroModule
from pyro.nn.module import _PyroModuleMeta


class _ProbabilisticModelMeta(_PyroModuleMeta, ABCMeta):
    """Combined metaclass that satisfies both PyroModule and ABCMeta constraints."""


class ProbabilisticModel(PyroModule, metaclass=_ProbabilisticModelMeta):
    """Abstract base class for probabilistic graphical models.

    All concrete PGM implementations (e.g. :class:`BayesianNetwork`) must
    inherit from this class and implement :meth:`forward`.
    """

    @abstractmethod
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        batch_size: Optional[int] = None,
        temperature: Optional[torch.Tensor] = None,
    ):
        """Run the PGM as a Pyro stochastic function. Subclasses must implement this."""
        ...
