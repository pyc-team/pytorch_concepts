"""Abstract base for PGM factors."""
from __future__ import annotations

from pyro.nn import PyroModule


class ParametricFactor(PyroModule):
    """Abstract base — not directly instantiable (§2.1)."""

    def __init__(self, *args, **kwargs):
        if type(self) is ParametricFactor:
            raise NotImplementedError(
                "ParametricFactor is abstract. Use ParametricCPD (or another concrete subclass)."
            )
        super().__init__()
