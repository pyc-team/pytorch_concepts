"""Abstract base for PGM factors."""
from __future__ import annotations

from pyro.nn import PyroModule


class ParametricFactor(PyroModule):
    """PGM factor whose parameters are produced by a neural network.

    Subclasses :class:`~pyro.nn.PyroModule`. Concrete factor types (e.g.
    ``ParametricCPD``) subclass this. Direct instantiation is not supported.
    """

    def __init__(self, *args, **kwargs):
        if type(self) is ParametricFactor:
            raise NotImplementedError(
                "ParametricFactor is abstract. Use ParametricCPD (or another concrete subclass)."
            )
        super().__init__()
