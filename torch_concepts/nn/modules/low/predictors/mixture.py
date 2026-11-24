"""Mixture-of-experts predictors for M-CBE models (De Santis et al., 2026)."""
from typing import List, Sequence

import sympy
import sympytorch
import torch


class MixtureOfExpertsConceptToConcept(torch.nn.Module):
    """Weighted mixture of expert predictors over the same concepts.

    Every expert maps the concepts to the task parameters; the outputs are
    averaged with the selector weights carried by ``embeddings``. A hard
    one-hot selector turns the average into a selection of one expert.

    Args:
        experts: ``M`` modules mapping ``(..., in_concepts)`` to
            ``(..., out_concepts)``.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import MixtureOfExpertsConceptToConcept
        >>> layer = MixtureOfExpertsConceptToConcept(
        ...     [torch.nn.Linear(3, 1), torch.nn.Linear(3, 1)])
        >>> selector = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        >>> layer(concepts=torch.randn(2, 3), embeddings=selector).shape
        torch.Size([2, 1])
    """

    def __init__(self, experts: Sequence[torch.nn.Module]):
        super().__init__()
        self.experts = torch.nn.ModuleList(experts)

    def forward(self, concepts: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Mix the experts' outputs ``(..., M, out)`` by the selector ``(..., M)``."""
        out = torch.stack([expert(concepts) for expert in self.experts], dim=-2)
        return (embeddings.unsqueeze(-1) * out).sum(dim=-2)


def _with_trainable_constants(expr: sympy.Expr, in_exponent: bool = False) -> sympy.Expr:
    # NOTE: sympytorch trains Floats only: constants -> Float (trainable),
    # exponents -> Rational (fixed), as in the paper.
    if expr.is_Number:
        return sympy.nsimplify(expr) if in_exponent else sympy.Float(expr)
    if not expr.args:
        return expr
    if isinstance(expr, sympy.Pow):
        base, exp = expr.args
        return sympy.Pow(_with_trainable_constants(base), _with_trainable_constants(exp, True),
                         evaluate=False)
    return expr.func(*[_with_trainable_constants(a, in_exponent) for a in expr.args],
                     evaluate=False)


class SymbolicConceptToConcept(torch.nn.Module):
    """Symbolic expressions over named concepts, with trainable constants.

    Each expression gives one output column. Numeric constants become
    trainable parameters, while exponents stay fixed.

    Args:
        expressions: One sympy expression (or string) per output.
        in_names: Symbol name of each input concept column, in order.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import SymbolicConceptToConcept
        >>> layer = SymbolicConceptToConcept(["2.0*c1 + c2**2"], ["c1", "c2"])
        >>> layer(torch.tensor([[1.0, 3.0]])).item()
        11.0
    """

    def __init__(self, expressions: Sequence, in_names: List[str]):
        super().__init__()
        self.in_names = list(in_names)
        # NOTE: one SymPyModule per expression, so outputs never share constants.
        self.expressions = torch.nn.ModuleList(
            sympytorch.SymPyModule(expressions=[_with_trainable_constants(sympy.sympify(e))])
            for e in expressions
        )

    def sympy(self) -> List[sympy.Expr]:
        """Current expressions, with the trained constants."""
        return [e.sympy()[0] for e in self.expressions]

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        symbols = {name: concepts[..., i] for i, name in enumerate(self.in_names)}
        # A constant expression has no batch dimension: broadcast it.
        batch = concepts.shape[:-1]
        return torch.cat([e(**symbols).expand(*batch, 1) for e in self.expressions], dim=-1)
