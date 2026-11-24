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


# NOTE: protected operators, as in the paper's code. Fine-tuning moves the
# constants and the concepts are now predicted, so an argument can leave the
# domain PySR saw. sympytorch cannot override its own functions, hence new names.
_EPS = 1e-6
_pos = sympy.Function("_pos")              # > 0: log and fractional powers
_nonzero = sympy.Function("_nonzero")      # != 0: negative powers (division)
_safe_exp = sympy.Function("_safe_exp")
_safe_tan = sympy.Function("_safe_tan")


def _nonzero_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 0, x.clamp(max=-_EPS), x.clamp(min=_EPS))


_SAFE_FUNCS = {
    _pos: lambda x: x.clamp(min=_EPS),
    _nonzero: _nonzero_torch,
    _safe_exp: lambda x: torch.exp(x.clamp(max=20.0)),
    _safe_tan: lambda x: torch.sin(x) / _nonzero_torch(torch.cos(x)),
}


def _protected(expr: sympy.Expr) -> sympy.Expr:
    expr = expr.replace(sympy.log, lambda a: sympy.log(_pos(a)))
    expr = expr.replace(sympy.exp, _safe_exp).replace(sympy.tan, _safe_tan)
    return expr.replace(
        lambda e: e.is_Pow and (e.exp.is_negative or not e.exp.is_integer),
        lambda e: sympy.Pow((_nonzero if e.exp.is_integer else _pos)(e.base), e.exp),
    )


def _unprotected(expr: sympy.Expr) -> sympy.Expr:
    for guard in (_pos, _nonzero):
        expr = expr.replace(guard, lambda a: a)
    return expr.replace(_safe_exp, sympy.exp).replace(_safe_tan, sympy.tan)


class SymbolicConceptToConcept(torch.nn.Module):
    """Symbolic expressions over named concepts, with trainable constants.

    Each expression gives one output column. Numeric constants become
    trainable parameters, while exponents stay fixed. ``log``, ``exp``,
    ``tan``, divisions and fractional powers are protected (clamped).

    Args:
        expressions: One sympy expression (or string) per output. Every
            number in it is trainable (``2*c1`` too; the ``-1`` of ``c1 - c2``
            too), except exponents; an implicit coefficient (``c1 + c2``) is not.
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
        # NOTE: concept names are always symbols (sympify alone reads E, I, S,
        # N, gamma, ... as sympy constants and functions).
        symbols = {name: sympy.Symbol(name) for name in self.in_names}
        # NOTE: one SymPyModule per expression, so outputs never share constants.
        self.expressions = torch.nn.ModuleList(
            sympytorch.SymPyModule(
                expressions=[_protected(_with_trainable_constants(
                    sympy.sympify(e, locals=symbols)))],
                extra_funcs=_SAFE_FUNCS,
            )
            for e in expressions
        )

    def sympy(self) -> List[sympy.Expr]:
        """Current expressions, with the trained constants."""
        return [_unprotected(e.sympy()[0]) for e in self.expressions]

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        symbols = {name: concepts[..., i] for i, name in enumerate(self.in_names)}
        # A constant expression has no batch dimension: broadcast it.
        batch = concepts.shape[:-1]
        return torch.cat([e(**symbols).expand(*batch, 1) for e in self.expressions], dim=-1)
