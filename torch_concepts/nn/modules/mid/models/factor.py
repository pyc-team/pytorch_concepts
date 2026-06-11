"""
Abstract class for PGM factors.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, List, Set, Union

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
    """Concatenate parent values along the last dim, preserving their shape.

    No flattening or reshaping is performed: every parent value keeps its full
    event shape and the tensors are concatenated along ``dim=-1``. This
    deliberately raises when the values have mismatched non-concatenation
    dimensions (e.g. a matrix-valued parent alongside a vector-valued one) —
    such combinations are ambiguous and must be resolved with a custom
    ``aggregate``. Values are cast to floating point so discrete parents can
    feed float layers, but their shape is left untouched.
    """
    vals = [
        v.float() if not v.is_floating_point() else v
        for v in inputs.values()
    ]
    return torch.cat(vals, dim=-1)


def _module_input_names(mod: nn.Module) -> Set[str]:
    """Return the explicit keyword/positional parameter names of ``mod.forward``."""
    sig = inspect.signature(mod.forward)
    return {
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


class ParametricFactor(nn.Module, ABC):
    """Abstract class for factors parameterised by torch.nn.Module.

    Concrete factor types (directed: :class:`ParametricCPD`; undirected:
    ``ParametricPotential``) must subclass this and implement :meth:`forward`.

    Subclasses call ``super().__init__(parametrization, aggregate)`` to store:

    - ``self.parametrization`` — an ``nn.ModuleDict`` mapping parameter names
      to ``nn.Module`` instances.
    - ``self._module_signatures`` — cached ``forward`` param-name sets, one per
      module in ``parametrization`` (computed once at construction time).
    - ``self._aggregators`` — per-parameter aggregation callables, resolved at
      construction time. Standard modules get :meth:`_standard_aggregate`;
      PyC modules get :meth:`_pyc_aggregate`; user-supplied callables override.

    ``aggregate`` accepts:

    - ``None`` — auto-select :meth:`_standard_aggregate` or :meth:`_pyc_aggregate`
      per module based on its ``forward`` signature.
    - A single ``Callable`` — use it for every parameter module.
    - A ``Dict[str, Callable]`` — use the keyed callable for the matching
      parameter module; auto-select the default for any missing key.
    """

    def __init__(
        self,
        parametrization: Dict[str, nn.Module],
        aggregate: Optional[
            Union[
                Callable,
                Dict[str, Callable],
            ]
        ] = None,
    ):
        super().__init__()

        parametrization = self._initialize_parametrization(parametrization)

        # Cache each module's forward parameter names once at construction time.
        self._module_signatures: Dict[str, Set[str]] = {
            pname: _module_input_names(mod)
            for pname, mod in parametrization.items()
        }

        if aggregate is None:
            self._aggregators: Dict[str, Callable] = {
                pname: self._select_default(pname) for pname in parametrization
            }
        elif callable(aggregate):
            self._aggregators = {pname: aggregate for pname in parametrization}
        elif isinstance(aggregate, dict):
            bad = [k for k, v in aggregate.items() if not callable(v)]
            if bad:
                raise TypeError(
                    f"ParametricFactor: aggregate dict contains non-callable "
                    f"values for keys {bad}."
                )
            self._aggregators = {
                pname: aggregate.get(pname, self._select_default(pname))
                for pname in parametrization
            }
        else:
            raise TypeError(
                "ParametricFactor: `aggregate` must be None, a callable, or a "
                f"dict mapping parameter names to callables, got {type(aggregate).__name__}."
            )
        self.parametrization = parametrization

    def _initialize_parametrization(
        self,
        parametrization: Dict[str, nn.Module],
    ) -> nn.ModuleDict:
        """Normalise ``parametrization`` into an ``nn.ModuleDict``.

        Accepts a plain dict (or an existing ``nn.ModuleDict``) mapping each
        parameter name to a ready ``nn.Module``. Concrete subclasses resolve any
        :class:`LazyConstructor` entries before calling ``super().__init__`` —
        the input/output sizes a lazy layer needs come from the factor's
        variables, which only the subclass knows (see
        :meth:`ParametricCPD._instantiate_lazy`). As a safeguard, an
        already-built ``LazyConstructor`` is unwrapped to its concrete module.
        """
        from ...low.lazy import LazyConstructor

        modules: Dict[str, nn.Module] = {}
        for pname, module in parametrization.items():
            if isinstance(module, LazyConstructor) and module.module is not None:
                module = module.module
            modules[pname] = module
        return nn.ModuleDict(modules)

    # For entries not covered by the user, pick _pyc_aggregate or
    # _standard_aggregate based on the cached module signature.
    def _select_default(self, pname: str) -> Callable:
        return (
            self._pyc_aggregate
            if self._module_signatures[pname] in _PYC_PARAM_SETS
            else self._standard_aggregate
        )

    def _standard_aggregate(
        self,
        inputs: Dict[Variable, torch.Tensor],
    ) -> torch.Tensor:
        """Default aggregation for standard torch modules.

        Concatenates the parent values along the last dim without reshaping
        (see :func:`_cat_parents`).
        """
        return _cat_parents(inputs)

    def _pyc_aggregate(
        self,
        inputs: Dict[Variable, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Default aggregation for PyC-style modules.

        Splits parent values by variable type (``'embedding'`` vs ``'concept'``),
        concatenates each group separately, and returns a dict with keys
        ``'embeddings'`` and/or ``'concepts'`` matching the module's ``forward``
        signature.
        """
        embeddings: Dict[str, torch.Tensor] = {}
        concepts: Dict[str, torch.Tensor] = {}
        for p in self.parents:
            if p.variable_type == "embedding":
                embeddings[p.name] = inputs[p]
            elif p.variable_type == "concept":
                concepts[p.name] = inputs[p]
            else:
                raise ValueError(
                    f"ParametricCPD({self.variable.name!r}): parent "
                    f"{p.name!r} has invalid type {p.variable_type!r}, "
                    "expected 'embedding' or 'concept'."
                )
        return {
            k: _cat_parents(v)
            for k, v in (("embeddings", embeddings), ("concepts", concepts))
            if v
        }

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