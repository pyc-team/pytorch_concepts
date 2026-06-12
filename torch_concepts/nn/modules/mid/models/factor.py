"""
Abstract class for PGM factors.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, List, Set, Tuple, Union

import torch
import torch.nn as nn
from .variable import Variable


# Known PyC parameter-name combinations
_PYC_PARAM_SETS = [
    {'concepts'},
    {'embeddings'},
    {'concepts', 'embeddings'},
]


def _identity(x: torch.Tensor) -> torch.Tensor:
    """No-op activation: return the module output unchanged."""
    return x


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
    """Return the explicit keyword/positional parameter names of ``mod.forward``.

    A PyC :class:`~torch_concepts.nn.Sequential` forwards its inputs straight to
    its first layer, so its input signature *is* that first layer's.
    """
    from ...low.sequential import Sequential

    while isinstance(mod, Sequential) and len(mod) > 0:
        mod = mod[0]
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

    A user-supplied aggregate is called with a signature that matches the
    parameter module's kind: for a **PyC** module it receives the parent values
    already split by type — ``agg(concepts, embeddings)``, each a
    ``Dict[Variable, Tensor]`` — and must return the ``{'concepts': ...,
    'embeddings': ...}`` dict the module expects; for a **standard** module it
    receives the single ``agg(inputs)`` dict and returns one concatenated
    tensor. See :meth:`_resolve_aggregator`.
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
        activate: Optional[
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

        # Normalise the user input to one entry per parameter (``None`` = use
        # the auto-selected default), then adapt each to the uniform
        # ``inputs -> result`` call site used by the CPD's forward.
        if aggregate is None:
            per_param: Dict[str, Optional[Callable]] = {pname: None for pname in parametrization}
        elif callable(aggregate):
            per_param = {pname: aggregate for pname in parametrization}
        elif isinstance(aggregate, dict):
            bad = [k for k, v in aggregate.items() if not callable(v)]
            if bad:
                raise TypeError(
                    f"ParametricFactor: aggregate dict contains non-callable "
                    f"values for keys {bad}."
                )
            per_param = {pname: aggregate.get(pname) for pname in parametrization}
        else:
            raise TypeError(
                "ParametricFactor: `aggregate` must be None, a callable, or a "
                f"dict mapping parameter names to callables, got {type(aggregate).__name__}."
            )
        self._aggregators: Dict[str, Callable] = {
            pname: self._resolve_aggregator(pname, agg) for pname, agg in per_param.items()
        }

        self._activations: Dict[str, Callable] = self._resolve_per_param(
            activate, parametrization, self._select_default_activation, "activate"
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

    def _is_pyc(self, pname: str) -> bool:
        """Whether the parameter module follows the PyC ``concepts``/``embeddings``
        calling convention (vs. a standard single-tensor module)."""
        return self._module_signatures[pname] in _PYC_PARAM_SETS

    # For entries not covered by the user, pick _pyc_aggregate or
    # _standard_aggregate based on the cached module signature.
    def _select_default(self, pname: str) -> Callable:
        return self._pyc_aggregate if self._is_pyc(pname) else self._standard_aggregate

    def _resolve_aggregator(
        self,
        pname: str,
        user_aggregate: Optional[Callable],
    ) -> Callable:
        """Adapt an aggregate to the uniform ``inputs -> result`` call site.

        ``None`` selects the auto-chosen default (:meth:`_select_default`). A
        user-supplied aggregate is dispatched by the parameter module's kind:

        - **PyC** module — called as ``agg(concepts, embeddings)`` over the
          type-split parent dicts (each ``Dict[Variable, Tensor]``); it returns
          the ``{'concepts': ..., 'embeddings': ...}`` dict the module expects.
        - **standard** module — called as ``agg(inputs)`` over the single
          parent dict; it returns one concatenated tensor.
        """
        if user_aggregate is None:
            return self._select_default(pname)
        if self._is_pyc(pname):
            def aggregator(inputs, _agg=user_aggregate):
                concepts, embeddings = self._split_by_type(inputs)
                return _agg(concepts, embeddings)
            return aggregator
        return user_aggregate

    def _standard_aggregate(
        self,
        inputs: Dict[Variable, torch.Tensor],
    ) -> torch.Tensor:
        """Default aggregation for standard torch modules.

        Concatenates the parent values along the last dim without reshaping
        (see :func:`_cat_parents`).
        """
        return _cat_parents(inputs)

    def _split_by_type(
        self,
        inputs: Dict[Variable, torch.Tensor],
    ) -> Tuple[Dict[Variable, torch.Tensor], Dict[Variable, torch.Tensor]]:
        """Partition parent values into ``(concepts, embeddings)`` dicts by
        variable type, preserving parent order."""
        concepts: Dict[Variable, torch.Tensor] = {}
        embeddings: Dict[Variable, torch.Tensor] = {}
        for p in self.parents:
            if p not in inputs:
                continue
            if p.variable_type == "concept":
                concepts[p] = inputs[p]
            elif p.variable_type == "embedding":
                embeddings[p] = inputs[p]
            else:
                raise ValueError(
                    f"ParametricCPD({self.variable.name!r}): parent "
                    f"{p.name!r} has invalid type {p.variable_type!r}, "
                    "expected 'embedding' or 'concept'."
                )
        return concepts, embeddings

    def _pyc_aggregate(
        self,
        inputs: Dict[Variable, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Default aggregation for PyC-style modules.

        Splits parent values by type (:meth:`_split_by_type`) and concatenates
        each group along the last dim (see :func:`_cat_parents`), returning a
        dict with keys ``'concepts'`` and/or ``'embeddings'`` matching the
        module's ``forward`` signature.
        """
        concepts, embeddings = self._split_by_type(inputs)
        out: Dict[str, torch.Tensor] = {}
        if concepts:
            out["concepts"] = _cat_parents(concepts)
        if embeddings:
            out["embeddings"] = _cat_parents(embeddings)
        return out

    @staticmethod
    def _resolve_per_param(
        spec: Optional[Union[Callable, Dict[str, Callable]]],
        names,
        default_selector: Callable[[str], Callable],
        what: str,
    ) -> Dict[str, Callable]:
        """Resolve a per-parameter callable spec into a ``{name: callable}`` dict.

        Shared by ``aggregate`` and ``activate``: ``None`` falls back to
        ``default_selector`` per name, a single callable applies to all, and a
        dict supplies per-name overrides (auto-default for missing keys).
        """
        if spec is None:
            return {pname: default_selector(pname) for pname in names}
        if callable(spec):
            return {pname: spec for pname in names}
        if isinstance(spec, dict):
            bad = [k for k, v in spec.items() if not callable(v)]
            if bad:
                raise TypeError(
                    f"ParametricFactor: {what} dict contains non-callable "
                    f"values for keys {bad}."
                )
            return {
                pname: spec.get(pname, default_selector(pname))
                for pname in names
            }
        raise TypeError(
            f"ParametricFactor: `{what}` must be None, a callable, or a "
            f"dict mapping parameter names to callables, got {type(spec).__name__}."
        )

    def _select_default_activation(self, pname: str) -> Callable:
        """Default activation for a parameter (identity unless overridden).

        Distribution-aware subclasses (e.g. :class:`ParametricCPD`) override this
        to map each parameter's raw module output into its natural domain.
        """
        return _identity

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