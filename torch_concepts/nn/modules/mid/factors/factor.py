"""
Abstract class for PGM factors.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Callable, Dict, Mapping, Optional, List, Set, Tuple, Union

import torch
import torch.nn as nn
from ..variable import Variable
from ...low.lazy import LazyConstructor


# Known PyC parameter-name combinations
_PYC_PARAM_SETS = [
    {'concepts'},
    {'embeddings'},
    {'concepts', 'embeddings'},
]


def _cat_parents(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate parent values along the last dim, preserving their shape.

    No flattening or reshaping is performed: every parent value keeps its full
    event shape and the tensors are concatenated along ``dim=-1``. 

    NOTE: this deliberately raises when the values have mismatched non-concatenation
    dimensions (e.g. a matrix-valued parent alongside a vector-valued one).
    """
    vals = [
        v.float() if not v.is_floating_point() else v
        for v in inputs.values()
    ]
    return torch.cat(vals, dim=-1)


def _module_input_names(mod: nn.Module) -> Set[str]:
    """Return the explicit keyword/positional parameter names of ``mod.forward``."""

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

    **Subclass contract.** Before calling ``super().__init__``, a subclass must
    set ``self.inputs``: the ordered list of variables the aggregation machinery
    feeds to the parametrization modules. For a :class:`ParametricCPD` these are
    the CPD's parents; for a ``ParametricPotential`` they are its ``scope``
    (an undirected factor has no parents, hence the neutral name). This is the
    only attribute the base class reads off the subclass.

    Subclasses call ``super().__init__(parametrization, aggregate)`` to store:

    - ``self.parametrization`` — an ``nn.ModuleDict`` mapping parameter names
      to ``nn.Module`` instances.
    - ``self._module_signatures`` — cached ``forward`` param-name sets, one per
      module in ``parametrization`` (computed once at construction time).
    - ``self._aggregators`` — per-parameter aggregation callables, resolved at
      construction time. Standard modules get :meth:`_standard_aggregate`;
      PyC modules get :meth:`_pyc_aggregate`; user-supplied callables override.

    Parameters
    ----------
    parametrization : dict[str, nn.Module]
        Maps each parameter name to the ``nn.Module`` producing it. Normalised
        into an ``nn.ModuleDict``; an already-built :class:`LazyConstructor` is
        unwrapped to its concrete module. Concrete subclasses resolve any
        *unbuilt* lazy entries before calling ``super().__init__``, since the
        sizes those need come from the factor's own variables.
    aggregate : callable or dict[str, callable], optional
        How the input values are combined into each parameter module's input.

        - ``None`` (the default) — auto-select :meth:`_standard_aggregate` or
          :meth:`_pyc_aggregate` per module, from its ``forward`` signature.
        - a single callable — used for every parameter module.
        - a dict — the keyed callable is used for the matching parameter
          module; any missing key falls back to the auto-selected default.

        A user-supplied aggregate is called with a signature matching the
        parameter module's kind: for a **PyC** module it receives the parent
        values already split by type — ``agg(concepts, embeddings)``, each a
        ``Dict[Variable, Tensor]`` — and must return the ``{'concepts': ...,
        'embeddings': ...}`` dict the module expects; for a **standard** module
        it receives the single ``agg(inputs)`` dict and returns one concatenated
        tensor. See :meth:`_resolve_aggregator`.

    Raises
    ------
    TypeError
        If a subclass reaches ``super().__init__`` without having set
        ``self.inputs``, or if ``aggregate`` is neither ``None``, a callable,
        nor a dict whose values are all callables.
    """

    # Ordered aggregation inputs, set by every concrete subclass before
    # ``super().__init__`` (see the class docstring).
    inputs: List[Variable]

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

        if not hasattr(self, "inputs"):
            raise TypeError(
                f"{type(self).__name__}: subclasses must set `self.inputs` (the "
                "ordered aggregation inputs) before calling super().__init__()."
            )

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

        self.parametrization = parametrization

    def _initialize_parametrization(
        self,
        parametrization: Dict[str, nn.Module],
    ) -> nn.ModuleDict:
        """Create a ``nn.ModuleDict`` from the parametrization.

        Accepts a plain dict (or an existing ``nn.ModuleDict``) mapping each
        parameter name to a ready ``nn.Module``. Concrete subclasses resolve any
        :class:`LazyConstructor` entries before calling ``super().__init__`` —
        the input/output sizes a lazy layer needs come from the factor's
        variables, which only the subclass knows (see
        :meth:`ParametricCPD._instantiate_lazy`). As a safeguard, an
        already-built ``LazyConstructor`` is unwrapped to its concrete module.
        """

        modules: Dict[str, nn.Module] = {}
        for pname, module in parametrization.items():
            if isinstance(module, LazyConstructor) and module.module is not None:
                module = module.module
            modules[pname] = module
        return nn.ModuleDict(modules)

    # ------------------------- Aggregation -------------------------

    def _is_pyc(self, pname: str) -> bool:
        """Whether the parameter module follows the PyC ``concepts``/``embeddings``
        calling convention."""
        return self._module_signatures[pname] in _PYC_PARAM_SETS

    # For entries not covered by the user, pick _pyc_aggregate or
    # _standard_aggregate based on the cached module signature.
    def _select_default(self, pname: str) -> Callable:
        """Select the default aggregation for a parameter module.
        
        If pyc module return _pyc_aggregate, else return _standard_aggregate.
        """
        return self._pyc_aggregate if self._is_pyc(pname) else self._standard_aggregate

    def _resolve_aggregator(
        self,
        pname: str,
        user_aggregate: Optional[Callable],
    ) -> Callable:
        """Adapt an aggregate to the uniform ``inputs -> result`` call site."""
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

        Concatenates the parent values along the last dim without reshaping.
        """
        return _cat_parents(inputs)

    def resolve_value(
        self, v: Variable, values: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Tensor for input ``v`` from a name-keyed ``values`` mapping.

        Looked up by ``v``'s exact name first (so a caller may key by the member
        handle directly), then by its owning plate's name, in which case the
        member's column span is sliced out (a view, no copy). A superset of keys
        is fine — unrelated entries are ignored.
        """
        value = values.get(v.name)
        if value is not None:
            return value
        owner = v.plate
        value = values.get(owner.name)
        if value is not None:
            return value[..., owner.column_of(v.name)]
        raise KeyError(
            f"{type(self).__name__}({self.name!r}): no value for input "
            f"{v.name!r} (owner {owner.name!r}) in keys {sorted(values)}."
        )

    def _split_by_type(
        self,
        inputs: Dict[Variable, torch.Tensor],
    ) -> Tuple[Dict[Variable, torch.Tensor], Dict[Variable, torch.Tensor]]:
        """Partition input values into ``(concepts, embeddings)`` dicts by
        variable type, preserving the declared input order."""
        concepts: Dict[Variable, torch.Tensor] = {}
        embeddings: Dict[Variable, torch.Tensor] = {}
        for p in self.inputs:
            if p not in inputs:
                continue
            if p.variable_type == "concept":
                concepts[p] = inputs[p]
            elif p.variable_type == "embedding":
                embeddings[p] = inputs[p]
            else:
                raise ValueError(
                    f"{type(self).__name__}({self.name!r}): input "
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

    # ------------------------- Abstract methods -------------------------

    @abstractmethod
    def forward(
        self,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the factor output given its input variable values.

        Subclasses define the precise signature and semantics:

        - :class:`ParametricCPD` accepts ``parent_values`` and returns a
          named distribution-parameter dict (e.g. ``{"probs": ...}``).
        - :class:`ParametricPotential` accepts clique variable values
          and returns a named energy-parameter dict.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable key under which a :class:`ProbabilisticModel` registers this factor.

        A :class:`ParametricCPD` uses its child variable's name (so a
        :class:`BayesianNetwork`'s ``factors`` keeps its historical
        ``{child_name: cpd}`` keying); a :class:`ParametricPotential` uses an
        explicit or scope-derived name.
        """

    @property
    @abstractmethod
    def scope(self) -> List[Variable]:
        """The variables this factor touches (its factor-graph edges).

        For a directed CPD this is ``[child, *parents]``; for an undirected
        potential it is the clique of variables the energy ranges over —
        including any that happen to be observed, since a potential has no
        separate notion of a conditioning input.
        """

    @abstractmethod
    def log_potential(
        self,
        assignment: Dict["Variable", torch.Tensor],
    ) -> torch.Tensor:
        """Log score ``log φ_f`` of a joint assignment over :attr:`scope`.

        - :class:`ParametricCPD` returns ``log p(child | parents)`` evaluated at
          the assignment.
        - :class:`ParametricPotential` returns ``-E(scope)``.

        Parameters
        ----------
        assignment : dict[Variable, Tensor]
            Value ``(*leading, *var.shape)`` for **every** variable in
            :attr:`scope` — free ones at the value being scored, observed ones
            at their evidence. Any number of leading (batch-like) dimensions is
            allowed, as long as every entry agrees on them. The assignment must
            be complete; the factor does not fall back to any other source.

        Returns
        -------
        torch.Tensor
            Log-potential of shape ``(*leading,)``.
        """