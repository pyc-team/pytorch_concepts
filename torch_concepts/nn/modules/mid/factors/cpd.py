"""
ParametricCPD — Conditional distribution parameterised by a neural network.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn

from ..distributions import spec_for
from .factor import ParametricFactor
from ..variable import Variable, Delta


class ParametricCPD(ParametricFactor):
    """Conditional distribution parameterised by a neural network
    :math:`p(c_i \\mid \\mathrm{PA}(c_i))`.

    Every parametrization module must already emit a value in the parameter's
    natural domain — no activation is applied on its output. Root CPDs (no
    parents) use the same interface: pass a
    :class:`~torch_concepts.nn.LearnablePrior`, which holds a learnable
    parameter and returns it from a no-argument ``forward()``.

    Parameters
    ----------
    variable : Variable or list of Variable
        The child variable this CPD parametrizes. A **list** builds one
        independent CPD per variable, all sharing the same ``parents``, and
        returns them as a list. A plate (a ``Variable`` with named members) is
        still a single variable: one CPD produces every member at once.
    parametrization : nn.Module or dict[str, nn.Module] or list of dict
        Required — no default is inferred. Accepts:

        - a **dict** ``{param_name: nn.Module}``, explicit and valid for any
          distribution;
        - a single **nn.Module**, shorthand for a family with exactly one
          parameter (Bernoulli → ``probs``, Delta → ``value``). Rejected for a
          family needing several distinct parameters (Normal needs both ``loc``
          and ``scale``) — pass a dict there;
        - a per-CPD **list of dicts**, only when ``variable`` is a list.

        A single dict or module passed alongside a list of variables is
        deep-copied per CPD, so the CPDs do not share weights. Any unbuilt
        :class:`LazyConstructor` entry is instantiated here, sized from the
        parents and from ``variable.param_sizes``.
    parents : list of Variable, optional
        The conditioning set — this *is* the graph structure. Entries may be
        whole variables or plate-member handles (``plate.member('c1')``), in
        which case only that member's column is sliced out of the plate's value.
        Empty or omitted makes this a **root** CPD, whose modules are called
        with no arguments.
    aggregate : callable or dict[str, callable], optional
        How parent values are combined into each module's input; see
        :class:`ParametricFactor`. Defaults to concatenating the parents along
        the last dimension (or splitting them by variable type for a PyC-style
        module).

    Raises
    ------
    ValueError
        If ``parametrization`` is omitted, is an empty dict, uses parameter
        names the variable's distribution family does not accept, or uses the
        single-module shorthand for a multi-parameter family.
    TypeError
        If ``variable`` is neither a ``Variable`` nor a list of them, if a
        parent is not a ``Variable``, or if a parametrization value is not an
        ``nn.Module``.

    Examples
    --------
    A root Bernoulli prior, with the logits held by a learnable parameter:

    >>> prior = ParametricCPD(
    ...     variable=z,
    ...     parametrization={"logits": LearnablePrior(z.size)},
    ... )

    A non-root Bernoulli CPD using the single-module shorthand, which expands
    to ``{'probs': ...}`` automatically:

    >>> cpd = ParametricCPD(
    ...     variable=c,
    ...     parametrization=nn.Linear(4, 1),
    ...     parents=[x],
    ... )
    """

    def __new__(
        cls,
        variable: Union[Variable, List[Variable]],
        parametrization: Optional[
            Union[nn.Module, Dict[str, nn.Module], List[Dict[str, nn.Module]]]
        ] = None,
        parents: Optional[List[Variable]] = None,
        aggregate: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
    ):
        # Single-Variable path: defer to normal __init__. (A variable with named
        # members — a plate — is still a single Variable and takes this path: one
        # CPD produces all its members stacked on the last dimension.)
        if isinstance(variable, Variable):
            return super().__new__(cls)

        # List path: broadcast into one independent CPD per variable.
        if not isinstance(variable, list) or not all(
            isinstance(v, Variable) for v in variable
        ):
            raise TypeError(
                "ParametricCPD: `variable` must be a Variable or a list of "
                f"Variables, got {type(variable).__name__}."
            )

        n = len(variable)
        if isinstance(parametrization, list):
            if len(parametrization) != n:
                raise ValueError(
                    f"ParametricCPD: `parametrization` list length "
                    f"{len(parametrization)} does not match `variable` length {n}."
                )
            if not all(isinstance(p, dict) for p in parametrization):
                raise TypeError(
                    "ParametricCPD: when `parametrization` is a list, every element "
                    "must be a dict mapping parameter names to nn.Module instances."
                )
            modules = list(parametrization)
        elif isinstance(parametrization, dict):
            modules = [copy.deepcopy(parametrization) for _ in range(n)]
        elif isinstance(parametrization, nn.Module):
            modules = [copy.deepcopy(parametrization) for _ in range(n)]
        else:
            raise TypeError(
                "ParametricCPD: `parametrization` must be an nn.Module, a dict "
                "mapping parameter names to nn.Module instances, or a list of such dicts; "
                f"got {type(parametrization).__name__}."
            )

        return [
            cls(v, modules[i], parents=parents, aggregate=aggregate)
            for i, v in enumerate(variable)
        ]

    def __init__(
        self,
        variable: Variable,
        parametrization: Optional[Union[nn.Module, Dict[str, nn.Module]]] = None,
        parents: Optional[List[Variable]] = None,
        aggregate: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
    ):
        # When __new__ returned a list, __init__ is also invoked once per
        # element with a singular Variable, so the list-path is a no-op here.
        if not isinstance(variable, Variable):
            return  # pragma: no cover

        parents = list(parents) if parents else []
        for p in parents:
            if not isinstance(p, Variable):
                raise TypeError(
                    f"ParametricCPD({variable.name!r}): every parent "
                    f"must be a Variable, got {type(p).__name__}."
                )

        D = variable.distribution

        # Get the distribution's parameter spec, which lists the valid parameter names.
        spec = spec_for(D, f"ParametricCPD({variable.name!r})")

        # --- Expand parametrization shorthands ---
        if parametrization is None:
            raise ValueError(
                f"ParametricCPD({variable.name!r}): `parametrization` is required. "
                "There is no default for any node (root or non-root); pass an "
                "nn.Module or a {param_name: nn.Module} dict whose output is already "
                "in the parameter's domain. For a root prior, pass a LearnablePrior, "
                "e.g. parametrization={'logits': LearnablePrior(size)}."
            )
        elif isinstance(parametrization, nn.Module):
            default_pnames = spec.default_params
            if len(default_pnames) > 1:
                raise ValueError(
                    f"ParametricCPD({variable.name!r}): {D.__name__} requires multiple "
                    f"parameters {list(default_pnames)}; pass a dict mapping each "
                    "parameter name to its nn.Module."
                )
            parametrization = {default_pnames[0]: parametrization}
        elif not isinstance(parametrization, dict):
            raise TypeError(
                f"ParametricCPD({variable.name!r}): `parametrization` must be None, an "
                "nn.Module, or a dict mapping parameter names to nn.Module instances; "
                f"got {type(parametrization).__name__}."
            )

        # --- Validate parameter keys against known distribution families ---
        got_keys = set(parametrization.keys())
        if not got_keys:
            raise ValueError(
                f"ParametricCPD({variable.name!r}): `parametrization` dict must not be empty."
            )
        valid_sets = spec.valid_param_sets
        if not any(got_keys == vs for vs in valid_sets):
            options = [sorted(vs) for vs in valid_sets]
            raise ValueError(
                f"ParametricCPD({variable.name!r}): invalid parametrization keys "
                f"{sorted(got_keys)} for {D.__name__}. "
                f"Expected one of: {options}."
            )
        for pname, mod in parametrization.items():
            if not isinstance(mod, nn.Module):
                raise TypeError(
                    f"ParametricCPD({variable.name!r}): parametrization[{pname!r}] must be an "
                    f"nn.Module, got {type(mod).__name__}."
                )

        # Instantiate any LazyConstructor entries now that the parent (input)
        # and target (output) variable sizes are known.
        parametrization = self._instantiate_lazy(parametrization, variable, parents)

        # Store the target variable and parents before super().__init__. These
        # are plain (non-nn.Module) objects, so assigning them prior to
        # nn.Module.__init__ is safe, and forward() reads self.variable.
        self.variable: Variable = variable
        self.parents: List[Variable] = parents

        super().__init__(
            parametrization=parametrization,
            aggregate=aggregate,
        )

    @staticmethod
    def _instantiate_lazy(
        parametrization: Dict[str, nn.Module],
        variable: Variable,
        parents: List[Variable],
    ) -> Dict[str, nn.Module]:
        """Build any unbuilt :class:`LazyConstructor` entries into concrete modules.

        Returns ``parametrization`` unchanged when there is nothing to build —
        the common, eagerly-constructed case skips all the work below.

        A :class:`LazyConstructor` defers module creation until the input/output
        sizes are known; those sizes come from this CPD's variables:

        * ``in_concepts``   — summed size of the ``"concept"`` parents;
        * ``in_embeddings`` — summed size of the ``"embedding"`` parents;
        * ``out_concepts``  — the *per-parameter* output size for the parameter
          this module produces (``variable.param_sizes[param]``), so e.g. a
          ``MultivariateNormal``'s ``scale_tril`` module is sized to its
          ``size * (size + 1) // 2`` Cholesky entries, not just ``size``.

        Input parents carry a multi-dimensional ``shape`` (a ``torch.Size``), but
        the default aggregators flatten every event into a single feature axis
        before a module sees it, so the relevant scalar is ``Variable.size``
        (``== math.prod(shape)``).
        """
        from ...low.lazy import LazyConstructor

        # Fast path: every module is already a concrete layer — nothing to build.
        if not any(
            isinstance(m, LazyConstructor) and m.module is None
            for m in parametrization.values()
        ):
            return parametrization

        in_concepts = sum(p.size for p in parents if p.variable_type == "concept")
        in_embeddings = sum(p.size for p in parents if p.variable_type == "embedding")
        out_sizes = variable.param_sizes

        resolved: Dict[str, nn.Module] = {}
        for pname, module in parametrization.items():
            if isinstance(module, LazyConstructor) and module.module is None:
                # build() instantiates the layer and returns the concrete module.
                module = module.build(
                    out_concepts=out_sizes.get(pname, variable.size),
                    in_concepts=in_concepts or None,
                    in_embeddings=in_embeddings or None,
                )
            resolved[pname] = module
        return resolved

    @property
    def is_root(self) -> bool:
        return len(self.parents) == 0

    def forward(
        self,
        parent_values: Optional[Mapping[str, torch.Tensor]] = None,
        **layer_kwargs,
    ):
        """Compute the distribution parameters by processing the parent values through the
        nn.Module(s).

        Root CPDs are called with no parent values: the parametrization module
        is called with no arguments and its output is returned under the named
        parameter dict for ``self.variable.distribution`` (typically a
        :class:`~torch_concepts.nn.LearnablePrior`).

        Non-root CPDs receive a ``parent_values`` dict. Each parent is resolved
        by :meth:`~ParametricFactor.resolve_value`: keyed either by its exact name
        or by its owning variable's name (a member column is sliced out), so callers
        may pass a superset (e.g. an engine's whole value cache) and unknown keys are
        ignored. The resolved tensors (shape ``(*batch, *parent.shape)``) are
        passed to ``self.aggregate`` (default: flatten event dims and concatenate
        along the last axis) to produce a single input tensor, which is then
        forwarded to each parameter module independently.

        No activation is applied: each module's output is used as the distribution
        parameter verbatim, so the module must already emit a value in the
        parameter's natural domain.

        Returns a ``Dict[str, Tensor]`` ready to pass to the distribution
        constructor (e.g. ``{"probs": ...}`` for Bernoulli,
        ``{"loc": ..., "scale": ...}`` for Normal).
        """
        if self.is_root:
            # Root CPD: no parents expected.
            return {
                pname: mod()
                for pname, mod in self.parametrization.items()
            }

        # Compose the Variable → Tensor dict for the parents (ordered by parent list).
        parent_variable_values = {
            p: self.resolve_value(p, parent_values) for p in self.parents
        }

        # Each parameter module uses its own pre-resolved aggregation function.
        # The aggregated inputs are merged into a *fresh* kwargs dict per
        # parameter: a PyC-style module contributes ``concepts``/``embeddings``
        # keys that a standard module in the same parametrization cannot accept,
        # so they must not leak across iterations.
        result = {}
        for pname, mod in self.parametrization.items():
            cat = self._aggregators[pname](parent_variable_values)
            if isinstance(cat, dict):
                out = mod(**{**layer_kwargs, **cat})
            else:
                out = mod(cat, **layer_kwargs)
            result[pname] = out
        return result

    def root_params(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Root (parent-less) params broadcast to a batch.

        A root CPD's parametrization produces a single batch-less prior; this runs
        it and expands each parameter to ``(batch_size, *param_shape)`` so the
        engine doesn't have to. Only meaningful for root CPDs.
        """
        return {
            key: value.unsqueeze(0).expand(batch_size, *value.shape)
            for key, value in self(parent_values={}).items()
        }

    # ---- unified factor-graph interface (see ParametricFactor) --------------
    @property
    def name(self) -> str:
        """This factor's key: the child variable's name (keeps ``BayesianNetwork``'s
        historical ``{child_name: cpd}`` factor keying)."""
        return self.variable.name

    @property
    def scope(self) -> List[Variable]:
        """``[child, *parents]`` — the variables this CPD wires together."""
        return [self.variable, *self.parents]

    def log_potential(
        self,
        assignment: Mapping["Variable", torch.Tensor],
        conditioning: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """``log p(child | parents)`` evaluated at ``assignment``.

        Directed factors act as ordinary factor-graph factors: the child value
        is scored against the distribution built from its parents' values. Parent
        (and child) values may come from ``assignment`` (keyed by ``Variable``) or
        from ``conditioning`` (keyed by name); the former wins on conflict.
        """
        from ..inference.utils import build_distribution  # local: avoid import cycle

        values: Dict[str, torch.Tensor] = dict(conditioning) if conditioning else {}
        child_value: Optional[torch.Tensor] = None
        for var, val in assignment.items():
            values[var.name] = val
            if var.name == self.variable.name:
                child_value = val
        if child_value is None:
            child_value = values.get(self.variable.name)
        if child_value is None:
            raise KeyError(
                f"ParametricCPD({self.variable.name!r}).log_potential: no value for "
                f"the child variable {self.variable.name!r} in the assignment."
            )

        B = child_value.shape[0]
        params = self.root_params(B) if self.is_root else self(parent_values=values)
        d = build_distribution(self.variable, params)
        param_dtype = next(iter(params.values())).dtype
        child_flat = child_value.reshape(B, self.variable.size).to(param_dtype)
        return d.log_prob(child_flat)

    # ---- member addressing (delegates to Variable; kept for back-compat) -----
    # Slicing a member's column span is a pure function of the variable's own
    # column layout, so the logic lives on ``Variable``. These thin delegators
    # preserve the historical CPD-level API every inference backend already calls.
    def select(
        self, params: Dict[str, torch.Tensor], name: str
    ) -> Dict[str, torch.Tensor]:
        """Distribution params for ``name`` (delegates to :meth:`Variable.select`)."""
        return self.variable.select(params, name)

    def select_value(self, value: torch.Tensor, name: str) -> torch.Tensor:
        """Realised value for ``name`` (delegates to :meth:`Variable.select_value`)."""
        return self.variable.select_value(value, name)

    def clamp_members(
        self, value: torch.Tensor, observed: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Clamp individually-observed members (delegates to
        :meth:`Variable.clamp_members`)."""
        return self.variable.clamp_members(value, observed)

