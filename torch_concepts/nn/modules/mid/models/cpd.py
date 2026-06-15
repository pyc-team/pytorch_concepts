"""
ParametricCPD — Conditional distribution parameterised by a neural network.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

import torch.distributions as dist

from .factor import ParametricFactor
from .variable import Variable, Delta


# ---------------------------------------------------------------------------
# Expected parameter names per distribution family, used to validate and
# dispatch per-parameter module dicts.
# ---------------------------------------------------------------------------
# Maps each distribution family to valid parameter-key sets.
# For families that accept both ``probs`` and ``logits``, both options are listed.
# The user's parametrization dict must match EXACTLY ONE of the listed sets.
_DIST_VALID_PARAM_SETS: Dict[type, List[set]] = {
    dist.Bernoulli:                [{"probs"}, {"logits"}],
    dist.RelaxedBernoulli:         [{"probs"}, {"logits"}],
    dist.Categorical:              [{"probs"}, {"logits"}],
    dist.OneHotCategorical:        [{"probs"}, {"logits"}],
    dist.RelaxedOneHotCategorical: [{"probs"}, {"logits"}],
    dist.Normal:                   [{"loc", "scale"}],
    dist.MultivariateNormal:       [{"loc", "scale_tril"}],
    Delta:                         [{"value"}],
}

# ---------------------------------------------------------------------------
# Default parameter-name list per distribution (used for the nn.Module shorthand).
# ---------------------------------------------------------------------------
# For distributions whose only ambiguity is probs-vs-logits, ``probs`` is the
# default.  For distributions that require *multiple distinct* parameters (e.g.
# Normal needs both ``loc`` and ``scale``), the single-module shorthand is
# rejected at construction time and the user must supply a dict.
_DIST_DEFAULT_PARAM_SET: Dict[type, List[str]] = {
    dist.Bernoulli:                ["probs"],
    dist.RelaxedBernoulli:         ["probs"],
    dist.Categorical:              ["probs"],
    dist.OneHotCategorical:        ["probs"],
    dist.RelaxedOneHotCategorical: ["probs"],
    dist.Normal:                   ["loc", "scale"],
    dist.MultivariateNormal:       ["loc", "scale_tril"],
    Delta:                         ["value"],
}


class ParametricCPD(ParametricFactor):
    """Conditional distribution parameterised by a neural network
    :math:`p(c_i \\mid \\mathrm{PA}(c_i))`.

    ``parametrization`` is **required** (no default is inferred) and accepts two
    forms:

    * **Dict** ``{param_name: nn.Module}`` — explicit, works for any distribution.
    * **Single** ``nn.Module`` — shorthand when the distribution has 
      one parameter (e.g. Bernoulli → ``probs``, Delta → ``value``). Raises if the
      distribution requires multiple distinct parameters (e.g. Normal needs both
      ``loc`` and ``scale``); pass a dict in that case.

    Every module must already emit a value in the parameter's natural domain. 
    Root CPDs (no parents) use the same interface:
    pass a :class:`~torch_concepts.nn.LearnablePrior` as the parametrization,
    which holds a learnable parameter and returns it on ``forward()``.

    Example — root Bernoulli prior (logits parametrized by a LearnablePrior):
    ```
        prior = ParametricCPD(variable=z, parametrization={"logits": LearnablePrior(z.size)})
    ```

    Example — non-root Bernoulli CPD (single-module shorthand):
    ```
        cpd = ParametricCPD(
            variable=c,
            parametrization=nn.Linear(4, 1),       # expanded to {'probs': ...} automatically
            parents=[x],
        )
    ```

    Passing a list of ``Variable`` instances returns a list of independent CPDs
    sharing the same parent list; ``parametrization`` may be a single dict /
    ``nn.Module`` (deep-copied per CPD), or a per-CPD list of dicts.
    """

    def __new__(
        cls,
        variable: Union[Variable, List[Variable]],
        parametrization: Optional[
            Union[nn.Module, Dict[str, nn.Module], List[Dict[str, nn.Module]]]
        ] = None,
        parents: Optional[List[Variable]] = None,
        aggregate: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
        shared_key: Optional[str] = None,
    ):
        # Single-Variable path: defer to normal __init__.
        if isinstance(variable, Variable):
            return super().__new__(cls)

        # List path: broadcast.
        if not isinstance(variable, list) or not all(
            isinstance(v, Variable) for v in variable
        ):
            raise TypeError(
                "ParametricCPD: `variable` must be a Variable or a list of "
                f"Variables, got {type(variable).__name__}."
            )

        # Shared path: one CPD parametrizes all variables jointly (stacked),
        # exposing per-variable facades that slice the shared output.
        if shared_key is not None:
            return _make_shared_cpd(variable, parametrization, parents, aggregate, shared_key)

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
        shared_key: Optional[str] = None,
    ):
        # When __new__ returned a list, __init__ is also invoked once per
        # element with a singular Variable, so the list-path is a no-op here.
        # ``shared_key`` is consumed by __new__ (shared path); on the single
        # path it is meaningless and ignored.
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
            default_pnames = next(
                (pnames for base, pnames in _DIST_DEFAULT_PARAM_SET.items()
                 if issubclass(D, base)),
                None,
            )
            if default_pnames is None or len(default_pnames) > 1:
                pnames_str = default_pnames or "unknown"
                raise ValueError(
                    f"ParametricCPD({variable.name!r}): {D.__name__} requires multiple "
                    f"parameters {pnames_str}; pass a dict mapping each parameter name "
                    "to its nn.Module."
                )
            parametrization = {default_pnames[0]: parametrization}
        elif not isinstance(parametrization, dict):
            raise TypeError(
                f"ParametricCPD({variable.name!r}): `parametrization` must be None, an "
                "nn.Module, or a dict mapping parameter names to nn.Module instances; "
                f"got {type(parametrization).__name__}."
            )

        # --- Validate parameter keys against known distribution families ---
        valid_sets = next(
            (sets for base, sets in _DIST_VALID_PARAM_SETS.items() if issubclass(D, base)),
            None,
        )
        got_keys = set(parametrization.keys())
        if not got_keys:
            raise ValueError(
                f"ParametricCPD({variable.name!r}): `parametrization` dict must not be empty."
            )
        if valid_sets is not None and not any(got_keys == vs for vs in valid_sets):
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
        parent_values: Optional[Dict[str, torch.Tensor]] = None,
        **layer_kwargs,
    ):
        """Compute the distribution parameters by processing the parent values through the
        nn.Module(s).

        Root CPDs are called with no parent values: the parametrization module
        is called with no arguments and its output is returned under the named
        parameter dict for ``self.variable.distribution`` (typically a
        :class:`~torch_concepts.nn.LearnablePrior`).

        Non-root CPDs receive a ``parent_values`` dict mapping each parent name
        to its tensor value (shape ``(*batch, *parent.shape)``). These tensors are
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
        parent_variable_values = {p: parent_values[p.name] for p in self.parents}

        # Each parameter module uses its own pre-resolved aggregation function.
        result = {}
        for pname, mod in self.parametrization.items():
            cat = self._aggregators[pname](parent_variable_values)
            if isinstance(cat, dict):
                layer_kwargs.update(cat)
                out = mod(**layer_kwargs)
            else:
                out = mod(cat, **layer_kwargs)
            result[pname] = out
        return result


# ---------------------------------------------------------------------------
# Shared CPDs: one parametrization producing several homogeneous variables
# stacked, with per-variable facades for individual addressing.
# ---------------------------------------------------------------------------
class _SharedCPD(ParametricCPD):
    """Core CPD of a shared group: parametrizes the stacked group variable and
    caches its forward output for one pass.

    The (potentially heavy) parametrization runs **once per forward pass** no
    matter how many member facades request their slice. The cache is keyed by
    the *identity* of the parent value tensors: within one inference pass every
    facade is called with the same parent tensors, so the first call computes
    and the rest reuse it; a new pass brings fresh tensors and recomputes. Root
    groups (no parents) are not cached — their recompute is trivial.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_parents: Optional[Dict[str, torch.Tensor]] = None
        self._cache_out: Optional[Dict[str, torch.Tensor]] = None

    def forward(self, parent_values=None, **layer_kwargs):
        if self._cache_hit(parent_values, layer_kwargs):
            return self._cache_out
        out = super().forward(parent_values, **layer_kwargs)
        self._cache_parents, self._cache_out = parent_values, out
        return out

    def _cache_hit(self, parent_values, layer_kwargs) -> bool:
        # Only cache the non-root, no-extra-kwargs case; require an exact
        # tensor-identity match on every parent so a new forward pass misses.
        if not parent_values or layer_kwargs or self._cache_parents is None:
            return False
        if parent_values.keys() != self._cache_parents.keys():
            return False
        return all(parent_values[k] is self._cache_parents[k] for k in parent_values)


class _SharedMemberCPD(ParametricFactor):
    """Facade CPD for one member of a shared group.

    Holds no parameters of its own: it delegates to the shared :class:`_SharedCPD`
    ``core`` and returns this member's slice of the cached stacked parameters
    (a *view*, so no memory is duplicated). It quacks like a CPD —
    ``variable``, ``parents``, ``is_root``, ``forward`` — so the graph and every
    inference backend treat it as an ordinary node with no changes.
    """

    def __init__(
        self,
        variable: Variable,
        parents: List[Variable],
        core: _SharedCPD,
        index: int,
        member_size: int,
    ):
        nn.Module.__init__(self)  # deliberately skip ParametricFactor.__init__
        self.variable = variable
        self.parents = list(parents)
        self.core = core  # shared module; registered once (parameters() dedups by identity)
        self._index = index
        self._member_size = member_size

    @property
    def is_root(self) -> bool:
        return len(self.parents) == 0

    def forward(self, parent_values=None, **layer_kwargs):
        stacked = self.core(parent_values, **layer_kwargs)
        start = self._index * self._member_size
        sl = slice(start, start + self._member_size)
        return {pname: value[..., sl] for pname, value in stacked.items()}


def _make_shared_cpd(variables, parametrization, parents, aggregate, shared_key):
    """Build a shared CPD over homogeneous ``variables``; return one facade per member.

    All members must share the same distribution family and size. The core
    parametrizes a synthetic stacked variable of size ``n * member_size``; each
    returned :class:`_SharedMemberCPD` slices out its member. The facades are
    ordinary graph nodes, so the PGM and inference engines need no changes.
    """
    if isinstance(parametrization, list):
        raise TypeError(
            f"ParametricCPD(shared_key={shared_key!r}): a shared group is driven by ONE "
            "parametrization; pass a single nn.Module / dict, not a per-member list."
        )
    dist0, size0, kw0 = variables[0].distribution, variables[0].size, variables[0].dist_kwargs
    for v in variables:
        if v.distribution is not dist0 or v.size != size0:
            raise ValueError(
                f"ParametricCPD(shared_key={shared_key!r}): shared members must be homogeneous "
                f"(same distribution and size); {v.name!r} is {v.distribution.__name__}/size "
                f"{v.size} vs {dist0.__name__}/size {size0}."
            )
    member_size = size0
    group_variable = type(variables[0])(
        shared_key,
        distribution=dist0,
        size=len(variables) * member_size,
        dist_kwargs=copy.deepcopy(kw0),
    )
    core = _SharedCPD(
        group_variable, parametrization, parents=parents,
        aggregate=aggregate,
    )
    return [
        _SharedMemberCPD(member, parents, core, i, member_size)
        for i, member in enumerate(variables)
    ]