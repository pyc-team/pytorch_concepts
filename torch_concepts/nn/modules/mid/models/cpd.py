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
from .variable import Variable, Delta, PARAM_DIM


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

    # ---- member addressing (a plate produces all members; these slice) ------
    # ``forward`` runs once and returns the whole stacked output; the methods
    # below pick out a single member's column span (a view, no copy). They are
    # the only thing that differs between addressing this CPD by its variable
    # name vs by one of its members — every inference backend reuses them.

    def select(
        self, params: Dict[str, torch.Tensor], name: str
    ) -> Dict[str, torch.Tensor]:
        """Distribution params for ``name``: the whole output for this CPD's own
        variable, or a member's column slice."""
        if name == self.variable.name:
            return params
        columns = self.variable.column_of(name)
        return {key: value[..., columns] for key, value in params.items()}

    def select_value(self, value: torch.Tensor, name: str) -> torch.Tensor:
        """Realised value for ``name``: the whole value, or a member's column slice."""
        if name == self.variable.name:
            return value
        return value[..., self.variable.column_of(name)]

    def clamp_members(
        self, value: torch.Tensor, observed: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Clamp individually-observed members to their observed values.

        Used for *partial observation* of a plate: the CPD has produced the whole
        stacked ``value``, and this overwrites the columns of the observed members
        with their observed tensors, leaving the unobserved members at the model's
        value. ``observed`` maps member name -> observed tensor. Returns a new
        tensor (the input is not mutated); a no-op when ``observed`` is empty.
        """
        if not observed:
            return value
        value = value.clone()
        for member, obs in observed.items():
            columns = self.variable.column_of(member)
            slot = value[..., columns]
            value[..., columns] = obs.to(value.dtype).reshape(slot.shape)
        return value
    

