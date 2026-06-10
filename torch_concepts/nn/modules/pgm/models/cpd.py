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
    dist.Bernoulli:         [{"probs"}, {"logits"}],
    dist.Categorical:       [{"probs"}, {"logits"}],
    dist.OneHotCategorical: [{"probs"}, {"logits"}],
    dist.Normal:            [{"loc", "scale"}],
    dist.MultivariateNormal:[{"loc", "scale_tril"}],
    Delta:                  [{"value"}],
}

class ParametricCPD(ParametricFactor):
    """Conditional distribution parameterised by a neural network
    :math:`p(c_i \\mid \\mathrm{PA}(c_i))`.

    The ``parametrization`` argument must be a dict mapping each distribution
    parameter name to an ``nn.Module``.
    Each module receives the concatenated parent values as input (or no input for 
    CPDs with no parents) and must produce a tensor already in the correct domain for that
    parameter.

    Example — root Bernoulli prior:
    ```
        class ConstantModule(nn.Module):
            def __init__(self, value): 
                super().__init__() 
                self.p = nn.Parameter(value)

            def forward(self): 
                return self.p

        prior = ParametricCPD(
            variable=z,
            parametrization={
                'probs':   ConstantModule(torch.zeros(z.size)),
            },
        )
    ```

    Example — non-root Bernoulli CPD:
    ```
        cpd = ParametricCPD(
            variable=c,
            parametrization={'probs': nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())},
            parents=[x],
        )
    ```
        
    Passing a list of ``Variable`` instances returns a list of independent CPDs
    sharing the same parent list; ``parametrization`` may be a single dict
    (deep-copied per CPD) or a per-CPD list of dicts.
    """

    def __new__(
        cls,
        variable: Union[Variable, List[Variable]],
        parametrization: Union[Dict[str, nn.Module], List[Dict[str, nn.Module]]] = None,
        parents: Optional[List[Variable]] = None,
        aggregate: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
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
        else:
            raise TypeError(
                "ParametricCPD: `parametrization` must be a dict mapping parameter "
                "names to nn.Module instances, or a list of such dicts."
            )

        return [
            cls(v, modules[i], parents=parents, aggregate=aggregate)
            for i, v in enumerate(variable)
        ]

    def __init__(
        self,
        variable: Variable,
        parametrization: Dict[str, nn.Module],
        parents: Optional[List[Variable]] = None,
        aggregate: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
    ):
        # When __new__ returned a list, __init__ is also invoked once per
        # element with a singular Variable, so the list-path is a no-op here.
        if not isinstance(variable, Variable):
            return  # pragma: no cover
        if not isinstance(parametrization, dict):
            raise TypeError(
                f"ParametricCPD({variable.name!r}): `parametrization` must be a dict "
                "mapping parameter names to nn.Module instances."
            )
        D = variable.distribution
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
        parametrization = nn.ModuleDict(parametrization)
        if parents is not None:
            for p in parents:
                if not isinstance(p, Variable):
                    raise TypeError(
                        f"ParametricCPD({variable.name!r}): every parent "
                        f"must be a Variable, got {type(p).__name__}."
                    )

        super().__init__(parametrization=parametrization, aggregate=aggregate)
        # Store the target (child) variable.
        self.variable: Variable = variable
        # Store the ordered list of parent variables whose values are the CPD inputs.
        self.parents: List[Variable] = list(parents) if parents else []

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
        is called with no arguments and its raw scalar output is split into the
        named parameter dict for ``self.variable.distribution``.

        Non-root CPDs receive a ``parent_values`` dict mapping each parent name
        to its tensor value (shape ``(*batch, *parent.shape)``). These tensors are
        passed to ``self.aggregate`` (default: flatten event dims and concatenate
        along the last axis) to produce a single input tensor, which is then
        forwarded to each parameter module independently.

        Returns a ``Dict[str, Tensor]`` ready to pass to the distribution
        constructor (e.g. ``{"probs": ...}`` for Bernoulli,
        ``{"loc": ..., "scale": ...}`` for Normal).

        NOTE: The caller is responsible for ensuring each module output is in the
        correct domain for its parameter.
        """
        if self.is_root:
            # Root CPD: no parents expected.
            return {pname: mod() for pname, mod in self.parametrization.items()}

        # Compose the Variable → Tensor dict for the parents (ordered by parent list).
        parent_variable_values = {p: parent_values[p.name] for p in self.parents}

        # Each parameter module uses its own pre-resolved aggregation function.
        result = {}
        for pname, mod in self.parametrization.items():
            cat = self._aggregators[pname](parent_variable_values)
            if isinstance(cat, dict):
                layer_kwargs.update(cat)
                result[pname] = mod(**layer_kwargs)
            else:
                result[pname] = mod(cat, **layer_kwargs)
        return result