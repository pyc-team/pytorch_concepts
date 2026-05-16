"""BayesianNetwork: a directed factor graph wiring a list of ``Variable``s to a list of ``ParametricCPD``s."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

from .cpd import ParametricCPD
from .variable import Variable


class BayesianNetwork(PyroModule):
    """Directed factor graph wiring a list of ``Variable``s to a list of
    ``ParametricCPD``s.

    Validates the structure (one factor per variable, no duplicate names,
    every parent reference resolves, DAG only), runs a topological sort,
    and exposes itself as a Pyro stochastic function: calling ``pgm(data)``
    emits one ``pyro.sample`` site per non-root variable (with
    ``obs=data[name]`` for variables present in ``data``) and one
    ``pyro.deterministic`` site per root.
    """

    def __init__(
        self,
        variables: List[Variable],
        factors: List[ParametricCPD],
    ):
        # Enable module-local Pyro parameters so the PyroModule's parameters
        # are owned by this instance (must be set BEFORE super().__init__()).
        # This allows multiple BayesianNetwork instances to coexist without parameter name clashes.
        pyro.settings.set(module_local_params=True)
        super().__init__()

        # ---- variable tables ------------------------------------------------
        self.variables: List[Variable] = list(variables)
        self.name_to_variable: Dict[str, Variable] = {
            v.name: v for v in self.variables
        }
        if len(self.name_to_variable) != len(self.variables):
            raise ValueError("Duplicate variable names in `variables`.")

        # ---- factors --------------------------------------------------------
        if len(factors) != len(variables):
            raise ValueError(
                f"Got {len(variables)} variables but {len(factors)} factors; "
                "exactly one factor per variable is required."
            )
        self.factors: nn.ModuleDict = nn.ModuleDict()
        for f in factors:
            if f.variable.name in self.factors:
                raise ValueError(f"Duplicate factor for variable {f.variable.name!r}.")
            if f.variable.name not in self.name_to_variable:
                raise ValueError(
                    f"Factor name {f.variable.name!r} has no matching Variable."
                )
            self.factors[f.variable.name] = f

        # Validate parent references against the variables table and dedup.
        self._validate_graph()

        # Validate: distribution=None is only allowed for roots.
        for v in self.variables:
            if (
                v.distribution is None
                and not self.factors[v.name].is_root
            ):
                raise ValueError(
                    f"Variable {v.name!r}: distribution=None is only allowed "
                    "for root nodes. Use dist.Delta for deterministic non-root "
                    "variables."
                )

        # ---- topological order ---------------------------------------------
        self.sorted_variables: List[Variable] = self._topological_sort()

    # ----------------------------------------------------------- validate
    def _validate_graph(self) -> None:
        """Validate every CPD's parents against ``self.variables`` and dedup.

        Each parent must be the exact same object as the one registered in
        ``self.name_to_variable``; a same-name-but-different-instance
        parent is rejected with ``ValueError``. After validation, each CPD's
        ``parents`` list is replaced by an order-preserving deduplicated copy
        to guard against accidental repetition.
        """
        for name, f in self.factors.items():
            for p in f.parents:
                if not isinstance(p, Variable):
                    # Defensive: CPD.__init__ already enforces this, but the
                    # user could mutate `f.parents` between construction and
                    # registration.
                    raise TypeError(
                        f"Factor {name!r}: parent must be a Variable, "
                        f"got {type(p).__name__}."
                    )
                if p.name not in self.name_to_variable:
                    raise ValueError(
                        f"Factor {name!r}: parent {p.name!r} not in variables list."
                    )
                if self.name_to_variable[p.name] is not p:
                    raise ValueError(
                        f"Factor {name!r}: parent {p.name!r} is a different "
                        "Variable instance than the one registered in "
                        "`variables`. Pass the same object."
                    )
            # Order-preserving dedup by identity.
            f.parents = list({id(p): p for p in f.parents}.values())

    # ----------------------------------------------------------- topo sort
    def _topological_sort(self) -> List[Variable]:
        indeg: Dict[str, int] = {v.name: 0 for v in self.variables}
        children: Dict[str, List[str]] = defaultdict(list)
        for name, f in self.factors.items():
            for p in f.parents:
                indeg[name] += 1
                children[p.name].append(name)

        queue = deque([n for n, d in indeg.items() if d == 0])
        out: List[Variable] = []
        while queue:
            n = queue.popleft()
            out.append(self.name_to_variable[n])
            for c in children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(out) != len(self.variables):
            raise ValueError(
                "BayesianNetwork: variables/factors form a cycle; the graph "
                "must be a DAG."
            )
        return out

    # ----------------------------------------------------------- distribution
    @staticmethod
    def _build_distribution(var: Variable, params: Dict[str, torch.Tensor]):
        """Build a Pyro distribution for a given variable and its parameters."""
        D = var.distribution
        if D is dist.Bernoulli:
            d = D(**params, **var.dist_kwargs)
            if var.size > 1:
                d = d.to_event(1)
            return d
        if D is dist.Normal:
            d = D(**params, **var.dist_kwargs)
            if var.size > 1:
                d = d.to_event(1)
            return d
        if D is dist.Delta:
            d = dist.Delta(**params, **var.dist_kwargs)
            if var.size > 1:
                d = d.to_event(1)
            return d
        # OneHotCategorical / Categorical / MultivariateNormal already carry the
        # right event shape from their K-vector / scale_tril parameter.
        return D(**params, **var.dist_kwargs)

    # ----------------------------------------------------------- forward
    def forward(self, data: Dict[str, torch.Tensor]):
        """Run the PGM as a Pyro stochastic function.

        Iterates variables in topological order; for each non-root variable,
        evaluates its CPD on cached parent values and registers a
        ``pyro.sample`` site with ``obs=data[name]`` if ``name`` is in
        ``data`` and ``None`` otherwise. Roots must appear in ``data`` and
        are recorded as ``pyro.deterministic`` sites. Returns a dict mapping
        every variable name to its propagated value.
        """
        cache: Dict[str, torch.Tensor] = {}
        for var in self.sorted_variables:
            name = var.name
            f: ParametricCPD = self.factors[name]
            if f.is_root:
                if name not in data:
                    raise ValueError(
                        f"BayesianNetwork.forward: root variable {name!r} "
                        "must appear in data."
                    )
                value = f(evidence_value=data[name])
                pyro.deterministic(name, value, event_dim=1)
                cache[name] = value
            else:
                parent_values: Dict[str, torch.Tensor] = {}
                for p in f.parents:
                    if p.name in cache:
                        parent_values[p.name] = cache[p.name]
                    elif p.name in data:
                        parent_values[p.name] = data[p.name]
                    else:
                        raise ValueError(
                            f"forward: parent {p.name!r} of {name!r} is "
                            "neither cached nor in data — graph order issue."
                        )
                params = f(parent_values=parent_values)
                d = self._build_distribution(var, params)
                obs = data.get(name, None)
                value = pyro.sample(name, d, obs=obs)
                cache[name] = value
        return cache


# Backwards-compatible alias.
ProbabilisticModel = BayesianNetwork
