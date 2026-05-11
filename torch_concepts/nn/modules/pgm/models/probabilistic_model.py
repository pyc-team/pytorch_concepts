"""ProbabilisticModel — owner of variables and factors."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

from .cpd import ParametricCPD
from .variable import ExogenousVariable, Variable


class ProbabilisticModel(PyroModule):
    """Top-level PGM (§4)."""

    def __init__(
        self,
        variables: List[Variable],
        factors: List[ParametricCPD],
    ):
        # Module-local params (§4.1) — must be set BEFORE super().__init__().
        pyro.settings.set(module_local_params=True)
        super().__init__()

        # ---- variable tables ------------------------------------------------
        self.variables: List[Variable] = list(variables)
        self.concept_to_variable: Dict[str, Variable] = {
            v.concept: v for v in self.variables
        }
        if len(self.concept_to_variable) != len(self.variables):
            raise ValueError("Duplicate variable names in `variables`.")

        # ---- factors --------------------------------------------------------
        if len(factors) != len(variables):
            raise ValueError(
                f"Got {len(variables)} variables but {len(factors)} factors; "
                "exactly one factor per variable is required."
            )
        self.factors: nn.ModuleDict = nn.ModuleDict()
        for f in factors:
            if f.concept in self.factors:
                raise ValueError(f"Duplicate factor for variable {f.concept!r}.")
            if f.concept not in self.concept_to_variable:
                raise ValueError(
                    f"Factor concept {f.concept!r} has no matching Variable."
                )
            self.factors[f.concept] = f

        # Resolve parent strings → Variables; deduplicate; build LazyConstructors.
        self._resolve_and_build()

        # Validate: distribution=None is only allowed for roots.
        for v in self.variables:
            if (
                v.distribution is None
                and not self.factors[v.concept].is_root
            ):
                raise ValueError(
                    f"Variable {v.concept!r}: distribution=None is only allowed "
                    "for root nodes. Use dist.Delta for deterministic non-root "
                    "variables."
                )

        # ---- topological order ---------------------------------------------
        self.sorted_variables: List[Variable] = self._topological_sort()

    # ----------------------------------------------------------- resolve
    def _resolve_and_build(self) -> None:
        """Resolve parent strings, deduplicate, and build LazyConstructors."""
        try:
            from ...low.lazy import LazyConstructor  # type: ignore
        except Exception:  # pragma: no cover
            LazyConstructor = None  # type: ignore

        for name, f in self.factors.items():
            v = self.concept_to_variable[name]
            resolved: List[Variable] = []
            for p in f._raw_parents:
                if isinstance(p, Variable):
                    if p.concept not in self.concept_to_variable:
                        raise ValueError(
                            f"Factor {name!r}: parent {p.concept!r} not in variables list."
                        )
                    resolved.append(self.concept_to_variable[p.concept])
                elif isinstance(p, str):
                    if p not in self.concept_to_variable:
                        raise ValueError(
                            f"Factor {name!r}: parent name {p!r} not in variables list."
                        )
                    resolved.append(self.concept_to_variable[p])
                else:
                    raise TypeError(
                        f"Factor {name!r}: parent must be Variable or str, got {type(p).__name__}."
                    )
            resolved = list({id(p): p for p in resolved}.values())

            if LazyConstructor is not None and isinstance(
                f.parametrization, LazyConstructor
            ):
                from .variable import ConceptVariable

                in_concepts = 0
                in_exogenous = 0
                in_latent = 0  # kept for backwards compat with low.lazy build API
                for pv in resolved:
                    if isinstance(pv, ExogenousVariable):
                        in_exogenous += pv.size
                    elif isinstance(pv, ConceptVariable):
                        in_concepts += pv.size
                    else:
                        in_concepts += pv.size
                f.parametrization = f.parametrization.build(
                    out_concepts=v.size,
                    in_concepts=in_concepts,
                    in_latent=in_latent,
                    in_exogenous=in_exogenous,
                )

            f._bind(v, resolved)

    # ----------------------------------------------------------- topo sort
    def _topological_sort(self) -> List[Variable]:
        indeg: Dict[str, int] = {v.concept: 0 for v in self.variables}
        children: Dict[str, List[str]] = defaultdict(list)
        for name, f in self.factors.items():
            for p in f.parents:
                indeg[name] += 1
                children[p.concept].append(name)

        queue = deque([n for n, d in indeg.items() if d == 0])
        out: List[Variable] = []
        while queue:
            n = queue.popleft()
            out.append(self.concept_to_variable[n])
            for c in children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(out) != len(self.variables):
            raise ValueError(
                "ProbabilisticModel: variables/factors form a cycle; the graph "
                "must be a DAG."
            )
        return out

    # ----------------------------------------------------------- distribution
    @staticmethod
    def _build_distribution(var: Variable, params: Dict[str, torch.Tensor]):
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
        """Run the PGM as a Pyro stochastic function (§4.3).

        Every variable in ``data`` becomes a ``pyro.sample`` site with
        ``obs=data[name]``; remaining variables sample from their prior.
        Roots are emitted via ``pyro.deterministic``.
        """
        cache: Dict[str, torch.Tensor] = {}
        for var in self.sorted_variables:
            name = var.concept
            f: ParametricCPD = self.factors[name]
            if f.is_root:
                if name not in data:
                    raise ValueError(
                        f"ProbabilisticModel.forward: root variable {name!r} "
                        "must appear in data."
                    )
                value = f(evidence_value=data[name])
                pyro.deterministic(name, value, event_dim=1)
                cache[name] = value
            else:
                parent_values = []
                for p in f.parents:
                    if p.concept in cache:
                        parent_values.append(cache[p.concept])
                    elif p.concept in data:
                        parent_values.append(data[p.concept])
                    else:
                        raise ValueError(
                            f"forward: parent {p.concept!r} of {name!r} is "
                            "neither cached nor in data — graph order issue."
                        )
                params = f(parent_values=parent_values)
                d = self._build_distribution(var, params)
                obs = data.get(name, None)
                value = pyro.sample(name, d, obs=obs)
                cache[name] = value
        return cache
