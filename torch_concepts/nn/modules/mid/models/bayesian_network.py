"""
BayesianNetwork: a directed factor graph wiring a list of ``Variable``s to a list of ``ParametricCPD``s.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .cpd import ParametricCPD
from .probabilistic_model import ProbabilisticModel
from .variable import Variable

class BayesianNetwork(ProbabilisticModel):
    """Directed factor graph wiring a list of ``Variable``s to a list of
    ``ParametricCPD``s.

    Validates the structure (one factor per variable, no duplicate names,
    every parent reference resolves, DAG only), runs a topological sort,
    and stores the graph ready for inference engines.

    Parameters
    ----------
    variables : list of Variable
        All random variables in the graph.
    factors : list of ParametricCPD
        One CPD per variable (in any order).
    """

    def __init__(
        self,
        variables: List[Variable],
        factors: List[ParametricCPD],
    ):
        super().__init__(variables)  # registers self.variables (dict), self.guides

        # ---- factors --------------------------------------------------------
        if len(factors) != len(variables):
            raise ValueError(
                f"Got {len(variables)} variables but {len(factors)} factors; "
                "exactly one factor per variable is required."
            )
        # ``_factors`` maps {variable name: ParametricCPD}; the key is taken from
        # each child ``f.variable.name``.
        # Exposed through the ``factors`` property (the abstract contract).
        self._factors: nn.ModuleDict = nn.ModuleDict()
        for f in factors:
            if f.variable.name in self._factors:
                raise ValueError(f"Duplicate factor for variable {f.variable.name!r}.")
            if f.variable.name not in self.variables:
                raise ValueError(
                    f"Factor name {f.variable.name!r} has no matching Variable."
                )
            self._factors[f.variable.name] = f

        # Validate parent references against the variables table and dedup.
        self._validate_graph()

        # Variables sorted in topological order.
        self.sorted_variables: List[Variable] = self._topological_sort()

        # Cache for topological levels (computed lazily on first access).
        self._levels_cache: Optional[List[List[Variable]]] = None

    @property
    def factors(self) -> nn.ModuleDict:
        """Mapping ``{child variable name: ParametricCPD}`` (one CPD per variable)."""
        return self._factors

    @property
    def levels(self) -> List[List[Variable]]:
        """Variables grouped by topological depth.

        ``levels[d]`` is the list of variables whose longest path from any
        root is exactly ``d``. All variables in ``levels[d]`` are mutually
        independent given ``levels[0..d-1]``, so their CPDs can be evaluated
        in parallel within a level.

        The result is cached after the first call. The DAG is immutable
        after construction, so this is safe.
        """
        if self._levels_cache is not None:
            return self._levels_cache

        depth: Dict[str, int] = {}
        for v in self.sorted_variables:
            parents = self._factors[v.name].parents
            depth[v.name] = 0 if not parents else 1 + max(
                depth[p.plate.name] for p in parents
            )

        groups: Dict[int, List[Variable]] = defaultdict(list)
        for v in self.sorted_variables:
            groups[depth[v.name]].append(v)

        self._levels_cache = [groups[d] for d in sorted(groups)]
        return self._levels_cache

    # ----------------------------------------------------------- validate
    def _validate_graph(self) -> None:
        """Validate every CPD's parents against ``self.variables`` and dedup.

        Each parent must be the exact same object as the one registered in
        ``self.variables``; a same-name-but-different-instance
        parent is rejected with ``ValueError``. After validation, each CPD's
        ``parents`` list is replaced by an order-preserving deduplicated copy
        to guard against accidental repetition.
        """
        for name, f in self._factors.items():
            for p in f.parents:
                if not isinstance(p, Variable):
                    # Defensive: CPD.__init__ already enforces this, but the
                    # user could mutate `f.parents` between construction and
                    # registration.
                    raise TypeError(
                        f"Factor {name!r}: parent must be a Variable, "
                        f"got {type(p).__name__}."
                    )
                plate = getattr(p, "_plate", None)
                if plate is not None:
                    # Member handle (``plate.member(name)``): the edge depends on a
                    # single member; validate the plate is registered and owns it.
                    if self.variables.get(plate.name) is not plate:
                        raise ValueError(
                            f"Factor {name!r}: parent {p.name!r} is a member of plate "
                            f"{plate.name!r}, which is not the registered variable. "
                            "Pass the same plate object via `variables`."
                        )
                    if p.name not in plate.members:
                        raise ValueError(
                            f"Factor {name!r}: {plate.name!r} has no member {p.name!r}."
                        )
                    continue
                if p.name not in self.variables:
                    raise ValueError(
                        f"Factor {name!r}: parent {p.name!r} not in variables list."
                    )
                if self.variables[p.name] is not p:
                    raise ValueError(
                        f"Factor {name!r}: parent {p.name!r} is a different "
                        "Variable instance than the one registered in "
                        "`variables`. Pass the same object."
                    )
            # Order-preserving dedup by identity.
            f.parents = list({id(p): p for p in f.parents}.values())

    # ----------------------------------------------------------- topo sort
    def _topological_sort(self) -> List[Variable]:
        indeg: Dict[str, int] = {name: 0 for name in self.variables}
        children: Dict[str, List[str]] = defaultdict(list)
        for name, f in self._factors.items():
            for p in f.parents:
                indeg[name] += 1
                children[p.plate.name].append(name)

        queue = deque([n for n, d in indeg.items() if d == 0])
        out: List[Variable] = []
        while queue:
            n = queue.popleft()
            out.append(self.variables[n])
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
