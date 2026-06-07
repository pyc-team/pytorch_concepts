"""BayesianNetwork: a directed factor graph wiring a list of ``Variable``s to a list of ``ParametricCPD``s."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist

from .cpd import ParametricCPD
from .probabilistic_model import ProbabilisticModel
from .utils import build_distribution, build_relaxed_distribution
from .variable import Variable

class BayesianNetwork(ProbabilisticModel):
    """Directed factor graph wiring a list of ``Variable``s to a list of
    ``ParametricCPD``s.

    Validates the structure (one factor per variable, no duplicate names,
    every parent reference resolves, DAG only), runs a topological sort,
    and exposes itself as a Pyro stochastic function: calling ``pgm(data)``
    emits one ``pyro.sample`` site per variable.

    Amortised variational guides are registered on this network by
    :class:`~torch_concepts.nn.modules.pgm.inference.variational.VariationalInference`
    at construction time.  Once registered, ``pgm.parameters()`` covers both
    the prior CPDs and the guides.
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

        # ---- topological order ---------------------------------------------
        self.sorted_variables: List[Variable] = self._topological_sort()

        # ---- guides (optional) ---------------------------------------------
        # Guide modules are stored here so pgm.parameters() includes them.
        # The latent/conditioning contract lives on the inference engine, not here.
        self.guides: nn.ModuleDict = nn.ModuleDict()

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

    # ----------------------------------------------------------- guides

    @property
    def has_guides(self) -> bool:
        """Whether any guide modules have been registered on this PGM."""
        return len(self.guides) > 0

    # ----------------------------------------------------------- guide forward
    def guide(
        self,
        data: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
        latent_names: List[str],
    ) -> None:
        """Pyro stochastic function for the variational posterior.

        For each name in ``latent_names``, retrieves the corresponding guide
        ``ParametricCPD`` from ``self.guides``, gathers the required parent
        values from ``data`` (using the CPD’s own ``parents`` list), calls the
        CPD to obtain raw distribution parameters, builds the reparameterised
        surrogate distribution, and emits the ``pyro.sample`` site.

        The latent contract (``latent_names``) is owned by the inference engine
        and passed in at call time; the PGM only stores the guide CPDs.
        """
        B = next(iter(data.values())).shape[0] if data else 1
        with pyro.plate("batch", B, dim=-1):
            for name in latent_names:
                cpd: ParametricCPD = self.guides[name]
                if cpd.is_root:
                    # Unconditional guide: no parent inputs.
                    params = cpd(parent_values={})
                    params = {
                        k: v.unsqueeze(0).expand(B, *v.shape)
                        for k, v in params.items()
                    }
                else:
                    # Conditional guide: gather parent values from the data dict.
                    parent_values: Dict[str, torch.Tensor] = {
                        p.name: data[p.name] for p in cpd.parents
                    }
                    params = cpd(parent_values=parent_values)
                q = build_relaxed_distribution(cpd.variable, params, temperature)
                pyro.sample(name, q)
        return None

    # ----------------------------------------------------------- forward
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        batch_size: Optional[int] = None,
        temperature: Optional[torch.Tensor] = None,
    ):
        """Run the PGM as a Pyro stochastic function.

        Iterates variables in topological order. Every variable is emitted as a
        ``pyro.sample`` site. The choice of distribution at each site depends
        on whether the site has an observation in ``data``:

        - With ``obs``: the *exact* declared distribution
          (e.g. ``Bernoulli``, ``Normal``) is used so the observation is
          scored against the true likelihood.
        - Without ``obs``: a reparameterised surrogate is used so gradients
          can flow through the sampled value (straight-through relaxations
          for discrete families; exact distribution for already-reparameterised
          families). ``temperature`` controls the relaxation sharpness;
          defaults to ``1.0`` when not supplied.

        Parameters
        ----------
        data
            Per-variable evidence tensors keyed by variable name.
        batch_size
            Fallback batch size when ``data`` is empty.
        temperature
            Optional temperature tensor for the relaxed surrogates.
        """
        # Resolve batch size.
        if data:
            B = next(iter(data.values())).shape[0]
        elif batch_size is not None:
            B = batch_size
        else:
            raise ValueError(
                "Cannot infer batch dimension: data is empty and "
                "batch_size was not provided."
            )

        if temperature is None:
            temperature = torch.tensor(1.0)

        cache: Dict[str, torch.Tensor] = {}

        # dim=-1 is correct even though your tensor batch axis is 0,
        # because Pyro plate dims refer to distribution batch dimensions from the right,
        # while event dimensions remain on the trailing tensor axes.
        with pyro.plate("batch", B, dim=-1):
            for var in self.sorted_variables:
                name = var.name
                f: ParametricCPD = self.factors[name]

                # --- 1. Build the parameter dict for this variable's distribution.
                if f.is_root:
                    params = f(parent_values={})
                    params = {
                        k: v.unsqueeze(0).expand(B, *v.shape)
                        for k, v in params.items()
                    }
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
                                "neither cached nor in data."
                            )
                    params = f(parent_values=parent_values)

                # --- 2. Build the right distribution depending on observability.
                obs = data.get(name, None)
                if obs is None:
                    d = build_relaxed_distribution(var, params, temperature)
                else:
                    d = build_distribution(var, params)

                # --- 3. Sample.
                value = pyro.sample(name, d, obs=obs)
                cache[name] = value
        return cache
