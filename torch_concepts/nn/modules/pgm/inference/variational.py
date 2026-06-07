"""Variational inference engine.

Thin orchestrator that configures and runs guides registered on the
:class:`~torch_concepts.nn.modules.pgm.models.bayesian_network.BayesianNetwork`.
Both the prior CPDs and the variational guides live on the PGM, so a single
``pgm.parameters()`` collects everything an optimiser needs.

Contract (different from the other engines!):

- The set of latent variables is declared when constructing this inference
    engine; the corresponding guide modules are registered on the PGM. The
    latent/conditioning contract itself lives on the engine, not the model.
- At call time pass all variables in ``query``: observed variables carry a
  tensor value, latent variables are absent or mapped to ``None``.
  ``evidence`` is still accepted and merged (``query`` values take priority).
- Querying with some non-latent variables omitted is allowed but unsupported:
  the engine emits a yellow warning and gives no guarantees on the result.
"""
from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional, Union

import pyro.poutine as poutine
import torch
import torch.nn as nn

from ..models.bayesian_network import BayesianNetwork
from ..models.cpd import ParametricCPD
from .base import (
    BaseInference,
    make_temperature_schedule,
    trace_to_params,
)
from .outputs import InferenceOutput


_YELLOW_START = "\033[33m"
_YELLOW_END = "\033[0m"


def _yellow_notice(msg: str) -> None:
    """Print a yellow notice on ``stderr``. Always shown — bypasses the
    standard ``warnings`` machinery so a global ``filterwarnings('ignore')``
    cannot hide it."""
    print(f"{_YELLOW_START}{msg}{_YELLOW_END}", file=sys.stderr, flush=True)


class VariationalInference(BaseInference):
    """Variational inference engine.

    The engine declares which PGM variables are latent through ``latents=``;
    the corresponding amortised guide CPDs are then registered as submodules on
    the PGM. Beyond that setup, the engine is stateless except for the
    temperature schedule. At call time it traces ``pgm.guide`` when guides are
    available, replays ``pgm`` under that trace and conditions on the supplied
    data, and returns the per-site parameters of both the model CPDs
    (``out.params``) and the guides (``out.guide_params``).

    Parameters
    ----------
    pgm : BayesianNetwork
        The probabilistic graphical model. Guide CPDs are registered on
        this object during engine construction.
    latents
        Declaration of latent (unobservable) variables and their guide CPDs.
        If omitted or empty, no guides are registered and the engine warns
        that variational inference may not behave as expected.

        Accepted form:

        ``Dict[str, ParametricCPD]``
            Maps each latent variable name to a user-provided
            :class:`~torch_concepts.nn.modules.pgm.models.cpd.ParametricCPD`
            that acts as the variational guide for that variable.  The CPD\'s
            ``variable`` must match the named latent, and its ``parents`` must
            all be non-latent variables already registered in the PGM.
            No default guide architectures are provided — the user must supply
            the full ``ParametricCPD`` for each latent.

        Example::

            engine = VariationalInference(
                pgm,
                latents={
                    "z": ParametricCPD(
                        variable=z_var,
                        parametrization=nn.Linear(x_dim, z_param_dim),
                        parents=[x_var],
                    ),
                },
            )

    initial_temperature, annealing, annealing_rate
        Temperature schedule for the relaxed-discrete sites; see
        :func:`make_temperature_schedule`.
    """

    name = "VariationalInference"

    def __init__(
        self,
        pgm: BayesianNetwork,
        latents: Optional[Dict[str, ParametricCPD]] = None,
        n_samples: int = 1,
        max_plate_nesting: int = 1,
        initial_temperature: float = 1.0,
        annealing: Union[str, Callable[[int], float]] = "constant",
        annealing_rate: float = 0.0,
    ):
        super().__init__(pgm)

        # Detect PGM device *before* building guides (only CPD params exist at this point).
        try:
            _pgm_device = next(pgm.parameters()).device
        except StopIteration:
            _pgm_device = torch.device("cpu")

        self._latent_names = self._build_guides(
            pgm, latents=latents or {}
        )

        # Move the newly registered guides to the same device as the PGM.
        pgm.guides.to(_pgm_device)

        self.n_samples = int(n_samples)
        self.max_plate_nesting = int(max_plate_nesting)
        self._warned_latent_evidence = False
        self._schedule = make_temperature_schedule(
            initial_temperature, annealing, annealing_rate
        )
        self._step: int = 0
        # Cached temperature: a buffer so it travels with .to(device) and is
        # mutated in place by ``step`` rather than re-allocated each call.
        self.register_buffer(
            "_temperature",
            torch.tensor(float(self._schedule(self._step))),
        )

        # --- Construction-time yellow notices ---------------------------
        if self._latent_names:
            # 1) Show which variables have been declared latent/unobservable.
            _yellow_notice(
                f"{self.name} Warning:\nDeclared latent (unobservable) variables: "
                f"{self._latent_names}. This inference algorithm expects to be queried "
                "with those variables absent from `query` (or mapped to None)."
                "No guarantees if you query with any of these variables observed, or if you "
                "query with other unobserved variables that are not declared latent."
            )
        else:
            _yellow_notice(
                f"{self.name} Warning:\nYou are using variational inference without "
                "declaring unobservable variables. The engine might not "
                "behave as expected."
            )
        # 2) Remind the caller about the query-primary contract.
        _yellow_notice(
            f"{self.name} Warning:\nContract — pass all variables in `query`: observed "
            "variables with tensor values, latent variables absent or set to None. "
            "`evidence` is still accepted and merged (`query` values take priority)."
        )

    # ------------------------------------------------------------------
    @classmethod
    def _build_guides(
        cls,
        pgm: BayesianNetwork,
        latents: Dict[str, ParametricCPD],
    ) -> List[str]:
        """Validate the provided guide CPDs, register them on ``pgm``, and
        return the ordered list of latent variable names."""
        if not latents:
            pgm.guides = nn.ModuleDict()
            return []

        if not isinstance(latents, dict):
            raise TypeError(
                "VariationalInference: `latents` must be a dict mapping latent "
                "variable names to ParametricCPD instances, "
                f"got {type(latents).__name__}."
            )

        all_var_names = set(pgm.name_to_variable.keys())
        latent_set = set(latents.keys())

        for lat_name, cpd in latents.items():
            # Each key must be a known PGM variable.
            if lat_name not in all_var_names:
                raise ValueError(
                    f"VariationalInference: unknown latent variable name {lat_name!r}."
                )
            # Each value must be a ParametricCPD.
            if not isinstance(cpd, ParametricCPD):
                raise TypeError(
                    f"VariationalInference: guide for {lat_name!r} must be a "
                    f"ParametricCPD, got {type(cpd).__name__}."
                )
            # The CPD's variable must match the declared latent name.
            if cpd.variable.name != lat_name:
                raise ValueError(
                    f"VariationalInference: guide CPD variable name "
                    f"{cpd.variable.name!r} does not match the dict key {lat_name!r}."
                )
            # All parents of the guide CPD must be non-latent PGM variables,
            # and must be the same object instances as those in the PGM.
            for p in cpd.parents:
                if p.name not in all_var_names:
                    raise ValueError(
                        f"VariationalInference: guide for {lat_name!r}: parent "
                        f"{p.name!r} is not a variable of the PGM."
                    )
                if p.name in latent_set:
                    raise ValueError(
                        f"VariationalInference: guide for {lat_name!r} cannot "
                        f"condition on latent variable {p.name!r}."
                    )
                if pgm.name_to_variable[p.name] is not p:
                    raise ValueError(
                        f"VariationalInference: guide for {lat_name!r}: parent "
                        f"{p.name!r} is a different Variable instance than the one "
                        "registered in the PGM. Pass the same object."
                    )

        pgm.guides = nn.ModuleDict(latents)
        return list(latents.keys())

    # ------------------------------------------------------------------
    @property
    def latent_names(self) -> List[str]:
        """Ordered list of latent variable names declared for this engine."""
        return list(self._latent_names)

    @property
    def guide_conditioning(self) -> Dict[str, List[str]]:
        """Per-latent conditioning variable names, derived from each guide CPD\'s
        ``parents`` list."""
        return {
            name: [p.name for p in self.pgm.guides[name].parents]
            for name in self._latent_names
        }

    @property
    def temperature(self) -> torch.Tensor:
        """Current relaxation temperature (a cached buffer)."""
        return self._temperature

    def step(self) -> None:
        """Advance the temperature schedule by one step (in place)."""
        self._step += 1
        self._temperature.fill_(float(self._schedule(self._step)))

    # ------------------------------------------------------------------
    def guide_fn(self, data: Dict[str, torch.Tensor]):
        """Plain Pyro-compatible guide callable bound to the engine\'s
        current temperature. Pass this directly to
        ``pyro.infer.Trace_ELBO`` (or any other Pyro inference object) when
        bypassing the engine\'s ``query``."""
        return self.pgm.guide(data, self.temperature, self._latent_names)

    # ------------------------------------------------------------------
    def _merge_observables(
        self,
        query: Dict[str, Optional[torch.Tensor]],
        evidence: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Merge ``evidence`` and non-None ``query`` entries into a single
        observables dict. Non-None ``query`` values take precedence over
        ``evidence`` values for the same name."""
        data: Dict[str, torch.Tensor] = dict(evidence)
        for name, val in query.items():
            if val is not None:
                data[name] = val
        return data

    # ------------------------------------------------------------------
    def query(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]], None] = None,
        evidence: Dict[str, torch.Tensor] = None,
    ) -> InferenceOutput:
        """Run variational inference and return model and guide parameters.

        **New contract (query-primary):** pass all variables in ``query``.
        Observed variables carry a tensor value; latent variables are absent
        or mapped to ``None``. ``evidence`` is still accepted for backwards
        compatibility and is merged after ``query`` (``query`` values take
        priority on duplicates).
        """
        if query is None:
            query = {}
        if evidence is None:
            evidence = {}
        query = self._normalize_query(query)
        self._validate_containers(query, evidence)

        data = self._merge_observables(query, evidence)

        supplied_latents = [name for name in self._latent_names if name in data]
        if supplied_latents and not self._warned_latent_evidence:
            _yellow_notice(
                f"{self.name} Warning:\nDeclared latent variables were supplied as "
                f"observations: {supplied_latents}. Variational inference is "
                "guaranteed only when declared latent variables match the "
                "unobserved variables."
            )
            self._warned_latent_evidence = True

        # Soft check on the implicit "observables = everything non-latent"
        # contract. We never error: at test time the user is allowed to
        # leave non-latents unobserved, with the understanding that the
        # result is no longer the ELBO.
        non_latent_missing = [
            v.name
            for v in self.pgm.variables
            if v.name not in self._latent_names and v.name not in data
        ]
        if non_latent_missing:
            _yellow_notice(
                f"{self.name} Warning:\nThe following non-latent variables were not "
                f"supplied: {non_latent_missing}. They will be sampled from "
                "the respective distributions; we cannot guarantee the result."
            )

        batch_size = next(iter(data.values())).shape[0] if data else None

        temperature = self.temperature
        if self.pgm.has_guides:
            # we execute all the guides in the model.
            guide_fn = lambda: self.pgm.guide(data, temperature, self._latent_names)
            # We get the trace of the guide execution.
            guide_tr = poutine.trace(guide_fn).get_trace()
            # replay allows to use the samples from the guide q(z|x) 
            # instead of the model samples p(z).
            replayed = poutine.replay(self.pgm, trace=guide_tr)
            # We get the trace of the model execution, which will use the guide samples
            #  for the latent variables.
            model_tr = poutine.trace(replayed).get_trace(
                data, batch_size=batch_size, temperature=temperature
            )
            # Get the guide parameters.
            guide_params = trace_to_params(guide_tr)
        else:
            model_tr = poutine.trace(self.pgm).get_trace(
                data, batch_size=batch_size, temperature=temperature
            )
            guide_params = {}

        return InferenceOutput(
            params=trace_to_params(model_tr),
            guide_params=guide_params,
        )
