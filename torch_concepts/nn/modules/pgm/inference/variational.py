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
from typing import Callable, Dict, List, Optional, Type, Union

import pyro.poutine as poutine
import pyro.distributions as dist
import torch

from ..models.bayesian_network import BayesianNetwork
from ..models.guides import _BaseGuide
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
    the corresponding amortised guides are then registered as submodules on
    the PGM. Beyond that setup, the engine is stateless except for the
    temperature schedule. At call time it traces ``pgm.guide`` when guides are
    available, replays ``pgm`` under that trace and conditions on the supplied
    data, and returns the per-site parameters of both the model CPDs
    (``out.params``) and the guides (``out.guide_params``).

    Parameters
    ----------
    pgm : BayesianNetwork
        The probabilistic graphical model. Guide modules are registered on
        this object during engine construction.
    latents
        Declaration of latent (unobservable) variables and the conditioning of
        their guides. If omitted or empty, no guides are registered and the
        engine warns that variational inference may not behave as expected.
        Accepted forms:

        - ``List[str]``: treat these variables as latent; every guide is
          conditioned on all root variables of the PGM.
        - ``Dict[str, List[str]]``: maps each latent variable name to the
          ordered list of (non-latent) PGM variables its guide conditions on.

    default_guides
        Optional override of the per-family guide class mapping.
    n_samples, max_plate_nesting
        Reserved for future use (Monte-Carlo ELBO with multiple samples and
        nested plates respectively).
    initial_temperature, annealing, annealing_rate
        Temperature schedule for the relaxed-discrete sites; see
        :func:`make_temperature_schedule`.
    """

    name = "VariationalInference"

    def __init__(
        self,
        pgm: BayesianNetwork,
        latents: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        default_guides: Optional[Dict[Type[dist.Distribution], Type[_BaseGuide]]] = None,
        n_samples: int = 1,
        max_plate_nesting: int = 1,
        initial_temperature: float = 1.0,
        annealing: Union[str, Callable[[int], float]] = "constant",
        annealing_rate: float = 0.0,
    ):
        super().__init__(pgm)

        self._latent_names, self._guide_conditioning = pgm.setup_guides(
            latents=[] if latents is None else latents,
            default_guides=default_guides,
        )

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
    @property
    def latent_names(self) -> List[str]:
        """Ordered list of latent variable names declared for this engine."""
        return list(self._latent_names)

    @property
    def guide_conditioning(self) -> Dict[str, List[str]]:
        """Per-latent conditioning variable names as declared at construction."""
        return {k: list(v) for k, v in self._guide_conditioning.items()}

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
        """Plain Pyro-compatible guide callable bound to the engine's
        current temperature. Pass this directly to
        ``pyro.infer.Trace_ELBO`` (or any other Pyro inference object) when
        bypassing the engine's ``query``."""
        return self.pgm.guide(data, self.temperature, self._latent_names, self._guide_conditioning)

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
            guide_fn = lambda: self.pgm.guide(data, temperature, self._latent_names, self._guide_conditioning)
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
