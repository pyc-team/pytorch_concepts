"""Variational inference engine (Pyro backend).

This engine uses Pyro's effect handlers (``poutine.trace`` and
``poutine.replay``) to collect distribution parameters from the generative
model and the variational guide.

The Pyro stochastic functions themselves are inherited from
:class:`PyroBaseInference` (``model_fn`` / ``guide_fn``); this class only
orchestrates the effect-handler plumbing.
"""
from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional, Union

import pyro.poutine as poutine
import torch
import torch.nn as nn

from ...models.bayesian_network import BayesianNetwork
from ...models.cpd import ParametricCPD
from ..utils import make_temperature_schedule
from ..outputs import InferenceOutput
from .base import PyroBaseInference, trace_to_params


_YELLOW_START = "\033[33m"
_YELLOW_END = "\033[0m"


def _yellow_notice(msg: str) -> None:
    """Print a yellow notice on ``stderr``."""
    print(f"{_YELLOW_START}{msg}{_YELLOW_END}", file=sys.stderr, flush=True)


class PyroVariationalInference(PyroBaseInference):
    """Variational inference engine.

    Uses Pyro's effect handlers to trace the generative model and the
    variational guide, intercept sample sites, and collect distribution
    parameters.  The Pyro stochastic functions are provided by
    :class:`PyroBaseInference`.

    Parameters
    ----------
    pgm : BayesianNetwork
        The probabilistic graphical model.
    latents : dict, optional
        Declaration of latent (unobservable) variables and their guide CPDs.
        Maps each latent variable name to a user-provided
        :class:`~torch_concepts.nn.modules.pgm.models.cpd.ParametricCPD` that
        acts as the variational guide for that variable. If omitted or empty,
        no guides are registered and the engine warns that variational
        inference may not behave as expected.
    initial_temperature, annealing, annealing_rate
        Temperature schedule for the relaxed-discrete sites; see
        :func:`~torch_concepts.nn.modules.pgm.inference.base.make_temperature_schedule`.
    """

    name = "PyroVariationalInference"

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

        # Detect PGM device before building guides.
        try:
            _pgm_device = next(pgm.parameters()).device
        except StopIteration:
            _pgm_device = torch.device("cpu")

        self._latent_names = self._build_guides(pgm, latents=latents or {})

        # Move the newly registered guides to the same device as the PGM.
        pgm.guides.to(_pgm_device)

        self.n_samples = int(n_samples)
        self.max_plate_nesting = int(max_plate_nesting)
        self._warned_latent_evidence = False
        self._schedule = make_temperature_schedule(
            initial_temperature, annealing, annealing_rate
        )
        self._step: int = 0
        self.register_buffer(
            "_temperature",
            torch.tensor(float(self._schedule(self._step))),
        )

        # Construction-time yellow notices.
        if self._latent_names:
            _yellow_notice(
                f"{self.name} Warning:\nDeclared latent (unobservable) variables: "
                f"{self._latent_names}. This inference algorithm expects to be queried "
                "with those variables absent from `query` (or mapped to None). "
                "No guarantees if you query with any of these variables observed, or if you "
                "query with other unobserved variables that are not declared latent."
            )
        else:
            _yellow_notice(
                f"{self.name} Warning:\nYou are using variational inference without "
                "declaring unobservable variables. The engine might not "
                "behave as expected."
            )
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
        if not latents:
            pgm.guides = nn.ModuleDict()
            return []

        if not isinstance(latents, dict):
            raise TypeError(
                f"{cls.__name__}: `latents` must be a dict mapping latent "
                "variable names to ParametricCPD instances, "
                f"got {type(latents).__name__}."
            )

        all_var_names = {v.name for v in pgm.variables}
        latent_set = set(latents.keys())

        for lat_name, cpd in latents.items():
            if lat_name not in all_var_names:
                raise ValueError(
                    f"{cls.__name__}: unknown latent variable name {lat_name!r}."
                )
            if not isinstance(cpd, ParametricCPD):
                raise TypeError(
                    f"{cls.__name__}: guide for {lat_name!r} must be a "
                    f"ParametricCPD, got {type(cpd).__name__}."
                )
            if cpd.variable.name != lat_name:
                raise ValueError(
                    f"{cls.__name__}: guide CPD variable name "
                    f"{cpd.variable.name!r} does not match the dict key {lat_name!r}."
                )
            for p in cpd.parents:
                if p.name not in all_var_names:
                    raise ValueError(
                        f"{cls.__name__}: guide for {lat_name!r}: parent "
                        f"{p.name!r} is not a variable of the PGM."
                    )
                if p.name in latent_set:
                    raise ValueError(
                        f"{cls.__name__}: guide for {lat_name!r} cannot "
                        f"condition on latent variable {p.name!r}."
                    )
                if pgm.name_to_variable(p.name) is not p:
                    raise ValueError(
                        f"{cls.__name__}: guide for {lat_name!r}: parent "
                        f"{p.name!r} is a different Variable instance than the one "
                        "registered in the PGM. Pass the same object."
                    )

        pgm.guides = nn.ModuleDict(latents)
        return list(latents.keys())

    # ------------------------------------------------------------------
    @property
    def latent_names(self) -> List[str]:
        return list(self._latent_names)

    @property
    def guide_conditioning(self) -> Dict[str, List[str]]:
        return {
            name: [p.name for p in self.pgm.guides[name].parents]
            for name in self._latent_names
        }

    @property
    def temperature(self) -> torch.Tensor:
        return self._temperature

    def step(self) -> None:
        self._step += 1
        self._temperature.fill_(float(self._schedule(self._step)))

    # ------------------------------------------------------------------
    def _merge_observables(
        self,
        query: Dict[str, Optional[torch.Tensor]],
        evidence: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        data: Dict[str, torch.Tensor] = dict(evidence)
        for name, val in query.items():
            if val is not None:
                data[name] = val
        return data

    def _align_param_keys(
        self,
        params: Dict[str, Dict[str, torch.Tensor]],
        use_guides: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Relabel ``'probs'`` ↔ ``'logits'`` to match each CPD's original key.

        ``dist_to_params`` always extracts ``probs`` from relaxed discrete
        distributions (since Pyro's internal representation is logits and
        Pyro reconstructs distribution objects during tracing). This method
        converts the extracted key back to what the user originally wrote in
        their ``parametrization`` dict, so ``guide_params['Y']`` contains
        ``'logits'`` when the guide CPD was built with ``{'logits': ...}``.
        """
        aligned = {}
        for name, pdict in params.items():
            cpd = (self.pgm.guides[name] if use_guides and name in self.pgm.guides
                   else self.pgm.name_to_factor(name) if not use_guides else None)
            if cpd is None:
                aligned[name] = pdict
                continue
            cpd_keys = set(cpd.parametrization.keys())
            pdict = dict(pdict)  # shallow copy — don't mutate caller's dict
            if "logits" in cpd_keys and "probs" in pdict and "logits" not in pdict:
                probs = pdict.pop("probs")
                pdict["logits"] = torch.log(probs.clamp(min=1e-8)) - torch.log(
                    (1.0 - probs).clamp(min=1e-8)
                )
            elif "probs" in cpd_keys and "logits" in pdict and "probs" not in pdict:
                pdict["probs"] = torch.sigmoid(pdict.pop("logits"))
            aligned[name] = pdict
        return aligned

    # ------------------------------------------------------------------
    def query(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]], None] = None,
        evidence: Dict[str, torch.Tensor] = None,
    ) -> InferenceOutput:
        """Run variational inference and return model and guide parameters."""
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

        temperature = self.temperature
        latent_names = self._latent_names

        if self.pgm.has_guides:
            guide_fn = lambda: self.guide_fn(data, temperature, latent_names)
            guide_tr = poutine.trace(guide_fn).get_trace()
            model_fn = lambda: self.model_fn(data, temperature, latent_names)
            replayed = poutine.replay(model_fn, trace=guide_tr)
            model_tr = poutine.trace(replayed).get_trace()
            guide_params = self._align_param_keys(
                trace_to_params(guide_tr), use_guides=True
            )
        else:
            model_fn = lambda: self.model_fn(data, temperature, latent_names)
            model_tr = poutine.trace(model_fn).get_trace()
            guide_params = {}

        return InferenceOutput(
            params=self._align_param_keys(trace_to_params(model_tr), use_guides=False),
            guide_params=guide_params,
        )
