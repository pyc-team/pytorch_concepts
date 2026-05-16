"""Ancestral inference engine — single-sample differentiable propagation."""
from __future__ import annotations

from typing import Callable, Dict, List, Union

import pyro.distributions as dist
import torch

from ..models.bayesian_network import BayesianNetwork
from .base import InferenceEngine, make_temperature_schedule
from .deterministic import _align_gt, _teacher_force
from .result import InferenceOutput


def _sample_from(
    var,
    params: Dict[str, torch.Tensor],
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Draw one differentiable sample from ``var``'s CPD: Straight-Through
    relaxed for ``Bernoulli`` / ``OneHotCategorical`` (at the given
    ``temperature``), reparameterised ``rsample`` for ``Normal`` /
    ``MultivariateNormal``, identity for ``Delta``."""
    D = var.distribution
    if D is dist.Bernoulli:
        # Need logits for the ST-relaxed family; recover from probs.
        probs = params["probs"]
        logits = torch.log(probs.clamp(min=1e-8)) - torch.log((1.0 - probs).clamp(min=1e-8))
        relaxed = dist.RelaxedBernoulliStraightThrough(
            temperature=temperature, logits=logits
        )
        return relaxed.rsample()
    if D is dist.OneHotCategorical:
        probs = params["probs"]
        logits = torch.log(probs.clamp(min=1e-8))
        relaxed = dist.RelaxedOneHotCategoricalStraightThrough(
            temperature=temperature, logits=logits
        )
        return relaxed.rsample()
    if D is dist.Categorical:
        raise ValueError(
            "AncestralInference: plain Categorical is rejected as the prior of "
            "an unobserved variable. Declare the variable as OneHotCategorical."
        )
    if D is dist.Normal:
        return dist.Normal(**params).rsample()
    if D is dist.MultivariateNormal:
        return dist.MultivariateNormal(**params).rsample()
    if D is dist.Delta:
        # Sampling is the identity.
        return params["v"]
    raise ValueError(f"Unsupported distribution {D!r}")


class AncestralInference(InferenceEngine):
    """Single-sample ancestral inference engine.

    Walks the PGM in topological order and propagates one differentiable
    sample per non-evidence variable to its children (Straight-Through for
    discrete families, ``rsample`` for Gaussian, identity for ``Delta``).
    When a non-evidence variable also appears in ``data``, its label is
    teacher-forced with per-sample probability ``p_int``. The relaxation
    temperature follows the ``annealing`` schedule (``constant`` /
    ``exponential`` / ``linear`` / callable); call ``engine.step()`` to
    advance it. ``InferenceOutput.model_params`` carries the CPD parameters
    and ``InferenceOutput.samples`` the realised values.
    """

    name = "AncestralInference"

    def __init__(
        self,
        pgm: BayesianNetwork,
        p_int: float = 1.0,
        initial_temperature: float = 1.0,
        annealing: Union[str, Callable[[int], float]] = "constant",
        annealing_rate: float = 0.0,
    ):
        super().__init__(pgm)
        if not 0.0 <= float(p_int) <= 1.0:
            raise ValueError(f"p_int must be in [0, 1], got {p_int!r}.")
        self.p_int = float(p_int)
        self._schedule = make_temperature_schedule(
            initial_temperature, annealing, annealing_rate
        )
        self._step: int = 0

    @property
    def temperature(self) -> torch.Tensor:
        return torch.tensor(float(self._schedule(self._step)))

    def step(self) -> None:
        self._step += 1

    def _run(
        self,
        query: List[str],
        evidence: List[str],
        data: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        out = InferenceOutput()
        cache: Dict[str, torch.Tensor] = {}
        evidence_set = set(evidence)
        query_set = set(query)
        temp = self.temperature

        for var in self.pgm.sorted_variables:
            name = var.name
            f = self.pgm.factors[name]

            if f.is_root:
                value = f(evidence_value=data[name])
                cache[name] = value
                # Roots have no probabilistic params; record as Delta-like for
                # query consistency.
                if name in query_set:
                    out.model_params[name] = {"v": value}
                out.samples[name] = value
                continue

            parent_values = {p.name: cache[p.name] for p in f.parents}
            params = f(parent_values=parent_values)
            sampled = _sample_from(var, params, temp)

            if name in evidence_set:
                propagated = _align_gt(data[name], sampled)
            elif name in data:
                propagated = _teacher_force(sampled, data[name], self.p_int)
            else:
                propagated = sampled

            cache[name] = propagated
            out.samples[name] = propagated
            if name in query_set:
                out.model_params[name] = params

        return out
