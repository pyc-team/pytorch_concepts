from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import pyro.distributions as dist
import torch

from ..models.bayesian_network import BayesianNetwork
from ..models.variable import Variable
from .base import BaseInference, make_temperature_schedule
from .outputs import InferenceOutput


def _bern_sampler(params: Dict[str, torch.Tensor], temperature: torch.Tensor) -> torch.Tensor:
    """Reparameterised sample from a Bernoulli via straight-through relaxation."""
    probs = params["probs"]
    logits = torch.log(probs.clamp(min=1e-8)) - torch.log((1.0 - probs).clamp(min=1e-8))
    return dist.RelaxedBernoulliStraightThrough(temperature=temperature, logits=logits).rsample()


def _ohc_sampler(params: Dict[str, torch.Tensor], temperature: torch.Tensor) -> torch.Tensor:
    """Reparameterised sample from a OneHotCategorical via straight-through relaxation."""
    logits = torch.log(params["probs"].clamp(min=1e-8))
    return dist.RelaxedOneHotCategoricalStraightThrough(temperature=temperature, logits=logits).rsample()


def _cat_sampler_or_reject(params: Dict[str, torch.Tensor], temperature: torch.Tensor) -> torch.Tensor:
    """Always raises: plain Categorical cannot be ancestrally sampled; use OneHotCategorical."""
    raise ValueError(
        "AncestralInference: plain Categorical is rejected as the prior of "
        "an unobserved variable. Declare the variable as OneHotCategorical."
    )


def _normal_sampler(params: Dict[str, torch.Tensor], temperature: torch.Tensor) -> torch.Tensor:
    """Reparameterised sample from a Normal distribution."""
    return dist.Normal(**params).rsample()


def _mvn_sampler(params: Dict[str, torch.Tensor], temperature: torch.Tensor) -> torch.Tensor:
    """Reparameterised sample from a MultivariateNormal distribution."""
    return dist.MultivariateNormal(**params).rsample()


def _delta_sampler(params: Dict[str, torch.Tensor], temperature: torch.Tensor) -> torch.Tensor:
    """Delta 'sample': returns the deterministic value v unchanged."""
    return params["v"]


_DIST_OPS = {
    dist.Bernoulli: ("probs", _bern_sampler),
    dist.OneHotCategorical: ("probs", _ohc_sampler),
    dist.Categorical: ("probs", _cat_sampler_or_reject),
    dist.Normal: ("loc", _normal_sampler),
    dist.MultivariateNormal: ("loc", _mvn_sampler),
    dist.Delta: ("v", _delta_sampler),
}


def _propagated_value(distribution, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the primary parameter tensor used as the deterministic propagated value."""
    try:
        param_name, _ = _DIST_OPS[distribution]
    except KeyError as exc:
        raise ValueError(f"Unsupported distribution {distribution!r}") from exc
    return params[param_name]


def _sample_from(
    variable: Variable,
    params: Dict[str, torch.Tensor],
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Dispatch to the family-specific sampler for the given variable."""
    try:
        _, sampler = _DIST_OPS[variable.distribution]
    except KeyError as exc:
        raise ValueError(f"Unsupported distribution {variable.distribution!r}") from exc
    return sampler(params, temperature)


def _align_gt(gt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Cast and reshape ground-truth tensor to match the dtype and shape of ref."""
    aligned = gt.to(ref.dtype) if gt.dtype != ref.dtype else gt
    if aligned.shape != ref.shape:
        if aligned.dim() == ref.dim() + 1 and aligned.shape[-1] == 1:
            aligned = aligned.squeeze(-1)
        elif aligned.dim() + 1 == ref.dim() and ref.shape[-1] == 1:
            aligned = aligned.unsqueeze(-1)
    return aligned.expand_as(ref) if aligned.shape != ref.shape else aligned


def _teacher_force(nn_value: torch.Tensor, gt: torch.Tensor, p_int: float) -> torch.Tensor:
    """Stochastically replace nn_value with ground truth at rate p_int."""
    aligned = _align_gt(gt, nn_value)
    if p_int >= 1.0:
        return aligned
    if p_int <= 0.0:
        return nn_value
    mask_shape = nn_value.shape[:1] + (1,) * (nn_value.dim() - 1)
    mask = (torch.rand(mask_shape, device=nn_value.device) < p_int).to(nn_value.dtype)
    return mask * aligned + (1.0 - mask) * nn_value


class ForwardInference(BaseInference):
    name = "ForwardInference"

    def __init__(
        self,
        pgm: BayesianNetwork,
        mode: str = "deterministic",
        p_int: float = 1.0,
        initial_temperature: float = 1.0,
        annealing: Union[str, Callable[[int], float]] = "constant",
        annealing_rate: float = 0.0,
    ):
        super().__init__(pgm)
        if mode not in {"deterministic", "ancestral"}:
            raise ValueError(f"mode must be 'deterministic' or 'ancestral', got {mode!r}.")
        if not 0.0 <= float(p_int) <= 1.0:
            raise ValueError(f"p_int must be in [0, 1], got {p_int!r}.")
        self.mode = mode
        self.p_int = float(p_int)
        self._schedule = make_temperature_schedule(initial_temperature, annealing, annealing_rate)
        self._step = 0

    @property
    def temperature(self) -> torch.Tensor:
        return torch.tensor(float(self._schedule(self._step)))

    def step(self) -> None:
        if self.mode == "ancestral":
            self._step += 1

    def query(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        query = self._normalize_query(query)
        self._validate_containers(query, evidence)
        all_tensors = list(evidence.values()) + [v for v in query.values() if v is not None]
        batch_size = all_tensors[0].shape[0] if all_tensors else 1
        out = InferenceOutput()
        cache: Dict[str, torch.Tensor] = {}
        query_names = set(query.keys())
        evidence_names = set(evidence.keys())
        temperature = self.temperature

        for variable in self.pgm.sorted_variables:
            name = variable.name
            cpd = self.pgm.factors[name]

            if cpd.is_root:
                params = cpd(parent_values={})
                params = {key: value.unsqueeze(0).expand(batch_size, *value.shape) for key, value in params.items()}
                value = self._propagate(variable, params, temperature)
            else:
                parent_values = {parent.name: cache[parent.name] for parent in cpd.parents}
                params = cpd(parent_values=parent_values)
                value = self._propagate(variable, params, temperature)

            sampled = self.mode == "ancestral"
            if name in evidence_names:
                propagated = _align_gt(evidence[name], value)
            elif name in query_names and query[name] is not None:
                propagated = _teacher_force(value, query[name], self.p_int)
            else:
                propagated = value

            cache[name] = propagated
            if name in query_names:
                out.params[name] = params
            if sampled and name not in evidence_names:
                out.samples[name] = propagated

        return out

    def _propagate(
        self,
        variable: Variable,
        params: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "deterministic":
            return _propagated_value(variable.distribution, params)
        return _sample_from(variable, params, temperature)
