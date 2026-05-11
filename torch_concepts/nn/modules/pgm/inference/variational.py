"""Variational inference engine — eager guide instantiation per family (§5.4)."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.nn import PyroModule

from ..models.probabilistic_model import ProbabilisticModel
from ..models.variable import ExogenousVariable, Variable, param_dim
from .base import (
    InferenceEngine,
    make_temperature_schedule,
    trace_to_params,
)
from .result import InferenceOutput


# --------------------------------------------------------------------------
# Default per-family guide modules.
#
# Each guide:
# - takes ``variable`` and ``parent_dim`` at construction time;
# - builds a 2-layer MLP (hidden=64) producing ``param_dim(distribution, size)``;
# - on ``forward(data, temperature)``, reads the concatenation of every
#   ExogenousVariable from ``data`` as its conditioning input, applies the
#   family-appropriate output bounding, and emits ``pyro.sample(name, q_dist)``
#   only if ``name`` is not already observed (handled by the engine via
#   ``poutine.condition``).
# --------------------------------------------------------------------------


def _default_mlp(parent_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(parent_dim, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim),
    )


class _BaseGuide(PyroModule):
    def __init__(self, variable: Variable, parent_dim: int):
        super().__init__()
        self.variable = variable
        self.parent_dim = parent_dim
        self.net = _default_mlp(parent_dim, param_dim(variable.distribution, variable.size))


class STBernoulliGuide(_BaseGuide):
    """ST-relaxed Bernoulli guide."""

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        if self.variable.size == 1:
            raw = raw.squeeze(-1)
        probs = torch.sigmoid(raw)
        logits = torch.log(probs.clamp(min=1e-8)) - torch.log((1 - probs).clamp(min=1e-8))
        q = dist.RelaxedBernoulliStraightThrough(temperature=temperature, logits=logits)
        if self.variable.size > 1:
            q = q.to_event(1)
        return pyro.sample(self.variable.concept, q)


class STOneHotGuide(_BaseGuide):
    """ST-relaxed OneHotCategorical guide."""

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        probs = torch.softmax(raw, dim=-1)
        logits = torch.log(probs.clamp(min=1e-8))
        q = dist.RelaxedOneHotCategoricalStraightThrough(
            temperature=temperature, logits=logits
        )
        return pyro.sample(self.variable.concept, q)


class NormalGuide(_BaseGuide):
    """Reparameterised Normal guide."""

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        s = self.variable.size
        loc, scale = raw[..., :s], raw[..., s:]
        if s == 1:
            loc = loc.squeeze(-1)
            scale = scale.squeeze(-1)
        scale = torch.nn.functional.softplus(scale) + 1e-6
        q = dist.Normal(loc=loc, scale=scale)
        if s > 1:
            q = q.to_event(1)
        return pyro.sample(self.variable.concept, q)


class MVNGuide(_BaseGuide):
    """Reparameterised MultivariateNormal guide."""

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        s = self.variable.size
        loc = raw[..., :s]
        tril_flat = raw[..., s:]
        tril = torch.zeros(*raw.shape[:-1], s, s, device=raw.device, dtype=raw.dtype)
        idx = torch.tril_indices(s, s)
        tril[..., idx[0], idx[1]] = tril_flat
        diag_idx = torch.arange(s)
        tril[..., diag_idx, diag_idx] = (
            torch.nn.functional.softplus(tril[..., diag_idx, diag_idx]) + 1e-6
        )
        q = dist.MultivariateNormal(loc=loc, scale_tril=tril)
        return pyro.sample(self.variable.concept, q)


DEFAULT_GUIDES: Dict[Type[dist.Distribution], Type[_BaseGuide]] = {
    dist.Bernoulli: STBernoulliGuide,
    dist.OneHotCategorical: STOneHotGuide,
    dist.Normal: NormalGuide,
    dist.MultivariateNormal: MVNGuide,
}


# --------------------------------------------------------------------------
class _GuideContainer(PyroModule):
    """Holds per-latent guides and exposes a callable ``forward(data, temperature)``."""

    def __init__(
        self,
        guides: Dict[str, _BaseGuide],
        exogenous_names: List[str],
    ):
        super().__init__()
        self._latent_order: List[str] = list(guides.keys())
        self._exogenous_names = list(exogenous_names)
        for name, g in guides.items():
            setattr(self, f"guide_{name}", g)

    def _conditioning_input(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for n in self._exogenous_names:
            t = data[n]
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            parts.append(t.float())
        return torch.cat(parts, dim=-1)

    def forward(self, data: Dict[str, torch.Tensor], temperature: torch.Tensor):
        x = self._conditioning_input(data)
        for name in self._latent_order:
            getattr(self, f"guide_{name}")(x, temperature)
        return None


# --------------------------------------------------------------------------
class VariationalInference(InferenceEngine):
    """Variational inference engine (§5.4)."""

    name = "VariationalInference"

    def __init__(
        self,
        pgm: ProbabilisticModel,
        latents: List[str],
        default_guides: Optional[Dict[Type[dist.Distribution], Type[_BaseGuide]]] = None,
        n_samples: int = 1,
        max_plate_nesting: int = 0,
        initial_temperature: float = 1.0,
        annealing: Union[str, Callable[[int], float]] = "constant",
        annealing_rate: float = 0.0,
    ):
        super().__init__(pgm)

        merged = dict(DEFAULT_GUIDES)
        if default_guides is not None:
            merged.update(default_guides)

        # Conditioning input: all exogenous variables (always in evidence).
        exog_names = [
            v.concept for v in pgm.variables if isinstance(v, ExogenousVariable)
        ]
        if not exog_names:
            raise ValueError(
                f"{self.name}: at least one ExogenousVariable is required to "
                "supply the guides' conditioning input."
            )
        # Compute parent_dim = sum of exogenous variable sizes.
        parent_dim = sum(pgm.concept_to_variable[n].size for n in exog_names)

        guides: Dict[str, _BaseGuide] = {}
        for name in latents:
            if name not in pgm.concept_to_variable:
                raise ValueError(
                    f"{self.name}: latent {name!r} is not a variable of the PGM."
                )
            v = pgm.concept_to_variable[name]
            if v.distribution is dist.Delta:
                # Delta guides are degenerate — skip silently (§5.4).
                continue
            if v.distribution is None:
                raise ValueError(
                    f"{self.name}: latent {name!r} has distribution=None; "
                    "such variables are roots and cannot be latent."
                )
            cls = merged.get(v.distribution)
            if cls is None:
                raise ValueError(
                    f"{self.name}: no default guide registered for "
                    f"distribution {v.distribution!r}. Pass `default_guides=` "
                    "with an entry for this family."
                )
            guides[name] = cls(variable=v, parent_dim=parent_dim)

        self.latents: List[str] = list(latents)
        self.guide: _GuideContainer = _GuideContainer(guides, exog_names)
        self.n_samples = int(n_samples)
        self.max_plate_nesting = int(max_plate_nesting)
        self._schedule = make_temperature_schedule(
            initial_temperature, annealing, annealing_rate
        )
        self._step: int = 0

    @property
    def temperature(self) -> torch.Tensor:
        return torch.tensor(float(self._schedule(self._step)))

    def step(self) -> None:
        self._step += 1

    # ------------------------------------------------------------------
    def guide_fn(self, data: Dict[str, torch.Tensor]):
        """Pyro-callable guide. Use this when calling Trace_ELBO directly."""
        return self.guide(data, self.temperature)

    # ------------------------------------------------------------------
    def _run(
        self,
        query: List[str],
        evidence: List[str],
        data: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        # Build the model-side observation dict from evidence only.
        obs_data = {k: data[k] for k in evidence}

        guide_fn = lambda: self.guide(data, self.temperature)
        guide_tr = poutine.trace(guide_fn).get_trace()
        replayed = poutine.replay(self.pgm, trace=guide_tr)
        model_tr = poutine.trace(replayed).get_trace(obs_data)

        out = InferenceOutput(
            model_params=trace_to_params(model_tr),
            guide_params=trace_to_params(guide_tr),
        )
        return out
