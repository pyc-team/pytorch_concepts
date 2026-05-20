"""Variational inference engine with amortised guides."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from pyro.nn import PyroModule

from ..models.bayesian_network import BayesianNetwork
from ..models.variable import Variable, param_dim
from .base import (
    BaseInference,
    make_temperature_schedule,
    trace_to_params,
)
from .outputs import InferenceOutput


# --------------------------------------------------------------------------
# Default per-family guide modules.
#
# Each guide:
# - takes ``variable`` and ``parent_dim`` at construction time;
# - builds a 2-layer MLP (hidden=64) producing ``param_dim(distribution, size)``;
# - on ``forward(x, temperature)``, receives the concatenated conditioning
#   input for that specific latent (selected by ``_GuideContainer``), applies
#   the family-appropriate output bounding, and emits ``pyro.sample(name, q_dist)``.
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
    """Amortised variational guide for ``Bernoulli`` latents.

    Reads the concatenated exogenous inputs, predicts per-sample probabilities
    through a 2-layer MLP (hidden=64) with sigmoid output, and emits a
    ``pyro.sample`` site drawn from
    ``RelaxedBernoulliStraightThrough(temperature, logits)``.
    """

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        if self.variable.size == 1:
            raw = raw.squeeze(-1)
        probs = torch.sigmoid(raw)
        logits = torch.log(probs.clamp(min=1e-8)) - torch.log((1 - probs).clamp(min=1e-8))
        q = dist.RelaxedBernoulliStraightThrough(temperature=temperature, logits=logits)
        if self.variable.size > 1:
            q = q.to_event(1)
        return pyro.sample(self.variable.name, q)


class STOneHotGuide(_BaseGuide):
    """Amortised variational guide for ``OneHotCategorical`` latents using a
    ``RelaxedOneHotCategoricalStraightThrough`` distribution; same MLP
    architecture as ``STBernoulliGuide``."""

    def forward(self, x: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        probs = torch.softmax(raw, dim=-1)
        logits = torch.log(probs.clamp(min=1e-8))
        q = dist.RelaxedOneHotCategoricalStraightThrough(
            temperature=temperature, logits=logits
        )
        return pyro.sample(self.variable.name, q)


class NormalGuide(_BaseGuide):
    """Amortised variational guide for ``Normal`` latents using a
    reparameterised ``Normal(loc, scale)`` distribution; same MLP
    architecture as ``STBernoulliGuide``, with softplus on the scale."""

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
        return pyro.sample(self.variable.name, q)


class MVNGuide(_BaseGuide):
    """Amortised variational guide for ``MultivariateNormal`` latents using a
    reparameterised ``MultivariateNormal(loc, scale_tril)`` distribution;
    same MLP architecture as ``STBernoulliGuide``, with softplus on the
    diagonal of the lower-triangular Cholesky factor."""

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
        return pyro.sample(self.variable.name, q)


DEFAULT_GUIDES: Dict[Type[dist.Distribution], Type[_BaseGuide]] = {
    dist.Bernoulli: STBernoulliGuide,
    dist.OneHotCategorical: STOneHotGuide,
    dist.Normal: NormalGuide,
    dist.MultivariateNormal: MVNGuide,
}


# --------------------------------------------------------------------------
class _GuideContainer(PyroModule):
    """Registers one guide module per latent and, when called, feeds each
    guide the conditioning input declared for it at construction time."""

    def __init__(
        self,
        guides: Dict[str, _BaseGuide],
        conditioning: Dict[str, List[str]],  # latent_name -> list of conditioning var names
    ):
        super().__init__()
        self._latent_order: List[str] = list(guides.keys())
        # Per-latent conditioning variable names (declared order).
        self._conditioning: Dict[str, List[str]] = conditioning
        for name, g in guides.items():
            setattr(self, f"guide_{name}", g)

    def _conditioning_input(
        self, data: Dict[str, torch.Tensor], latent_name: str
    ) -> torch.Tensor:
        """Concatenate the tensors named in the conditioning list for ``latent_name``.
        Returns a ``(B, sum_sizes)`` tensor, or ``(B, 0)`` for an unconditional guide.
        """
        names = self._conditioning[latent_name]
        if not names:
            # Unconditional guide: return a (B, 0) dummy so the MLP still runs.
            B = next(iter(data.values())).shape[0] if data else 1
            device = next(iter(data.values())).device if data else torch.device("cpu")
            return torch.zeros(B, 0, device=device)
        parts = []
        for n in names:
            if n not in data:
                raise ValueError(
                    f"_GuideContainer: conditioning variable {n!r} for guide "
                    f"of {latent_name!r} is missing from data."
                )
            t = data[n]
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            parts.append(t.float())
        return torch.cat(parts, dim=-1)

    def forward(self, data: Dict[str, torch.Tensor], temperature: torch.Tensor):
        for name in self._latent_order:
            x = self._conditioning_input(data, name)
            getattr(self, f"guide_{name}")(x, temperature)
        return None


# --------------------------------------------------------------------------
class VariationalInference(BaseInference):
    """Variational inference engine with amortised guides.

    At construction time, auto-detects non-root latent variables and builds one guide
    per latent using ``DEFAULT_GUIDES`` (overridable via ``default_guides``).
    The conditioning input for each guide is declared via ``condition_on``:

    - ``None``: condition every latent on root variables (raises if none).
    - ``List[str]``: apply the same list to every latent.
    - ``Dict[str, List[str]]``: per-latent, keys must equal the latent set.

    At call time, traces the guide, replays the model under that trace,
    conditions on evidence, and populates ``InferenceOutput.params`` and
    ``InferenceOutput.guide_params``. The relaxation temperature follows the
    ``annealing`` schedule; call ``engine.step()`` to advance it.
    """

    name = "VariationalInference"

    def __init__(
        self,
        pgm: BayesianNetwork,
        condition_on: Optional[Union[List[str], Dict[str, List[str]]]] = None,
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

        latents = [v.name for v in pgm.variables if not pgm.factors[v.name].is_root]
        latent_set = set(latents)

        # Resolve condition_on to Dict[latent_name, List[str]].
        if condition_on is None:
            roots = [v.name for v in pgm.variables if pgm.factors[v.name].is_root]
            if not roots:
                raise ValueError(
                    f"{self.name}: no condition_on provided and no "
                    "root variables exist; cannot build guides."
                )
            conditioning: Dict[str, List[str]] = {lat: list(roots) for lat in latents}
        elif isinstance(condition_on, list):
            conditioning = {lat: list(condition_on) for lat in latents}
        elif isinstance(condition_on, dict):
            expected_keys = set(latents)
            got_keys = set(condition_on.keys())
            missing = expected_keys - got_keys
            extra = got_keys - expected_keys
            if missing or extra:
                raise ValueError(
                    f"{self.name}: condition_on dict keys do not match latent set. "
                    f"Missing: {sorted(missing)}, Extra: {sorted(extra)}."
                )
            conditioning = {k: list(v) for k, v in condition_on.items()}
        else:
            raise TypeError(
                f"{self.name}: condition_on must be None, a list, or a dict, "
                f"got {type(condition_on).__name__}."
            )

        # Validate each conditioning list: names must exist and must not be latents.
        for latent_name, cond_names in conditioning.items():
            for cname in cond_names:
                if cname not in pgm.name_to_variable:
                    raise ValueError(
                        f"{self.name}: conditioning name {cname!r} for guide of "
                        f"{latent_name!r} is not a variable of the PGM."
                    )
                if cname in latent_set:
                    raise ValueError(
                        f"Guide for {latent_name!r} cannot condition on "
                        f"latent variable {cname!r}."
                    )

        # Build one guide per latent with its own parent_dim.
        guides: Dict[str, _BaseGuide] = {}
        for name in latents:
            v = pgm.name_to_variable[name]
            # Latents are non-Delta (filtered above); no extra guard needed.
            cls = merged.get(v.distribution)
            if cls is None:
                raise ValueError(
                    f"{self.name}: no default guide registered for "
                    f"distribution {v.distribution!r}. Pass `default_guides=` "
                    "with an entry for this family."
                )
            parent_dim = sum(
                pgm.name_to_variable[n].size for n in conditioning[name]
            )
            guides[name] = cls(variable=v, parent_dim=parent_dim)

        self.latents: List[str] = latents
        self.guide: _GuideContainer = _GuideContainer(guides, conditioning)
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
        """Plain Pyro-compatible guide callable bound to the engine's current
        temperature. Pass this directly to ``pyro.infer.Trace_ELBO`` (or any
        other Pyro inference object) when bypassing the engine's ``_run``."""
        return self.guide(data, self.temperature)

    # ------------------------------------------------------------------
    def _run(
        self,
        query: Dict[str, Optional[torch.Tensor]],
        evidence: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        # data = evidence + any non-None query values (used to condition the guide).
        data = dict(evidence)
        for name, val in query.items():
            if val is not None:
                data[name] = val

        batch_size = next(iter(evidence.values())).shape[0] if evidence else None

        guide_fn = lambda: self.guide(data, self.temperature)
        guide_tr = poutine.trace(guide_fn).get_trace()
        replayed = poutine.replay(self.pgm, trace=guide_tr)
        model_tr = poutine.trace(replayed).get_trace(evidence, batch_size=batch_size)

        return InferenceOutput(
            params=trace_to_params(model_tr),
            guide_params=trace_to_params(guide_tr),
        )
