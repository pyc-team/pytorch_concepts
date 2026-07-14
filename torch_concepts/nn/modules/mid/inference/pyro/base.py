"""PyroBaseInference — base class for Pyro-backed inference engines.

Provides the shared Pyro plumbing required by any engine that uses Pyro's
effect handlers (``poutine.trace``, ``poutine.replay``, ``pyro.infer.SVI``):

- ``model_fn`` / ``guide_fn``: bound Pyro stochastic functions that traverse
  the wrapped PGM topologically and emit ``pyro.sample`` sites.
- ``_pyro_relaxed_distribution``: pyro-compatible straight-through relaxation
  for the discrete distribution families.
- ``dist_to_params`` / ``trace_to_params``: helpers to harvest distribution
  parameters from a Pyro trace into the engine-agnostic
  :class:`InferenceOutput.params` schema.

Parameter sharing with the wrapped PGM is inherited from
:class:`BaseInference` (the engine holds a reference to ``pgm``, so
``engine.parameters()`` enumerates the same tensors as ``pgm.parameters()``).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.distributions as td

from ...models.bayesian_network import BayesianNetwork
from ...models.variable import Delta
from ..base import BaseInference
from ..utils import build_distribution, reshape_value_to_event
from .utils import dist_to_params, trace_to_params


def _import_pyro():
    """Lazily import Pyro, raising a clear error if it is not installed."""
    try:
        import pyro
        import pyro.distributions as pyro_dist
        import pyro.poutine as poutine
        return pyro, pyro_dist, poutine
    except ImportError as exc:
        raise ImportError(
            "Pyro-based inference requires the `pyro-ppl` package. "
            "Install it with: pip install pyro-ppl"
        ) from exc


# -----------------------------------------------------------------------------
class PyroBaseInference(BaseInference):
    """Base class for inference engines backed by Pyro.

    Bundles the model/guide stochastic functions and the Pyro-side parameter
    harvesters. Subclasses (e.g. :class:`VariationalInference`) supply
    their own ``query`` method that orchestrates effect handlers.
    """

    name = "PyroBaseInference"

    def __init__(self, pgm: BayesianNetwork):
        super().__init__(pgm)

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pyro_relaxed_distribution(
        variable,
        params: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
    ) -> pyro_dist.Distribution:
        """Build a Pyro-compatible relaxed distribution for ``pyro.sample`` sites.

        Returns a ``pyro.distributions`` instance (subclass of
        ``TorchDistribution``) — required by ``pyro.sample`` for unobserved
        sites. Plain ``torch.distributions`` objects are not callable and would
        raise ``TypeError: 'X' object is not callable`` at runtime.

        Uses Pyro's own straight-through estimators (which register correctly
        with Pyro's effect-handler stack) for the discrete families.
        """
        # Parameters are flat (*batch, size); the single size axis is reinterpreted
        # as the event (``to_event(1)`` / ``event_dim=1``) so batch_shape stays
        # (*batch,) and the ``pyro.plate("batch", ...)`` dim lines up. The
        # variable's declared shape is restored on the sampled realization.
        _, pyro_dist, _ = _import_pyro()
        D = variable.distribution
        if issubclass(D, td.Bernoulli):
            d = pyro_dist.RelaxedBernoulliStraightThrough(temperature=temperature, **params)
            return d.to_event(1)
        if issubclass(D, td.OneHotCategorical):
            d = pyro_dist.RelaxedOneHotCategoricalStraightThrough(temperature=temperature, **params)
            return d
        if issubclass(D, td.Normal):
            d = pyro_dist.Normal(**params)
            return d.to_event(1)
        if issubclass(D, td.MultivariateNormal):
            return pyro_dist.MultivariateNormal(**params)
        if D.__name__ == "Delta":
            # Map ``value`` (our Delta convention) to ``v`` (Pyro Delta convention).
            v = params["value"]
            return pyro_dist.Delta(v, event_dim=1)
        # Fallback for any other family: try the exact torch distribution.
        return build_distribution(variable, params)

    # ------------------------------------------------------------------
    # Plate (member) addressing — shared by the Pyro engines, reusing the
    # CPD's slicing so a plate behaves the same as under the torch engine.
    # ------------------------------------------------------------------
    def _gather_parents(self, cpd, cache, data):
        """Parent values for ``cpd``, slicing member-handle parents out of their
        plate's (sampled or observed) value — the Pyro counterpart of the torch
        engine's member-as-parent handling."""
        parents: Dict[str, torch.Tensor] = {}
        for p in cpd.parents:
            src = p.plate.name  # owning plate for a member handle, else p.name
            value = cache.get(src, data.get(src))
            if value is None:
                raise ValueError(
                    f"{self.name}: parent {p.name!r} of {cpd.variable.name!r} is "
                    "neither sampled nor in data."
                )
            if p.name != src:  # member handle -> slice its column from the plate value
                value = self.pgm.factors[src].select_value(value, p.name)
            parents[p.name] = value
        return parents

    def _expose_members(self, params, query_names):
        """Add per-member entries for queried plate members, sliced (a view) from
        their plate's params, so members are addressable by name in the output."""
        for name in query_names:
            var = self.pgm.resolve(name)
            if name != var.name and var.name in params:
                params[name] = self.pgm.factors[var.name].select(params[var.name], name)
        return params

    # ------------------------------------------------------------------
    # Stochastic functions (bound to ``self.pgm``)
    # ------------------------------------------------------------------
    def model_fn(
        self,
        data: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
        latent_names: List[str],
        batch_size: Optional[int] = None,
        layer_kwargs: Dict[str, Dict] = {},
    ) -> Dict[str, torch.Tensor]:
        """Pyro stochastic function for the generative model.

        Iterates ``self.pgm.sorted_variables`` in topological order. Each
        variable becomes a ``pyro.sample`` site:

        - Variables present in ``data`` are scored against their exact
          distribution (``obs=`` keyword to ``pyro.sample``).
        - Variables absent from ``data`` are sampled via a straight-through
          relaxation so gradients flow through the discrete sites.

        Registers ``self.pgm`` with Pyro's param store via ``pyro.module`` on
        every call so SVI updates flow back into the original PGM's
        ``nn.Parameter`` tensors (no parameter duplication).
        """
        pyro, _, _ = _import_pyro()
        pgm = self.pgm
        pyro.module("pgm", pgm)

        if data:
            B = next(iter(data.values())).shape[0]
        elif batch_size is not None:
            B = batch_size
        else:
            raise ValueError(
                "Cannot infer batch size: `data` is empty and `batch_size` was not provided."
            )

        cache: Dict[str, torch.Tensor] = {}

        with pyro.plate("batch", B, dim=-1):
            for level in pgm.levels:
                for var in level:
                    cpd = pgm.factors[var.name]
                    if cpd.is_root:
                        params = cpd.root_params(B)
                    else:
                        parent_values = self._gather_parents(cpd, cache, data)
                        params = cpd(parent_values=parent_values, **layer_kwargs.get(var.name, {}))

                    obs = data.get(var.name, None)
                    if obs is not None:
                        # The distribution's event is the flat size axis, so match
                        # the observation to it: (*batch, *shape) -> (*batch, size).
                        obs = obs.reshape(obs.shape[0], var.size)
                    d = (
                        build_distribution(var, params)
                        if obs is not None
                        else self._pyro_relaxed_distribution(var, params, temperature)
                    )
                    sample = pyro.sample(var.name, d, obs=obs)
                    # Cache the realization in the variable's event shape; downstream
                    # CPD aggregation re-flattens it as needed.
                    cache[var.name] = reshape_value_to_event(var, sample)

        return cache

    def guide_fn(
        self,
        data: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
        latent_names: List[str],
        layer_kwargs: Dict[str, Dict] = {},
    ) -> None:
        """Pyro stochastic function for the variational posterior.

        Runs a ``pyro.sample`` site for each latent variable using its
        registered guide CPD from ``self.pgm.guides``.

        Registers the guide ``nn.ModuleDict`` with Pyro's param store via
        ``pyro.module`` on every call so SVI updates flow back into the
        original guide CPDs' ``nn.Parameter`` tensors.
        """
        pyro, _, _ = _import_pyro()
        pgm = self.pgm
        pyro.module("pgm_guides", pgm.guides)
        B = next(iter(data.values())).shape[0] if data else 1

        with pyro.plate("batch", B, dim=-1):
            for name in latent_names:
                cpd = pgm.guides[name]

                if cpd.is_root:
                    params = cpd(parent_values={})
                    params = {
                        k: v.unsqueeze(0).expand(B, *v.shape) for k, v in params.items()
                    }
                else:
                    parent_values = {p.name: data[p.name] for p in cpd.parents}
                    params = cpd(parent_values=parent_values, **layer_kwargs.get(name, {}))

                q = self._pyro_relaxed_distribution(cpd.variable, params, temperature)
                pyro.sample(name, q)
