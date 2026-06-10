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

import pyro
import pyro.distributions as pyro_dist
import torch
import torch.distributions as td

from ...models.bayesian_network import BayesianNetwork
from ...models.variable import Delta
from ..base import BaseInference
from ..utils import build_distribution
from .utils import dist_to_params, trace_to_params


# -----------------------------------------------------------------------------
class PyroBaseInference(BaseInference):
    """Base class for inference engines backed by Pyro.

    Bundles the model/guide stochastic functions and the Pyro-side parameter
    harvesters. Subclasses (e.g. :class:`PyroVariationalInference`) supply
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
        D = variable.distribution
        if issubclass(D, td.Bernoulli):
            d = pyro_dist.RelaxedBernoulliStraightThrough(temperature=temperature, **params)
            return d.to_event(len(variable.shape))
        if issubclass(D, td.OneHotCategorical):
            d = pyro_dist.RelaxedOneHotCategoricalStraightThrough(temperature=temperature, **params)
            return d
            return d
        if issubclass(D, td.Normal):
            d = pyro_dist.Normal(**params)
            return d.to_event(len(variable.shape))
        if issubclass(D, td.MultivariateNormal):
            return pyro_dist.MultivariateNormal(**params)
        if D.__name__ == "Delta":
            # Map ``value`` (our Delta convention) to ``v`` (Pyro Delta convention).
            v = params["value"]
            event_dim = len(variable.shape)
            return pyro_dist.Delta(v, event_dim=event_dim)
        # Fallback for any other family: try the exact torch distribution.
        return build_distribution(variable, params)

    # ------------------------------------------------------------------
    # Stochastic functions (bound to ``self.pgm``)
    # ------------------------------------------------------------------
    def model_fn(
        self,
        data: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
        latent_names: List[str],
        batch_size: Optional[int] = None,
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
                    cpd = pgm.name_to_factor(var.name)

                    if cpd.is_root:
                        params = cpd(parent_values={})
                        params = {
                            k: v.unsqueeze(0).expand(B, *v.shape) for k, v in params.items()
                        }
                    else:
                        parent_values: Dict[str, torch.Tensor] = {}
                        for p in cpd.parents:
                            if p.name in cache:
                                parent_values[p.name] = cache[p.name]
                            elif p.name in data:
                                parent_values[p.name] = data[p.name]
                            else:
                                raise ValueError(
                                    f"model_fn: parent {p.name!r} of {var.name!r} is "
                                    "neither cached nor in data."
                                )
                        params = cpd(parent_values=parent_values)

                    obs = data.get(var.name, None)
                    d = (
                        build_distribution(var, params)
                        if obs is not None
                        else self._pyro_relaxed_distribution(var, params, temperature)
                    )
                    cache[var.name] = pyro.sample(var.name, d, obs=obs)

        return cache

    def guide_fn(
        self,
        data: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
        latent_names: List[str],
    ) -> None:
        """Pyro stochastic function for the variational posterior.

        Runs a ``pyro.sample`` site for each latent variable using its
        registered guide CPD from ``self.pgm.guides``.

        Registers the guide ``nn.ModuleDict`` with Pyro's param store via
        ``pyro.module`` on every call so SVI updates flow back into the
        original guide CPDs' ``nn.Parameter`` tensors.
        """
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
                    params = cpd(parent_values=parent_values)

                q = self._pyro_relaxed_distribution(cpd.variable, params, temperature)
                pyro.sample(name, q)
