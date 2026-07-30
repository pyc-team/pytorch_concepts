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

from collections import ChainMap
from typing import Dict, List, Optional

import torch
import torch.distributions as td

from ...graph.bayesian_network import BayesianNetwork
from ...variable import Delta
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

    Parameters
    ----------
    pgm : BayesianNetwork
        The directed model to run inference on, held by reference (see
        :class:`~torch_concepts.nn.modules.mid.inference.base.BaseInference`).
        Pyro's model/guide traces walk the topological order, so a general
        (undirected or mixed) ``ProbabilisticModel`` is not supported here.
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
        # A CPD for a matrix-valued variable (e.g. an ``(n_states, emb)`` concept
        # embedding from ``LinearEmbeddingEncoder``) emits the full event shape,
        # which would leave the event dims in ``batch_shape`` and collide with the
        # batch plate — so flatten those back onto the single size axis here.
        n_event = len(variable.shape)
        if n_event > 1:
            params = {
                key: value.reshape(*value.shape[:value.dim() - n_event], variable.size)
                for key, value in params.items()
            }
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
    # Stochastic functions (bound to ``self.pgm``)
    # ------------------------------------------------------------------
    def model_fn(
        self,
        data: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
        latent_names: List[str],
        batch_size: Optional[int] = None,
        layer_kwargs: Dict[str, Dict] = {},
        member_evidence: Dict[str, Dict[str, torch.Tensor]] = {},
    ) -> Dict[str, torch.Tensor]:
        """Pyro stochastic function for the generative model.

        Iterates ``self.pgm.sorted_variables`` in topological order. Each
        variable becomes a ``pyro.sample`` site:

        - Variables present in ``data`` are scored against their exact
          distribution (``obs=`` keyword to ``pyro.sample``).
        - Variables absent from ``data`` are sampled via a straight-through
          relaxation so gradients flow through the discrete sites.

        ``member_evidence`` forces individually-observed plate members onto the
        sampled value (value forcing; no likelihood term). Registers ``self.pgm``
        with Pyro's param store via ``pyro.module`` on every call so SVI updates
        flow back into the original PGM's ``nn.Parameter`` tensors (no parameter
        duplication).
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
                        # cache (sampled/observed values) wins over raw data; the
                        # CPD resolves member-handle parents from the plate value.
                        # ChainMap avoids an O(#variables) dict copy per site.
                        params = cpd(parent_values=ChainMap(cache, data), **layer_kwargs.get(var.name, {}))

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
                    # CPD aggregation re-flattens it as needed. Partial-plate evidence
                    # is forced onto the observed members here.
                    value = reshape_value_to_event(var, sample)
                    cache[var.name] = cpd.clamp_members(
                        value, member_evidence.get(var.name, {})
                    )

        return cache

    def guide_fn(
        self,
        data: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
        latent_names: List[str],
        layer_kwargs: Dict[str, Dict] = {},
        member_evidence: Dict[str, Dict[str, torch.Tensor]] = {},
    ) -> None:
        """Pyro stochastic function for the variational posterior.

        Runs a ``pyro.sample`` site for each latent variable using its
        registered guide CPD from ``self.pgm.guides``.

        Registers the guide ``nn.ModuleDict`` with Pyro's param store via
        ``pyro.module`` on every call so SVI updates flow back into the
        original guide CPDs' ``nn.Parameter`` tensors. ``member_evidence`` is
        threaded for symmetry with ``model_fn``; the guide conditions on
        observed ``data`` (member evidence included by name), so it clamps
        nothing itself.
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
                    # The CPD resolves member-handle parents from ``data``.
                    params = cpd(parent_values=data, **layer_kwargs.get(name, {}))

                q = self._pyro_relaxed_distribution(cpd.variable, params, temperature)
                pyro.sample(name, q)
