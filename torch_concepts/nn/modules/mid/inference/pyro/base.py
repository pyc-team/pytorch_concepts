"""PyroBaseInference — base class for Pyro-backed inference engines."""

from __future__ import annotations

from collections import ChainMap
from typing import Dict, List, Optional

import torch

from ...graph.bayesian_network import BayesianNetwork
from ..base import BaseInference
from ..utils import build_distribution, teacher_force
from .utils import build_relaxed_pyro_distribution, dist_to_params, trace_to_params


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
    """

    name = "PyroBaseInference"

    def _flatten_multidim_events(
        self, per_variable: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Report a multi-dimensional event as one flat row, as this backend always has.

        The torch engines keep a parameter's own rank (an ``(n_states, emb)``
        embedding stays a matrix); the Pyro engines have always flattened it onto
        the annotated axis instead. Keeping that difference here rather than at
        the distribution boundary means the member layout still reaches Pyro,
        which is what makes a categorical plate ``k`` independent draws.
        """
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, params in per_variable.items():
            var = self.pgm.variables.get(name)
            if var is not None and len(var.shape) > 1:
                params = {
                    key: var.to_flat(var.to_member(value, key))
                    for key, value in params.items()
                }
            out[name] = params
        return out

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
        teacher_forced: Dict[str, torch.Tensor] = {},
    ) -> Dict[str, torch.Tensor]:
        """Pyro stochastic function for the generative model.

        Iterates ``self.pgm.sorted_variables`` in topological order. Each
        variable becomes a ``pyro.sample`` site:

        - Variables present in ``data`` are scored against their exact
          distribution (``obs=`` keyword to ``pyro.sample``).
        - Variables absent from ``data`` are sampled.

        ``member_evidence`` forces individually-observed plate members onto the
        sampled value (value forcing; no likelihood term). Registers ``self.pgm``
        with Pyro's param store via ``pyro.module`` on every call so SVI updates
        flow back into the original PGM's ``nn.Parameter`` tensors (no parameter
        duplication).

        ``teacher_forced`` carries the ground truth for variables that are to be
        forced *stochastically* at rate ``self.p_int`` (RandInt). Such a variable
        is a **latent** site — there is nothing to pass as ``obs=`` when only
        some rows will take the ground truth — and the blend happens after the
        draw. The caller supplies it only when ``p_int < 1``; at ``p_int == 1``
        every site keeps the plain ``obs=`` path.
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

                    # A stochastically-forced variable must be *sampled* — only
                    # some rows will take the ground truth — so it never becomes
                    # an `obs=` site, whatever `data` holds for it.
                    gt = teacher_forced.get(var.name, None)
                    obs = None if gt is not None else data.get(var.name, None)
                    if obs is not None:
                        # Match the observation to the distribution's event.
                        obs = var.to_member(obs)
                    d = (
                        build_distribution(var, params)
                        if obs is not None
                        else build_relaxed_pyro_distribution(var, params, temperature)
                    )
                    sample = pyro.sample(var.name, d, obs=obs)
                    if gt is not None:
                        sample = teacher_force(
                            sample, var.to_member(gt), self.p_int, 1, var.name,
                        )
                    # The realisation is already in the member layout the cache
                    # holds. Partial-plate evidence is forced on here.
                    cache[var.name] = cpd.clamp_members(
                        sample, member_evidence.get(var.name, {})
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

                q = build_relaxed_pyro_distribution(cpd.variable, params, temperature)
                pyro.sample(name, q)
