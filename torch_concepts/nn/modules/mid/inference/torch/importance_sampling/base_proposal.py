"""BaseProposal — modular proposal distribution for importance sampling.

An importance-sampling proposal is a *full joint* distribution
:math:`q_\\phi(z \\mid e)` over **all** variables of a
:class:`BayesianNetwork`, conditioned on the per-query evidence ``e``. It is
factorised autoregressively along the network's topological order::

    q_phi(z | e) = prod_i  q_phi( x_i | PA_i, e )         (i ranges over non-evidence vars)

The two properties that make this useful for the "evidence-on-the-leaves,
query-near-the-roots" regime are:

* **Evidence-awareness.** Every factor receives the *entire* evidence dict
  ``e`` (not only its topological ancestors), so a root-level node may condition
  on a leaf-level observation. This is the backward information flow that prior
  / ancestral sampling cannot provide.
* **Clamping.** Evidence variables are clamped to their observed value and are
  *not* sampled, so they contribute no proposal density — exactly mirroring the
  importance weight assembled by :class:`ImportanceSampling`.

Subclassing
-----------
The common case only requires implementing :meth:`propose`, which returns the
*distribution parameters* (the same ``{param_name: tensor}`` schema a
:class:`ParametricCPD` produces) for one non-evidence variable. The base
:meth:`sample` then handles the topological traversal, evidence clamping,
reparameterised sampling, and ``log q`` accumulation — building the relaxed
distribution from those parameters so the proposal stays in the *same family*
as the model (which keeps the ``p / q`` density ratio well defined). For full
control over the joint (e.g. a non-model proposal family or a non-BN
factorisation), override :meth:`sample` directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Set, Tuple

import torch
import torch.nn as nn

import torch.distributions as dist

from ....models.bayesian_network import BayesianNetwork
from ....models.variable import Variable
from ...utils import reshape_value_to_event
from ..utils import build_relaxed_distribution


_BERNOULLI = (dist.Bernoulli, dist.RelaxedBernoulli)
_ONEHOT = (dist.OneHotCategorical, dist.RelaxedOneHotCategorical)


def _stabilize_relaxed(variable: Variable, sample: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Nudge a relaxed discrete draw off the support boundary.

    At low temperature a Gumbel-Softmax / Concrete sample underflows to exact
    0 / 1, where ``log_prob`` evaluates ``log(0) = -inf``. Clamping (and, for the
    one-hot family, renormalising back onto the simplex) keeps ``log q`` and the
    model ``log p`` finite. Because the *same* stabilised sample is scored by
    both, the importance ratio is unaffected. Continuous families are returned
    untouched (their support is unbounded).
    """
    D = variable.distribution
    if issubclass(D, _BERNOULLI):
        return sample.clamp(eps, 1.0 - eps)
    if issubclass(D, _ONEHOT):
        s = sample.clamp_min(eps)
        return s / s.sum(dim=-1, keepdim=True)
    return sample


class BaseProposal(nn.Module, ABC):
    """Abstract proposal :math:`q_\\phi(z \\mid e)` for importance sampling.

    This is an abstract base class: it cannot be instantiated directly. Concrete
    proposals must subclass it and implement :meth:`propose` (or override
    :meth:`sample` for full control). See :class:`MutilatedNetworkProposal` for a
    parameter-free reference implementation (likelihood weighting).

    Parameters
    ----------
    pgm : BayesianNetwork
        The network whose joint the proposal approximates. It is stored as a
        *non-registered* reference (wrapped in a tuple) so that the proposal's
        own learnable parameters — and only those — are exposed through
        ``proposal.parameters()``; the PGM's parameters are owned by the
        inference engine, avoiding double registration in an optimizer.
    """

    name: str = "BaseProposal"

    def __init__(self, pgm: BayesianNetwork) -> None:
        super().__init__()
        # Hide the PGM from nn.Module's submodule registration (a tuple is not a
        # Module), so its parameters are not double-counted by the engine.
        self._pgm_ref: Tuple[BayesianNetwork] = (pgm,)

    @property
    def pgm(self) -> BayesianNetwork:
        return self._pgm_ref[0]

    # ------------------------------------------------------------------
    # Primary extension hook
    # ------------------------------------------------------------------
    @abstractmethod
    def propose(
        self,
        variable: Variable,
        parent_values: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor],
        batch_size: int,
        temperature: torch.Tensor,
        layer_kwargs: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Return the proposal-distribution parameters for one variable.

        Implementations return a ``{param_name: tensor}`` dict using the same
        schema as the variable's distribution family (e.g. ``{'probs': ...}``
        for Bernoulli / OneHotCategorical, ``{'loc': ..., 'scale': ...}`` for
        Normal), with a leading batch dimension of ``batch_size``.

        Parameters
        ----------
        variable : Variable
            The (non-evidence) variable to propose.
        parent_values : dict[str, Tensor]
            Already-resolved values of this variable's **BN parents** — both
            clamped evidence parents and previously sampled ones — each shaped
            ``(batch_size, *parent.shape)``. Empty for root variables.
        evidence : dict[str, Tensor]
            The full evidence dict for this query, each shaped
            ``(batch_size, *var.shape)``. Available to *every* factor so a root
            proposal may condition on a leaf observation.
        batch_size : int
            Leading batch dimension the returned parameters must carry; needed
            e.g. to broadcast an unbatched root parameter to all rows.
        temperature : Tensor
            Current relaxation temperature (scalar buffer).
        layer_kwargs : dict
            Per-variable extra keyword arguments forwarded to any module.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Joint sampling (overridable template)
    # ------------------------------------------------------------------
    def sample(
        self,
        query_names: Set[str],
        evidence: Dict[str, torch.Tensor],
        batch_size: int,
        temperature: torch.Tensor,
        layer_kwargs: Dict[str, Dict] = {},
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Sample the joint :math:`q_\\phi(z \\mid e)` and accumulate ``log q``.

        Traverses ``self.pgm.sorted_variables`` in topological order. Evidence
        variables are clamped to their observed value (no density contribution);
        every other variable is proposed via :meth:`propose`, sampled with a
        reparameterised (differentiable) draw, and scored under the *same*
        relaxed distribution.

        Parameters
        ----------
        query_names : set[str]
            Names of the query variables. Provided for proposals that wish to
            special-case query nodes; the default template treats query and
            hidden variables identically (both are non-evidence, hence sampled).
        evidence : dict[str, Tensor]
            Evidence values, each shaped ``(batch_size, *var.shape)``.
        batch_size : int
            Number of independent rows to draw (the engine packs
            ``n_samples * B`` importance draws into this single leading axis).
        temperature : Tensor
            Relaxation temperature.
        layer_kwargs : dict[str, dict]
            Per-variable extra module kwargs.

        Returns
        -------
        samples : dict[str, Tensor]
            One entry per variable (clamped evidence included), each shaped
            ``(batch_size, *var.shape)``.
        log_q : Tensor
            ``(batch_size,)`` joint log-density of the sampled (non-evidence)
            variables under the proposal.
        """
        pgm = self.pgm
        device = temperature.device
        log_q = torch.zeros(batch_size, device=device)
        samples: Dict[str, torch.Tensor] = {}

        for var in pgm.sorted_variables:
            name = var.name
            if name in evidence:
                # Clamped observation: carry the value forward, no q-density.
                value = evidence[name].reshape(evidence[name].shape[0], var.size)
                samples[name] = reshape_value_to_event(var, value)
                continue

            cpd = pgm.name_to_factor(name)
            parent_values = {p.name: samples[p.name] for p in cpd.parents}
            params = self.propose(
                var, parent_values, evidence, batch_size, temperature,
                layer_kwargs.get(name, {}),
            )
            # validate_args=False: at low temperature a relaxed draw lands on
            # the simplex / unit-interval boundary, which torch rejects in
            # log_prob even though it is expected here.
            d = build_relaxed_distribution(var, params, temperature, validate_args=False)
            s = _stabilize_relaxed(var, d.rsample())
            log_q = log_q + d.log_prob(s)
            samples[name] = reshape_value_to_event(var, s.reshape(batch_size, var.size))

        return samples, log_q
