r"""Importance-sampling inference engine (Pyro backend).

Estimates ``P(Q=q | E=e)`` for a :class:`BayesianNetwork` by importance sampling,
using Pyro's effect handlers (``poutine.trace`` / ``poutine.replay``) — the same
primitives underlying :class:`pyro.infer.importance.Importance`. Unlike the
pure-PyTorch :class:`ImportanceSampling`, this engine is **not differentiable**:
discrete sites are sampled from their *exact* distributions (hard samples) and
the query is matched by exact equality, all under :func:`torch.no_grad`.

Proposal
--------
The proposal is a Pyro **guide**, supplied as ``proposal`` — a dict mapping each
non-evidence variable name to a :class:`ParametricCPD` (same idiom as
:class:`VariationalInference`'s ``latents``). A guide CPD ``q(x_i | ...)`` may
condition on evidence (and on earlier-sampled variables), which is what lets a
root-level query be informed by leaf-level evidence. **Any non-evidence variable
without a declared proposal is proposed from the model's own prior** — so an
empty ``proposal`` reduces this engine to *likelihood weighting* (the mutilated
network of :class:`pyro.infer.importance.Importance`'s default guide).

Weight (Pyro convention)
------------------------
For each particle, ``log w = model_trace.log_prob - guide_trace.log_prob``::

    log w = sum_{i in E} log p(e_i | PA_i)                       # evidence likelihood
          + sum_{i not in E} [ log p(x_i | PA_i) - log q(x_i) ]  # proposal correction

Variables proposed from the prior have ``log p = log q`` and cancel exactly
(the guide and the replayed model score the *same* value under the *same* CPD),
so only the evidence likelihood and the declared-proposal corrections remain.

The estimate is self-normalised over the ``N`` particles::

    P(Q=q | E=e) ~= sum_n softmax(log w)_n * 1[Q_n = q].

``out.probabilities`` is a ``(B,)`` tensor. Query variables must be discrete
(Bernoulli / OneHotCategorical); evidence may be continuous.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Set

import pyro
import pyro.distributions as pyro_dist
import pyro.poutine as poutine
import torch
import torch.distributions as td
import torch.nn as nn

from ...models.bayesian_network import BayesianNetwork
from ...models.cpd import ParametricCPD
from ..utils import build_distribution, reshape_value_to_event
from ....outputs import InferenceOutput
from .base import PyroBaseInference


# Discrete families admissible as query variables (relaxed variants included:
# a variable declared RelaxedBernoulli is conceptually a binary node).
_DISCRETE = (
    td.Bernoulli, td.RelaxedBernoulli,
    td.Categorical,
    td.OneHotCategorical, td.RelaxedOneHotCategorical,
)
_BERNOULLI = (td.Bernoulli, td.RelaxedBernoulli)
_ONEHOT = (td.OneHotCategorical, td.RelaxedOneHotCategorical)


def _pyro_exact_distribution(variable, params: Dict[str, torch.Tensor]):
    """Build the *exact* (non-relaxed) Pyro distribution for a variable.

    Relaxed discrete declarations are mapped to their hard counterparts so the
    importance weights are exact and the samples are genuine ``{0,1}`` / one-hot
    values (this engine does not need reparameterised gradients). Univariate
    families are wrapped with ``to_event(1)`` so the single ``size`` axis is the
    event and ``batch_shape`` stays ``(*batch,)`` — matching the ``pyro.plate``.
    """
    D = variable.distribution
    if issubclass(D, _BERNOULLI):
        return pyro_dist.Bernoulli(**params).to_event(1)
    if issubclass(D, _ONEHOT):
        return pyro_dist.OneHotCategorical(**params)
    if issubclass(D, td.Normal):
        return pyro_dist.Normal(**params).to_event(1)
    if issubclass(D, td.MultivariateNormal):
        return pyro_dist.MultivariateNormal(**params)
    if D.__name__ == "Delta":
        return pyro_dist.Delta(params["value"], event_dim=1)
    if issubclass(D, td.Categorical):
        raise NotImplementedError(
            "PyroImportanceSampling: plain Categorical is not supported; declare "
            "the variable as OneHotCategorical instead."
        )
    # Fallback for any other family: the exact torch distribution.
    return build_distribution(variable, params)


def _hard_match(sample: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact-equality membership of ``sample`` in ``target``.

    Both are shaped ``(N, B, *event)``; returns ``(N, B)`` floats in ``{0, 1}``
    (all event entries must match). Samples are hard ``{0,1}`` / one-hot draws,
    so float equality is exact.
    """
    eq = sample == target.to(sample.dtype)
    while eq.dim() > 2:
        eq = eq.all(dim=-1)
    return eq.to(sample.dtype)


class PyroImportanceSampling(PyroBaseInference):
    """Pyro-backed importance-sampling estimator of ``P(Q=q | E=e)``.

    Parameters
    ----------
    pgm : BayesianNetwork
        The probabilistic graphical model to query. It is shared by reference
        (parameter sharing is inherited from :class:`PyroBaseInference`), so the
        same torch PGM wrapper used by every other engine works here unchanged.
    proposal : dict[str, ParametricCPD], optional
        Per-variable guide CPDs forming the proposal. Each key is a non-evidence
        variable name and each value a :class:`ParametricCPD` proposing it. Omit
        (or pass empty) to propose every non-evidence node from the prior, i.e.
        likelihood weighting.
    n_samples : int
        Number of importance particles drawn per observation (default ``1000``).
    warn_low_ess : float
        Warn when the effective sample size falls below this fraction of
        ``n_samples`` (default ``0.01``).
    """

    name = "PyroImportanceSampling"
    _DISCRETE = _DISCRETE

    def __init__(
        self,
        pgm: BayesianNetwork,
        proposal: Optional[Dict[str, ParametricCPD]] = None,
        n_samples: int = 1_000,
        warn_low_ess: float = 0.01,
    ) -> None:
        super().__init__(pgm)
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
        self.n_samples = int(n_samples)
        self.warn_low_ess = float(warn_low_ess)
        # Store the proposal CPDs on the engine (not on pgm.guides) so this
        # engine does not clobber a VariationalInference guide on the same PGM.
        self.proposal: nn.ModuleDict = self._build_proposal(pgm, proposal or {})

    # ------------------------------------------------------------------
    @classmethod
    def _build_proposal(
        cls, pgm: BayesianNetwork, proposal: Dict[str, ParametricCPD]
    ) -> nn.ModuleDict:
        if not isinstance(proposal, dict):
            raise TypeError(
                f"{cls.__name__}: `proposal` must be a dict mapping variable "
                f"names to ParametricCPD instances, got {type(proposal).__name__}."
            )
        all_names = {v.name for v in pgm.variables}
        for name, cpd in proposal.items():
            if name not in all_names:
                raise ValueError(f"{cls.__name__}: unknown proposal variable {name!r}.")
            if not isinstance(cpd, ParametricCPD):
                raise TypeError(
                    f"{cls.__name__}: proposal for {name!r} must be a ParametricCPD, "
                    f"got {type(cpd).__name__}."
                )
            if cpd.variable.name != name:
                raise ValueError(
                    f"{cls.__name__}: proposal CPD variable {cpd.variable.name!r} "
                    f"does not match the dict key {name!r}."
                )
            for p in cpd.parents:
                if p.name not in all_names:
                    raise ValueError(
                        f"{cls.__name__}: proposal for {name!r}: parent {p.name!r} "
                        "is not a variable of the PGM."
                    )
        return nn.ModuleDict(proposal)

    # ------------------------------------------------------------------
    # Stochastic functions (exact, non-differentiable)
    # ------------------------------------------------------------------
    @staticmethod
    def _params_with_batch(cpd: ParametricCPD, parent_values: Dict, batch: int,
                           layer_kwargs: Dict) -> Dict[str, torch.Tensor]:
        """Evaluate a CPD, broadcasting an unbatched root parameter to ``batch``."""
        if cpd.is_root:
            params = cpd(parent_values={})
            return {
                k: v.unsqueeze(0).expand(batch, *v.shape) for k, v in params.items()
            }
        return cpd(parent_values=parent_values, **layer_kwargs)

    def _model_fn(self, data: Dict[str, torch.Tensor], batch: int,
                  layer_kwargs: Dict[str, Dict]) -> None:
        """Generative model: evidence observed (``obs=``), the rest sampled."""
        pgm = self.pgm
        cache: Dict[str, torch.Tensor] = {}
        with pyro.plate("batch", batch, dim=-1):
            for level in pgm.levels:
                for var in level:
                    cpd = pgm.name_to_factor(var.name)
                    parent_values = {
                        p.name: cache.get(p.name, data.get(p.name)) for p in cpd.parents
                    }
                    params = self._params_with_batch(
                        cpd, parent_values, batch, layer_kwargs.get(var.name, {})
                    )
                    obs = data.get(var.name)
                    if obs is not None:
                        obs = obs.reshape(obs.shape[0], var.size)
                    d = _pyro_exact_distribution(var, params)
                    value = pyro.sample(var.name, d, obs=obs)
                    cache[var.name] = reshape_value_to_event(var, value)

    def _guide_fn(self, data: Dict[str, torch.Tensor], batch: int,
                  layer_kwargs: Dict[str, Dict]) -> None:
        """Proposal: sample every non-evidence node (declared guide, else prior)."""
        pgm = self.pgm
        cache: Dict[str, torch.Tensor] = {}
        with pyro.plate("batch", batch, dim=-1):
            for level in pgm.levels:
                for var in level:
                    if var.name in data:
                        continue  # evidence: observed, not proposed
                    if var.name in self.proposal:
                        cpd = self.proposal[var.name]
                    else:
                        cpd = pgm.name_to_factor(var.name)  # prior (mutilated)
                    parent_values = {
                        p.name: cache.get(p.name, data.get(p.name)) for p in cpd.parents
                    }
                    params = self._params_with_batch(
                        cpd, parent_values, batch, layer_kwargs.get(var.name, {})
                    )
                    d = _pyro_exact_distribution(cpd.variable, params)
                    value = pyro.sample(var.name, d)
                    cache[var.name] = reshape_value_to_event(var, value)

    # ------------------------------------------------------------------
    @staticmethod
    def _trace_log_prob(trace) -> torch.Tensor:
        """Sum per-site ``log_prob`` (shape ``(M,)``) over all sample sites."""
        trace.compute_log_prob()
        total = None
        for node in trace.nodes.values():
            if node["type"] != "sample":
                continue
            lp = node["log_prob"]
            total = lp if total is None else total + lp
        return total

    @torch.no_grad()
    def query(
        self,
        query: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor] = None,
        layer_kwargs: Dict[str, Dict] = {},
    ) -> InferenceOutput:
        """Estimate ``P(Q=q | E=e)`` for a batch via Pyro importance sampling."""
        if evidence is None:
            evidence = {}
        B = self._validate(query, evidence)

        N = self.n_samples
        M = N * B
        query_names: Set[str] = set(query.keys())

        # Pack N particles x B observations into one plate axis of size M
        # (row m -> particle n = m // B, observation b = m % B).
        def _expand(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0).expand(N, *t.shape).reshape(M, *t.shape[1:])

        data = {name: _expand(val) for name, val in evidence.items()}

        guide_tr = poutine.trace(
            lambda: self._guide_fn(data, M, layer_kwargs)
        ).get_trace()
        model = lambda: self._model_fn(data, M, layer_kwargs)
        model_tr = poutine.trace(
            poutine.replay(model, trace=guide_tr)
        ).get_trace()

        log_w = (self._trace_log_prob(model_tr) - self._trace_log_prob(guide_tr))
        log_w = log_w.reshape(N, B)
        w_tilde = torch.softmax(log_w, dim=0)  # self-normalised over particles

        match = torch.ones(N, B, device=log_w.device)
        for name, target in query.items():
            var = self.pgm.name_to_variable(name)
            sample_nb = model_tr.nodes[name]["value"].reshape(N, B, *var.shape)
            target_nb = _expand(target).reshape(N, B, *var.shape)
            match = match * _hard_match(sample_nb, target_nb)

        prob = (w_tilde * match).sum(dim=0)  # (B,)
        self._warn_ess(w_tilde)

        out = InferenceOutput()
        out.probabilities = prob
        return out

    # ------------------------------------------------------------------
    def _warn_ess(self, w_tilde: torch.Tensor) -> None:
        ess = 1.0 / (w_tilde.pow(2).sum(dim=0).clamp(min=1e-12))  # (B,)
        threshold = self.warn_low_ess * self.n_samples
        for b in (ess < threshold).nonzero(as_tuple=False).flatten().tolist():
            warnings.warn(
                f"{self.name} [row {b}]: low effective sample size "
                f"(ESS = {float(ess[b]):.1f} of {self.n_samples}). The proposal "
                "poorly approximates the posterior; refine it or increase n_samples.",
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    def _validate(
        self, query: Dict[str, torch.Tensor], evidence: Dict[str, torch.Tensor]
    ) -> int:
        if not isinstance(query, dict) or not query:
            raise ValueError(
                f"{self.name}.query() requires a non-empty 'query' dict mapping "
                "query variable names to their target Tensor values with a leading "
                "batch dimension, e.g. {'Y': torch.tensor([[1.], [0.]])}."
            )
        for role, d in (("query", query), ("evidence", evidence)):
            for vname, val in d.items():
                if not isinstance(val, torch.Tensor):
                    raise ValueError(
                        f"{self.name}: {role}[{vname!r}] must be a Tensor, "
                        f"got {type(val).__name__}."
                    )
        overlap = query.keys() & evidence.keys()
        if overlap:
            raise ValueError(
                f"{self.name}: variables {sorted(overlap)} appear in both query "
                "and evidence; a variable is either queried or observed, not both."
            )
        all_tensors = {**query, **evidence}
        for vname, val in all_tensors.items():
            if val.dim() < 2:
                raise ValueError(
                    f"{self.name}: tensor for '{vname}' has shape {tuple(val.shape)} "
                    "but a leading batch dimension is required, e.g. (B, *event)."
                )
        batch_sizes = {name: v.shape[0] for name, v in all_tensors.items()}
        if len(set(batch_sizes.values())) > 1:
            raise ValueError(f"{self.name}: mismatched batch sizes {batch_sizes}.")
        all_names = {v.name for v in self.pgm.variables}
        unknown = set(all_tensors.keys()) - all_names
        if unknown:
            raise ValueError(f"{self.name}: unknown variable names {sorted(unknown)}.")
        self._require_discrete(list(query.keys()))
        return next(iter(batch_sizes.values()))

    def _require_discrete(self, names: List[str]) -> None:
        for name in names:
            v = self.pgm.name_to_variable(name)
            if not issubclass(v.distribution, self._DISCRETE):
                raise ValueError(
                    f"{self.name}: query variable {name!r} has distribution "
                    f"{v.distribution.__name__!r}, which is continuous. Only "
                    "Bernoulli and OneHotCategorical query variables are supported "
                    "(exact equality matching needs a discrete target)."
                )
