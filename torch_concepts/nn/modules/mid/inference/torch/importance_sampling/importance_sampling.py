"""ImportanceSampling — approximate conditional inference via importance sampling.

Estimates ``P(Q=q | E=e)`` for a :class:`BayesianNetwork` using a modular,
user-defined proposal :math:`q_\\phi(z \\mid e)` (see :class:`BaseProposal`).
Unlike :class:`RejectionSampling`, which draws from the prior and rejects on the
evidence, this engine draws the non-evidence variables from a proposal that is
*conditioned on the evidence* and corrects the mismatch with importance weights.
This is the right tool when the evidence sits on the leaves while the query
variables sit near the roots: there is no forward path to propagate the
observation, but a proposal can be built to approximate the posterior directly.

Algorithm
---------
For a batch of ``B`` observations, draw ``N`` importance samples each.
Let ``z`` denote the non-evidence variables (query ``Q`` and hidden ``H``).

1. Sample ``z ~ q_phi(z | e)`` from the proposal (evidence ``e`` clamped),
   obtaining values for every variable and the joint log-density ``log q``.
2. Score the model joint with the evidence clamped::

       log p(z, e) = sum_{i in E} log p(e_i | PA_i)        # exact distribution
                   + sum_{i in z} log p(x_i | PA_i)        # relaxed distribution

3. Importance log-weight ``log w = log p(z, e) - log q``.
4. Self-normalised weights ``w_tilde = softmax(log w)`` over the ``N`` axis.
5. Differentiable soft indicator ``match in [0, 1]`` that the relaxed query
   samples equal the target ``q`` (exact as temperature -> 0).
6. ``P(Q=q | E=e) ~= sum_n w_tilde_n * match_n``.

Inputs / Outputs
----------------
``query`` maps each query variable to its **target** tensor ``q`` (shape
``(B, *event)``); ``evidence`` maps each observed variable to ``e``. Query
variables must be discrete (Bernoulli / Categorical / OneHotCategorical
families). Evidence may be continuous. ``out.probabilities`` is a ``(B,)``
tensor of ``P(Q=q | E=e)``.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Set, Union

import torch
import torch.distributions as dist

from ....models.bayesian_network import BayesianNetwork
from ...utils import build_distribution, make_temperature_schedule
from .....outputs import InferenceOutput
from ..base import TorchBaseInference
from ..utils import build_relaxed_distribution
from .base_proposal import BaseProposal


# Discrete families admissible as query variables (relaxed variants included:
# a variable declared as RelaxedBernoulli is conceptually a binary node).
_DISCRETE = (
    dist.Bernoulli,
    dist.RelaxedBernoulli,
    dist.Categorical,
    dist.OneHotCategorical,
    dist.RelaxedOneHotCategorical,
)
_BERNOULLI = (dist.Bernoulli, dist.RelaxedBernoulli)
_ONEHOT = (dist.OneHotCategorical, dist.RelaxedOneHotCategorical)


def _soft_match(
    variable, sample: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Differentiable membership of a relaxed ``sample`` in the target value.

    ``sample`` and ``target`` are shaped ``(N, B, *event)``; the return is
    ``(N, B)`` in ``[0, 1]`` and converges to the hard indicator
    ``1[sample == target]`` as the relaxation temperature goes to zero.

    * Bernoulli family: ``prod_d  s_d if t_d == 1 else (1 - s_d)``.
    * OneHotCategorical family: ``<s, t>`` over the class (last) axis.
    """
    D = variable.distribution
    s = sample
    t = target.to(s.dtype)
    if issubclass(D, _BERNOULLI):
        m = t * s + (1.0 - t) * (1.0 - s)
        return m.flatten(2).prod(dim=-1)
    if issubclass(D, _ONEHOT):
        m = (s * t).sum(dim=-1)  # contract the class axis
        while m.dim() > 2:       # product over any remaining event axes
            m = m.prod(dim=-1)
        return m
    raise ValueError(
        f"ImportanceSampling: query variable {variable.name!r} has distribution "
        f"{D.__name__!r}, which has no soft match. Query variables must be "
        "Bernoulli, Categorical or OneHotCategorical."
    )


class ImportanceSampling(TorchBaseInference):
    """Importance-sampling estimator of ``P(Q=q | E=e)``.

    Parameters
    ----------
    pgm : BayesianNetwork
        The probabilistic graphical model to query.
    proposal : BaseProposal
        The modular proposal :math:`q_\\phi(z \\mid e)`. Its learnable
        parameters are registered with this engine (and therefore trainable),
        while the PGM is shared by reference.
    n_samples : int
        Number of importance samples drawn per observation (default ``1000``).
    initial_temperature, annealing, annealing_rate
        Relaxation-temperature schedule for the differentiable discrete
        distributions; see
        :func:`~torch_concepts.nn.modules.mid.inference.utils.make_temperature_schedule`.
    warn_low_ess : float
        Warn when the effective sample size drops below this fraction of
        ``n_samples`` (default ``0.01``); a symptom of a poor proposal.
    """

    name = "ImportanceSampling"
    _DISCRETE = _DISCRETE

    def __init__(
        self,
        pgm: BayesianNetwork,
        proposal: BaseProposal,
        n_samples: int = 1_000,
        initial_temperature: float = 1.0,
        annealing: Union[str, Callable[[int], float]] = "constant",
        annealing_rate: float = 0.0,
        warn_low_ess: float = 0.01,
    ) -> None:
        super().__init__(pgm)
        if not isinstance(proposal, BaseProposal):
            raise TypeError(
                f"{self.name}: `proposal` must be a BaseProposal subclass, "
                f"got {type(proposal).__name__}."
            )
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
        self.proposal = proposal
        self.n_samples = int(n_samples)
        self.warn_low_ess = float(warn_low_ess)
        # Retained for repr/introspection; the live schedule lives in ``_schedule``.
        self.initial_temperature = float(initial_temperature)
        self.annealing = annealing
        self.annealing_rate = float(annealing_rate)

        self._schedule = make_temperature_schedule(
            initial_temperature, annealing, annealing_rate
        )
        self._step = 0
        self.register_buffer(
            "_temperature", torch.tensor(float(self._schedule(self._step)))
        )

    def __repr__(self) -> str:
        return self._format_repr(
            proposal=self.proposal,
            n_samples=self.n_samples,
            initial_temperature=self.initial_temperature,
            annealing=self.annealing,
            annealing_rate=self.annealing_rate,
            warn_low_ess=self.warn_low_ess,
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self._temperature

    def step(self) -> None:
        """Advance the temperature schedule by one step."""
        self._step += 1
        self._temperature.fill_(float(self._schedule(self._step)))

    # ------------------------------------------------------------------
    def _model_log_joint(
        self,
        samples: Dict[str, torch.Tensor],
        evidence_names: Set[str],
        temperature: torch.Tensor,
        batch_size: int,
        layer_kwargs: Dict[str, Dict],
    ) -> torch.Tensor:
        """``log p(z, e)`` over all variables, evidence clamped.

        Evidence variables are scored with their **exact** distribution (a
        fixed observed value is in-support and the exact likelihood is what the
        weight requires). Non-evidence variables are scored with the **relaxed**
        distribution, matching the proposal's family so the ``p / q`` ratio is
        consistent. Returns a ``(batch_size,)`` tensor.
        """
        pgm = self.pgm
        log_p = torch.zeros(batch_size, device=temperature.device)
        for var in pgm.sorted_variables:
            name = var.name
            cpd = pgm.factors[name]
            if cpd.is_root:
                params = cpd(parent_values={})
                params = {
                    k: v.unsqueeze(0).expand(batch_size, *v.shape)
                    for k, v in params.items()
                }
            else:
                parent_values = {p.name: samples[p.name] for p in cpd.parents}
                params = cpd(parent_values=parent_values, **layer_kwargs.get(name, {}))

            value = samples[name].reshape(batch_size, var.size)
            if name in evidence_names:
                d = build_distribution(var, params)
            else:
                # validate_args=False: relaxed samples may sit on the support
                # boundary at low temperature (see build_relaxed_distribution).
                d = build_relaxed_distribution(var, params, temperature, validate_args=False)
            log_p = log_p + d.log_prob(value)
        return log_p

    # ------------------------------------------------------------------
    def query(
        self,
        query: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor] = None,
        layer_kwargs: Dict[str, Dict] = {},
    ) -> InferenceOutput:
        """Estimate ``P(Q=q | E=e)`` for a batch via importance sampling."""
        if evidence is None:
            evidence = {}
        B = self._validate(query, evidence)

        N = self.n_samples
        M = N * B
        temperature = self.temperature
        query_names: Set[str] = set(query.keys())
        evidence_names: Set[str] = set(evidence.keys())

        # Pack the N importance draws and the B observations into one leading
        # axis of size M = N * B (row m corresponds to sample n = m // B,
        # observation b = m % B), so every CPD / distribution call sees a single
        # batch dimension. Reshape to (N, B) only for the final reductions.
        def _expand(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0).expand(N, *t.shape).reshape(M, *t.shape[1:])

        evidence_M = {name: _expand(val) for name, val in evidence.items()}

        samples, log_q = self.proposal.sample(
            query_names, evidence_M, M, temperature, layer_kwargs
        )
        log_p = self._model_log_joint(
            samples, evidence_names, temperature, M, layer_kwargs
        )

        log_w = (log_p - log_q).reshape(N, B)
        w_tilde = torch.softmax(log_w, dim=0)  # self-normalised over samples

        match = torch.ones(N, B, device=temperature.device)
        for name, target in query.items():
            var = self.pgm.variables[name]
            sample_nb = samples[name].reshape(N, B, *var.shape)
            target_nb = _expand(target).reshape(N, B, *var.shape)
            match = match * _soft_match(var, sample_nb, target_nb)

        prob = (w_tilde * match).sum(dim=0)  # (B,)

        self._warn_ess(w_tilde)

        out = InferenceOutput()
        out.probabilities = prob
        return out

    # ------------------------------------------------------------------
    def _warn_ess(self, w_tilde: torch.Tensor) -> None:
        """Warn per row when the effective sample size is low."""
        ess = 1.0 / (w_tilde.pow(2).sum(dim=0).clamp(min=1e-12))  # (B,)
        threshold = self.warn_low_ess * self.n_samples
        low = (ess < threshold).nonzero(as_tuple=False).flatten().tolist()
        for b in low:
            warnings.warn(
                f"{self.name} [row {b}]: low effective sample size "
                f"(ESS = {float(ess[b]):.1f} of {self.n_samples}). The proposal "
                "poorly approximates the posterior; refine it or increase n_samples.",
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    def _validate(
        self,
        query: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor],
    ) -> int:
        """Validate inputs and return ``B`` (the batch size)."""
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
                    "but a leading batch dimension is required, e.g. (B, *event). "
                    "Use tensor.unsqueeze(0) for a single observation."
                )

        batch_sizes = {name: v.shape[0] for name, v in all_tensors.items()}
        if len(set(batch_sizes.values())) > 1:
            raise ValueError(f"{self.name}: mismatched batch sizes {batch_sizes}.")

        all_names = {v.name for v in self.pgm.variables.values()}
        unknown = set(all_tensors.keys()) - all_names
        if unknown:
            raise ValueError(f"{self.name}: unknown variable names {sorted(unknown)}.")

        self._require_discrete(list(query.keys()))
        return next(iter(batch_sizes.values()))

    def _require_discrete(self, names: List[str]) -> None:
        for name in names:
            v = self.pgm.variables[name]
            if not issubclass(v.distribution, self._DISCRETE):
                raise ValueError(
                    f"{self.name}: query variable {name!r} has distribution "
                    f"{v.distribution.__name__!r}, which is continuous. Only "
                    "Bernoulli, Categorical and OneHotCategorical query variables "
                    "are supported (the soft indicator needs a discrete target)."
                )
