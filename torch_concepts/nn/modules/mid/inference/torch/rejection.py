"""RejectionSampling — approximate conditional inference via rejection sampling.

Algorithm
---------
1. Draw ``n_samples`` joint samples from the PGM by calling
   :class:`AncestralInference` (ancestral mode) with every variable
   declared as a query and no evidence.
2. For each row b in the batch:
   a. Build an **evidence mask**: samples where every E variable equals e_b.
   b. Build a **full mask**: evidence mask AND every Q variable equals q_b.
   c. Estimate P(Q=q_b | E=e_b) = |full mask| / |evidence mask|.
   d. Return hidden-variable samples from the full-masked pool as draws from
      P(H | Q=q_b, E=e_b).

Inputs / Outputs
----------------
Query and evidence tensors must always have a leading batch dimension ``(B, *event)``.
Outputs are ragged because different rows may accept different numbers of samples:

- ``out.probabilities``  — ``(B,)`` tensor of P(Q=q_b | E=e_b).

Constraints
-----------
- All query and evidence variables **must** be discrete (Bernoulli,
  Categorical, OneHotCategorical).  Exact equality matching is used.
- Hidden variables (neither query nor evidence) may be continuous.
"""

from __future__ import annotations

import warnings
from typing import Dict, List

import torch
import torch.distributions as dist

from ...models.bayesian_network import BayesianNetwork
from ....outputs import InferenceOutput
from ..utils import reshape_value_to_event
from .base import TorchBaseInference


_DISCRETE = frozenset({dist.Bernoulli, dist.Categorical, dist.OneHotCategorical})


def _match(sampled: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Boolean mask ``(N,)`` — rows of *sampled* that equal *observed*.

    *sampled* has shape ``(N, *event)``; *observed* has shape ``(*event,)``.
    Dims are prepended to *observed* until they align, then broadcast.
    """
    obs = observed.to(sampled.dtype)
    while obs.dim() < sampled.dim():
        obs = obs.unsqueeze(0)
    eq = sampled == obs.expand_as(sampled)
    while eq.dim() > 1:
        eq = eq.all(dim=-1)
    return eq


class RejectionSampling(TorchBaseInference):
    """Approximate conditional inference via pure rejection sampling.

    Internally delegates joint sampling to :class:`AncestralInference`
    so the topological ordering logic is not duplicated.

    Parameters
    ----------
    pgm : BayesianNetwork
        The probabilistic graphical model to query.
    n_samples : int
        Number of joint samples drawn per observation on each ``query`` call.
    warn_low_acceptance : float
        Warn when the acceptance rate drops below this fraction (default 1 %).
    """

    name = "RejectionSampling"
    _DISCRETE = _DISCRETE

    def __init__(
        self,
        pgm: BayesianNetwork,
        n_samples: int = 1_000,
        warn_low_acceptance: float = 0.01,
    ) -> None:
        super().__init__(pgm)
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
        self.n_samples = int(n_samples)
        self.warn_low_acceptance = float(warn_low_acceptance)

    def __repr__(self) -> str:
        return self._format_repr(
            n_samples=self.n_samples,
            warn_low_acceptance=self.warn_low_acceptance,
        )

    # ------------------------------------------------------------------
    def _require_discrete(self, names: List[str], role: str) -> None:
        for name in names:
            v = self.pgm.variables[name]
            if not any(issubclass(v.distribution, d) for d in self._DISCRETE):
                raise ValueError(
                    f"{self.name}: {role} variable {name!r} has "
                    f"distribution {v.distribution.__name__!r} which is "
                    "continuous. Only Bernoulli, Categorical and "
                    "OneHotCategorical are supported for query/evidence variables."
                )

    def _require_tensor_values(self, d: Dict[str, object], role: str) -> None:
        for name, val in d.items():
            if not isinstance(val, torch.Tensor):
                raise ValueError(
                    f"{self.name}: {role}[{name!r}] must be a Tensor "
                    f"with the target value, got {type(val).__name__!r}."
                )

    # ------------------------------------------------------------------
    def _draw_joint(
        self,
        root_evidence: Dict[str, torch.Tensor],
        layer_kwargs: Dict[str, Dict],
    ) -> Dict[str, torch.Tensor]:
        """Draw ``n_samples`` hard joint samples conditioned on root evidence.

        Root evidence variables are clamped so that all ``n_samples`` samples
        already agree with those observations. Every other variable is sampled
        from its exact (non-relaxed) discrete or continuous distribution using
        the topological order of the PGM. Hard sampling (``dist.sample()``) is
        used so that exact equality matching in :meth:`_build_mask` works.
        """
        N = self.n_samples
        samples: Dict[str, torch.Tensor] = {}

        # Pre-expand root-clamped evidence to the sample dimension.
        for name, val in root_evidence.items():
            samples[name] = val.unsqueeze(0).expand(N, *val.shape)

        with torch.no_grad():
            for level in self.pgm.levels:
                for var in level:
                    name = var.name
                    if name in samples:
                        continue  # already set (root evidence)
                    cpd = self.pgm.factors[name]
                    if cpd.is_root:
                        params = cpd(parent_values={})
                        params = {k: v.unsqueeze(0).expand(N, *v.shape)
                                  for k, v in params.items()}
                    else:
                        parent_values = {p.name: samples[p.name] for p in cpd.parents}
                        params = cpd(parent_values=parent_values,
                                     **layer_kwargs.get(name, {}))

                    D = var.distribution
                    if issubclass(D, (dist.Bernoulli, dist.RelaxedBernoulli)):
                        s = dist.Bernoulli(**params).sample()
                    elif issubclass(D, (dist.OneHotCategorical,
                                        dist.RelaxedOneHotCategorical)):
                        s = dist.OneHotCategorical(**params).sample()
                    elif issubclass(D, dist.Categorical):
                        s = dist.Categorical(**params).sample()
                    else:
                        # Continuous: use exact reparameterised draw.
                        from ..utils import build_distribution
                        s = build_distribution(var, params).rsample()

                    samples[name] = reshape_value_to_event(var, s)

        return samples

    def _build_mask(
        self,
        stacked_samples: Dict[str, torch.Tensor],
        obs_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build an ``(N,)`` boolean mask for a single-observation dict."""
        mask = torch.ones(self.n_samples, dtype=torch.bool)
        for name, val in obs_dict.items():
            mask = mask & _match(stacked_samples[name], val)
        return mask

    # ------------------------------------------------------------------
    def query(
        self,
        query: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor] = None,
        layer_kwargs: Dict[str, Dict] = {},
    ) -> InferenceOutput:
        """Run rejection sampling to estimate P(Q=q_b | E=e_b) for a batch."""
        if evidence is None:
            evidence = {}

        B = self._validate(query, evidence)

        # Partition evidence into root vars (conditioned during generation)
        # and non-root vars (handled by rejection filtering). The PGM might
        # require constant evidence on certain roots (e.g. a root image).
        root_names = {
            v.name for v in self.pgm.variables.values()
            if self.pgm.factors[v.name].is_root
        }
        root_evidence_names = set(evidence.keys()) & root_names
        nonroot_evidence_names = set(evidence.keys()) - root_names

        probs: List[float] = []

        for b in range(B):
            root_evidence_b    = {name: evidence[name][b] for name in root_evidence_names}
            nonroot_evidence_b = {name: evidence[name][b] for name in nonroot_evidence_names}
            query_b            = {name: v[b] for name, v in query.items()}

            stacked_samples = self._draw_joint(root_evidence_b, layer_kwargs)

            e_mask  = self._build_mask(stacked_samples, nonroot_evidence_b)
            qe_mask = e_mask & self._build_mask(stacked_samples, query_b)

            n_e = int(e_mask.sum())
            n_qe = int(qe_mask.sum())

            if n_e == 0:
                warnings.warn(
                    f"{self.name} [row {b}]: P(E=e) ≈ 0 — no samples matched "
                    "the evidence. Increase n_samples or check the evidence values.",
                    stacklevel=2,
                )
                prob_b = 0.0
            else:
                prob_b = n_qe / n_e
                joint_rate = n_qe / self.n_samples
                if joint_rate < self.warn_low_acceptance:
                    warnings.warn(
                        f"{self.name} [row {b}]: low joint acceptance rate "
                        f"({n_qe}/{self.n_samples} = {joint_rate:.4f}). "
                        "Consider increasing n_samples.",
                        stacklevel=2,
                    )
            probs.append(prob_b)

        out = InferenceOutput()
        out.probabilities = torch.tensor(probs)
        return out

    def _validate(
        self,
        query: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor],
    ) -> int:
        """Validate inputs and return ``B`` (the batch size)."""
        if not isinstance(query, dict):
            raise ValueError(
                f"{self.name}.query() requires 'query' to be a dict mapping "
                "variable names to their target Tensor values with a leading batch "
                "dimension, e.g. {'Y': torch.tensor([[1.], [0.]])}."
            )

        self._require_tensor_values(query, "query")
        self._require_tensor_values(evidence, "evidence")

        all_tensors = {**query, **evidence}

        for name, v in all_tensors.items():
            if v.dim() < 2:
                raise ValueError(
                    f"{self.name}: tensor for '{name}' has shape {tuple(v.shape)} "
                    "but a leading batch dimension is required, e.g. shape (B, *event). "
                    "Use tensor.unsqueeze(0) for a single observation."
                )

        batch_sizes = {name: v.shape[0] for name, v in all_tensors.items()}
        if len(set(batch_sizes.values())) > 1:
            raise ValueError(
                f"{self.name}: mismatched batch sizes {batch_sizes}."
            )
        B = next(iter(batch_sizes.values()))

        all_names = {v.name for v in self.pgm.variables.values()}
        unknown = set(all_tensors.keys()) - all_names
        if unknown:
            raise ValueError(f"{self.name}: unknown variable names {sorted(unknown)}.")

        self._require_discrete(list(query.keys()), "query")
        self._require_discrete(list(evidence.keys()), "evidence")

        return B
