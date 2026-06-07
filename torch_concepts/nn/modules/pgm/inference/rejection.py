"""RejectionSampling — approximate conditional inference via rejection sampling.

Algorithm
---------
1. Draw ``n_samples`` joint samples from the PGM by calling
   :class:`ForwardInference` (ancestral mode) with every variable declared as
   a query and no evidence. 
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

import pyro.distributions as dist
import torch

from ..models.bayesian_network import BayesianNetwork
from .base import BaseInference
from .ancestral import AncestralInference
from .forward import ForwardInference
from .outputs import InferenceOutput


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


class RejectionSampling(BaseInference):
    """Approximate conditional inference via pure rejection sampling.

    Internally delegates joint sampling to :class:`ForwardInference`
    (ancestral mode) so the topological ordering logic is not duplicated.

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
        self._sampler = AncestralInference(pgm, p_int=0.0)

    # ------------------------------------------------------------------
    def _require_discrete(self, names: List[str], role: str) -> None:
        for name in names:
            v = self.pgm.name_to_variable[name]
            if v.distribution not in self._DISCRETE:
                raise ValueError(
                    f"RejectionSampling: {role} variable {name!r} has "
                    f"distribution {v.distribution.__name__!r} which is "
                    "continuous. Only Bernoulli, Categorical and "
                    "OneHotCategorical are supported for query/evidence variables."
                )

    def _require_tensor_values(self, d: Dict[str, object], role: str) -> None:
        for name, val in d.items():
            if not isinstance(val, torch.Tensor):
                raise ValueError(
                    f"RejectionSampling: {role}[{name!r}] must be a Tensor "
                    f"with the target value, got {type(val).__name__!r}."
                )

    # ------------------------------------------------------------------
    def _draw_joint(
        self,
        root_evidence: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Draw ``n_samples`` joint samples conditioned on root evidence.

        Root evidence variables are clamped directly in :class:`ForwardInference`
        so that all ``n_samples`` samples already agree with those observations.
        Non-root evidence variables are left to vary freely and are filtered
        later via rejection.

        Parameters
        ----------
        root_evidence : dict[name → Tensor(*event)]
            Single-row (no batch dim) values for evidence variables that are
            roots.  Expanded to ``(n_samples, *event)`` internally.

        Returns
        -------
        dict[name → Tensor(n_samples, *event)]
        """
        # Expand each root evidence value to the full sample batch size.
        batched_root_evidence = {
            name: val.unsqueeze(0).expand(self.n_samples, *val.shape)
            for name, val in root_evidence.items()
        }

        # Dummy query just to set batch_size=n_samples; values ignored (p_int=0).
        dummy_query = {
            v.name: torch.zeros(self.n_samples, v.size)
            for v in self.pgm.sorted_variables
            if v.name not in root_evidence
        }

        with torch.no_grad():
            fwd = self._sampler.query(
                query=dummy_query,
                evidence=batched_root_evidence,
            )

        # ForwardInference stores evidence variables in out.samples only when
        # sampled (mode=ancestral and not in evidence_names). Reconstruct the
        # full dict so root-clamped variables are also present for masking.
        samples = dict(fwd.samples)
        for name, val in batched_root_evidence.items():
            samples[name] = val
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
    ) -> InferenceOutput:
        """Run rejection sampling to estimate P(Q=q_b | E=e_b) for a batch.

        Parameters
        ----------
        query : dict[str, Tensor]
            Target values for each query variable.  Tensors must have shape
            ``(B, *event)``, e.g. ``{'Y': torch.tensor([[1.], [0.], [1.]])}``.
            All variables must be discrete.
        evidence : dict[str, Tensor]
            Conditioning values.  Tensors must have shape ``(B, *event)``.
            All variables must be discrete.

        Returns
        -------
        InferenceOutput
            - ``out.probabilities`` — ``(B,)`` tensor of P(Q=q_b | E=e_b).
        """
        if evidence is None:
            evidence = {}

        B = self._validate(query, evidence)

        # Partition evidence into root vars (conditioned during generation)
        # and non-root vars (handled by rejection filtering).
        # This step is done since the PGM we instantiate might require constant 
        # evidence on certain roots (e.g. a root representing an image).
        root_names = {name for name, cpd in self.pgm.factors.items() if cpd.is_root}
        root_evidence_names = set(evidence.keys()) & root_names
        nonroot_evidence_names = set(evidence.keys()) - root_names

        probs: List[float] = []

        for b in range(B):
            root_evidence_b    = {name: evidence[name][b] for name in root_evidence_names}
            nonroot_evidence_b = {name: evidence[name][b] for name in nonroot_evidence_names}
            query_b            = {name: v[b] for name, v in query.items()}

            # Condition generation on root evidence; sample the rest freely.
            stacked_samples = self._draw_joint(root_evidence_b)

            # Filter only by non-root evidence and query values.
            e_mask  = self._build_mask(stacked_samples, nonroot_evidence_b)
            qe_mask = e_mask & self._build_mask(stacked_samples, query_b)

            n_e = int(e_mask.sum())
            n_qe = int(qe_mask.sum())

            if n_e == 0:
                warnings.warn(
                    f"RejectionSampling [row {b}]: P(E=e) ≈ 0 — no samples matched "
                    "the evidence. Increase n_samples or check the evidence values.",
                    stacklevel=2,
                )
                prob_b = 0.0
            else:
                prob_b = n_qe / n_e
                joint_rate = n_qe / self.n_samples
                if joint_rate < self.warn_low_acceptance:
                    warnings.warn(
                        f"RejectionSampling [row {b}]: low joint acceptance rate "
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
        """Validate inputs and return ``B`` (the batch size).

        Raises ``ValueError`` on any of:
        - query is not a dict
        - any value is not a Tensor
        - any tensor is missing the leading batch dimension
        - mismatched batch sizes across query and evidence
        - unknown variable names
        - non-discrete query or evidence variables
        """
        if not isinstance(query, dict):
            raise ValueError(
                "RejectionSampling.query() requires 'query' to be a dict mapping "
                "variable names to their target Tensor values with a leading batch "
                "dimension, e.g. {'Y': torch.tensor([[1.], [0.]])}."
            )

        self._require_tensor_values(query, "query")
        self._require_tensor_values(evidence, "evidence")

        all_tensors = {**query, **evidence}

        for name, v in all_tensors.items():
            if v.dim() < 2:
                raise ValueError(
                    f"RejectionSampling: tensor for '{name}' has shape {tuple(v.shape)} "
                    "but a leading batch dimension is required, e.g. shape (B, *event). "
                    "Use tensor.unsqueeze(0) for a single observation."
                )

        batch_sizes = {name: v.shape[0] for name, v in all_tensors.items()}
        if len(set(batch_sizes.values())) > 1:
            raise ValueError(
                f"RejectionSampling: mismatched batch sizes {batch_sizes}."
            )
        B = next(iter(batch_sizes.values()))

        all_names = set(self.pgm.name_to_variable.keys())
        unknown = set(all_tensors.keys()) - all_names
        if unknown:
            raise ValueError(f"RejectionSampling: unknown variable names {sorted(unknown)}.")

        self._require_discrete(list(query.keys()), "query")
        self._require_discrete(list(evidence.keys()), "evidence")

        return B

