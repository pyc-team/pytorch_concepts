"""RejectionSampling — approximate conditional inference via rejection sampling.

Algorithm
---------
1. Draw ``n_samples`` joint samples from the PGM in topological order
   (:meth:`RejectionSampling._draw_joint`), clamping any root evidence.
2. For each row b in the batch:
   a. Build an **evidence mask**: samples where every E variable equals e_b.
   b. Build a **full mask**: evidence mask AND every Q variable equals q_b.
   c. Estimate P(Q=q_b | E=e_b) = |full mask| / |evidence mask|.
   d. Return hidden-variable samples from the full-masked pool as draws from
      P(H | Q=q_b, E=e_b).

Inputs / Outputs
----------------
Query and evidence tensors are ``(*leading, *event)`` with at least one leading
(batch-like) dimension. Outputs are ragged because different rows may accept
different numbers of samples:

- ``out.probabilities``  — ``(*leading,)`` tensor of P(Q=q_b | E=e_b).

Constraints
-----------
- Query variables and **non-root** evidence variables **must** be discrete
  (Bernoulli, Categorical, OneHotCategorical): they are matched by exact
  equality.
- Evidence on a **root** variable may be continuous (e.g. an input embedding):
  it is clamped into every joint draw, never matched.
- Hidden variables (neither query nor evidence) may be continuous.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Set

import torch

from ...graph.bayesian_network import BayesianNetwork
from ...distributions import spec_for
from ....outputs import InferenceOutput
from .ancestral import AncestralSamplingInference
from .base import TorchBaseInference


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

    The joint draw is :class:`AncestralSamplingInference` in ``exact=True`` mode:
    the same topological traversal, but hard draws from the exact family. The
    relaxed surrogate would propagate *soft* parent values, which leaves the
    marginals right and the joint wrong — and rejection filters on the joint.

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

    def __init__(
        self,
        pgm: BayesianNetwork,
        n_samples: int = 1_000,
        warn_low_acceptance: float = 0.01,
    ) -> None:
        super().__init__(pgm)
        self._require_directed()
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
    def _root_names(self) -> Set[str]:
        """Whole-variable names of the roots.

        Evidence on these is clamped during generation rather than matched.
        A plate member's name is never in this set, so member evidence is
        always matched, even on a root plate.
        """
        return {
            v.name for v in self.pgm.variables.values()
            if self.pgm.factors[v.name].is_root
        }

    def _require_discrete(self, names: List[str], role: str) -> None:
        for name in names:
            v = self.pgm.resolve(name)  # a member's family is its plate's family
            spec = spec_for(v.distribution, f"{self.name}: {name!r}")
            if not spec.is_discrete:
                raise ValueError(
                    f"{self.name}: {role} variable {name!r} has "
                    f"distribution {v.distribution.__name__!r} which is not "
                    "discrete. Exact equality matching needs a discrete family "
                    "(Bernoulli / Categorical / OneHotCategorical, or their "
                    "relaxed variants) for query/evidence variables."
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
        sampler: AncestralSamplingInference,
        root_evidence: Dict[str, torch.Tensor],
        layer_kwargs: Dict[str, Dict],
    ) -> Dict[str, torch.Tensor]:
        """Draw ``n_samples`` hard joint samples conditioned on root evidence.

        Root evidence is expanded to the sample dimension and clamped, so all
        ``n_samples`` already agree with those observations; every other
        variable is drawn from its exact family in topological order. Returns
        the raw per-variable cache, in the member layout.
        """
        N = self.n_samples
        evidence = {}
        for name, val in root_evidence.items():
            val = self.pgm.variables[name].to_member(val)
            evidence[name] = val.unsqueeze(0).expand(N, *val.shape)
        with torch.no_grad():
            _, cache, _ = sampler._run(
                query={v.name: None for v in self.pgm.variables.values()},
                evidence=evidence,
                layer_kwargs=layer_kwargs,
                n_samples=N,
            )
        return cache

    def _build_mask(
        self,
        stacked_samples: Dict[str, torch.Tensor],
        obs_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build an ``(N,)`` boolean mask for a single-observation dict."""
        mask = torch.ones(self.n_samples, dtype=torch.bool)
        for name, val in obs_dict.items():
            # ``extract`` reads a whole variable or a member column uniformly, so
            # member evidence joins the rejection mask (exact conditioning).
            mask = mask & _match(self.pgm.extract(name, stacked_samples), val)
        return mask

    # ------------------------------------------------------------------
    def query(
        self,
        query: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor] = None,
        layer_kwargs: Dict[str, Dict] = {},
    ) -> InferenceOutput:
        """Run rejection sampling to estimate P(Q=q_b | E=e_b) for a batch.

        Query and evidence accept plate-member names as well as variable names.
        Member evidence joins the rejection mask, so for this engine it is **exact
        conditioning** (not the value forcing the other engines apply).

        Tensors may carry any number of leading (batch-like) dimensions. Because
        the estimator loops over observations independently, those dimensions are
        collapsed into one batch axis for the run and restored on
        ``out.probabilities``, which comes back shaped ``(*leading,)``.
        """
        if evidence is None:
            evidence = {}

        self._validate(query, evidence)
        leading = self._query_leading_shape(query, evidence)
        query = self._collapse_leading(query, leading)
        evidence = self._collapse_leading(evidence, leading)
        B = math.prod(leading)

        # Partition evidence into root vars (conditioned during generation)
        # and non-root vars (handled by rejection filtering). The PGM might
        # require constant evidence on certain roots (e.g. a root image).
        root_names = self._root_names()
        root_evidence_names = set(evidence.keys()) & root_names
        nonroot_evidence_names = set(evidence.keys()) - root_names

        probs: List[float] = []
        # Local, so it never becomes a submodule of self (state_dict untouched).
        sampler = AncestralSamplingInference(self.pgm, exact=True)

        for b in range(B):
            root_evidence_b    = {name: evidence[name][b] for name in root_evidence_names}
            nonroot_evidence_b = {name: evidence[name][b] for name in nonroot_evidence_names}
            query_b            = {name: v[b] for name, v in query.items()}

            stacked_samples = self._draw_joint(sampler, root_evidence_b, layer_kwargs)

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

        return InferenceOutput(
            probabilities=self._restore_leading(torch.tensor(probs), leading)
        )

    def _validate(
        self,
        query: Dict[str, torch.Tensor],
        evidence: Dict[str, torch.Tensor],
    ) -> None:
        """Validate the query/evidence containers."""
        if not isinstance(query, dict):
            raise ValueError(
                f"{self.name}.query() requires 'query' to be a dict mapping "
                "variable names to their target Tensor values with a leading batch "
                "dimension, e.g. {'Y': torch.tensor([[1.], [0.]])}."
            )

        self._require_tensor_values(query, "query")
        self._require_tensor_values(evidence, "evidence")

        all_tensors = {**query, **evidence}

        unknown = set(all_tensors.keys()) - self.pgm.queryable_names
        if unknown:
            raise ValueError(f"{self.name}: unknown variable names {sorted(unknown)}.")

        for name, v in all_tensors.items():
            if v.dim() < 2:
                raise ValueError(
                    f"{self.name}: tensor for '{name}' has shape {tuple(v.shape)} "
                    "but at least one leading batch dimension is required, e.g. shape "
                    "(*leading, *event). Use tensor.unsqueeze(0) for a single observation."
                )

        leadings = {
            name: tuple(self._leading_shape(name, v)) for name, v in all_tensors.items()
        }
        if len(set(leadings.values())) > 1:
            raise ValueError(
                f"{self.name}: mismatched leading (batch) dimensions {leadings}."
            )

        self._require_discrete(list(query.keys()), "query")
        # Root evidence is clamped into the draw, never matched, so it may be
        # continuous; only evidence the mask compares must be discrete.
        root_names = self._root_names()
        self._require_discrete(
            [name for name in evidence if name not in root_names], "non-root evidence"
        )
