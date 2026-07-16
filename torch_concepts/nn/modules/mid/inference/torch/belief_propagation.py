"""BeliefPropagation — loopy sum-product inference over a :class:`ProbabilisticModel`.

Consumes the unified factor interface (``scope`` + ``log_potential``), so it runs
on directed (:class:`BayesianNetwork`), undirected (:class:`MarkovNetwork`), and
mixed (chain) :class:`ProbabilisticModel` graphs identically.
It returns per-variable **marginals**; training is cross-entropy on those
marginals, which never touches the partition function ``Z``.

Preconditions (see FACTOR_GRAPH_INSTRUCTIONS.md §1)
---------------------------------------------------
Every **free** (non-evidence) variable must be discrete with finite cardinality
(Bernoulli with ``size == 1``, or a Categorical/OneHot family). Continuous
variables are supported only as **observed evidence / conditioning inputs** to
the factors' energies. A continuous free variable raises a clear error; lifting
that needs a future MCMC/variational engine (the energy representation already
supports it).

The whole pass is differentiable (unrolled message passing), so gradients flow
from a marginal loss into every factor/CPD parametrization.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple, Union

import torch

from ...models.potential import enumerable_cardinality
from ...models.probabilistic_model import ProbabilisticModel
from ...models.variable import Variable
from ....outputs import InferenceOutput
from ..utils import reshape_value_to_event
from .base import TorchBaseInference


class BeliefPropagation(TorchBaseInference):
    """Loopy belief propagation (sum-product) over a :class:`ProbabilisticModel`.

    Parameters
    ----------
    pgm : ProbabilisticModel
        Any factor graph (directed, undirected, or mixed).
    iters : int, default 5
        Number of synchronous message-passing rounds. On a tree/forest, enough
        iterations make BP exact; with loops it is the standard approximation.
    damping : float, default 0.0
        Message damping in ``[0, 1)``. ``0`` is no damping; larger values mix in
        the previous message (in probability space) for stability on loopy graphs.
    tol : float, optional
        If set, stop early once the max change in factor->variable messages
        between rounds drops below ``tol``.
    """

    name = "BeliefPropagation"

    def __init__(
        self,
        pgm: FactorGraph,
        iters: int = 5,
        damping: float = 0.0,
        tol: Optional[float] = None,
    ):
        if not isinstance(pgm, ProbabilisticModel):
            raise TypeError(
                f"BeliefPropagation requires a ProbabilisticModel, got {type(pgm).__name__}."
            )
        super().__init__(pgm)
        if int(iters) < 1:
            raise ValueError(f"iters must be >= 1, got {iters!r}.")
        if not 0.0 <= float(damping) < 1.0:
            raise ValueError(f"damping must be in [0, 1), got {damping!r}.")
        self.iters = int(iters)
        self.damping = float(damping)
        self.tol = None if tol is None else float(tol)

    def __repr__(self) -> str:
        return self._format_repr(iters=self.iters, damping=self.damping, tol=self.tol)

    # ------------------------------------------------------------------ utils
    def _dtype(self) -> torch.dtype:
        try:
            return next(self.pgm.parameters()).dtype
        except StopIteration:
            return torch.get_default_dtype()

    def _format_evidence(self, variable: Variable, value: torch.Tensor) -> torch.Tensor:
        """Cast to the model dtype and reshape to ``(batch, *variable.shape)``."""
        return reshape_value_to_event(variable, value.to(self._dtype()))

    def _encode_state(
        self, v: Variable, state: int, batch: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Value tensor for discrete ``state`` of ``v``: scalar {0,1} (binary) or one-hot."""
        card = enumerable_cardinality(v)
        if card == 2 and v.size == 1:
            return torch.full((batch, 1), float(state), dtype=dtype, device=device)
        val = torch.zeros(batch, v.size, dtype=dtype, device=device)
        val[:, state] = 1.0
        return val

    def _factor_table(
        self,
        factor,
        free_vars: List[Variable],
        fixed: Dict[Variable, torch.Tensor],
        conditioning: Dict[str, torch.Tensor],
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Log-potential table over ``free_vars`` (axis order preserved).

        Observed scope variables in ``fixed`` are baked in (per-batch). Returns
        ``None`` when the factor has no free variable (a constant w.r.t. the
        active variables, contributing nothing to messages).
        """
        free_cards = [enumerable_cardinality(v) for v in free_vars]
        if not free_cards:
            return None
        # Fast path: a tabular potential with all its scope free hands us the table.
        if not fixed and hasattr(factor, "log_potential_table"):
            return factor.log_potential_table(conditioning=conditioning, batch_size=batch)
        # Generic path: enumerate the free grid, evaluate log_potential per cell.
        cells: List[torch.Tensor] = []
        for combo in itertools.product(*[range(c) for c in free_cards]):
            assignment: Dict[Variable, torch.Tensor] = dict(fixed)
            for v, s in zip(free_vars, combo):
                assignment[v] = self._encode_state(v, s, batch, dtype, device)
            cells.append(factor.log_potential(assignment, conditioning))
        return torch.stack(cells, dim=-1).reshape(batch, *free_cards)

    # ------------------------------------------------------------------ query
    def query(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        """Run loopy BP and return per-variable marginals.

        ``out.params[name] = {'probs': marginal, 'logits': belief}`` where
        ``marginal`` is the ``(batch, cardinality)`` posterior over the
        variable's discrete states and ``logits`` is the un-normalized belief
        (``softmax(logits) == probs``), so a plain
        ``cross_entropy(out.params[name]['logits'], labels)`` trains the model.

        Only queried names that are *active* (free, computed) variables appear in
        ``params``; a fully-observed queried variable emits none (its value is
        its evidence), mirroring the directed engines.
        """
        query = self._normalize_query(query)
        self._validate_containers(query, evidence)
        query_names = set(query)

        tensors = list(evidence.values()) + [v for v in query.values() if v is not None]
        batch = tensors[0].shape[0] if tensors else 1
        dtype = self._dtype()
        device = tensors[0].device if tensors else torch.device("cpu")

        # Whole-variable evidence becomes conditioning; member (partial-plate)
        # evidence is not supported on active plates in v1.
        whole_ev, member_ev = self._split_evidence(evidence)
        if member_ev:
            raise NotImplementedError(
                "BeliefPropagation: partial-plate (member) evidence "
                f"{sorted(m for d in member_ev.values() for m in d)} is not supported "
                "in v1; observe the whole plate variable instead."
            )
        evidence_names = set(whole_ev)
        conditioning: Dict[str, torch.Tensor] = {
            name: self._format_evidence(self.pgm.variables[name], val)
            for name, val in whole_ev.items()
        }

        # Active = free variables (not observed) that participate in some factor.
        active_vars = [
            v for v in self.pgm.variables.values()
            if v.name not in evidence_names and self.pgm._var_factors[v.name]
        ]
        active_names = {v.name for v in active_vars}
        # Enumerability precondition (raises a clear error for continuous free vars).
        cards: Dict[str, int] = {v.name: enumerable_cardinality(v) for v in active_vars}

        # Build factor tables over each factor's free scope variables.
        factor_free: Dict[str, List[Variable]] = {}
        factor_table: Dict[str, torch.Tensor] = {}
        for fname, f in self.pgm.factors.items():
            self._check_conditioning(fname, f, conditioning)
            free_vars = [v for v in f.scope if v.name in active_names]
            fixed = {
                v: conditioning[v.name] for v in f.scope if v.name in evidence_names
            }
            table = self._factor_table(f, free_vars, fixed, conditioning, batch, dtype, device)
            if table is None:
                continue
            factor_free[fname] = free_vars
            factor_table[fname] = table

        # Adjacency restricted to factors that actually carry a table.
        var_factors: Dict[str, List[str]] = {
            vn: [fn for fn in self.pgm._var_factors[vn] if fn in factor_table]
            for vn in active_names
        }

        m_fv = self._run_message_passing(
            active_names, cards, factor_free, factor_table, var_factors, batch, dtype, device
        )

        # Beliefs -> marginals.
        out = InferenceOutput()
        for vn in active_names:
            belief = sum(m_fv[(fn, vn)] for fn in var_factors[vn])
            out.params[vn] = {
                "probs": torch.softmax(belief, dim=-1),
                "logits": belief,
            }

        out.params = self._expose_params(out.params, query_names)
        out.params = {n: p for n, p in out.params.items() if n in query_names}
        return out

    def _check_conditioning(self, fname: str, factor, conditioning: Dict[str, torch.Tensor]) -> None:
        """A potential's conditioning inputs must all be observed evidence."""
        for v in getattr(factor, "conditioning", []):
            if v.name not in conditioning:
                raise ValueError(
                    f"BeliefPropagation: factor {fname!r} conditions on {v.name!r}, "
                    "which must be provided as evidence."
                )

    # --------------------------------------------------------- message passing
    def _run_message_passing(
        self,
        active_names,
        cards,
        factor_free,
        factor_table,
        var_factors,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """Synchronous log-domain sum-product; returns final factor->variable messages."""
        # Init factor->variable messages to uniform (log 1 == 0).
        m_fv: Dict[Tuple[str, str], torch.Tensor] = {
            (fn, v.name): torch.zeros(batch, cards[v.name], dtype=dtype, device=device)
            for fn, free_vars in factor_free.items()
            for v in free_vars
        }
        log_d = math.log(self.damping) if self.damping > 0 else None
        log_1md = math.log(1.0 - self.damping) if self.damping > 0 else None

        for _ in range(self.iters):
            # variable -> factor: product (log-sum) of all other factors' messages.
            m_vf: Dict[Tuple[str, str], torch.Tensor] = {}
            for vn in active_names:
                fns = var_factors[vn]
                if not fns:
                    continue
                total = sum(m_fv[(fn, vn)] for fn in fns)
                for fn in fns:
                    msg = total - m_fv[(fn, vn)]
                    m_vf[(vn, fn)] = msg - torch.logsumexp(msg, dim=-1, keepdim=True)

            # factor -> variable: combine table with incoming, marginalize others.
            new_m_fv: Dict[Tuple[str, str], torch.Tensor] = {}
            max_delta = 0.0
            for fn, free_vars in factor_free.items():
                n = len(free_vars)
                combined = factor_table[fn]
                for a, w in enumerate(free_vars):
                    shape = [batch] + [1] * n
                    shape[a + 1] = cards[w.name]
                    combined = combined + m_vf[(w.name, fn)].reshape(shape)
                for a, v in enumerate(free_vars):
                    shape = [batch] + [1] * n
                    shape[a + 1] = cards[v.name]
                    tmp = combined - m_vf[(v.name, fn)].reshape(shape)
                    reduce_axes = [ax for ax in range(1, n + 1) if ax != a + 1]
                    msg = torch.logsumexp(tmp, dim=reduce_axes) if reduce_axes else tmp
                    msg = msg - torch.logsumexp(msg, dim=-1, keepdim=True)
                    if log_d is not None:
                        old = m_fv[(fn, v.name)]
                        msg = torch.logaddexp(msg + log_1md, old + log_d)
                        msg = msg - torch.logsumexp(msg, dim=-1, keepdim=True)
                    if self.tol is not None:
                        max_delta = max(
                            max_delta, (msg - m_fv[(fn, v.name)]).abs().max().item()
                        )
                    new_m_fv[(fn, v.name)] = msg
            m_fv = new_m_fv
            if self.tol is not None and max_delta < self.tol:
                break

        return m_fv
