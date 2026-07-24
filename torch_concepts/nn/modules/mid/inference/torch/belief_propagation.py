"""BeliefPropagation — loopy sum-product inference over a :class:`ProbabilisticModel`."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch

from ...graph.probabilistic_model import ProbabilisticModel
from ...variable import Variable
from ....outputs import InferenceOutput
from ..utils import enumerable_cardinality, reshape_value_to_event
from .base import TorchBaseInference


#: A numerically safe stand-in for ``log 0`` in float32. Real ``-inf`` poisons
#: the "total minus self" exclusive sums with ``nan`` (``-inf - -inf``); ``-1e4``
#: is small enough that ``exp`` of it underflows to ``0`` in every ``logsumexp``
#: while staying finite under subtraction.
LOG0 = -1e4

#: One compiled bucket: ``(arity, cardinalities, log-potential stack, edge ids)``.
_Bucket = Tuple[int, Tuple[int, ...], torch.Tensor, torch.Tensor]


class BeliefPropagation(TorchBaseInference):
    """Loopy belief propagation (sum-product) over a :class:`ProbabilisticModel`.

    Consumes the unified factor interface (``scope`` + ``log_potential``), so
    directed (:class:`~..graph.bayesian_network.BayesianNetwork`), undirected
    (:class:`~..graph.markov_network.MarkovNetwork`) and mixed graphs all run
    through the same code path. Every **free** (non-evidence) variable must be
    discrete with finite cardinality; continuous variables are supported as
    **observed evidence** feeding the factors' energies (the CRF case). A
    continuous free variable raises a clear error.

    The whole pass is differentiable (unrolled message passing), so gradients
    flow from a marginal loss into every factor/CPD parametrization.

    Parameters
    ----------
    pgm : ProbabilisticModel
        Any factor graph (directed, undirected, or mixed).
    iters : int, default 5
        Maximum number of synchronous (flooding) message-passing rounds. On a
        tree/forest, enough iterations make BP exact; with loops it is the
        standard approximation.
    damping : float, default 0.0
        Weight in ``[0, 1)`` given to the *previous* factor->variable message,
        applied in **log space** (a geometric mean in probability space), which
        is the form the convergence analyses assume. ``0`` is no damping;
        larger values trade convergence speed for stability on loopy graphs.
        Only the factor->variable family is damped — the variable->factor
        messages are a deterministic function of it, so damping one damps the
        whole iteration.
    tol : float, optional
        If set, stop early once the max absolute change in factor->variable
        messages drops below ``tol``.
    check_every : int, default 1
        How many rounds between convergence tests. The test needs the residual
        on the host, which forces a device synchronisation; on small graphs that
        sync can cost more than the arithmetic, so raising this to ~10 trades at
        most ``check_every - 1`` wasted rounds for far fewer syncs. Ignored when
        ``tol`` is ``None`` (no test, no sync).
    init_noise : float, default 0.0
        Standard deviation of the perturbation applied to the initial
        factor->variable messages. ``0`` gives the classic uniform start. A
        non-zero value only matters on *symmetric* graphs, where it lets the
        iteration reach symmetry-broken fixed points instead of sitting on the
        symmetric one; on a model with asymmetric local fields it is wasted
        work.
    """

    name = "BeliefPropagation"

    def __init__(
        self,
        pgm: ProbabilisticModel,
        iters: int = 5,
        damping: float = 0.0,
        tol: Optional[float] = None,
        check_every: int = 1,
        init_noise: float = 0.0,
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
        if int(check_every) < 1:
            raise ValueError(f"check_every must be >= 1, got {check_every!r}.")
        if float(init_noise) < 0.0:
            raise ValueError(f"init_noise must be >= 0, got {init_noise!r}.")
        self.iters = int(iters)
        self.damping = float(damping)
        self.tol = None if tol is None else float(tol)
        self.check_every = int(check_every)
        self.init_noise = float(init_noise)

    def __repr__(self) -> str:
        return self._format_repr(
            iters=self.iters,
            damping=self.damping,
            tol=self.tol,
            check_every=self.check_every,
            init_noise=self.init_noise,
        )

    # ------------------------------------------------------------------ utils
    def _dtype(self) -> torch.dtype:
        try:
            return next(self.pgm.parameters()).dtype
        except StopIteration:
            return torch.get_default_dtype()

    def _format_evidence(self, variable: Variable, value: torch.Tensor) -> torch.Tensor:
        """Cast to the model dtype and reshape to ``(*leading, *variable.shape)``."""
        return reshape_value_to_event(variable, value.to(self._dtype()))

    @staticmethod
    def _norm(v: torch.Tensor) -> torch.Tensor:
        """``norm(v)`` of the pseudocode: subtract the log-partition of the state axis."""
        return v - torch.logsumexp(v, dim=-1, keepdim=True)

    @staticmethod
    def _pad_states(v: torch.Tensor, card: int, states: int) -> torch.Tensor:
        """Widen a ``(..., card)`` block to ``(..., states)``, padding with :data:`LOG0`."""
        if card == states:
            return v
        pad = torch.full(
            (*v.shape[:-1], states - card), LOG0, dtype=v.dtype, device=v.device
        )
        return torch.cat([v, pad], dim=-1)

    def _encode_states(
        self,
        v: Variable,
        states: torch.Tensor,
        leading: torch.Size,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Value tensor for a whole *grid* of discrete states of ``v``.

        ``states`` is a ``(grid,)`` vector of state indices; the result is
        ``(grid, *leading, width)`` — scalar ``{0., 1.}`` for a binary variable,
        one-hot otherwise — broadcast (as a view) over the leading dimensions.
        Batching the grid into a leading axis is what lets a factor's whole
        table come out of a *single* ``log_potential`` call instead of one call
        per cell.
        """
        card = enumerable_cardinality(v)
        if card == 2 and v.size == 1:
            width = 1
            flat = states.to(dtype).unsqueeze(-1)
        else:
            width = v.size
            flat = torch.nn.functional.one_hot(states, v.size).to(dtype)
        grid = int(states.shape[0])
        return flat.reshape(grid, *([1] * len(leading)), width).expand(
            grid, *leading, width
        )

    def _factor_table(
        self,
        factor,
        free_vars: List[Variable],
        fixed: Dict[Variable, torch.Tensor],
        leading: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Log-potential table over ``free_vars`` (axis order preserved).

        The free grid is enumerated into a **leading** axis and scored in one
        ``factor.log_potential`` call — uniform for CPDs and energy-based
        potentials alike, since both accept any number of leading dimensions.
        Observed scope variables in ``fixed`` are baked in at their evidence
        (factor reduction), which is also how *continuous* evidence enters.
        Returns ``None`` when the factor has no free variable: it is then a
        constant w.r.t. the active variables and contributes nothing.

        The result is shaped ``(*leading, *free_cards)`` — the state axes are
        appended after however many leading dimensions the query carries.

        NOTE: folding the grid into the batch is transparent to any module that
        acts **per element**, including ``nn.Dropout`` — its mask has the shape
        of its input, so every cell of the table still gets an independent mask,
        exactly as when the cells were scored one call at a time. What *does*
        change is a module that couples across the batch (``BatchNorm`` in
        training mode, or anything reducing over the batch axis): its statistics
        are now taken over ``grid * leading`` rows rather than ``leading``. Such
        a module makes ``log_potential`` batch-dependent, which is outside the
        factor contract to begin with.
        """
        free_cards = [enumerable_cardinality(v) for v in free_vars]
        if not free_cards:
            return None
        n_leading = len(leading)
        grid = math.prod(free_cards)

        # State index of slot ``a`` at each grid position, in C order (the last
        # slot varies fastest) so the final reshape maps axis ``a`` to slot ``a``.
        assignment: Dict[Variable, torch.Tensor] = {}
        inner = grid
        for a, v in enumerate(free_vars):
            card = free_cards[a]
            inner //= card
            outer = grid // (inner * card)
            states = (
                torch.arange(card, device=device).repeat_interleave(inner).repeat(outer)
            )
            assignment[v] = self._encode_states(v, states, leading, dtype)
        for v, value in fixed.items():
            assignment[v] = value.unsqueeze(0).expand(grid, *value.shape)

        logp = factor.log_potential(assignment).reshape(grid, *leading)
        # (grid, *leading) -> (*leading, grid) -> (*leading, *free_cards)
        return logp.permute(*range(1, n_leading + 1), 0).reshape(*leading, *free_cards)

    # ------------------------------------------------------------------ query
    def query(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        """Run loopy BP and return per-variable marginals.

        ``out.params`` follows the same contract as every other engine: it is
        keyed by parameter name, each entry an annotated tensor shaped
        ``(*leading, width)`` and sliceable by variable name. The BP marginal is
        therefore reported in the variable's *own* parametrization rather than
        as a state-space belief — a binary concept gets Bernoulli
        ``{'probs': P(x=1), 'logits': log-odds}`` of width 1, a ``k``-way
        categorical gets ``{'probs', 'logits'}`` of width ``k``. The same
        ``binary_cross_entropy_with_logits`` / ``cross_entropy`` call that trains
        a :class:`~..forward.ForwardInference` model therefore trains this one.

        Only queried names that are *active* (free, computed) variables appear in
        ``params``; a fully-observed queried variable emits none (its value is
        its evidence), mirroring the directed engines.

        Query and evidence tensors may carry any number of leading (batch-like)
        dimensions; messages and marginals carry the same ones.

        On a loopy graph the marginals are **approximate** — see the class
        docstring.
        """
        query = self._normalize_query(query)
        self._validate_containers(query, evidence)
        query_names = list(query)

        tensors = list(evidence.values()) + [v for v in query.values() if v is not None]
        leading = self._query_leading_shape(query, evidence)
        n_leading = len(leading)
        dtype = self._dtype()
        device = tensors[0].device if tensors else torch.device("cpu")

        # Member (partial-plate) evidence is not supported on active plates in v1.
        whole_ev, member_ev = self._split_evidence(evidence)
        if member_ev:
            raise NotImplementedError(
                "BeliefPropagation: partial-plate (member) evidence "
                f"{sorted(m for d in member_ev.values() for m in d)} is not supported "
                "in v1; observe the whole plate variable instead."
            )
        evidence_names = set(whole_ev)
        observed: Dict[str, torch.Tensor] = {
            name: self._format_evidence(self.pgm.variables[name], val)
            for name, val in whole_ev.items()
        }

        # Active = free variables (not observed) that participate in some factor.
        # Model order, so the variable axis is deterministic across queries.
        active_vars = [
            v for v in self.pgm.variables.values()
            if v.name not in evidence_names and self.pgm.factor_names_of(v.name)
        ]
        if not active_vars:
            return InferenceOutput(params=self._assemble_params({}, query_names))
        # Enumerability precondition (raises a clear error for continuous free vars).
        cards = [enumerable_cardinality(v) for v in active_vars]
        vindex = {v.name: i for i, v in enumerate(active_vars)}
        n_states = max(cards)

        bias, vid, pad, buckets = self._compile(
            active_vars, cards, vindex, n_states, evidence_names, observed,
            leading, dtype, device,
        )

        # ══ R. READOUT ═════════════════════════════════════════════════════
        if vid.numel():
            m_fv = self._run_message_passing(
                bias, vid, pad, buckets, n_leading, dtype, device
            )
            belief = self._norm(bias.index_add(n_leading, vid, m_fv))
        else:
            # Every factor was degree-1 after reduction: the bias *is* the answer.
            belief = self._norm(bias)

        computed = {
            v.name: self._canonical_params(v, belief[..., i, : cards[i]])
            for i, v in enumerate(active_vars)
        }
        return InferenceOutput(params=self._assemble_params(computed, query_names))

    # ---------------------------------------------------------------- compile
    def _compile(
        self,
        active_vars: List[Variable],
        cards: List[int],
        vindex: Dict[str, int],
        n_states: int,
        evidence_names,
        observed: Dict[str, torch.Tensor],
        leading: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[_Bucket]]:
        """Precompute stage ``P`` — everything the message loop reads.

        Returns ``(bias, vid, pad, buckets)``:

        - ``bias`` ``(*leading, V, n_states)`` — padding mask (:data:`LOG0` on
          states a variable does not have) plus every degree-1 factor's
          normalised table. Absorbing the unaries here removes them from the
          edge list entirely, so the loop has no "is this factor unary" branch.
        - ``vid`` ``(E,)`` — the variable index of each edge.
        - ``pad`` ``(E, n_states)`` — which message entries are padding.
        - ``buckets`` — one entry per ``(arity, cardinalities)`` signature,
          holding the stacked log-potentials ``(*leading, n_b, *cards)`` and the
          edge ids ``(n_b, arity)`` of each factor's slots.
        """
        n_leading = len(leading)
        unary: List[List[torch.Tensor]] = [[] for _ in active_vars]
        edge_var: List[int] = []
        raw: Dict[Tuple[int, Tuple[int, ...]], Tuple[List[torch.Tensor], List[List[int]]]] = {}

        for f in self.pgm.factors.values():
            free_vars = [v for v in f.scope if v.name in vindex]
            fixed = {v: observed[v.name] for v in f.scope if v.name in evidence_names}
            table = self._factor_table(f, free_vars, fixed, leading, dtype, device)
            if table is None:
                continue
            if len(free_vars) == 1:
                # P5-P7: a degree-1 factor's outgoing message is constant.
                unary[vindex[free_vars[0].name]].append(self._norm(table))
                continue
            signature = (
                len(free_vars),
                tuple(enumerable_cardinality(v) for v in free_vars),
            )
            tables, idx = raw.setdefault(signature, ([], []))
            tables.append(table)
            slots = []
            for v in free_vars:
                slots.append(len(edge_var))
                edge_var.append(vindex[v.name])
            idx.append(slots)

        # P2-P4/P6: one padded, unary-loaded row per variable.
        rows = []
        for i, card in enumerate(cards):
            row = (
                torch.stack(unary[i], dim=0).sum(dim=0) if unary[i]
                else torch.zeros(*leading, card, dtype=dtype, device=device)
            )
            rows.append(self._pad_states(row, card, n_states))
        bias = torch.stack(rows, dim=n_leading)

        vid = torch.tensor(edge_var, dtype=torch.long, device=device)
        # Which message entries are states their variable does not have. Built
        # from the Python edge list rather than from ``vid``, so no device
        # round-trip is needed to know it.
        edge_cards = torch.tensor(
            [cards[i] for i in edge_var], dtype=torch.long, device=device
        )
        pad = torch.arange(n_states, device=device) >= edge_cards.unsqueeze(-1)
        buckets: List[_Bucket] = [
            (
                arity,
                bcards,
                torch.stack(tables, dim=n_leading),
                torch.tensor(idx, dtype=torch.long, device=device),
            )
            for (arity, bcards), (tables, idx) in raw.items()
        ]
        return bias, vid, pad, buckets

    # --------------------------------------------------------- message passing
    def _run_message_passing(
        self,
        bias: torch.Tensor,
        vid: torch.Tensor,
        pad: torch.Tensor,
        buckets: List[_Bucket],
        n_leading: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Synchronous (flooding) log-domain sum-product; returns the final
        factor->variable messages, shaped ``(*leading, E, n_states)``.

        The two half-steps have disjoint read/write sets — the factor half reads
        only the ``m_vf`` written by the variable half — so a whole round is one
        parallel update and no message is read "fresh" within its own round.
        """
        leading = bias.shape[:n_leading]
        vdim = n_leading  # the edge / variable axis
        lam = 1.0 - self.damping
        n_states = int(pad.shape[-1])

        # I1-I2: uniform (log 1 == 0) start, optionally perturbed, with the
        # padded states pinned at LOG0 so they stay inert for the whole run.
        m_fv = torch.zeros(*leading, *pad.shape, dtype=dtype, device=device)
        if self.init_noise:
            m_fv = m_fv + self.init_noise * torch.randn_like(m_fv)
        m_fv = m_fv.masked_fill(pad, LOG0)

        # I4: on-device residual — a Python float would force a host sync per round.
        residual = torch.zeros((), dtype=dtype, device=device)

        for step in range(self.iters):
            residual = torch.zeros_like(residual)

            # ─ variable -> factor: exclusive sum as total minus self ─────────
            total = bias.index_add(vdim, vid, m_fv)
            m_vf = self._norm(total.index_select(vdim, vid) - m_fv).clamp(min=LOG0)

            # ─ factor -> variable: one batched logsumexp per bucket ──────────
            edge_ids: List[torch.Tensor] = []
            values: List[torch.Tensor] = []
            for arity, bcards, phi, idx in buckets:
                n_b = int(idx.shape[0])
                gathered = m_vf.index_select(vdim, idx.reshape(-1))
                gathered = gathered.reshape(*leading, n_b, arity, n_states)

                # L8-L10: the full theta, built once per bucket.
                theta = phi
                slots: List[torch.Tensor] = []
                for a, card in enumerate(bcards):
                    shape = [*leading, n_b] + [1] * arity
                    shape[vdim + 1 + a] = card
                    slot = gathered[..., a, :card].reshape(shape)
                    slots.append(slot)
                    theta = theta + slot

                # L11-L18: exclusive by subtraction, marginalise the other slots.
                for a, card in enumerate(bcards):
                    axes = [vdim + 1 + j for j in range(arity) if j != a]
                    new = torch.logsumexp(theta - slots[a], dim=axes)
                    eid = idx[:, a]
                    old = m_fv.index_select(vdim, eid)[..., :card]
                    if lam != 1.0:
                        new = lam * new + self.damping * old
                    new = self._norm(new).clamp(min=LOG0)
                    residual = torch.maximum(residual, (new - old).abs().amax())
                    edge_ids.append(eid)
                    values.append(self._pad_states(new, card, n_states))

            # Every edge is written by exactly one (bucket, slot) pair, so the
            # concatenated indices are a permutation of ``0..E-1`` and this is a
            # single out-of-place scatter (in-place would break autograd).
            m_fv = m_fv.index_copy(
                vdim, torch.cat(edge_ids), torch.cat(values, dim=vdim)
            )

            # L19-L20: the only host sync, and only when a tolerance is asked for.
            if (
                self.tol is not None
                and (step + 1) % self.check_every == 0
                and residual.item() < self.tol
            ):
                break

        return m_fv

    @staticmethod
    def _canonical_params(
        variable: Variable, belief: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Express a state-space belief in ``variable``'s own parametrization.

        ``belief`` is the (log-)belief over the variable's discrete states,
        shape ``(*leading, cardinality)``. The result matches what a forward
        engine puts in ``out.params`` for the same variable:

        - **binary** (2 states, width 1) -> Bernoulli ``probs`` = ``P(x=1)`` and
          ``logits`` = the log-odds, both ``(*leading, 1)``, satisfying
          ``sigmoid(logits) == probs``;
        - **categorical** (one state per class) -> ``probs`` and normalised
          ``logits`` of width ``variable.size``, satisfying
          ``softmax(logits) == probs``.
        """
        log_norm = belief - torch.logsumexp(belief, dim=-1, keepdim=True)
        if enumerable_cardinality(variable) == 2 and variable.size == 1:
            logits = log_norm[..., 1:2] - log_norm[..., 0:1]
            return {"probs": torch.sigmoid(logits), "logits": logits}
        return {"probs": log_norm.exp(), "logits": log_norm}
