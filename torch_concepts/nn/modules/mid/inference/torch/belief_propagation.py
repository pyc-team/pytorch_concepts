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

#: One compiled bucket: ``(cardinalities, log-potential stack, edge ids)``.
#: The arity is ``len(cardinalities)`` — never stored twice.
_Bucket = Tuple[Tuple[int, ...], torch.Tensor, torch.Tensor]


class BeliefPropagation(TorchBaseInference):
    """Loopy belief propagation (sum-product) over a :class:`ProbabilisticModel`.

    Consumes the unified factor interface (``scope`` + ``log_potential``), so
    directed (:class:`~..graph.bayesian_network.BayesianNetwork`), undirected
    (:class:`~..graph.markov_network.MarkovNetwork`) and mixed graphs all run
    through the same code path. 

    On a tree, enough iterations make BP exact; with loops it is the
    standard approximation.
    
    The whole pass is differentiable (unrolled message passing), so gradients
    flow from a marginal loss into every factor/CPD parametrization.
    
    NOTE: Every **free** (non-evidence) variable must be
    discrete with finite cardinality; continuous variables are supported as
    **observed evidence** feeding the factors' energies (the CRF case).

    Parameters
    ----------
    pgm : ProbabilisticModel
        Any factor graph (directed, undirected, or mixed).
    iters : int, default 5
        Maximum number of synchronous (flooding) message-passing rounds.
    damping : float, default 0.0
        Weight in ``[0, 1)`` given to the *previous* factor->variable message. 
        ``0`` is no damping; larger values trade convergence speed for stability 
        on loopy graphs.
    tol : float, optional
        If set, stop early once the max absolute change in factor->variable
        messages drops below ``tol``.
    check_every : int, default 1
        How many rounds between convergence tests. Ignored when ``tol`` is 
        ``None`` (no test, no sync).
    init_noise : float, default 0.0
        Standard deviation of the perturbation applied to the initial
        factor->variable messages. ``0`` gives the classic uniform start.
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
        """normalize by subtracting the log-partition of the state axis."""
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

    @staticmethod
    def _on_state_axis(msg: torch.Tensor, a: int, arity: int) -> torch.Tensor:
        """``(..., card)`` -> ``(..., 1, …, card, …, 1)`` with ``card`` on state axis ``a``.

        How a message is broadcast onto its own axis of a factor's table.
        """
        block = [1] * arity
        block[a] = msg.shape[-1]
        return msg.reshape(*msg.shape[:-1], *block)

    @staticmethod
    def _scope_entries(factor) -> List[Tuple[Variable, List[str]]]:
        """Each scope entry paired with the member names it covers.

        The unit of inference is a plate **member**, not a plate: a whole-plate
        scope entry covers all of its members, a member handle covers just
        itself, and an ordinary variable covers itself (its ``members`` is
        ``[name]``, so it is a degenerate plate and needs no special case).
        """
        return [
            (s, list(s.members) if s is s.plate else [s.name])
            for s in factor.scope
        ]

    def _free_members(self, factor, node_index: Dict[str, int]) -> List[str]:
        """The factor's scope members that are free, in scope order, deduped."""
        free: List[str] = []
        for _, names in self._scope_entries(factor):
            free.extend(m for m in names if m in node_index and m not in free)
        return free

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
        free_members: List[str],
        handles: Dict[str, Variable],
        member_blocks: Dict[str, torch.Tensor],
        leading: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Log-potential table over ``free_members`` (axis order preserved).

        The free grid is enumerated into a **leading** axis and scored in one
        ``factor.log_potential`` call — uniform for CPDs and energy-based
        potentials alike, since both accept any number of leading dimensions.
        Observed members read their value from ``member_blocks`` and are baked
        in (factor reduction), which is also how *continuous* evidence enters.
        Returns ``None`` when the factor has no free member: it is then a
        constant w.r.t. the active variables and contributes nothing.

        Because the enumeration is per **member**, a scope entry naming a whole
        plate has to be handed back the plate's full-width value: its members'
        blocks — enumerated or observed, in member order — are concatenated on
        the event axis before scoring. A member handle in the scope is keyed by
        its own name, which ``ParametricFactor.resolve_value`` looks up first,
        so no reassembly is needed there.

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
        free_cards = [enumerable_cardinality(handles[m]) for m in free_members]
        if not free_cards:
            return None
        grid = math.prod(free_cards)

        # ``cartesian_prod`` enumerates in C order (last slot varies fastest),
        # which is what makes the final reshape map axis ``a`` to slot ``a``.
        states = torch.cartesian_prod(
            *[torch.arange(c, device=device) for c in free_cards]
        ).reshape(grid, len(free_cards))
        blocks: Dict[str, torch.Tensor] = {
            m: self._encode_states(handles[m], states[:, a], leading, dtype)
            for a, m in enumerate(free_members)
        }

        def block(name: str) -> torch.Tensor:
            """This member's ``(grid, *leading, width)`` value: enumerated or observed."""
            if name in blocks:
                return blocks[name]
            observed = member_blocks[name]
            return observed.unsqueeze(0).expand(grid, *observed.shape)

        assignment: Dict[Variable, torch.Tensor] = {}
        for entry, names in self._scope_entries(factor):
            parts = [block(m) for m in names]
            assignment[entry] = (
                parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
            )

        logp = factor.log_potential(assignment).reshape(grid, *leading)
        # (grid, *leading) -> (*leading, grid) -> (*leading, *free_cards)
        return torch.movedim(logp, 0, -1).reshape(*leading, *free_cards)

    # ------------------------------------------------------------------ query
    def query(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        """Run loopy BP and return per-variable marginals."""
        
        query = self._normalize_query(query)
        self._validate_containers(query, evidence)
        query_names = list(query)

        tensors = list(evidence.values()) + [v for v in query.values() if v is not None]
        leading = self._query_leading_shape(query, evidence)
        dtype = self._dtype()
        device = tensors[0].device if tensors else torch.device("cpu")

        # Evidence is flattened to one block per observed **member**, so a whole
        # variable, a whole plate and a subset of a plate's members are all the
        # same thing downstream: some members have a value, the rest are free.
        whole_ev, member_ev = self._split_evidence(evidence)
        member_blocks: Dict[str, torch.Tensor] = {}
        for name, value in whole_ev.items():
            var = self.pgm.variables[name]
            observed = self._format_evidence(var, value)
            for m in var.members:
                member_blocks[m] = var.select_value(observed, m)
        for owner_name, members in member_ev.items():
            owner = self.pgm.variables[owner_name]
            for m, value in members.items():
                member_blocks[m] = reshape_value_to_event(
                    owner.member(m), value.to(dtype)
                )

        # Active = free members of a variable that participates in some factor.
        # Model order, so the node axis is deterministic across queries. An
        # ordinary variable is a one-member plate, so it needs no special case.
        nodes: List[Variable] = [
            v.member(m) if v.is_plate else v
            for v in self.pgm.variables.values()
            if self.pgm.factor_names_of(v.name)
            for m in v.members
            if m not in member_blocks
        ]
        if not nodes:
            return InferenceOutput(params=self._assemble_params({}, query_names))
        # Enumerability precondition (raises a clear error for continuous free
        # vars). Asking a *member* is also what fixes a plate: a member of a
        # Bernoulli plate is one bit (2 states), not a size-k bit vector.
        cards = [enumerable_cardinality(n) for n in nodes]

        bias, vid, pad, buckets = self._compile(
            nodes, cards, member_blocks, leading, dtype, device
        )

        # ══ R. READOUT ═════════════════════════════════════════════════════
        # No normalisation here: ``_marginal``'s softmax normalises anyway.
        # Every factor being degree-1 leaves no edges, and the bias is the answer.
        belief = bias
        if vid.numel():
            m_fv = self._run_message_passing(bias, vid, pad, buckets)
            belief = bias.index_add(-2, vid, m_fv)

        marginals = {
            n.name: self._marginal(n, belief[..., i, : cards[i]])
            for i, n in enumerate(nodes)
        }
        return InferenceOutput(
            params=self._assemble_params(
                self._regroup(marginals, member_blocks), query_names
            )
        )

    def _regroup(
        self,
        marginals: Dict[str, torch.Tensor],
        member_blocks: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Per-member marginals -> the per-*variable* dict ``_assemble_params`` wants.

        A variable's members are concatenated back into one ``(*leading, size)``
        tensor in member order. On a partially observed plate the observed
        members contribute their evidence, which is exactly their posterior:
        conditioning makes ``P(x_m | e)`` a point mass at the observed value.
        A variable with no free member at all emits nothing, mirroring the rule
        that a fully-observed queried variable reports no parameters.
        """
        computed: Dict[str, Dict[str, torch.Tensor]] = {}
        for var in self.pgm.variables.values():
            if not any(m in marginals for m in var.members):
                continue
            parts = [
                marginals[m] if m in marginals else member_blocks[m]
                for m in var.members
            ]
            computed[var.name] = {
                "probs": parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
            }
        return computed

    # ---------------------------------------------------------------- compile
    def _compile(
        self,
        nodes: List[Variable],
        cards: List[int],
        member_blocks: Dict[str, torch.Tensor],
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
        - ``vid`` ``(E,)`` — the node index of each edge.
        - ``pad`` ``(E, n_states)`` — which message entries are padding.
        - ``buckets`` — one entry per ``(arity, cardinalities)`` signature,
          holding the stacked log-potentials ``(*leading, n_b, *cards)`` and the
          edge ids ``(n_b, arity)`` of each factor's slots.
        """
        n_states = max(cards)
        vindex = {n.name: i for i, n in enumerate(nodes)}
        handles = {n.name: n for n in nodes}
        unary: List[List[torch.Tensor]] = [[] for _ in nodes]
        edge_var: List[int] = []
        raw: Dict[Tuple[int, ...], Tuple[List[torch.Tensor], List[List[int]]]] = {}

        for f in self.pgm.factors.values():
            free_vars = self._free_members(f, vindex)
            table = self._factor_table(
                f, free_vars, handles, member_blocks, leading, dtype, device
            )
            if table is None:
                continue
            if len(free_vars) == 1:
                # A degree-1 factor's outgoing message is constant.
                unary[vindex[free_vars[0]]].append(self._norm(table))
                continue
            # Signature = the cardinality tuple; its length is the arity.
            tables, idx = raw.setdefault(
                tuple(cards[vindex[m]] for m in free_vars), ([], [])
            )
            tables.append(table)
            idx.append(list(range(len(edge_var), len(edge_var) + len(free_vars))))
            edge_var.extend(vindex[m] for m in free_vars)

        # One padded, unary-loaded row per variable.
        bias = torch.stack(
            [
                self._pad_states(
                    torch.stack(rows, dim=0).sum(dim=0) if rows
                    else torch.zeros(*leading, card, dtype=dtype, device=device),
                    card, n_states,
                )
                for card, rows in zip(cards, unary)
            ],
            dim=-2,
        )

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
                bcards,
                torch.stack(tables, dim=-len(bcards) - 1),
                torch.tensor(idx, dtype=torch.long, device=device),
            )
            for bcards, (tables, idx) in raw.items()
        ]
        return bias, vid, pad, buckets

    # --------------------------------------------------------- message passing
    def _run_message_passing(
        self,
        bias: torch.Tensor,
        vid: torch.Tensor,
        pad: torch.Tensor,
        buckets: List[_Bucket],
    ) -> torch.Tensor:
        """Synchronous (flooding) log-domain sum-product; returns the final
        factor->variable messages, shaped ``(*leading, E, n_states)``.

        The two half-steps have disjoint read/write sets — the factor half reads
        only the ``m_vf`` written by the variable half — so a whole round is one
        parallel update and no message is read "fresh" within its own round.

        Every axis is addressed from the *right*: the edge axis is always ``-2``
        and a bucket's ``d`` state axes are always the last ``d``. That is why
        the number of leading (batch-like) dimensions never appears here.
        """
        n_states = int(pad.shape[-1])
        lam = 1.0 - self.damping
        # The previous message is only needed to damp with or to measure
        # convergence against. On the defaults neither is asked for, so it is
        # not gathered at all.
        need_old = self.damping > 0.0 or self.tol is not None

        # Uniform (log 1 == 0) start, optionally perturbed, with the
        # padded states pinned at LOG0 so they stay inert for the whole run.
        m_fv = torch.zeros(
            *bias.shape[:-2], *pad.shape, dtype=bias.dtype, device=bias.device
        )
        if self.init_noise:
            m_fv = m_fv + self.init_noise * torch.randn_like(m_fv)
        m_fv = m_fv.masked_fill(pad, LOG0)

        # On-device residual — a Python float would force a host sync per round.
        residual = torch.zeros((), dtype=bias.dtype, device=bias.device)

        for step in range(self.iters):
            residual = torch.zeros_like(residual)

            # ─ variable -> factor: exclusive sum as total minus self ─────────
            # No clamp is needed on the subtraction. Padding is applied *last*
            # when a row is built, so a padded state holds exactly LOG0 in both
            # operands and never a finite value plus LOG0 — there is nothing to
            # cancel catastrophically. Those states stay hugely negative (so
            # every logsumexp ignores them) and are never read back anyway:
            # each gather below slices to its slot's own cardinality.
            total = bias.index_add(-2, vid, m_fv)
            m_vf = self._norm(total.index_select(-2, vid) - m_fv)

            # ─ factor -> variable: one batched logsumexp per bucket ──────────
            edge_ids: List[torch.Tensor] = []
            values: List[torch.Tensor] = []
            for bcards, phi, idx in buckets:
                arity = len(bcards)
                gathered = m_vf.index_select(-2, idx.reshape(-1)).reshape(
                    *m_vf.shape[:-2], idx.shape[0], arity, n_states
                )
                slots = [
                    self._on_state_axis(gathered[..., a, :card], a, arity)
                    for a, card in enumerate(bcards)
                ]

                # The full theta, built once per bucket.
                theta = phi
                for slot in slots:
                    theta = theta + slot

                # Exclusive by subtraction, marginalise the other slots.
                for a, card in enumerate(bcards):
                    new = torch.logsumexp(
                        theta - slots[a],
                        dim=[j - arity for j in range(arity) if j != a],
                    )
                    eid = idx[:, a]
                    old = m_fv.index_select(-2, eid)[..., :card] if need_old else None
                    if lam != 1.0:
                        new = lam * new + self.damping * old
                    new = self._norm(new)
                    if old is not None:
                        residual = torch.maximum(residual, (new - old).abs().amax())
                    edge_ids.append(eid)
                    values.append(self._pad_states(new, card, n_states))

            # Every edge is written by exactly one (bucket, slot) pair, so the
            # concatenated indices are a permutation of ``0..E-1`` and this is a
            # single out-of-place scatter (in-place would break autograd).
            m_fv = m_fv.index_copy(-2, torch.cat(edge_ids), torch.cat(values, dim=-2))

            # The only host sync, and only when a tolerance is asked for.
            if (
                self.tol is not None
                and (step + 1) % self.check_every == 0
                and residual.item() < self.tol
            ):
                break

        return m_fv

    @staticmethod
    def _marginal(variable: Variable, belief: torch.Tensor) -> torch.Tensor:
        """The marginal ``P(x)``, in the width ``variable``'s own family uses.

        ``belief`` is the un-normalised log-belief over the variable's discrete
        states, shape ``(*leading, cardinality)``. BP computes exactly one
        thing — a probability — so that is all this reports: a ``k``-way
        categorical gets the full ``(*leading, k)`` distribution, and a binary
        variable gets Bernoulli ``P(x=1)`` at ``(*leading, 1)``, since its
        ``P(x=0)`` is redundant.
        """
        probs = torch.softmax(belief, dim=-1)
        if enumerable_cardinality(variable) == 2 and variable.size == 1:
            return probs[..., 1:2]
        return probs
