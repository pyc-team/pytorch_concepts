"""PgmpyVariableElimination — exact marginals by variable elimination (pgmpy)."""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ...graph.probabilistic_model import ProbabilisticModel
from ...variable import Variable
from ....outputs import InferenceOutput
from ..utils import (
    enumerable_cardinality, factor_table, query_targets, unpack_plates,
)
from .base import PgmpyBaseInference, _import_pgmpy


class PgmpyVariableElimination(PgmpyBaseInference):
    """Exact marginals by variable elimination, over any factor graph.

    A drop-in replacement for :class:`~torch_concepts.nn.BeliefPropagation` at
    **evaluation** time: the same ``query`` / ``evidence`` contract, the same
    ``out.probs`` layout, but the answer is *exact* on a loopy graph rather than
    the BP approximation. On a tree the two agree, BP being exact there too.
    Unlike BP, a query may name several variables at once.

    A value in the query is a target, as for the sampling estimators:
    ``out.probabilities`` then holds the exact ``P(Q=q | E)`` of the whole
    assignment, shaped ``(*leading,)``, over one queried variable or several.
    It is computed as ``Z(Q=q, E) / Z(E)`` — two eliminations per observation,
    whatever the number of queried variables. It needs a value for *every*
    queried variable: in a mixed query the values are ignored and
    ``out.probabilities`` stays ``None``.

    Consumes the unified factor interface (``scope`` + ``log_potential``), so
    directed (:class:`~..graph.bayesian_network.BayesianNetwork`), undirected
    (:class:`~..graph.markov_network.MarkovNetwork`) and mixed graphs all run
    through the same code path: each factor is enumerated into an unnormalised
    table and handed to pgmpy as a ``DiscreteFactor``, which needs no
    per-parent-configuration normalisation and so never has to know whether the
    model was directed.

    Runs under :func:`torch.no_grad` and crosses into NumPy, so **nothing here
    is differentiable** — train with :class:`BeliefPropagation` and evaluate
    with this. The cost is one pgmpy solve per observation per queried variable
    (sub-millisecond each, but a Python loop), which is fine for evaluation and
    unusable inside a training loop.

    NOTE: as in BP, every **free** (non-evidence) variable must be discrete with
    finite cardinality; continuous variables are supported as **observed
    evidence** feeding the factors' tables (the CRF case). Evidence is baked
    into the tables by factor reduction, which is the only route continuous
    evidence could take — pgmpy's own ``evidence=`` argument takes state names
    and cannot express a ``Normal`` observation.

    Parameters
    ----------
    pgm : ProbabilisticModel
        Any factor graph (directed, undirected, or mixed).

    Notes
    -----
    ``p_int`` and the temperature schedule are inherited from
    :class:`BaseInference` and never read: this engine draws no samples and
    realises no variable, so there is nothing to teacher-force or to relax.
    There is likewise no ``iters`` / ``damping`` / ``tol`` — variable
    elimination is exact, so there is nothing to converge.
    """

    name = "PgmpyVariableElimination"

    def __init__(self, pgm: ProbabilisticModel):
        if not isinstance(pgm, ProbabilisticModel):
            raise TypeError(
                f"{self.name} requires a ProbabilisticModel, "
                f"got {type(pgm).__name__}."
            )
        super().__init__(pgm)

    def __repr__(self) -> str:
        return self._format_repr()

    # ------------------------------------------------------------------ query
    @torch.no_grad()
    def query(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        """Run exact variable elimination and return per-variable marginals.

        A query that carries values also gets ``out.probabilities``: the exact
        ``P(Q=q | E)`` of the whole assignment, over one variable or several.
        """
        query = self._normalize_query(query)
        self._validate_containers(query, evidence)
        query_names = list(query)

        tensors = list(evidence.values()) + [v for v in query.values() if v is not None]
        leading = self._query_leading_shape(query, evidence)
        dtype = self._dtype()
        device = tensors[0].device if tensors else torch.device("cpu")

        # Plates are not handled, they are erased: this runs on a model where
        # every member is its own variable, exactly as BP does. Local, never a
        # submodule — see unpack_plates.
        model = unpack_plates(self.pgm)
        member_blocks = self.member_evidence(evidence, dtype)
        targets = query_targets(self, query, member_blocks, model, dtype)

        # ══ P. PROBABILITY ═════════════════════════════════════════════════
        # P(Q=q | E) = Z(Q=q, E) / Z(E): clamp the query to its target and sum
        # everything else out, once with and once without it. Two numbers per
        # row — the joint table over Q is never built, so the cost does not grow
        # with the number of queried variables.
        probabilities = None
        if targets is not None:
            log_p = self._log_partition(
                model, {**member_blocks, **targets}, set(targets), leading, device
            ) - self._log_partition(model, member_blocks, set(), leading, device)
            probabilities = log_p.exp().to(dtype).reshape(leading)

        nodes = self._free_nodes(model, member_blocks)
        if not nodes:
            return InferenceOutput(
                params=self._assemble_params({}, query_names),
                probabilities=probabilities,
            )
        # Enumerability precondition: raises a clear error for a continuous free
        # variable, before any table is built.
        cards = {n.name: enumerable_cardinality(n) for n in nodes}
        handles = {n.name: n for n in nodes}
        arrays, used, edges, _ = self._build_tables(
            model, cards, handles, member_blocks, set(), leading, device
        )

        # ══ W. WANTED ══════════════════════════════════════════════════════
        # The classic VE contract: name the query variables and the evidence,
        # and everything else is summed out. Eliminating the unasked-for
        # variables *is* what pgmpy does internally, so the only choice here is
        # which variables to ask a marginal for — the queried ones, closed over
        # their owning plate because regroup_members stacks a variable's members
        # as a whole and would KeyError on a free member left uncomputed.
        resolve = self.pgm.resolve
        owners = {resolve(name).name for name in query_names}
        wanted = [n.name for n in nodes if resolve(n.name).name in owners]
        if not wanted:
            return InferenceOutput(
                params=self._assemble_params({}, query_names),
                probabilities=probabilities,
            )

        # ══ S. SOLVE ═══════════════════════════════════════════════════════
        rows: Dict[str, List[np.ndarray]] = {m: [] for m in wanted}
        for b, engine in enumerate(self._row_engines(arrays, cards, used, edges, leading)):
            for m in wanted:
                # One variable per call, deliberately. pgmpy's default "greedy"
                # path builds an opt_einsum contraction whose output indices are
                # exactly the requested variables.
                values = engine.query([m], joint=False, show_progress=False)[m].values
                rows[m].append(values / self._partition(values, m, b))

        # ══ R. READOUT ═════════════════════════════════════════════════════
        marginals = {}
        for m in wanted:
            probs = torch.as_tensor(
                np.stack(rows[m]), dtype=dtype, device=device
            ).reshape(*leading, -1)
            variable = handles[m]
            # BP's convention: a k-way categorical reports the full
            # distribution, a binary variable only P(x=1), since P(x=0) is
            # redundant in the width its family uses.
            marginals[m] = (
                probs[..., 1:2]
                if enumerable_cardinality(variable) == 2 and variable.size == 1
                else probs
            )
        return InferenceOutput(
            params=self._assemble_params(
                self.regroup_members(marginals, member_blocks), query_names
            ),
            probabilities=probabilities,
        )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _free_nodes(
        model: ProbabilisticModel, member_blocks: Dict[str, torch.Tensor]
    ) -> List[Variable]:
        """Free variables that participate in some factor, in model order, so
        the node axis is deterministic across queries."""
        return [
            v for v in model.variables.values()
            if model.factor_names_of(v.name) and v.name not in member_blocks
        ]

    def _build_tables(
        self,
        model: ProbabilisticModel,
        cards: Dict[str, int],
        handles: Dict[str, Variable],
        member_blocks: Dict[str, torch.Tensor],
        clamped: set,
        leading: torch.Size,
        device: torch.device,
    ) -> Tuple[List[tuple], set, set, torch.Tensor]:
        """Every factor as a per-row probability table, ready for pgmpy.

        Returns ``(arrays, used, edges, log_const)``. ``log_const`` is the
        ``(batch,)`` log-mass the tables leave out: each table's max shift, plus
        the log-potential of every fully observed factor that touches a member
        in ``clamped``. Marginals are normalised and ignore it; a partition
        function needs it. Factors observed by the evidence alone are left out
        of it even then — they are the same constant in both terms of
        ``Z(q, E) / Z(E)`` and cancel.
        """
        # pgmpy is a product-space engine, so unlike BP — which stays in log
        # space throughout — the tables must be exponentiated here.
        dtype = self._dtype()
        batch = math.prod(leading) if len(leading) else 1
        arrays: List[tuple] = []
        used: set = set()
        edges: set = set()
        log_const = torch.zeros(batch, dtype=torch.float64, device=device)
        for f in model.factors.values():
            # Free scope entries in scope order; ``fromkeys`` dedupes a
            # variable named twice (e.g. a plate and one of its members).
            free = list(dict.fromkeys(v.name for v in f.scope if v.name in cards))
            table = factor_table(
                f, free, handles, member_blocks, leading, dtype, device
            )
            if table is None:
                # No free variable: a constant, which only a partition function
                # over a clamped query needs.
                if any(v.name in clamped for v in f.scope):
                    log_const = log_const + f.log_potential(
                        {v: member_blocks[v.name] for v in f.scope}
                    ).to(torch.float64).reshape(batch)
                continue
            # (*leading, *cards) -> (batch, *cards). Folding the leading axes
            # here rather than with _collapse_leading keeps ``leading`` truthful
            # all the way into factor_table and leaves the evidence untouched.
            table = table.to(torch.float64).reshape(
                batch, *table.shape[len(leading):]
            )
            # Per-row, per-table max subtraction, applied in log space *before*
            # exp. Both halves matter: float64 because float32 exp(-104) is
            # exactly 0, so a table spanning more than ~104 nats would silently
            # lose its tail; the shift because exp(+120) overflows to inf and
            # inf/inf is nan. Harmless to the answer — each table's max becomes
            # exactly 1.0 and the marginals are normalised at the end; the
            # shift is kept in ``log_const`` for the partition function.
            shift = table.flatten(1).amax(-1).reshape(batch, *[1] * (table.dim() - 1))
            log_const = log_const + shift.reshape(batch)
            arrays.append((free, (table - shift).exp().cpu().numpy()))
            used.update(free)
            edges.update(itertools.combinations(free, 2))  # clique -> all pairs
        return arrays, used, edges, log_const

    @staticmethod
    def _row_engines(
        arrays: List[tuple],
        cards: Dict[str, int],
        used: set,
        edges: set,
        leading: torch.Size,
    ):
        """One pgmpy elimination engine per batch row, built lazily."""
        MarkovNetwork, DiscreteFactor, VariableElimination = _import_pgmpy()
        # Sorted, so the graph — and hence pgmpy's greedy elimination order —
        # is deterministic across rows and runs.
        node_list, edge_list = sorted(used), sorted(edges)
        for b in range(math.prod(leading) if len(leading) else 1):
            network = MarkovNetwork()
            network.add_nodes_from(node_list)
            network.add_edges_from(edge_list)
            network.add_factors(*[
                DiscreteFactor(free, [cards[m] for m in free], a[b])
                for free, a in arrays
            ])
            yield VariableElimination(network)

    def _partition(self, values: np.ndarray, m: str, b: int) -> float:
        """The sum of ``m``'s unnormalised marginal on row ``b``: the partition
        function, checked to be representable."""
        # pgmpy normalises for a BayesianNetwork/JunctionTree but not
        # for a MarkovNetwork, so what comes back is unnormalised and
        # its sum is the partition function.
        total = values.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise FloatingPointError(
                f"{self.name}: the partition function for {m!r} on batch "
                f"row {b} is {total!r}. The factors' log-potentials span "
                "more than float64 can hold even after per-table max "
                "subtraction (~745 nats); rescale the energies."
            )
        return total

    def _log_partition(
        self,
        model: ProbabilisticModel,
        member_blocks: Dict[str, torch.Tensor],
        clamped: set,
        leading: torch.Size,
        device: torch.device,
    ) -> torch.Tensor:
        """``log Z`` of the model reduced by ``member_blocks``, ``(batch,)`` float64.

        Asking pgmpy for any one free variable eliminates every other one —
        across all connected components — so the sum of that unnormalised
        marginal is the whole partition function.
        """
        nodes = self._free_nodes(model, member_blocks)
        cards = {n.name: enumerable_cardinality(n) for n in nodes}
        handles = {n.name: n for n in nodes}
        arrays, used, edges, log_const = self._build_tables(
            model, cards, handles, member_blocks, clamped, leading, device
        )
        if not nodes:
            return log_const
        probe = nodes[0].name
        log_z = [
            math.log(self._partition(
                engine.query([probe], joint=False, show_progress=False)[probe].values,
                probe, b,
            ))
            for b, engine in enumerate(self._row_engines(arrays, cards, used, edges, leading))
        ]
        return log_const + torch.tensor(log_z, dtype=torch.float64, device=device)
