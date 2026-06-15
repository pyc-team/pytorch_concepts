"""ForwardInference — pytorch forward pass through a :class:`BayesianNetwork`.

Two modes:

- ``"deterministic"``: every variable is propagated by its "canonical"
  parameter (``loc`` for Normal/MVN, ``probs`` for Bernoulli/OneHotCat,
  ``value`` for Delta).
- ``"ancestral"``: every variable is sampled with the same reparameterised
  distributions used by :class:`BayesianNetwork.forward` for unobserved sites
  (straight-through relaxations for the discrete families).

Evidence variables (root or non-root) are hard-conditioned: the observed
value is propagated to children directly and their CPD is never evaluated,
so no parameters are produced for them.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from ...models.bayesian_network import BayesianNetwork
from ...models.variable import Variable
from ..utils import make_temperature_schedule, reshape_value_to_event
from ....outputs import InferenceOutput
from .utils import propagated_value, sample_from
from .base import TorchBaseInference


def _align_gt(gt: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Cast and reshape ground-truth tensor to match the dtype and shape of ref.

    Step-by-step:
    1. Cast ``gt`` to ``ref``\'s dtype so arithmetic ops don\'t raise type errors
       (e.g. LongTensor label vs FloatTensor network output).
    2. If shapes already match after the cast, return immediately.
    3. Handle the common "extra trailing 1" mismatches that arise when some
       code paths squeeze scalars and others don\'t:
       - ``gt`` has one more dim than ``ref`` and its last dim is 1 → squeeze it off.
       - ``gt`` has one fewer dim than ``ref`` and ``ref``\'s last dim is 1 → unsqueeze.
    4. Finally, broadcast ``gt`` to exactly ``ref``\'s shape so downstream ops
       (e.g. per-element masking) can use ``gt`` in place of ``ref``.
    """
    aligned = gt.to(ref.dtype) if gt.dtype != ref.dtype else gt
    if aligned.shape != ref.shape:
        if aligned.dim() == ref.dim() + 1 and aligned.shape[-1] == 1:
            aligned = aligned.squeeze(-1)
        elif aligned.dim() + 1 == ref.dim() and ref.shape[-1] == 1:
            aligned = aligned.unsqueeze(-1)
    return aligned.expand_as(ref) if aligned.shape != ref.shape else aligned


def _teacher_force(nn_value: torch.Tensor, gt: torch.Tensor, p_int: float) -> torch.Tensor:
    """Stochastically replace nn_value with ground truth at rate p_int."""
    aligned = _align_gt(gt, nn_value)
    if p_int >= 1.0:
        return aligned
    if p_int <= 0.0:
        return nn_value
    mask_shape = nn_value.shape[:1] + (1,) * (nn_value.dim() - 1)
    mask = (torch.rand(mask_shape, device=nn_value.device) < p_int).to(nn_value.dtype)
    return mask * aligned + (1.0 - mask) * nn_value


class ForwardInference(TorchBaseInference):
    """Abstract base class for torch based, forward-pass inference engines.

    Concrete subclasses (:class:`DeterministicInference`,
    :class:`AncestralInference`) implement the :meth:`_propagate` method to
    decide whether each variable is resolved deterministically (MAP estimate)
    or by ancestral sampling.  All shared logic (topological traversal,
    evidence clamping, teacher forcing, temperature schedule) lives here.
    """

    name = "ForwardInference"

    def __init__(
        self,
        pgm: BayesianNetwork,
        mode: str = "deterministic",
        p_int: float = 1.0,
        initial_temperature: float = 1.0,
        annealing: Union[str, Callable[[int], float]] = "constant",
        annealing_rate: float = 0.0,
        parallelize_levels: bool = False,
    ):
        super().__init__(pgm)
        if mode not in {"deterministic", "ancestral"}:
            raise ValueError(f"mode must be either 'deterministic' or 'ancestral', got {mode!r}.")
        if not 0.0 <= float(p_int) <= 1.0:
            raise ValueError(f"p_int must be in [0, 1], got {p_int!r}.")
        self.mode = mode
        self.p_int = float(p_int)
        # When True, variables in the same topological level (conditionally
        # independent given the previous levels) are evaluated concurrently.
        self.parallelize_levels = bool(parallelize_levels)
        self._schedule = make_temperature_schedule(initial_temperature, annealing, annealing_rate)
        self._step = 0
        self.register_buffer(
            "_temperature",
            torch.tensor(float(self._schedule(self._step))),
        )
        # Memoized required-variable sets, keyed by the (query, evidence) name
        # signature. The DAG is immutable, so a given signature always yields
        # the same set — for a training loop the signature is constant.
        self._required_cache: Dict[Tuple[frozenset, frozenset], set] = {}

    @property
    def temperature(self) -> torch.Tensor:
        return self._temperature

    def step(self) -> None:
        if self.mode == "ancestral":
            self._step += 1
            self._temperature.fill_(float(self._schedule(self._step)))

    # ------------------------------------------------------------------
    # Per-variable and per-level prediction
    # ------------------------------------------------------------------

    def _format_evidence(self, variable: Variable, value: torch.Tensor) -> torch.Tensor:
        """Cast and reshape an observed value to the cached-value contract.

        Evidence bypasses the CPD, so there is no network output to align
        against: the value is cast to the PGM's parameter dtype (what child
        CPDs expect as input) and reshaped to ``(batch, *variable.shape)``.
        A numel mismatch raises instead of silently broadcasting.
        """
        try:
            dtype = next(self.pgm.parameters()).dtype
        except StopIteration:
            dtype = torch.get_default_dtype()
        return reshape_value_to_event(variable, value.to(dtype))

    def _required_variables(
        self,
        query_names: set,
        evidence_names: set,
    ) -> set:
        """Variables whose value must be resolved to answer the query.

        The forward pass propagates strictly parent → child, so the value of
        a query variable depends only on its ancestors. Evidence is an
        absorbing barrier: an observed node is clamped to its value, so its
        parents are never reached. The required set is therefore the ancestral
        closure of the query set, with the upward walk halting at evidence
        nodes. Everything outside it (barren subtrees, ancestors hidden behind
        evidence) is pruned: its CPD — a neural-net forward pass — never fires.

        The set is closed under "parents of every non-evidence member", which
        guarantees every parent lookup in :meth:`predict_variable` is cached.
        Results are memoized per (query, evidence) name signature.
        """
        key = (frozenset(query_names), frozenset(evidence_names))
        cached = self._required_cache.get(key)
        if cached is not None:
            return cached

        required: set = set()
        stack: List[str] = list(query_names)
        while stack:
            name = stack.pop()
            if name in required:
                continue
            required.add(name)
            if name in evidence_names:
                # Evidence clamps the value; its parents are unreachable.
                continue
            stack.extend(p.name for p in self.pgm.name_to_factor(name).parents)

        self._required_cache[key] = required
        return required

    def predict_variable(
        self,
        variable: Variable,
        cache: Dict[str, torch.Tensor],
        batch_size: int,
        temperature: torch.Tensor,
        evidence: Dict[str, torch.Tensor],
        query: Dict[str, Optional[torch.Tensor]],
        query_names: set,
        evidence_names: set,
        layer_kwargs: Dict,
    ) -> Tuple[str, Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """Evaluate the CPD of a single variable and apply evidence / teacher forcing.

        Observed variables are hard-conditioned: the evidence value is
        propagated directly and the CPD is skipped entirely, so ``params``
        is ``None`` for them.

        Returns ``(name, params, propagated_value)``.
        """
        name = variable.name
        if name in evidence_names:
            # Pure evidence: clamp to the observed value, skip the CPD.
            return name, None, self._format_evidence(variable, evidence[name])

        cpd = self.pgm.name_to_factor(name)
        if cpd.is_root:
            params = cpd(parent_values={})
            params = {
                key: value.unsqueeze(0).expand(batch_size, *value.shape)
                for key, value in params.items()
            }
        else:
            parent_values = {p.name: cache[p.name] for p in cpd.parents}
            params = cpd(parent_values=parent_values, **layer_kwargs)
        value = self._propagate(variable, params, temperature)
        if name in query_names and query[name] is not None:
            propagated = _teacher_force(value, query[name], self.p_int)
        else:
            propagated = value
        return name, params, propagated

    def predict_level(
        self,
        level: List[Variable],
        cache: Dict[str, torch.Tensor],
        batch_size: int,
        temperature: torch.Tensor,
        evidence: Dict[str, torch.Tensor],
        query: Dict[str, Optional[torch.Tensor]],
        query_names: set,
        evidence_names: set,
        layer_kwargs: Dict[str, Dict],
    ) -> List[Tuple[str, Optional[Dict[str, torch.Tensor]], torch.Tensor]]:
        """Evaluate all variables in a topological level sequentially.

        Returns a list of ``(name, params, propagated_value)`` tuples, one per
        variable in ``level``; ``params`` is ``None`` for evidence variables
        (their CPD is skipped).

        When :attr:`parallelize_levels` is enabled and the level holds
        more than one variable, each call is dispatched with
        :func:`torch.jit.fork`, which runs them on PyTorch's interop thread pool —
        real multi-core parallelism on CPU and concurrent kernel launches on GPU,
        while staying autograd-aware so gradients still flow. Otherwise the
        variables are evaluated sequentially.

        NOTE: with stochastic (``mode="ancestral"``) inference the per-thread
        order of global-RNG consumption is not deterministic, so enabling
        ``parallelize_levels`` trades exact run-to-run reproducibility for speed;
        deterministic inference is unaffected.
        """
        if not self.parallelize_levels or len(level) == 1:
            return [
                self.predict_variable(
                    var, cache, batch_size, temperature,
                    evidence, query, query_names, evidence_names, layer_kwargs.get(var.name, {})
                )
                for var in level
            ]

        futures = [
            torch.jit.fork(
                self.predict_variable,
                var, cache, batch_size, temperature,
                evidence, query, query_names, evidence_names, layer_kwargs.get(var.name, {}),
            )
            for var in level
        ]
        return [torch.jit.wait(f) for f in futures]

    def query(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
        layer_kwargs: Dict[str, Dict] = {},
    ) -> InferenceOutput:
        """Run a forward pass through the network in topological order.

        Only the variables actually needed to answer the query are resolved:
        the ancestral closure of the query set, with the upward walk halting at
        evidence (see :meth:`_required_variables`). Barren subtrees and
        ancestors hidden behind evidence are pruned and their CPDs never fire.

        Evidence variables in that set are clamped to their observed values and
        their CPDs are skipped; every other required variable is resolved by
        :meth:`_propagate`. CPD parameters are collected in ``out.params`` for
        query variables and — in ancestral mode — samples are collected in
        ``out.samples`` for every required non-evidence variable (i.e. the
        simulated sub-network, not necessarily the whole graph). A variable
        should appear in either ``query`` or ``evidence``, not both.
        """
        query = self._normalize_query(query)
        self._validate_containers(query, evidence)
        all_tensors = list(evidence.values()) + [v for v in query.values() if v is not None]
        batch_size = all_tensors[0].shape[0] if all_tensors else 1
        out = InferenceOutput()
        cache: Dict[str, torch.Tensor] = {}
        query_names = set(query.keys())
        evidence_names = set(evidence.keys())
        temperature = self.temperature
        sampled = self.mode == "ancestral"
        required = self._required_variables(query_names, evidence_names)

        for level in self.pgm.levels:
            active = [v for v in level if v.name in required]
            if not active:
                continue
            results = self.predict_level(
                active, cache, batch_size, temperature,
                evidence, query, query_names, evidence_names, layer_kwargs
            )
            for name, params, propagated in results:
                cache[name] = propagated
                if name in query_names:
                    out.params[name] = params
                if sampled and name not in evidence_names:
                    out.samples[name] = propagated

        return out

    def _propagate(
        self,
        variable: Variable,
        params: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "deterministic":
            value = propagated_value(variable.distribution, params)
        else:
            value = sample_from(variable, params, temperature)
        # Reshape the realization to the variable's event shape. Samples are then
        # returned and cached (as parent values for downstream CPDs) as
        # (batch, *shape); the flat parameter dict is left as the CPD produced it.
        return reshape_value_to_event(variable, value)