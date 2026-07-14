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
    :class:`AncestralSamplingInference`) implement the :meth:`_propagate` method to
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
        activate_before_propagation: bool = True,
    ):
        super().__init__(pgm)
        if mode not in {"deterministic", "ancestral"}:
            raise ValueError(f"mode must be either 'deterministic' or 'ancestral', got {mode!r}.")
        if not 0.0 <= float(p_int) <= 1.0:
            raise ValueError(f"p_int must be in [0, 1], got {p_int!r}.")
        self.mode = mode
        self.p_int = float(p_int)
        # When True (deterministic mode only), the propagated parameter is passed
        # through its default activation before being fed to child CPDs. The
        # parameters reported in the inference output stay the raw (non-activated)
        # values produced by the CPD.
        self.activate_before_propagation = bool(activate_before_propagation)
        # When True, variables in the same topological level (conditionally
        # independent given the previous levels) are evaluated concurrently.
        self.parallelize_levels = bool(parallelize_levels)
        # Retained for repr/introspection; the live schedule lives in ``_schedule``.
        self.initial_temperature = float(initial_temperature)
        self.annealing = annealing
        self.annealing_rate = float(annealing_rate)
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

    def __repr__(self) -> str:
        return self._format_repr(
            mode=self.mode,
            p_int=self.p_int,
            initial_temperature=self.initial_temperature,
            annealing=self.annealing,
            annealing_rate=self.annealing_rate,
            parallelize_levels=self.parallelize_levels,
            activate_before_propagation=self.activate_before_propagation,
        )

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

    def _required_variables(self, query_names: set, evidence_names: set) -> set:
        """Variables whose value must be resolved to answer the query.

        Ancestral closure of the variables behind the queried names, walking
        parent -> variable and halting at a fully-observed variable (its value is
        clamped, so its parents are unreachable). A plate is a single variable, so
        the closure is O(number of CPDs), not O(members). Names that are plate
        members resolve to their owning variable. Memoized per name signature.
        """
        key = (frozenset(query_names), frozenset(evidence_names))
        cached = self._required_cache.get(key)
        if cached is not None:
            return cached

        resolve = self.pgm.resolve
        required: set = set()
        stack = [resolve(name) for name in query_names]
        while stack:
            var = stack.pop()
            if var in required:
                continue
            required.add(var)
            if var.name in evidence_names:
                continue  # whole-variable evidence clamps it; its parents are unreachable
            cpd = self.pgm.factors[var.name]
            stack.extend(resolve(p.name) for p in cpd.parents)

        self._required_cache[key] = required
        return required

    def _parent_value(self, parent: Variable, cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Cached value of a parent variable.

        Ordinary parents are read straight from the cache. A parent that is a
        plate member is sliced out of its owning variable's cached value by that
        variable's CPD, so a child can depend on a subset of a plate's members.
        """
        value = cache.get(parent.name)
        if value is not None:
            return value
        var = self.pgm.resolve(parent.name)
        return self.pgm.factors[var.name].select_value(cache[var.name], parent.name)

    def predict_variable(
        self,
        variable: Variable,
        cache: Dict[str, torch.Tensor],
        batch_size: int,
        temperature: torch.Tensor,
        evidence: Dict[str, torch.Tensor],
        query: Dict[str, Optional[torch.Tensor]],
        evidence_names: set,
        layer_kwargs: Dict,
        member_evidence: Dict[str, torch.Tensor],
    ) -> Tuple[str, Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """Evaluate one variable's CPD, applying evidence / teacher forcing.

        A fully-observed variable is clamped and its CPD is skipped (``params``
        is ``None``). ``member_evidence`` carries any individually-observed plate
        members (precomputed by the caller), which are spliced over the computed
        value (partial observation). Returns ``(name, params, value)``.
        """
        name = variable.name
        if name in evidence_names:
            # Pure (whole-variable) evidence: clamp to the observed value, skip the CPD.
            return name, None, self._format_evidence(variable, evidence[name])

        cpd = self.pgm.factors[name]
        if cpd.is_root:
            params = cpd.root_params(batch_size)
        else:
            parent_values = {p.name: self._parent_value(p, cache) for p in cpd.parents}
            params = cpd(parent_values=parent_values, **layer_kwargs)

        value = self._propagate(variable, params, temperature)
        target = query.get(name)
        if target is not None:
            value = _teacher_force(value, target, self.p_int)
        # Partial-plate observation: splice the observed members over the computed
        # value (the CPD owns the column write). ``member_evidence`` is {} unless
        # this variable has individually-observed members.
        value = cpd.clamp_members(value, member_evidence)
        return name, params, value

    def predict_level(
        self,
        level: List[Variable],
        cache: Dict[str, torch.Tensor],
        batch_size: int,
        temperature: torch.Tensor,
        evidence: Dict[str, torch.Tensor],
        query: Dict[str, Optional[torch.Tensor]],
        evidence_names: set,
        layer_kwargs: Dict[str, Dict],
        observed_members: Dict[str, Dict[str, torch.Tensor]],
    ) -> List[Tuple[str, Optional[Dict[str, torch.Tensor]], torch.Tensor]]:
        """Evaluate every variable in a topological level.

        Returns one ``(name, params, value)`` tuple per variable; ``params`` is
        ``None`` for fully-observed variables (their CPD is skipped).
        ``observed_members`` maps a variable name to its individually-observed
        plate members (precomputed once per query).

        When :attr:`parallelize_levels` is enabled and the level holds more than
        one variable, each call is dispatched with :func:`torch.jit.fork` (real
        interop-thread parallelism, autograd-aware); otherwise they run
        sequentially. With ``mode="ancestral"`` the per-thread RNG order is not
        deterministic, so parallelism trades reproducibility for speed.
        """
        if not self.parallelize_levels or len(level) == 1:
            return [
                self.predict_variable(
                    var, cache, batch_size, temperature, evidence, query,
                    evidence_names, layer_kwargs.get(var.name, {}),
                    observed_members.get(var.name, {}),
                )
                for var in level
            ]

        futures = [
            torch.jit.fork(
                self.predict_variable,
                var, cache, batch_size, temperature, evidence, query,
                evidence_names, layer_kwargs.get(var.name, {}),
                observed_members.get(var.name, {}),
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
        """Run a forward pass in topological order, looping over variables.

        A plate is a single variable: one forward produces all its members'
        parameters stacked together. Each queried name reads its result from the
        owning variable — the whole stacked output for the variable/plate name, or
        a column slice for an individual member. Only variables in the ancestral
        closure of the query run (evidence halts the upward walk), so the pass is
        O(number of CPDs), independent of how many members a plate has.

        A name should appear in either ``query`` or ``evidence``, not both.
        """
        query = self._normalize_query(query)
        self._validate_containers(query, evidence)

        query_names = set(query)
        evidence_names = set(evidence)
        resolve = self.pgm.resolve

        # Queried names grouped by the variable whose CPD produces them. The CPD
        # turns a name into its slice (whole output for the variable/plate name,
        # a column for a member), so the engine never slices here.
        requested: Dict[str, List[str]] = {}
        for q_name in query_names:
            requested.setdefault(resolve(q_name).name, []).append(q_name)

        # Individually-observed plate members, grouped by their variable. Built by
        # walking the (small) evidence set — never the members — so it stays O(#evidence),
        # not O(plate size). Whole-variable evidence is handled by the clamp instead.
        observed_members: Dict[str, Dict[str, torch.Tensor]] = {}
        for e_name in evidence_names:
            var = resolve(e_name)
            if e_name != var.name:
                observed_members.setdefault(var.name, {})[e_name] = evidence[e_name]

        required = self._required_variables(query_names, evidence_names)
        tensors = list(evidence.values()) + [v for v in query.values() if v is not None]
        batch_size = tensors[0].shape[0] if tensors else 1
        temperature = self.temperature
        sampled = self.mode == "ancestral"

        out = InferenceOutput()
        cache: Dict[str, torch.Tensor] = {}
        for level in self.pgm.levels:
            active = [var for var in level if var in required]
            if not active:
                continue
            for name, params, value in self.predict_level(
                active, cache, batch_size, temperature, evidence, query,
                evidence_names, layer_kwargs, observed_members,
            ):
                cache[name] = value
                if params is None:
                    continue  # fully-observed variable: clamped, no params emitted
                cpd = self.pgm.factors[name]

                for q_name in requested.get(name, ()):
                    out.params[q_name] = cpd.select(params, q_name)
                if sampled:
                    out.samples[name] = value
                    for q_name in requested.get(name, ()):
                        if q_name != name:
                            out.samples[q_name] = cpd.select_value(value, q_name)

                # for q_name in requested.get(name, ()):
                #     out.params[q_name] = cpd.select(params, q_name)
                #     if sampled:
                #         if q_name != name:
                #             out.samples[q_name] = cpd.select_value(value, q_name)
                # if sampled:
                #     out.samples[name] = value

        return out

    def _propagate(
        self,
        variable: Variable,
        params: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "deterministic":
            value = propagated_value(
                variable.distribution, params, activate=self.activate_before_propagation,
            )
        else:
            value = sample_from(variable, params, temperature)
        # Reshape the realization to the variable's event shape. Samples are then
        # returned and cached (as parent values for downstream CPDs) as
        # (batch, *shape); the flat parameter dict is left as the CPD produced it.
        return reshape_value_to_event(variable, value)