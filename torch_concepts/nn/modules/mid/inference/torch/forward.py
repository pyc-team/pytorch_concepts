"""ForwardInference — pytorch forward pass through a :class:`BayesianNetwork`."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from ...graph.bayesian_network import BayesianNetwork
from ...variable import Variable
from ..utils import make_temperature_schedule, reshape_value_to_event
from ....outputs import InferenceOutput
from .base import TorchBaseInference


def _align_gt(
    gt: torch.Tensor, ref: torch.Tensor, name: Optional[str] = None
) -> torch.Tensor:
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

    Step 4 also silently rescues genuinely wrong targets — e.g. a ``(B,)`` label
    stretched across a ``(B, k)`` output — so a broadcast that is not one of the
    trailing-1 cases warns once per (name, shape) pair.
    """
    aligned = gt.to(ref.dtype) if gt.dtype != ref.dtype else gt
    if aligned.shape == ref.shape:
        return aligned

    original_shape = tuple(aligned.shape)
    if aligned.dim() == ref.dim() + 1 and aligned.shape[-1] == 1:
        aligned = aligned.squeeze(-1)
    elif aligned.dim() + 1 == ref.dim() and ref.shape[-1] == 1:
        aligned = aligned.unsqueeze(-1)
    if aligned.shape == ref.shape:
        return aligned

    _warn_broadcast(name, original_shape, tuple(ref.shape))
    return aligned.expand_as(ref)


# (name, target shape, reference shape) triples already warned about, so a
# training loop reports a suspicious target once rather than every step.
_BROADCAST_WARNED: set = set()


def _warn_broadcast(name: Optional[str], gt_shape: tuple, ref_shape: tuple) -> None:
    key = (name, gt_shape, ref_shape)
    if key in _BROADCAST_WARNED:
        return
    _BROADCAST_WARNED.add(key)
    target = f"for {name!r}" if name else "for a query variable"
    warnings.warn(
        f"Teacher forcing {target}: the target of shape {gt_shape} does not match "
        f"the predicted shape {ref_shape} and is being broadcast to fit. This is "
        "usually a mis-shaped label (e.g. class indices where a one-hot or "
        "per-element target is expected); pass a target of shape "
        f"{ref_shape} to silence this.",
        UserWarning,
        stacklevel=3,
    )


def _teacher_force(
    nn_value: torch.Tensor,
    gt: torch.Tensor,
    p_int: float,
    n_leading: int,
    name: Optional[str] = None,
) -> torch.Tensor:
    """Stochastically replace nn_value with ground truth at rate p_int.

    The draw is per leading (batch-like) element: ``n_leading`` says how many of
    ``nn_value``'s dimensions are leading, so a variable is forced or not as a
    whole, whatever its event shape and however many batch axes there are.
    """
    aligned = _align_gt(gt, nn_value, name)
    if p_int >= 1.0:
        return aligned
    if p_int <= 0.0:
        return nn_value
    mask_shape = nn_value.shape[:n_leading] + (1,) * (nn_value.dim() - n_leading)
    mask = (torch.rand(mask_shape, device=nn_value.device) < p_int).to(nn_value.dtype)
    return mask * aligned + (1.0 - mask) * nn_value


class ForwardInference(TorchBaseInference, ABC):
    """Abstract base class for torch based, forward-pass inference engines.

    Concrete subclasses (:class:`DeterministicInference`,
    :class:`AncestralSamplingInference`,
    :class:`~.map_forward.MAPForwardInference`) implement :meth:`_resolve` to
    decide how each variable is realised — by propagating its canonical
    parameter, by ancestral sampling, or by taking the CPD's mode — and declare
    their behaviour with the class attributes below. All shared logic
    (topological traversal, evidence clamping, teacher forcing, temperature
    schedule) lives here.

    Subclasses declare their behaviour with two class attributes:

    - :attr:`is_stochastic` — whether :meth:`_resolve` produces a *realisation*
      rather than a propagated parameter. Controls whether realisations are
      reported in ``out.samples``, the reported :attr:`mode`, and whether
      :meth:`temperature_step` advances the temperature schedule. Set by the
      sampling engines and by :class:`~.map_forward.MAPForwardInference`, whose
      hard modes are realisations too (it overrides :attr:`mode` and ignores the
      temperature).
    - :attr:`name` — the engine name used in messages and ``repr``.

    Parameters
    ----------
    pgm : BayesianNetwork
        The model to query. Must be directed: the pass walks ``pgm.levels`` in
        topological order, which only a ``BayesianNetwork`` provides.
    p_int : float, optional
        Teacher-forcing probability in ``[0, 1]``: the per-sample chance of
        propagating a query variable's ground-truth target instead of the
        model's own prediction. ``1.0`` recovers sequential/independent CBM
        training, ``0.0`` joint training, and an intermediate value the
        CEM-style random-intervention regime. Only applies to query entries
        that carry a target tensor.
    initial_temperature : float, optional
        Starting temperature of the relaxed-discrete distributions. Ignored by
        the deterministic engines, which never sample.
    annealing : str or callable, optional
        Temperature schedule: ``'constant'``, ``'exponential'``, ``'linear'``,
        or a custom ``f(step) -> float``. Advanced by :meth:`step`.
    annealing_rate : float, optional
        Decay rate consumed by the built-in ``annealing`` schedules.
    parallelize_levels : bool, optional
        Evaluate the conditionally independent variables of one topological
        level concurrently via ``torch.jit.fork``. For a stochastic engine this
        makes the RNG draw order non-deterministic, trading reproducibility for
        speed.
    activate_before_propagation : bool, optional
        Deterministic engines only: pass each propagated parameter through its
        family's default activation before feeding it to child CPDs, so a CPD
        emitting ``logits`` propagates probabilities downstream. The parameters
        reported in ``out.params`` stay the raw, non-activated values.

    Raises
    ------
    ValueError
        If ``p_int`` falls outside ``[0, 1]``.
    TypeError
        If ``pgm`` is not a directed :class:`BayesianNetwork`.
    """

    name = "ForwardInference"

    #: Whether :meth:`_propagate` draws samples (vs. propagating a point estimate).
    is_stochastic: bool = False

    def __init__(
        self,
        pgm: BayesianNetwork,
        p_int: float = 1.0,
        initial_temperature: float = 1.0,
        annealing: Union[str, Callable[[int], float]] = "constant",
        annealing_rate: float = 0.0,
        parallelize_levels: bool = False,
        activate_before_propagation: bool = True,
    ):
        super().__init__(pgm)
        self._require_directed()
        if not 0.0 <= float(p_int) <= 1.0:
            raise ValueError(f"p_int must be in [0, 1], got {p_int!r}.")
        self.p_int = float(p_int)
        # When True (deterministic engines only), the propagated parameter is
        # passed through its default activation before being fed to child CPDs.
        # The parameters reported in the inference output stay the raw
        # (non-activated) values produced by the CPD.
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
    def mode(self) -> str:
        """``"ancestral"`` for a sampling engine, ``"deterministic"`` otherwise.

        Derived from :attr:`is_stochastic` — the engine's behaviour is fixed by
        its class, not by a constructor flag.
        """
        return "ancestral" if self.is_stochastic else "deterministic"

    @property
    def temperature(self) -> torch.Tensor:
        return self._temperature

    def temperature_step(self) -> None:
        """Advance the temperature schedule (no-op for deterministic engines).

        Rebinds the ``_temperature`` buffer to a fresh scalar rather than
        filling it in place.
        """
        if self.is_stochastic and self.training:
            self._step += 1
            self._temperature = self._temperature.new_full(
                (), float(self._schedule(self._step))
            )

    # ------------------------------------------------------------------
    # Per-variable and per-level prediction
    # ------------------------------------------------------------------

    def _format_evidence(self, variable: Variable, value: torch.Tensor) -> torch.Tensor:
        """Cast and reshape an observed value to the cached-value contract.

        Evidence bypasses the CPD, so there is no network output to align
        against: the value is cast to the PGM's parameter dtype (what child
        CPDs expect as input) and reshaped to ``(*leading, *variable.shape)``.
        A numel mismatch raises instead of silently broadcasting.
        """
        if value.is_floating_point():
            try:
                dtype = next(self.pgm.parameters()).dtype
            except StopIteration:
                dtype = torch.get_default_dtype()
            value = value.to(dtype)
        return reshape_value_to_event(variable, value)

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

    def predict_variable(
        self,
        variable: Variable,
        cache: Dict[str, torch.Tensor],
        leading: torch.Size,
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
            params = cpd.root_params(leading)
        else:
            # ``cache`` is keyed by whole-variable names; the CPD resolves each
            # parent (slicing member-handle parents out of their plate's value).
            params = cpd(parent_values=cache, **layer_kwargs)

        value = self._propagate(variable, params, temperature)
        target = query.get(name)
        if target is not None:
            value = _teacher_force(value, target, self.p_int, len(leading), name)
        # Partial-plate observation: splice the observed members over the computed
        # value (the CPD owns the column write). ``member_evidence`` is {} unless
        # this variable has individually-observed members.
        value = cpd.clamp_members(value, member_evidence)
        return name, params, value

    def predict_level(
        self,
        level: List[Variable],
        cache: Dict[str, torch.Tensor],
        leading: torch.Size,
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
        sequentially. For a stochastic engine the per-thread RNG order is not
        deterministic, so parallelism trades reproducibility for speed.
        """
        if not self.parallelize_levels or len(level) == 1:
            return [
                self.predict_variable(
                    var, cache, leading, temperature, evidence, query,
                    evidence_names, layer_kwargs.get(var.name, {}),
                    observed_members.get(var.name, {}),
                )
                for var in level
            ]

        futures = [
            torch.jit.fork(
                self.predict_variable,
                var, cache, leading, temperature, evidence, query,
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
        layer_kwargs: Optional[Dict[str, Dict]] = None,
    ) -> InferenceOutput:
        """Run a forward pass in topological order, looping over variables.

        A plate is a single variable: one forward produces all its members'
        parameters stacked together. Each queried name reads its result from the
        owning variable — the whole stacked output for the variable/plate name, or
        a column slice for an individual member. Only variables in the ancestral
        closure of the query run (evidence halts the upward walk), so the pass is
        O(number of CPDs), independent of how many members a plate has.

        A name should appear in either ``query`` or ``evidence``, not both.

        Member (partial-plate) evidence is **value forcing**: the member's column
        is overwritten after the plate is produced and the forced value propagates
        to descendants, contributing no likelihood of its own.

        Every tensor may carry any number of leading (batch-like) dimensions —
        ``(*leading, *event)`` — and the results come back with that same leading
        shape. The event always lives on the last axis.
        """
        query = self._normalize_query(query)
        self._validate_containers(query, evidence)
        layer_kwargs = layer_kwargs or {}

        query_names = list(query)
        leading = self._query_leading_shape(query, evidence)

        # Whole-variable evidence clamps-and-skips its CPD; member evidence is
        # threaded to ``clamp_members`` (a no-op for the empty dict).
        evidence, observed_members = self._split_evidence(evidence)
        evidence_names = set(evidence)

        required = self._required_variables(set(query_names), evidence_names)
        temperature = self.temperature

        cache: Dict[str, torch.Tensor] = {}
        computed: Dict[str, Dict[str, torch.Tensor]] = {}
        for level in self.pgm.levels:
            active = [var for var in level if var in required]
            if not active:
                continue
            for name, params, value in self.predict_level(
                active, cache, leading, temperature, evidence, query,
                evidence_names, layer_kwargs, observed_members,
            ):
                cache[name] = value
                if params is None:
                    continue  # fully-observed variable: clamped, no params emitted
                computed[name] = params

        # advance the temperature schedule if stochastic and training mode.
        self.temperature_step()  

        # Assemble once. ``params`` covers the queried names; ``samples`` covers
        # every variable the pass actually realised, queried or not — an ancestor
        # resolved only to reach the query is still a value the caller may want.
        return InferenceOutput(
            params=self._assemble_params(computed, query_names),
            samples=(
                self._assemble_samples(
                    {name: cache[name] for name in computed}, list(computed)
                )
                if self.is_stochastic else None
            ),
        )

    @abstractmethod
    def _resolve(
        self,
        variable: Variable,
        params: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Turn a CPD's parameters into a flat ``(batch, size)`` realisation.

        The one behavioural difference between the forward engines: a point
        estimate (:class:`DeterministicInference`), a reparameterised draw
        (:class:`AncestralSamplingInference`), or the CPD's mode
        (:class:`~.map_forward.MAPForwardInference`).
        """

    def _propagate(
        self,
        variable: Variable,
        params: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        value = self._resolve(variable, params, temperature)
        # Reshape the realization to the variable's event shape. Samples are then
        # returned and cached (as parent values for downstream CPDs) as
        # (batch, *shape); the flat parameter dict is left as the CPD produced it.
        return reshape_value_to_event(variable, value)