"""ForwardInference — non-Pyro forward pass through a :class:`BayesianNetwork`.

Two modes:

- ``"deterministic"``: every variable is propagated by its canonical
  parameter (``loc`` for Normal/MVN, ``probs`` for Bernoulli/OneHotCat,
  ``v`` for Delta).
- ``"ancestral"``: every variable is sampled with the same reparameterised
  distributions used by ``BayesianNetwork.forward`` for unobserved sites
  (straight-through relaxations for the discrete families).

Reparameterised samplers live in
:mod:`torch_concepts.nn.modules.pgm.models.utils` so the model and this
engine stay in lockstep.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import torch

from ..models.bayesian_network import BayesianNetwork
from ..models.utils import propagated_value, sample_from
from ..models.variable import Variable
from .base import BaseInference, make_temperature_schedule
from .outputs import InferenceOutput


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


class ForwardInference(BaseInference):
    """Abstract base class for non-Pyro forward-pass inference engines.

    Concrete subclasses (:class:`DeterministicInference`,
    :class:`AncestralInference`) implement the :meth:`_propagate` method to
    decide whether each variable is resolved deterministically (MAP estimate)
    or by ancestral sampling.  All shared logic (topological traversal, teacher
    forcing, temperature schedule) lives here.

    .. note::
        Do not instantiate this class directly; use
        :class:`DeterministicInference` or :class:`AncestralInference`.
    """

    def __init__(
        self,
        pgm: BayesianNetwork,
        mode: str = "deterministic",
        p_int: float = 1.0,
        initial_temperature: float = 1.0,
        annealing: Union[str, Callable[[int], float]] = "constant",
        annealing_rate: float = 0.0,
    ):
        super().__init__(pgm)
        if mode not in {"deterministic", "ancestral"}:
            raise ValueError(f"mode must be 'deterministic' or 'ancestral', got {mode!r}.")
        if not 0.0 <= float(p_int) <= 1.0:
            raise ValueError(f"p_int must be in [0, 1], got {p_int!r}.")
        self.mode = mode
        self.p_int = float(p_int)
        self._schedule = make_temperature_schedule(initial_temperature, annealing, annealing_rate)
        self._step = 0
        # Cached temperature: a buffer so it travels with .to(device) and is
        # mutated in place by ``step`` rather than re-allocated each call.
        self.register_buffer(
            "_temperature",
            torch.tensor(float(self._schedule(self._step))),
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self._temperature

    def step(self) -> None:
        if self.mode == "ancestral":
            self._step += 1
            self._temperature.fill_(float(self._schedule(self._step)))

    def query(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
    ) -> InferenceOutput:
        query = self._normalize_query(query)
        self._validate_containers(query, evidence)
        all_tensors = list(evidence.values()) + [v for v in query.values() if v is not None]
        batch_size = all_tensors[0].shape[0] if all_tensors else 1
        out = InferenceOutput()
        cache: Dict[str, torch.Tensor] = {}
        query_names = set(query.keys())
        evidence_names = set(evidence.keys())
        temperature = self.temperature

        for variable in self.pgm.sorted_variables:
            name = variable.name
            cpd = self.pgm.factors[name]

            if cpd.is_root:
                params = cpd(parent_values={})
                params = {key: value.unsqueeze(0).expand(batch_size, *value.shape) for key, value in params.items()}
                value = self._propagate(variable, params, temperature)
            else:
                parent_values = {parent.name: cache[parent.name] for parent in cpd.parents}
                params = cpd(parent_values=parent_values)
                value = self._propagate(variable, params, temperature)

            sampled = self.mode == "ancestral"
            if name in evidence_names:
                propagated = _align_gt(evidence[name], value)
            elif name in query_names and query[name] is not None:
                propagated = _teacher_force(value, query[name], self.p_int)
            else:
                propagated = value

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
            return propagated_value(variable.distribution, params)
        return sample_from(variable, params, temperature)
