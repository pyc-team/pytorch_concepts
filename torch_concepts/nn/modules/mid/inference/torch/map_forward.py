"""MAPForwardInference — greedy per-node MAP forward pass through a :class:`BayesianNetwork`."""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch

from ...graph.bayesian_network import BayesianNetwork
from ...variable import Variable
from ....outputs import InferenceOutput
from .forward import ForwardInference
from .utils import mode_value


class MAPForwardInference(ForwardInference):
    """Forward inference that propagates each CPD's *most likely* value.

    Evidence is clamped, then the network is swept root to leaf: every variable
    is resolved to the mode of its CPD **given the already-resolved (hard)
    values of its parents**, and that hard value is what its children see. A
    binary node contributes ``0.`` or ``1.``, a categorical one a one-hot row, a
    Normal its ``loc``.

    The realisations are reported in ``out.samples``; ``out.params`` still
    carries the raw CPD parameters they were derived from, so a caller can read
    both the decision and the confidence behind it. As with every forward
    engine, a fully observed variable emits no parameters and does not appear in
    ``out.samples`` — its value is the one the caller supplied. An individually
    observed *plate member* does appear, carrying its observed value.

    This is a *greedy, per-node* MAP sweep, **not** the joint MAP: each node
    commits to its own mode given its parents' commitments, which in general is
    not the single most probable joint assignment (that needs max-product or an
    enumeration engine). It is the hard counterpart of
    :class:`DeterministicInference`, which propagates the soft parameter
    (a Bernoulli's ``probs``, a categorical's whole simplex row) instead.

    Test-time only: the whole pass runs under :func:`torch.no_grad`. Taking a
    mode is a hard, non-differentiable step, so there is no gradient to keep —
    train with :class:`DeterministicInference` or
    :class:`AncestralSamplingInference` and evaluate with this one.

    Parameters
    ----------
    pgm : BayesianNetwork
        The model to query. Must be directed: the sweep walks ``pgm.levels`` in
        topological order, which only a ``BayesianNetwork`` provides.
    parallelize_levels : bool, optional
        Evaluate the conditionally independent variables of one topological
        level concurrently (see :meth:`ForwardInference.predict_level`). Unlike
        a sampling engine this pass consumes no RNG, so parallelism costs no
        reproducibility.

    Raises
    ------
    TypeError
        If ``pgm`` is not a directed :class:`BayesianNetwork`.

    Examples
    --------
    Read the decision from ``samples`` and the confidence behind it from the
    parameters::

        engine = MAPForwardInference(bn)
        out = engine.query(query=["dysp"], evidence={"asia": asia, "smoke": smoke})
        out.samples["dysp"]     # the committed value: hard 0. or 1.
        out.logits["dysp"]      # the raw CPD output it was thresholded from

    See ``examples/utilization/1_pgm/4_map_forward_inference.py`` for a runnable
    comparison against :class:`DeterministicInference`.
    """

    name = "MAPForwardInference"

    #: The mode is a *realisation* — a hard value the pass commits to — so it is
    #: reported in ``out.samples``. Nothing is actually drawn (the mode is a
    #: deterministic function of the CPD), which is why :attr:`mode` is overridden
    #: below and the inherited temperature schedule is never read.
    is_stochastic = True

    def __init__(self, pgm: BayesianNetwork, parallelize_levels: bool = False, **temperature_kwargs):
        # Teacher forcing and the temperature schedule are training-time knobs
        # with no meaning at test time, so neither is exposed: ``p_int`` is
        # pinned to 0 (the model follows its own trajectory) and the
        # temperature, which ``_resolve`` ignores, never advances.
        super().__init__(pgm, p_int=0.0, parallelize_levels=parallelize_levels,
                         **temperature_kwargs)

    def __repr__(self) -> str:
        # Only this engine's own knobs — the inherited annealing/teacher-forcing
        # parameters are pinned and inert, so printing them would mislead.
        return self._format_repr(
            mode=self.mode, parallelize_levels=self.parallelize_levels
        )

    @property
    def mode(self) -> str:
        """``"map"`` — neither a soft propagation nor a draw."""
        return "map"

    @torch.no_grad()
    def query(
        self,
        query: Union[List[str], Dict[str, Optional[torch.Tensor]]],
        evidence: Dict[str, torch.Tensor],
        layer_kwargs: Optional[Dict[str, Dict]] = None,
    ) -> InferenceOutput:
        """Run the MAP sweep; see :meth:`ForwardInference.query` for the contract.

        Wrapped in :func:`torch.no_grad` so the CPD forwards build no graph at
        all, not just the mode step — nothing in ``out`` carries a ``grad_fn``.
        ``BaseInference.__call__`` delegates here, so both entry points are
        covered. This also covers ``parallelize_levels=True``, but only because
        eager ``torch.jit.fork`` runs synchronously: were ``predict_level`` ever
        scripted, grad mode would have to be propagated to the worker threads.
        """
        return super().query(query, evidence, layer_kwargs)

    def _resolve(
        self,
        variable: Variable,
        params: Dict[str, torch.Tensor],
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """The CPD's most likely value — a hard mode (``temperature`` unused)."""
        return mode_value(variable, params)
