"""
DynamicInference: any inference engine on a :class:`DynamicBayesianNetwork`,
through its unrolled network.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from ..graph.dynamic_bayesian_network import DynamicBayesianNetwork, _at
from ...outputs import InferenceOutput
from .....tensor import AnnotatedTensor
from .base import BaseInference
from .torch.deterministic import DeterministicInference

#: A query or evidence key: a name, for every step at which it exists, or ``(name, t)``
#: for step ``t``.
Key = Union[str, Tuple[str, int]]

#: The options of the temperature schedule, which DynamicInference sets for its engines.
_SCHEDULE = ("initial_temperature", "annealing", "annealing_rate", "final_temperature")


def _flat(per_step: List[Dict]) -> Dict[str, Optional[torch.Tensor]]:
    """The per-step dicts as one dict of the unrolled network, keyed ``name_t``."""
    return {_at(n, t): v for t, step in enumerate(per_step) for n, v in step.items()}


class DynamicInference(BaseInference):
    """Any inference engine on a :class:`~torch_concepts.nn.DynamicBayesianNetwork`.

    Each query names a horizon: ``DynamicInference`` unrolls the DBN over
    steps ``0..horizon``
    (:meth:`~torch_concepts.nn.DynamicBayesianNetwork.unroll`), with one copy
    of each variable per step at which it exists, and runs a static engine on
    the unrolled network, :class:`~torch_concepts.nn.DeterministicInference`
    by default. The network and its engine are built once per horizon.

    A key of the query or the evidence is the name of a variable as passed, or
    of one of its members (``c``, ``c_a``), for every step at which its
    variable exists, or a pair ``(name, t)``, for step ``t`` only. The tensor
    of a name stacks those steps before the event,
    ``(batch, steps, *shape)``; that of a pair holds one step, ``(batch, *shape)``,
    as does that of a variable of step 0 only, such as an input window. A pair
    overrides the name at its step: for a variable ``z``,
    ``{"z": None, ("z", 4): z4}`` queries ``z`` at every step, with the target
    ``z4`` at step 4. All queried names must be queried at the same steps.

    The output is an :class:`~torch_concepts.nn.InferenceOutput`, as from a
    static engine, with the queried steps on the last leading axis:
    ``out.value["z"]`` has shape ``(batch, steps, width)`` and the labels of
    ``z``. ``out.probabilities`` has no step axis: it is the joint
    probability of the query over all its steps, ``(batch,)``.

    A forward engine reads the evidence as an intervention, which acts on its
    step and on the later ones. An engine that conditions on the evidence,
    such as :class:`~torch_concepts.nn.RejectionSampling`, reads it as an
    observation, so the evidence at a step also changes the belief about the
    steps before.

    Parameters
    ----------
    pgm : DynamicBayesianNetwork
        The DBN; each query runs the engine on ``pgm.unroll(horizon)``.
    engine : callable, optional
        An engine class, or a function that builds an engine from a
        :class:`~torch_concepts.nn.BayesianNetwork`. Defaults to
        :class:`~torch_concepts.nn.DeterministicInference`. A function
        receives the same ``engine_kwargs`` as a class, so write it as
        ``lambda bn, **kw: ImportanceSampling(bn, proposal=Q(bn), **kw)``,
        with ``Q`` a proposal class such as
        :class:`~torch_concepts.nn.MutilatedNetworkProposal`. ``bn`` is a new
        network at each horizon, so build inside the function what refers to
        its variables, such as a proposal or the guide CPDs of
        ``VariationalInference``, but the learnable modules they hold once,
        outside it. Those modules are not in the ``parameters()``,
        ``state_dict()``, ``.to()`` or ``eval()`` of ``DynamicInference``:
        train them together with ``pgm.parameters()``.
    **engine_kwargs
        Passed to ``engine``, e.g. ``p_int=1.0`` or ``n_samples=1000``.
        ``p_int`` and the options of the temperature schedule, such as
        ``annealing``, also set those of ``DynamicInference``. The engine runs
        on the unrolled network, so an option that names variables uses their
        names there, one per step up to the longest horizon ``H``:
        ``activations={f"z_{t}": fn for t in range(H + 1)}``.

    Notes
    -----
    Like any engine, ``DynamicInference`` is an ``nn.Module`` that holds the
    DBN as ``pgm``: ``parameters()`` are those of the DBN, and
    ``state_dict()`` holds the DBN and the temperature reached in training, not
    the engines of the horizons. The engines of the horizons read the
    temperature of ``DynamicInference``, so set the schedule, and call
    ``eval()`` and ``temperature_step()``, on ``DynamicInference`` itself.

    References
    ----------
    Murphy, K. "Dynamic Bayesian Networks: Representation, Inference and
    Learning", PhD thesis, UC Berkeley, 2002.
    https://www.cs.ubc.ca/~murphyk/Thesis/thesis.html

    Examples
    --------
    The example of :class:`~torch_concepts.nn.DynamicBayesianNetwork`: an
    input ``x`` gives a concept ``c`` at step 0, and a linear map moves ``c``
    forward. A forecast of ``c`` at steps 0 to 4, for 8 inputs:

    >>> import torch
    >>> from torch_concepts.distributions import Delta
    >>> from torch_concepts.nn import (ConceptVariable, DynamicBayesianNetwork,
    ...     DynamicInference, EmbeddingVariable, FixedPrior, ParametricCPD)
    >>> x = EmbeddingVariable("x", distribution=Delta, size=3)
    >>> c = ConceptVariable("c", distribution=Delta)
    >>> step_0 = [
    ...     ParametricCPD(x, parametrization=FixedPrior(torch.zeros(3))),
    ...     ParametricCPD(c, parents=[x], parametrization=torch.nn.Linear(3, 1)),
    ... ]
    >>> transition = [
    ...     ParametricCPD(c, parents=[c.previous()], parametrization=torch.nn.Linear(1, 1)),
    ... ]
    >>> dbn = DynamicBayesianNetwork([x, c], step_0, transition)
    >>> engine = DynamicInference(dbn)
    >>> window = torch.randn(8, 3)
    >>> out = engine.query(["c"], evidence={"x": window}, horizon=4)
    >>> out.value["c"].shape  # (batch, steps, width)
    torch.Size([8, 5, 1])

    Steps 3 and 4 only, from a value of ``c`` given at step 2:

    >>> evidence = {"x": window, ("c", 2): torch.zeros(8, 1)}
    >>> out = engine.query([("c", 3), ("c", 4)], evidence, horizon=4)
    >>> out.value["c"].shape
    torch.Size([8, 2, 1])
    """

    name = "DynamicInference"

    def __init__(
        self,
        pgm: DynamicBayesianNetwork,
        engine: Optional[Callable[..., BaseInference]] = None,
        **engine_kwargs,
    ):
        # --- Checks ---
        # Only a DBN can be unrolled; a static network takes an engine directly.
        if not isinstance(pgm, DynamicBayesianNetwork):
            raise TypeError(
                f"DynamicInference runs on a DynamicBayesianNetwork, got {type(pgm).__name__}; "
                "query a static network with an engine directly."
            )

        # --- Engine options ---
        # p_int and the schedule are those of DynamicInference too: engines read its temperature.
        shared = {k: engine_kwargs[k] for k in (*_SCHEDULE, "p_int") if k in engine_kwargs}
        super().__init__(pgm, **shared)
        self.engine = DeterministicInference if engine is None else engine
        self.engine_kwargs = engine_kwargs
        self._engines: Dict[int, BaseInference] = {}  # one per horizon, built on first use

    def __repr__(self) -> str:
        return self._format_repr(engine=self.engine)

    def clear_cache(self) -> None:
        """Drop the query caches and the engines, which the next queries build again."""
        super().clear_cache()
        self._engines.clear()

    def query(
        self,
        query: Union[List[Key], Dict[Key, Optional[torch.Tensor]]],
        evidence: Optional[Dict[Key, torch.Tensor]] = None,
        *,
        horizon: int,
        n_samples: Optional[int] = None,
    ) -> InferenceOutput:
        """Query the DBN over steps ``0..horizon``.

        The keys of ``query`` and ``evidence``, and the shapes of their tensors,
        are described in the class docstring. ``n_samples`` sets the batch of a
        query without any tensor, as for a static forward engine; otherwise the
        tensors of the query and the evidence set the leading shape.
        """
        # --- Query and evidence per step ---
        engine = self._engine(horizon)  # first, so that unroll() checks the horizon
        q = self._by_step(self._normalize_query(query), horizon)
        e = self._by_step(evidence or {}, horizon)
        names = list(dict.fromkeys(n for step in q for n in step))
        steps = [t for t in range(horizon + 1) if q[t]]

        # --- Checks ---
        # Nothing to stack without a queried step. The engine checks the rest: the names per
        # step, the tensors and their leading shapes.
        if not steps:
            raise ValueError("The query is empty.")

        # --- Query of the unrolled network ---
        fq, fe = _flat(q), _flat(e)
        out = engine.query(fq, fe, **({} if n_samples is None else {"n_samples": n_samples}))
        # The leading shape as the engine reads it, once the query has checked the tensors.
        return self._stack(out, names, steps, engine._query_leading_shape(fq, fe, n_samples))

    __call__ = query

    def _engine(self, horizon: int) -> BaseInference:
        """The engine on the network unrolled over ``horizon``, built on first use."""
        engine = self._engines.get(horizon)
        if engine is None:
            engine = self.engine(self.pgm.unroll(horizon), **self.engine_kwargs)

            # --- Checks ---
            # No schedule of its own: the engine reads the temperature of DynamicInference
            # (below), so a schedule set by an engine function would be silently ignored.
            own = {k: getattr(engine, k) for k in _SCHEDULE
                   if getattr(engine, k) != getattr(self, k)}
            if own:
                raise ValueError(
                    f"The engine has a temperature schedule of its own, {own}, but the engines "
                    "read the temperature of DynamicInference: pass the schedule to "
                    "DynamicInference, which passes it on to the engine."
                )

            self._engines[horizon] = engine
        # Shared at every query: .to() and load_state_dict(assign=True) replace the buffer.
        engine._temperature = self._temperature
        return engine

    def _by_step(self, container: Dict[Key, Optional[torch.Tensor]], horizon: int) -> List[Dict]:
        """One ``{name: value}`` dict per step ``0..horizon``.

        A name sets every step at which its variable exists, a pair ``(name, t)`` step ``t``.
        """
        dbn, per_step = self.pgm, [{} for _ in range(horizon + 1)]
        # Names first, then pairs, which override them at their step.
        for key, value in sorted(container.items(), key=lambda item: isinstance(item[0], tuple)):
            name, t = key if isinstance(key, tuple) else (key, None)

            # --- Checks ---
            # A name of the model as passed (c), not a name per step (c_1).
            if name not in dbn._templates:
                raise ValueError(f"Unknown name {name!r}; the names are {sorted(dbn._templates)}.")
            steps = dbn.steps(name, horizon)
            # A pair names a step at which its variable exists.
            if t is not None and t not in steps:
                raise ValueError(
                    f"{name!r} exists at steps {steps.start}..{steps.stop - 1}, not at {t!r}."
                )

            # --- Value at each step ---
            value = getattr(value, "tensor", value)  # an AnnotatedTensor as its tensor
            if t is not None:  # a pair: its step
                per_step[t][name] = value
            elif value is None or steps == range(1):  # no tensor, or no time axis (step 0 only)
                for s in steps:
                    per_step[s][name] = value
            else:  # a name: its tensor stacks its steps
                for s, v in zip(steps, self._unstack(name, value, steps)):
                    per_step[s][name] = v
        return per_step

    def _unstack(self, name: str, value: torch.Tensor, steps: range) -> Tuple[torch.Tensor, ...]:
        """The per-step tensors of ``value``, whose time axis precedes the event of ``name``."""
        v = self.pgm._templates[name]
        event = tuple(v.shape if name == v.name else v.member_shape)
        axis = value.dim() - len(event) - 1  # the time axis

        # --- Checks ---
        # The event last, and one entry per step on the axis before it.
        if axis < 0 or tuple(value.shape[axis + 1:]) != event or value.shape[axis] != len(steps):
            raise ValueError(
                f"{name!r} has shape {tuple(value.shape)}, but it exists at steps "
                f"{steps.start}..{steps.stop - 1}, so its tensor stacks them before the event: "
                f"(*leading, {len(steps)}, {', '.join(map(str, event))}). For one step, key it "
                f"as ({name!r}, t)."
            )

        return value.unbind(axis)

    def _stack(
        self, out: InferenceOutput, names: List[str], steps: List[int], leading: torch.Size
    ) -> InferenceOutput:
        """The output of the unrolled network, with the queried steps on the last leading axis.

        Labelled as in the model: ``c_a`` for ``c_a_0``, ``c_a_1``, ...
        """
        # Each queried label once, as a static engine reports it (a plate and one of its members).
        resolve = self.pgm._templates.__getitem__
        members = {n: resolve(n).members if n == resolve(n).name else [n] for n in names}
        labels = dict.fromkeys(m for n in names for m in members[n])
        suffix = len(_at("", steps[0]))

        def stack(kind: str, tensors: Dict[str, AnnotatedTensor]) -> Dict[str, AnnotatedTensor]:
            """``tensors`` with the steps stacked, and the labels of the model."""
            stacked = {}
            for quantity, tensor in tensors.items():
                # The queried labels that the tensor holds at each step.
                known = tensor.annotations.label_to_index
                found = {t: [m for m in labels if _at(m, t) in known] for t in steps}
                first = found[steps[0]]

                # --- Checks ---
                # Every step holds the same labels, so that the steps stack.
                for t in steps:
                    if found[t] != first:
                        raise ValueError(
                            f"{kind}[{quantity!r}] holds {found[t]} at step {t} and {first} at "
                            f"step {steps[0]}, but the queried steps must report the same "
                            "parameters. Names queried at different steps, or observed at some "
                            "of them, or CPDs parametrized differently in `initial` and "
                            "`transition`, make them differ: query those steps separately."
                        )

                # --- Steps stacked ---
                if first:  # none for a quantity of names that are not queried
                    views = [tensor[[_at(m, t) for m in first]] for t in steps]
                    data = torch.stack([v.tensor for v in views], dim=len(leading))
                    renamed = views[0].annotations.rename(lambda label: label[:-suffix])
                    stacked[quantity] = AnnotatedTensor(data, renamed, axis=-1)
            return stacked

        samples = stack("samples", {} if out.samples is None else {"samples": out.samples})
        return InferenceOutput(params=stack("params", out.params),
                               guide_params=stack("guide_params", out.guide_params),
                               samples=samples.get("samples"), probabilities=out.probabilities)
