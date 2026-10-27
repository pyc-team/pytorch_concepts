"""
DynamicInference: any inference engine on a :class:`DynamicBayesianNetwork`,
through its unrolled network.
"""

from __future__ import annotations

import numbers
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


def _stack(outs: Dict[int, InferenceOutput], leading: torch.Size) -> InferenceOutput:
    """One output from the outputs of the queried steps, with time as the last leading axis."""

    def layout(out: InferenceOutput) -> dict:
        """What a step reports: its quantities, and the labels of each."""
        return {kind: {q: t.annotations.labels for q, t in getattr(out, kind).items()}
                for kind in ("params", "guide_params")}

    (t0, first), *rest = outs.items()
    for t, out in rest:
        if layout(out) != layout(first):
            raise ValueError(
                f"Step {t} reports {layout(out)} and step {t0} reports {layout(first)}, but the "
                "queried names must report the same parameters at every step. A plate observed "
                "whole at some steps, or CPDs parametrized differently in `initial` and "
                "`transition`, make them differ: query those steps separately."
            )

    def stack(tensors: List[AnnotatedTensor]) -> AnnotatedTensor:
        data = torch.stack([t.tensor for t in tensors], dim=len(leading))
        return AnnotatedTensor(data, tensors[0].annotations, axis=-1)

    per_step = list(outs.values())
    return InferenceOutput(
        params={q: stack([o.params[q] for o in per_step]) for q in first.params},
        guide_params={q: stack([o.guide_params[q] for o in per_step]) for q in first.guide_params},
        samples=None if first.samples is None else stack([o.samples for o in per_step]),
    )


class DynamicInference:
    """Any inference engine on a :class:`~torch_concepts.nn.DynamicBayesianNetwork`.

    Each query names a horizon: ``DynamicInference`` unrolls the DBN over
    steps ``0..horizon``
    (:meth:`~torch_concepts.nn.DynamicBayesianNetwork.unroll`), with one copy
    of each variable per step, and runs a static engine on the unrolled
    network, :class:`~torch_concepts.nn.DeterministicInference` by default.
    The network and its engine are built once per horizon.

    A key of the query or the evidence is a name, for every step at which its
    variable exists, or a pair ``(name, t)``, for step ``t`` only. The tensor
    of a name stacks those ``T`` steps before the event,
    ``(batch, T, *shape)``; that of a pair holds one step, ``(batch, *shape)``,
    as does that of a variable of step 0 only, such as an input window. A pair
    overrides the name at its step: for a variable ``z``,
    ``{"z": None, ("z", 4): z4}`` queries ``z`` at every step, with the target
    ``z4`` at step 4. All queried names must be queried at the same steps.

    The output is an :class:`~torch_concepts.nn.InferenceOutput`, as from a
    static engine, with the queried steps on the last leading axis:
    ``out.value["z"]`` has shape ``(batch, T, width)`` and the labels of
    ``z``.

    A forward engine reads the evidence as an intervention, which acts on its
    step and on the later ones. An engine that conditions on the evidence,
    such as :class:`~torch_concepts.nn.RejectionSampling`, reads it as an
    observation, so the evidence at a step also changes the steps before.

    Parameters
    ----------
    dbn : DynamicBayesianNetwork
        The model to query.
    engine : callable, optional
        An engine class, or a function that builds an engine from a
        :class:`~torch_concepts.nn.BayesianNetwork`, such as
        ``lambda bn: ImportanceSampling(bn, proposal=...)``. Defaults to
        :class:`~torch_concepts.nn.DeterministicInference`. Modules that the
        function creates, such as the guides of ``VariationalInference``, are
        not part of the DBN: build them once, outside the function, and train
        them together with ``dbn.parameters()``.
    **engine_kwargs
        Passed to ``engine``, e.g. ``p_int=1.0`` or ``n_samples=1000``. The
        engine runs on the unrolled network, so an option that names variables
        uses their names there, one per step:
        ``activations={f"z_{t}": fn for t in range(horizon + 1)}``.

    Attributes
    ----------
    engines : dict
        The engines built so far, one per horizon.

    Notes
    -----
    ``DynamicInference`` is not an ``nn.Module``: the parameters live in the
    DBN, so an optimizer trains ``dbn.parameters()``.

    References
    ----------
    Murphy, K. "Dynamic Bayesian Networks: Representation, Inference and
    Learning", PhD thesis, UC Berkeley, 2002.
    https://www.cs.ubc.ca/~murphyk/Thesis/thesis.html

    Examples
    --------
    The first example of :class:`~torch_concepts.nn.DynamicBayesianNetwork`:
    an input ``x`` gives a concept ``c`` at step 0, and a linear map moves
    ``c`` forward. A forecast of ``c`` at steps 0 to 4, for 8 inputs:

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

    def __init__(
        self,
        dbn: DynamicBayesianNetwork,
        engine: Optional[Callable[..., BaseInference]] = None,
        **engine_kwargs,
    ):
        if not isinstance(dbn, DynamicBayesianNetwork):
            raise TypeError(
                f"DynamicInference runs on a DynamicBayesianNetwork, got {type(dbn).__name__}; "
                "query a static network with an engine directly."
            )
        self.dbn = dbn
        self.engine = DeterministicInference if engine is None else engine
        self.engine_kwargs = engine_kwargs
        self.engines: Dict[int, BaseInference] = {}
        self._temperature_steps = 0

    def __repr__(self) -> str:
        name = getattr(self.engine, "__name__", type(self.engine).__name__)
        return f"{type(self).__name__}(engine={name})"

    def temperature_step(self) -> None:
        """Advance the temperature schedule of the engines by one training step.

        A no-op outside training, as for a static engine. An engine built later,
        for a new horizon, starts from the same point of the schedule, which is
        not saved with the DBN.
        """
        if not self.dbn.training:
            return
        self._temperature_steps += 1
        for engine in self.engines.values():
            engine.temperature_step()

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
        are those of the class docstring. ``n_samples`` sets the batch of a
        query without any tensor, as for a static forward engine; otherwise the
        tensors of the query and the evidence set the leading shape.

        Raises
        ------
        ValueError
            If ``horizon`` is below 1, a name is unknown, a step is outside the
            steps of its name, a value is not a tensor or lacks its time axis,
            the tensors disagree on their leading shape, the queried names are
            queried at different steps, a name is both queried and observed at
            a step, or the evidence of a step names a plate and its members.
        TypeError
            If a key is neither a name nor a ``(name, t)`` pair.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}.")
        if isinstance(query, list):
            query = dict.fromkeys(query)
        q = self._by_step(query, horizon, "query")
        e = self._by_step(evidence or {}, horizon, "evidence")
        names = list(dict.fromkeys(n for step in q for n in step))
        steps = [t for t in range(horizon + 1) if q[t]]
        if not steps:
            raise ValueError("The query is empty.")
        if any(len(q[t]) != len(names) for t in steps):
            asked = {n: [t for t in steps if n in q[t]] for n in names}
            raise ValueError(
                f"Queried names must be queried at the same steps, got {asked}. Query "
                "them in separate calls."
            )
        resolve = self.dbn.resolve
        for t in range(horizon + 1):
            observed = [n for n in q[t] if n in e[t] or resolve(n).name in e[t]]
            if observed:
                raise ValueError(f"{observed} at step {t} are both queried and observed.")
            both = [n for n in e[t] if resolve(n).name != n and resolve(n).name in e[t]]
            if both:
                raise ValueError(
                    f"The evidence at step {t} names {both} and their plate: give the "
                    "whole plate or some of its members."
                )
        leading = self._leading(q, e, n_samples)

        def flat(per_step: List[Dict]) -> Dict[str, Optional[torch.Tensor]]:
            """The per-step dicts as one dict of the unrolled network, keyed ``name_t``."""
            return {_at(n, t): v for t, step in enumerate(per_step) for n, v in step.items()}

        kwargs = {} if n_samples is None else {"n_samples": n_samples}
        out = self._engine(horizon).query(flat(q), flat(e), **kwargs)
        return self._split(out, names, steps, leading)

    __call__ = query

    def _engine(self, horizon: int) -> BaseInference:
        """The engine on the network unrolled over ``horizon``, built on first use."""
        engine = self.engines.get(horizon)
        if engine is None:
            engine = self.engine(self.dbn.unroll(horizon), **self.engine_kwargs)
            self.engines[horizon] = engine
            for _ in range(self._temperature_steps):  # where the schedule of the other engines is
                engine.temperature_step()
        return engine

    def _by_step(
        self, container: Dict[Key, Optional[torch.Tensor]], horizon: int, what: str
    ) -> List[Dict]:
        """One ``{name: value}`` dict per step ``0..horizon``.

        A pair ``(name, t)`` overrides the name at step ``t``.
        """
        dbn, per_step = self.dbn, [{} for _ in range(horizon + 1)]
        known, pairs = sorted(dbn.queryable_names), []
        for key, value in container.items():
            # An AnnotatedTensor as its tensor; the labels are not read.
            value = getattr(value, "tensor", value)
            if not isinstance(key, str):
                pairs.append((key, value))
                continue
            if key not in dbn.queryable_names:
                raise ValueError(f"Unknown {what} name {key!r}; the names are {known}.")
            steps = dbn.slices(key, horizon)
            if value is None or steps == range(1):  # a variable of step 0 only has no time axis
                values = [value] * len(steps)
            else:
                values = self._unstack(key, value, steps, what)
            for t, v in zip(steps, values):
                per_step[t][key] = v
        for key, value in pairs:
            if not (isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str)
                    and isinstance(key[1], numbers.Integral) and not isinstance(key[1], bool)):
                raise TypeError(
                    f"A {what} key is a name or a (name, step) pair with an integer step, "
                    f"got {key!r}."
                )
            name, t = key[0], int(key[1])
            if name not in dbn.queryable_names:
                raise ValueError(f"Unknown {what} name {name!r}; the names are {known}.")
            steps = dbn.slices(name, horizon)
            if t not in steps:
                raise ValueError(
                    f"{name!r} exists at steps {steps.start}..{steps.stop - 1}, not at {t}."
                )
            per_step[t][name] = value
        return per_step

    def _unstack(
        self, name: str, value: torch.Tensor, steps: range, what: str
    ) -> Tuple[torch.Tensor, ...]:
        """The per-step tensors of ``value``, whose time axis precedes the event of ``name``."""
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{what}[{name!r}] must be a Tensor, got {type(value).__name__}.")
        v = self.dbn.resolve(name)
        event = tuple(v.shape if name == v.name else v.member_shape)
        axis = value.dim() - len(event) - 1
        if axis < 0 or tuple(value.shape[axis + 1:]) != event or value.shape[axis] != len(steps):
            raise ValueError(
                f"{what}[{name!r}] has shape {tuple(value.shape)}, but {name!r} exists at steps "
                f"{steps.start}..{steps.stop - 1}, so its tensor stacks them before the event: "
                f"(*leading, {len(steps)}, {', '.join(map(str, event))}). For one step, key it "
                f"as ({name!r}, t)."
            )
        return value.unbind(axis)

    def _leading(self, q: List[Dict], e: List[Dict], n_samples: Optional[int]) -> torch.Size:
        """The leading shape of the query: that of its tensors, which must agree.

        Without any tensor, ``n_samples`` observations.
        """
        found = {}
        for per_step in (e, q):
            for t, step in enumerate(per_step):
                for name, value in step.items():
                    if isinstance(value, torch.Tensor):
                        v = self.dbn.resolve(name)
                        shape = tuple(v.leading_of(value, member=None if name == v.name else name))
                        found.setdefault(shape, (name, t))
        if len(found) > 1:
            raise ValueError("Mismatched leading (batch) dimensions: " + ", ".join(
                f"{name!r} at step {t} has {shape}" for shape, (name, t) in found.items()) + ".")
        if found:
            return torch.Size(next(iter(found)))
        return torch.Size([1 if n_samples is None else int(n_samples)])

    def _split(
        self, out: InferenceOutput, names: List[str], steps: List[int], leading: torch.Size
    ) -> InferenceOutput:
        """The output of the unrolled network, split back into the queried steps."""
        if not out.params and out.samples is None:  # an estimator reports a probability only
            return InferenceOutput(probabilities=out.probabilities)

        # Each queried label once, as a static engine reports it (a plate and one of its members).
        resolve = self.dbn.resolve
        members = {n: resolve(n).members if n == resolve(n).name else [n] for n in names}

        def at_step(tensor: Optional[AnnotatedTensor], t: int) -> Optional[AnnotatedTensor]:
            """The queried labels of step ``t`` in ``tensor``, renamed to the names of the model."""
            if tensor is None:
                return None
            known = tensor.annotations.label_to_index
            labels = [label for label in dict.fromkeys(_at(m, t) for n in names for m in members[n])
                      if label in known]
            if not labels:
                return None
            view, suffix = tensor[labels], len(_at("", t))
            renamed = view.annotations.rename(lambda label: label[:-suffix])
            return AnnotatedTensor(view.tensor, renamed, axis=-1)

        def split(tensors: Dict[str, AnnotatedTensor], t: int) -> Dict[str, AnnotatedTensor]:
            views = {quantity: at_step(tensor, t) for quantity, tensor in tensors.items()}
            return {quantity: view for quantity, view in views.items() if view is not None}

        stacked = _stack({
            t: InferenceOutput(params=split(out.params, t), guide_params=split(out.guide_params, t),
                               samples=at_step(out.samples, t))
            for t in steps
        }, leading)
        stacked.probabilities = out.probabilities
        return stacked
