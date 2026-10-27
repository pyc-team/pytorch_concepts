"""
DynamicBayesianNetwork: a directed graphical model over discrete time steps.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn

from .....distributions import Delta
from ...low.priors import FixedPrior
from ..factors.cpd import ParametricCPD
from ..variable import Variable
from .bayesian_network import BayesianNetwork


def _at(name: str, t: int) -> str:
    """The name of ``name`` at step ``t`` in the unrolled network."""
    return f"{name}_{t}"


def _prev(name: str, lag: int = 1) -> str:
    """The name of ``name`` ``lag`` steps before, in the network of one step."""
    return f"{name}_prev" if lag == 1 else f"{name}_prev{lag}"


def _lagged(p: Variable) -> Tuple[Variable, int]:
    """The variable a parent reads and how many steps back: ``(c, 2)`` for
    ``c.previous(2)`` or its members, ``(c, 0)`` for ``c`` at the same step."""
    base = p.plate._previous_of
    return (base, p.plate._lag) if base is not None else (p.plate, 0)


def _copy(v: Variable, rename: Callable[[str], str], distribution=None) -> Variable:
    """``v`` with ``rename`` applied to its name and members: the same variable at another step."""
    layout = (
        {"size": v.member_size, "members": [rename(m) for m in v.members]}
        if v.is_plate else {"shape": v.member_shape}
    )
    return type(v)(
        rename(v.name),
        distribution=distribution or v.distribution,
        dist_kwargs=None if distribution else copy.deepcopy(v.dist_kwargs),
        **layout,
    )


def _clone(cpd: ParametricCPD, child: Variable, parents: List[Variable]) -> ParametricCPD:
    """``cpd`` on another child and other parents, with the same modules and parameters."""
    return ParametricCPD(
        child,
        parametrization=dict(cpd.parametrization),
        parents=parents,
        aggregate=cpd._aggregate_arg,
        trunk=cpd.trunk,
    )


class DynamicBayesianNetwork(nn.Module):
    """Directed graphical model over discrete time steps (dynamic Bayesian network).

    A dynamic Bayesian network (DBN) models variables that evolve over time
    steps ``t = 0, 1, 2, ...``. With :math:`Z_t` the variables of step ``t``,
    it factorizes as

    .. math::

        p(Z_0, \\dots, Z_T) = p(Z_0) \\prod_{t=1}^{T} p(Z_t \\mid Z_{t-1}),

    where each factor is a product of CPDs, one per variable of the step: the
    DBN takes those of :math:`p(Z_0)` as ``initial``, and those of
    :math:`p(Z_t \\mid Z_{t-1})` as ``transition``. Every step from 1 on uses
    the same transition CPDs, and so the same parameters.

    A transition CPD reads the step before through ``previous()``: for a
    variable ``z``, the parent ``z.previous()`` is ``z`` at step ``t - 1``,
    while a parent without ``previous()`` is at step ``t``, as in a
    :class:`BayesianNetwork`.

    :math:`Z_t` can hold any variable, concept or embedding, observed or
    latent, discrete or continuous, and any of them can pass to the next
    step: only the concepts, as in a concept bottleneck model over time, or
    also an embedding that holds what the concepts miss, as in
    ``examples/mid_level/09_dynamic_cbm_with_residual.py``. A variable exists
    at the steps where it has a CPD: with a CPD in ``initial`` only, at step
    0, such as an input window; in ``transition`` only, from step 1 on; in
    both, at every step.

    From the two lists the DBN builds the :class:`BayesianNetwork` of a single
    step, to run the model one step at a time: :attr:`initial` for step 0,
    and :attr:`transition` for a later step, where the step before is an
    observed root, ``z_prev`` for ``z.previous()``. :meth:`unroll` builds
    instead the network of a whole horizon, with a copy of each variable per
    step (``z_0``, ``z_1``, ...), which
    :class:`~torch_concepts.nn.DynamicInference` queries with any inference
    engine.

    A transition can also read further back: ``z.previous(2)`` is ``z`` at
    step ``t - 2``, and the transition then gives
    :math:`p(Z_t \\mid Z_{t-1}, Z_{t-2})`. Such a transition cannot run at
    step 1, which has no step ``-1``, so the first steps get CPDs of their
    own: ``initial`` is then one list of CPDs per step, and the transition
    starts after the last one.

    Parameters
    ----------
    variables : list of Variable
        All random variables of the model, each once. Names must be unique.
    initial : list of ParametricCPD
        The CPDs of step 0, one per variable that exists at step 0; or a list
        of such lists, one per step, when the first steps have CPDs of their
        own.
    transition : list of ParametricCPD
        The CPDs of the later steps, one per variable that exists there.

    Attributes
    ----------
    variables : dict
        The variables, ``{name: variable}``.
    interface : list of str
        Names of the variables read through ``previous()``: the state that each
        step passes to the next.
    initial : ModuleList of BayesianNetwork
        The networks of the first steps, ``initial[0]`` for step 0.
    transition : BayesianNetwork
        The network of a later step, where the steps before are observed
        roots: ``z_prev`` for ``z.previous()``, ``z_prev2`` for
        ``z.previous(2)``.

    Raises
    ------
    ValueError
        If a variable has no CPD, or a CPD reads a variable at a step where it
        does not exist, such as ``z.previous()`` in ``initial``.

    Notes
    -----
    The networks that the DBN builds rename the parents of each CPD,
    ``z_prev`` or ``z_3`` for ``z``, so an ``aggregate`` function should read
    its parents by position, e.g. with ``parents.values()``.

    To set a variable at one step, give its value at that step as evidence: a
    forward engine reads it as an intervention (see
    :class:`~torch_concepts.nn.DynamicInference`). ``intervention()`` hooks
    the module of a CPD, so it acts at every step that uses that CPD; its
    target is :attr:`transition` or a network of :attr:`initial`, not the DBN.

    References
    ----------
    Murphy, K. "Dynamic Bayesian Networks: Representation, Inference and
    Learning", PhD thesis, UC Berkeley, 2002.
    https://www.cs.ubc.ca/~murphyk/Thesis/thesis.html

    Examples
    --------
    A concept ``c`` that an input ``x`` gives at step 0, and that a linear map
    moves forward one step at a time, so :math:`Z_0 = \\{x_0, c_0\\}` and
    :math:`Z_t = \\{c_t\\}`:

    >>> import torch
    >>> from torch_concepts.distributions import Delta
    >>> from torch_concepts.nn import (ConceptVariable, DynamicBayesianNetwork,
    ...     EmbeddingVariable, FixedPrior, ParametricCPD)
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
    >>> list(dbn.transition.variables)
    ['c_prev', 'c']
    >>> list(dbn.unroll(2).variables)
    ['x_0', 'c_0', 'c_1', 'c_2']

    A transition that reads two steps back, after a step 1 with a CPD of its
    own:

    >>> step_1 = [ParametricCPD(c, parents=[c.previous()], parametrization=torch.nn.Linear(1, 1))]
    >>> transition = [ParametricCPD(c, parents=[c.previous(), c.previous(2)],
    ...     parametrization=torch.nn.Linear(2, 1))]
    >>> dbn = DynamicBayesianNetwork([x, c], [step_0, step_1], transition)
    >>> list(dbn.transition.variables)
    ['c_prev', 'c_prev2', 'c']
    >>> [p.name for p in dbn.unroll(3).factors["c_3"].parents]
    ['c_2', 'c_1']
    """

    def __init__(
        self,
        variables: List[Variable],
        initial: List[ParametricCPD],
        transition: List[ParametricCPD],
    ):
        super().__init__()
        variables, initial, transition = list(variables), list(initial), list(transition)
        # The CPDs of the first steps: one list for step 0 alone, or one list per step.
        if initial and isinstance(initial[0], (list, tuple)):
            initial = [list(cpds) for cpds in initial]
        else:
            initial = [initial]
        S = len(initial)
        self.variables: Dict[str, Variable] = {v.name: v for v in variables}
        if len(self.variables) != len(variables):
            raise ValueError("Duplicate variable names in `variables`.")

        # Every name addresses one variable, at whatever step it is read.
        self._addressable: Dict[str, Variable] = {}
        for v in variables:
            for name in dict.fromkeys([v.name, *v.members]):
                if name in self._addressable:
                    raise ValueError(
                        f"Name {name!r} is used by both {self._addressable[name].name!r} and "
                        f"{v.name!r}. Variable and member names must be unique."
                    )
                self._addressable[name] = v
        self._queryable_names = frozenset(self._addressable)

        # lists[s] holds the CPDs of step s < S, lists[S] those of every step t >= S.
        lists = [*initial, transition]
        for cpd in (cpd for cpds in lists for cpd in cpds):
            if not isinstance(cpd, ParametricCPD):
                raise TypeError(
                    "DynamicBayesianNetwork takes ParametricCPD instances, "
                    f"got {type(cpd).__name__}."
                )
            if self.variables.get(cpd.name) is not cpd.variable:
                raise ValueError(
                    f"CPD {cpd.name!r}: its variable is not the one registered in "
                    "`variables`. Pass the same object."
                )
        has = [{cpd.name for cpd in cpds} for cpds in lists]
        for n in self.variables:
            steps = [s for s, names in enumerate(has) if n in names]
            if not steps:
                raise ValueError(f"Variable {n!r} has no CPD in `initial` or `transition`.")
            if steps != list(range(steps[0], steps[-1] + 1)):
                raise ValueError(
                    f"Variable {n!r} has a CPD at steps {steps} but not at the steps between: "
                    "a variable exists over consecutive steps."
                )

        def where(s: int) -> str:
            return "transition" if s == S else "initial" if S == 1 else f"initial[{s}]"

        lagged = set()
        for s, cpds in enumerate(lists):
            names = [cpd.name for cpd in cpds]
            repeated = sorted({n for n in names if names.count(n) > 1})
            if repeated:
                raise ValueError(f"Variables {repeated} have more than one CPD in `{where(s)}`.")
            for cpd in cpds:
                for p in cpd.parents:
                    base, lag = _lagged(p)
                    if not lag:
                        if base.name in self.variables and base.name not in has[s]:
                            raise ValueError(
                                f"CPD {cpd.name!r} of `{where(s)}` reads {base.name!r} at its "
                                f"own step, but {base.name!r} has no CPD in `{where(s)}`, so it "
                                "does not exist at that step."
                            )
                        continue
                    if self.variables.get(base.name) is not base:
                        raise ValueError(
                            f"CPD {cpd.name!r}: parent {p.name!r} comes from previous() of a "
                            "variable that is not the one registered in `variables`. Pass the "
                            "same object."
                        )
                    # Step s reads base at s - lag; the transition, which runs at every step
                    # from S on, reads it at every step from S - lag on.
                    at = [s - lag] if s < S else range(S - lag, S + 1)
                    missing = [t for t in at if t < 0 or base.name not in has[min(t, S)]]
                    if missing:
                        hint = (f": the transition starts at step {S}, so give steps "
                                f"0..{lag - 1} their own CPDs, one list per step in `initial`"
                                if s == S and missing[0] < 0 else "")
                        raise ValueError(
                            f"CPD {cpd.name!r} of `{where(s)}` reads {base.name!r} at step "
                            f"{missing[0]} (lag {lag}), where it does not exist{hint}."
                        )
                    lagged.add(base.name)
        self.interface: List[str] = [n for n in self.variables if n in lagged]

        self.initial = nn.ModuleList([self._network(cpds) for cpds in initial])
        self.transition = self._network(transition)
        # Kept for unroll(). Plain lists, so not registered: the networks above hold the modules.
        self._cpds = lists

    def _network(self, cpds: List[ParametricCPD]) -> BayesianNetwork:
        """The static network of one step: its CPDs, with their modules, where a parent
        ``c.previous(lag)`` is an observed root, ``c_prev`` or ``c_prev2`` and so on."""
        prev: Dict[Tuple[str, int], Variable] = {}

        def read(p: Variable) -> Variable:
            base, lag = _lagged(p)
            if not lag:
                return p
            if (base.name, lag) not in prev:
                prev[(base.name, lag)] = _copy(base, lambda m: _prev(m, lag), Delta)
            v = prev[(base.name, lag)]
            return v if p._plate is None else v.member(_prev(p.name, lag))

        clones = [_clone(cpd, cpd.variable, [read(p) for p in cpd.parents]) for cpd in cpds]
        roots = list(prev.values())
        taken = [
            m for v in roots for m in dict.fromkeys([v.name, *v.members])
            if m in self._addressable
        ]
        if taken:
            raise ValueError(
                f"Names {taken} are taken: the network of a step names the steps before with "
                "the suffix '_prev'. Rename those variables or members."
            )
        here = {cpd.name for cpd in cpds}
        return BayesianNetwork(
            [*roots, *(v for v in self.variables.values() if v.name in here)],
            [ParametricCPD(v, parametrization=FixedPrior(torch.zeros(v.shape))) for v in roots]
            + clones,
        )

    @property
    def queryable_names(self) -> frozenset:
        """Names that a query or the evidence can use, the variables and their members."""
        return self._queryable_names

    def resolve(self, name: str) -> Variable:
        """The variable called ``name``, or the plate of the member called ``name``."""
        return self._addressable[name]

    def slices(self, name: str, horizon: int) -> range:
        """The time slices (steps) of ``0..horizon`` at which ``name`` exists, as a ``range``."""
        name, nets = self.resolve(name).name, [*self.initial, self.transition]
        steps = [s for s, net in enumerate(nets) if name in net.variables]
        last = horizon if steps[-1] == len(self.initial) else steps[-1]
        return range(steps[0], min(last, horizon) + 1)

    def unroll(self, horizon: int) -> BayesianNetwork:
        """The network of steps ``0..horizon``, with one copy of each variable per step.

        Variable ``c`` at step ``t`` is ``c_t``, and its member ``c1`` is
        ``c1_t``. The CPDs of every step hold the modules of this DBN, so the
        unrolled network computes with the same parameters and trains them. As
        with :func:`~torch_concepts.nn.modules.mid.inference.utils.unpack_plates`,
        do not register the result as a submodule: ``state_dict()`` would list
        every module once per step.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}.")
        copies = {
            (n, t): _copy(v, lambda m: _at(m, t))
            for t in range(horizon + 1)
            for n, v in self.variables.items()
            if t in self.slices(n, horizon)
        }

        def read(p: Variable, t: int) -> Variable:
            """A parent of a CPD at step ``t`` as the unrolled network holds it."""
            base, lag = _lagged(p)
            v = copies[(base.name, t - lag)]
            return v if p._plate is None else v.member(_at(p.name, t - lag))

        S = len(self.initial)
        factors = [
            _clone(cpd, copies[(cpd.name, t)], [read(p, t) for p in cpd.parents])
            for t in range(horizon + 1)
            for cpd in self._cpds[min(t, S)]
        ]
        return BayesianNetwork(list(copies.values()), factors)
