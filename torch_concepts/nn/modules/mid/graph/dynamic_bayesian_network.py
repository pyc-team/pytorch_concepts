"""
DynamicBayesianNetwork: a directed graphical model over discrete time steps.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple, Union

from ..factors.cpd import ParametricCPD
from ..variable import Variable
from .bayesian_network import BayesianNetwork


def _at(name: str, t: int) -> str:
    """The name of ``name`` at step ``t`` in the unrolled network."""
    return f"{name}_{t}"


def _lagged(p: Variable) -> Tuple[Variable, int]:
    """The variable that a CPD reads through parent ``p``, and how many steps back:
    ``(c, 2)`` for ``c.previous(2)`` or its members, ``(c, 0)`` for ``c`` or its members."""
    base = p.plate._previous_of
    return (base, p.plate._lag) if base is not None else (p.plate, 0)


def _copy(v: Variable, t: int) -> Variable:
    """The copy of ``v`` at step ``t``: ``c`` is ``c_t``, and its member ``c1`` is ``c1_t``."""
    layout = (
        {"size": v.member_size, "members": [_at(m, t) for m in v.members]}
        if v.is_plate else {"shape": v.member_shape}
    )
    return type(v)(
        _at(v.name, t),
        distribution=v.distribution,
        dist_kwargs=copy.deepcopy(v.dist_kwargs),
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


def _unrolled(
    lists: List[List[ParametricCPD]], templates: Dict[str, Variable], horizon: int
) -> Tuple[List[Variable], List[ParametricCPD]]:
    """The variables and CPDs of steps ``0..horizon``: ``lists[s]`` for step ``s``, and the
    last list for every later step, all with the modules of the CPDs passed."""
    S, has = len(lists) - 1, [{cpd.name for cpd in cpds} for cpds in lists]
    variables = dict.fromkeys(templates.values())  # each variable once, in the order passed
    copies = {
        (v.name, t): _copy(v, t)
        for t in range(horizon + 1)
        for v in variables
        if v.name in has[min(t, S)]
    }

    def read(p: Variable, t: int) -> Variable:
        """A parent of a CPD at step ``t`` as the unrolled network holds it."""
        base, lag = _lagged(p)
        v = copies[(base.name, t - lag)]
        return v if p._plate is None else v.member(_at(p.name, t - lag))

    factors = [
        _clone(cpd, copies[(cpd.name, t)], [read(p, t) for p in cpd.parents])
        for t in range(horizon + 1)
        for cpd in lists[min(t, S)]
    ]
    return list(copies.values()), factors


class DynamicBayesianNetwork(BayesianNetwork):
    """Directed graphical model over discrete time steps (dynamic Bayesian network).

    A dynamic Bayesian network (DBN) models variables that evolve over time
    steps ``t = 0, 1, 2, ...``. With :math:`Z_t` the variables of step ``t``,
    it factorizes as

    .. math::

        p(Z_0, \\dots, Z_T) = p(Z_0) \\prod_{t=1}^{T} p(Z_t \\mid Z_{t-1}),

    where each factor is a product of CPDs, one per variable of the step: the
    DBN takes those of :math:`p(Z_0)` as ``initial``, and those of
    :math:`p(Z_t \\mid Z_{t-1})` as ``transition``. Every later step uses the
    same transition CPDs, and so the same parameters.

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

    :meth:`unroll` builds the network of a whole horizon, with a copy of each
    variable per step at which it exists (``z_0``, ``z_1``, ...), which
    :class:`~torch_concepts.nn.DynamicInference` queries with any inference
    engine.

    Unrolled over any number of steps, a DBN is a Bayesian network (Murphy,
    2002), so the DBN is a :class:`BayesianNetwork`.
    As a :class:`BayesianNetwork`, the DBN is the network of its first steps:
    :meth:`unroll` over the initial steps and one step of the transition,
    steps 0 and 1 in the first example below. It is the shortest unrolling
    that uses every list of CPDs, and it gives the distribution of the DBN
    over those steps. Its :attr:`variables` and :attr:`factors` are named per
    step (``x_0``, ``c_0``, ``c_1``), so a static engine, or
    :func:`~torch_concepts.nn.plot_pgm`, works on it as on any network.
    :class:`~torch_concepts.nn.DynamicInference` takes the names of the
    variables as passed (the *template variables* of Koller and Friedman,
    2009), at any horizon.

    A transition can also read further back: ``z.previous(2)`` is ``z`` at
    step ``t - 2``, and the transition then gives
    :math:`p(Z_t \\mid Z_{t-1}, Z_{t-2})`. Such a transition cannot run at
    step 1, which has no step ``-1``, so the initial steps get CPDs of their
    own: ``initial`` is then one list of CPDs per initial step, and the
    transition starts after the last one.

    Parameters
    ----------
    variables : list of Variable
        The variables of the model, each once, not once per step (``c``, not
        ``c_0``, ``c_1``). Names must be unique.
    initial : list of ParametricCPD, or list of lists
        The CPDs of step 0, one per variable that exists at step 0; or a list
        of such lists, one per initial step.
    transition : list of ParametricCPD
        The CPDs of the later steps, one per variable that exists there.

    Attributes
    ----------
    variables : dict
        The variables of the network of the first steps, one copy per step at
        which each exists: ``x_0``, ``c_0``, ``c_1``.
    factors : ModuleDict
        Their CPDs, keyed like :attr:`variables`, with the modules of the CPDs
        passed.

    Raises
    ------
    ValueError
        If a variable has no CPD, or a CPD reads a variable at a step where it
        does not exist, such as ``z.previous()`` at step 0.

    Notes
    -----
    The DBN, and the networks that it builds, rename the parents of each CPD,
    ``z_0`` or ``z_3`` for ``z``, so an ``aggregate`` function should read
    its parents by position, e.g. with ``parents.values()``.

    To set a variable at one step, give its value at that step as evidence: a
    forward engine reads it as an intervention (see
    :class:`~torch_concepts.nn.DynamicInference`). ``intervention()`` hooks
    the module of a CPD instead, so it acts at every step that runs that
    module, whichever CPD holds it. Its target can be the DBN, or a
    ``DynamicInference`` on it, with the names of the network of the first
    steps: in the first example below, ``c_0`` hooks the CPD of step 0, and
    ``c_1`` that of the transition, which runs at every step from 1 on; in
    the second, ``c_1`` hooks the CPD of step 1, and ``c_2`` that of the
    transition. On a CPD in every list, it acts at every step.

    References
    ----------
    Murphy, K. "Dynamic Bayesian Networks: Representation, Inference and
    Learning", PhD thesis, UC Berkeley, 2002.
    https://www.cs.ubc.ca/~murphyk/Thesis/thesis.html

    Koller, D. and Friedman, N. "Probabilistic Graphical Models: Principles
    and Techniques", MIT Press, 2009, chapter 6.

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
    >>> list(dbn.variables)
    ['x_0', 'c_0', 'c_1']
    >>> list(dbn.unroll(2).variables)
    ['x_0', 'c_0', 'c_1', 'c_2']

    A transition that reads two steps back, after a step 1 with a CPD of its
    own:

    >>> step_1 = [ParametricCPD(c, parents=[c.previous()], parametrization=torch.nn.Linear(1, 1))]
    >>> transition = [ParametricCPD(c, parents=[c.previous(), c.previous(2)],
    ...     parametrization=torch.nn.Linear(2, 1))]
    >>> dbn = DynamicBayesianNetwork([x, c], [step_0, step_1], transition)
    >>> list(dbn.variables)
    ['x_0', 'c_0', 'c_1', 'c_2']
    >>> [p.name for p in dbn.unroll(3).factors["c_3"].parents]
    ['c_2', 'c_1']
    """

    def __init__(
        self,
        variables: List[Variable],
        initial: Union[List[ParametricCPD], List[List[ParametricCPD]]],
        transition: List[ParametricCPD],
    ):
        variables, initial, transition = list(variables), list(initial), list(transition)
        # The CPDs of the initial steps: one list for step 0 alone, or one list per step.
        if initial and isinstance(initial[0], (list, tuple)):
            initial = [list(cpds) for cpds in initial]
        else:
            initial = [initial]
        S = len(initial)
        by_name: Dict[str, Variable] = {v.name: v for v in variables}
        if len(by_name) != len(variables):
            raise ValueError("Duplicate variable names in `variables`.")

        # Every name addresses one variable, at whatever step it is read.
        templates: Dict[str, Variable] = {}
        for v in variables:
            for name in dict.fromkeys([v.name, *v.members]):
                if name in templates:
                    raise ValueError(
                        f"Name {name!r} is used by both {templates[name].name!r} and "
                        f"{v.name!r}. Variable and member names must be unique."
                    )
                templates[name] = v

        # lists[s] holds the CPDs of step s < S, lists[S] those of every step t >= S.
        lists = [*initial, transition]
        for cpd in (cpd for cpds in lists for cpd in cpds):
            if not isinstance(cpd, ParametricCPD):
                raise TypeError(
                    "DynamicBayesianNetwork takes ParametricCPD instances, "
                    f"got {type(cpd).__name__}."
                )
            if by_name.get(cpd.name) is not cpd.variable:
                raise ValueError(
                    f"CPD {cpd.name!r}: its variable is not one of `variables`. "
                    "Pass the same object."
                )
        has = [{cpd.name for cpd in cpds} for cpds in lists]
        for n in by_name:
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

        for s, cpds in enumerate(lists):
            names = [cpd.name for cpd in cpds]
            repeated = sorted({n for n in names if names.count(n) > 1})
            if repeated:
                raise ValueError(f"Variables {repeated} have more than one CPD in `{where(s)}`.")
            for cpd in cpds:
                for p in cpd.parents:
                    base, lag = _lagged(p)
                    if by_name.get(base.name) is not base:
                        raise ValueError(
                            f"CPD {cpd.name!r}: parent {p.name!r} is not one of `variables`, a "
                            "member of one, or previous() of one. Pass the same object."
                        )
                    # Step s reads base at s - lag (lag 0 at s itself); the transition, which
                    # runs at every step from S on, reads it at every step from S - lag on.
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

        super().__init__(*_unrolled(lists, templates, S))  # the network of steps 0..S
        # A plain list, so not registered: state_dict() would list its modules again.
        self._templates, self._cpds = templates, lists

    def steps(self, name: str, horizon: int) -> range:
        """The steps of ``0..horizon`` at which ``name`` exists, as a ``range``.

        ``name`` is the name of a variable as passed, or of one of its members, as in the keys
        of :class:`~torch_concepts.nn.DynamicInference`, not a name per step such as ``c_1``.
        """
        name, S = self._templates[name].name, len(self._cpds) - 1
        steps = [s for s, cpds in enumerate(self._cpds) if any(cpd.name == name for cpd in cpds)]
        last = horizon if steps[-1] == S else steps[-1]
        return range(steps[0], min(last, horizon) + 1)

    def unroll(self, horizon: int) -> BayesianNetwork:
        """The network of steps ``0..horizon``, with one copy of each variable per
        step at which it exists.

        Variable ``c`` at step ``t`` is ``c_t``, and its member ``c1`` is
        ``c1_t``. The CPDs of every step hold the modules of this DBN, so the
        unrolled network computes with the same parameters and trains them. As
        with :func:`~torch_concepts.nn.modules.mid.inference.utils.unpack_plates`,
        do not register the result as a submodule: ``state_dict()`` would list
        every module once per step.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}.")
        return BayesianNetwork(*_unrolled(self._cpds, self._templates, horizon))
