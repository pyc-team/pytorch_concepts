"""
DynamicBayesianNetwork: a directed graphical model over discrete time steps.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Dict, List, Tuple

from ..factors.cpd import ParametricCPD
from ..variable import Variable
from .bayesian_network import BayesianNetwork


def _at(name: str, t: int) -> str:
    """The name of ``name`` at step ``t`` in the unrolled network."""
    return f"{name}_{t}"


def _lagged(p: Variable) -> Tuple[Variable, int]:
    """The variable that a CPD reads through parent ``p``, and how many steps back:
    ``(c, 1)`` for ``c.previous()`` or its members, ``(c, 0)`` for ``c`` or its members."""
    base = p.plate._previous_of
    return (base, 1) if base is not None else (p.plate, 0)


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
    lists: Tuple[List[ParametricCPD], List[ParametricCPD]],
    templates: Dict[str, Variable],
    horizon: int,
) -> Tuple[List[Variable], List[ParametricCPD]]:
    """The variables and CPDs of steps ``0..horizon``, from ``lists = (initial, transition)``:
    ``initial`` at step 0, ``transition`` at every later step (``lists[min(t, 1)]`` at step
    ``t``), all with the modules of the CPDs passed."""
    has = [{cpd.name for cpd in cpds} for cpds in lists]
    variables = dict.fromkeys(templates.values())  # each variable once, in the order passed
    # A copy of each variable at each step where it has a CPD.
    copies = {
        (v.name, t): _copy(v, t)
        for t in range(horizon + 1)
        for v in variables
        if v.name in has[min(t, 1)]
    }

    def read(p: Variable, t: int) -> Variable:
        """A parent of a CPD at step ``t`` as the unrolled network holds it."""
        base, lag = _lagged(p)
        v = copies[(base.name, t - lag)]
        return v if p._plate is None else v.member(_at(p.name, t - lag))

    factors = [
        _clone(cpd, copies[(cpd.name, t)], [read(p, t) for p in cpd.parents])
        for t in range(horizon + 1)
        for cpd in lists[min(t, 1)]
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
    2002), so the DBN is a :class:`BayesianNetwork`: the network of steps 0
    and 1, ``unroll(1)``, the shortest unrolling that uses both lists of CPDs.
    Its :attr:`variables` and :attr:`factors` are named per step (``x_0``,
    ``c_0``, ``c_1``), so a static engine, or
    :func:`~torch_concepts.nn.plot_pgm`, works on it as on any network.
    :class:`~torch_concepts.nn.DynamicInference` takes the names of the
    variables as passed (the *template variables* of Koller and Friedman,
    2009), at any horizon.

    Parameters
    ----------
    variables : list of Variable
        The variables of the model, each once, not once per step (``c``, not
        ``c_0``, ``c_1``). Names must be unique.
    initial : list of ParametricCPD
        The CPDs of step 0, one per variable that exists at step 0.
    transition : list of ParametricCPD
        The CPDs of every later step, one per variable that exists there.

    Attributes
    ----------
    variables : dict
        The variables of the network of steps 0 and 1, one copy per step at
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
    ``DynamicInference`` on it, with the names of the network of steps 0 and
    1: in the example below, ``c_0`` hooks the CPD of step 0, and ``c_1``
    that of the transition, which runs at every step from 1 on. On a CPD in
    both lists, it acts at every step.

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
    >>> list(dbn.unroll(3).variables)
    ['x_0', 'c_0', 'c_1', 'c_2', 'c_3']
    """

    def __init__(
        self,
        variables: List[Variable],
        initial: List[ParametricCPD],
        transition: List[ParametricCPD],
    ):
        variables, lists = list(variables), (list(initial), list(transition))
        # Every name of the model, of a variable or of a member, and its variable. The base
        # class indexes the names per step instead (c_0, c_a_1).
        templates = {name: v for v in variables for name in [v.name, *v.members]}

        # --- Checks ---
        # On the model as passed. The base class then checks the network of steps 0 and 1
        # (one CPD per variable and step, no cycle), whose copies per step hide these.

        # Names are unique, of variables and members alike: each addresses one variable.
        names = Counter(name for v in variables for name in dict.fromkeys([v.name, *v.members]))
        repeated = sorted(name for name, count in names.items() if count > 1)
        if repeated:
            raise ValueError(
                f"Names {repeated} are used more than once. Variable and member names "
                "must be unique."
            )

        # Every CPD is a ParametricCPD of a variable of the model, the same object.
        for cpd in lists[0] + lists[1]:
            if not isinstance(cpd, ParametricCPD):
                raise TypeError(
                    "DynamicBayesianNetwork takes ParametricCPD instances, "
                    f"got {type(cpd).__name__}."
                )
            if templates.get(cpd.name) is not cpd.variable:
                raise ValueError(
                    f"CPD {cpd.name!r}: its variable is not one of `variables`. "
                    "Pass the same object."
                )

        # Every variable has a CPD, without which it would be missing from the network.
        has = [{cpd.name for cpd in cpds} for cpds in lists]  # the variables of each list
        no_cpd = [v.name for v in variables if v.name not in has[0] | has[1]]
        if no_cpd:
            raise ValueError(f"Variables {no_cpd} have no CPD in `initial` or `transition`.")

        # Every parent is a variable of the model, or a member or previous() of one, read at
        # a step where it exists. Step 0 reads it at step -lag, the transition at steps
        # 1 - lag, 2 - lag, ...: there is no step before 0, step 0 is `initial`, and every
        # later step is `transition`.
        for s, cpds in enumerate(lists):
            for cpd in cpds:
                for p in cpd.parents:
                    base, lag = _lagged(p)
                    if templates.get(base.name) is not base:
                        raise ValueError(
                            f"CPD {cpd.name!r}: parent {p.name!r} is not one of `variables`, a "
                            "member of one, or previous() of one. Pass the same object."
                        )
                    read = [-lag] if s == 0 else [1 - lag, 1]
                    missing = [t for t in read if t < 0 or base.name not in has[min(t, 1)]]
                    if missing:
                        raise ValueError(
                            f"CPD {cpd.name!r} of `{('initial', 'transition')[s]}` reads "
                            f"{base.name!r} at step {missing[0]}, where it does not exist."
                        )

        # --- Network of steps 0 and 1 ---
        super().__init__(*_unrolled(lists, templates, 1))
        # Plain containers, so not registered: state_dict() would list their modules again.
        self._templates, self._cpds = templates, lists

    def steps(self, name: str, horizon: int) -> range:
        """The steps of ``0..horizon`` at which ``name`` exists, as a ``range``:
        step 0 with a CPD in ``initial``, steps ``1..horizon`` with one in ``transition``.

        ``name`` is the name of a variable as passed, or of one of its members, as in the keys
        of :class:`~torch_concepts.nn.DynamicInference`, not a name per step such as ``c_1``.
        """
        name = self._templates[name].name  # a member exists where its plate does
        in_initial, in_transition = (any(cpd.name == name for cpd in cpds) for cpds in self._cpds)
        return range(0 if in_initial else 1, horizon + 1 if in_transition else 1)

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
        # --- Checks ---
        # Step 0 and at least one step of the transition.
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}.")

        # --- Network of steps 0..horizon ---
        return BayesianNetwork(*_unrolled(self._cpds, self._templates, horizon))
