"""Variable spec objects — metadata describing each node of the PGM."""
from __future__ import annotations

import copy
from typing import List, Optional, Type, Union

import pyro.distributions as dist


def param_dim(distribution: Optional[Type[dist.Distribution]], size: int) -> int:
    """Return the number of scalar outputs (parameters) a CPD must produce for ``distribution``
    of dimension ``size``.

    For example: ``Bernoulli`` / ``Categorical`` / ``OneHotCategorical`` need
    ``size``; ``Normal`` needs ``2 * size`` (loc, scale); ``MultivariateNormal``
    needs ``size + size * (size + 1) // 2`` (loc + lower-triangular Cholesky);
    ``Delta`` needs ``size``; ``None`` (exogenous root) returns ``size``.
    """
    if distribution is None:
        return size  # exogenous root (parametrisation is not constrained)
    if distribution in (dist.Bernoulli, dist.Categorical, dist.OneHotCategorical):
        return size
    if distribution is dist.Normal:
        return 2 * size
    if distribution is dist.MultivariateNormal:
        return size + size * (size + 1) // 2
    if distribution is dist.Delta:
        return size
    raise ValueError(
        f"param_dim: unsupported distribution {distribution!r}. Supported: "
        "Bernoulli, Categorical, OneHotCategorical, Normal, MultivariateNormal, Delta."
    )


def _broadcast(value, n: int, name: str):
    """Return a list of length ``n``: broadcast scalar or check list length."""
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(
                f"{name}: expected a single value or a list of length {n}, "
                f"got list of length {len(value)}."
            )
        return list(value)
    return [value] * n


class Variable:
    """Random variable.

    Holds the node name (``name``), its distribution family, its event size,
    and any extra distribution kwargs. Does not own parameters and is not a
    Pyro primitive — it is consumed by ``ParametricCPD`` and
    ``ProbabilisticModel`` to build the actual stochastic function.

    Passing a list of names to the constructor returns a list of independent
    ``Variable`` instances (one per name); ``distribution``, ``size`` and
    ``dist_kwargs`` may then be scalars (broadcast) or per-name lists.
    """

    _variable_type: str = ""

    def __new__(
        cls,
        names: Union[str, List[str]],
        distribution: Optional[Type[dist.Distribution]] = None,
        size: Union[int, List[int]] = 1,
        dist_kwargs: Optional[Union[dict, List[Optional[dict]]]] = None,
    ):
        if isinstance(names, str):
            return super().__new__(cls)
        if not isinstance(names, list) or not all(
            isinstance(n, str) for n in names
        ):
            raise TypeError(
                "`names` must be a string or a list of strings, "
                f"got {type(names).__name__}."
            )
        n = len(names)
        dists = _broadcast(distribution, n, "distribution")
        sizes = _broadcast(size, n, "size")
        kwargs_list = _broadcast(dist_kwargs, n, "dist_kwargs")
        return [
            cls(
                name,
                distribution=dists[i],
                size=sizes[i],
                dist_kwargs=copy.deepcopy(kwargs_list[i]),
            )
            for i, name in enumerate(names)
        ]

    def __init__(
        self,
        names: Union[str, List[str]],
        distribution: Optional[Type[dist.Distribution]] = None,
        size: Union[int, List[int]] = 1,
        dist_kwargs: Optional[Union[dict, List[Optional[dict]]]] = None,
    ):
        if not isinstance(names, str):
            return  
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive int, got {size!r}.")
        self.name: str = names
        self.distribution = distribution
        self.size: int = size
        self.dist_kwargs: dict = dict(dist_kwargs) if dist_kwargs else {}
        self.metadata: dict = {"variable_type": self._variable_type}

    def __repr__(self) -> str:  
        d = self.distribution.__name__ if self.distribution is not None else "None"
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"distribution={d}, size={self.size})"
        )


class ConceptVariable(Variable):
    """An interpretable random variable.

    May be observed, latent, or deterministic (via ``dist.Delta``); the engine
    decides on a per-call basis whether the variable is observed by checking
    whether it appears in the ``data`` dict.
    """
    _variable_type = "concept"


class ExogenousVariable(Variable):
    """A non-interpretable input of the PGM.

    Always a root, always observed (must appear in ``evidence``/``data`` at
    every engine call), and used as the conditioning input for variational
    guides. ``distribution`` is ignored.
    """
    _variable_type = "exogenous"


# NOTE: The latent variable has been removed as the fact that a variable is latent is an inference-time property.

# Readability alias: a synonym for ConceptVariable, used to emphasise that
# the node is generated by the model rather than supplied as input.
EndogenousVariable = ConceptVariable
