"""Variable spec objects — metadata describing each node of the PGM."""
from __future__ import annotations

import copy
from typing import List, Optional, Type, Union

import pyro.distributions as dist


def param_dim(distribution: Type[dist.Distribution], size: int) -> int:
    """Return the number of scalar outputs (parameters) a CPD must produce for ``distribution``
    of dimension ``size``.

    For example: ``Bernoulli`` / ``Categorical`` / ``OneHotCategorical`` need
    ``size``; ``Normal`` needs ``2 * size`` (loc, scale); ``MultivariateNormal``
    needs ``size + size * (size + 1) // 2`` (loc + lower-triangular Cholesky);
    ``Delta`` needs ``size`` (sampling is identity on the NN output).
    """
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
        if distribution is None:
            raise ValueError(
                f"{type(self).__name__}({names!r}): `distribution` is required. "
                "Pass an explicit distribution (e.g. dist.Normal, dist.Bernoulli, "
                "or dist.Delta)."
            )
        self.distribution = distribution
        self.size: int = size
        self.dist_kwargs: dict = dict(dist_kwargs) if dist_kwargs else {}
        self.metadata: dict = {
            "variable_type": self._variable_type,
            "interpretable": getattr(type(self), "interpretable", None),
        }

    def __repr__(self) -> str:
        s = (
            f"{type(self).__name__}(name={self.name!r}, "
            f"distribution={self.distribution.__name__}, size={self.size}"
        )
        return s + ")"


class ConceptVariable(Variable):
    """An interpretable random variable.

    May be observed, latent, or deterministic (via ``dist.Delta``); the engine
    decides on a per-call basis whether the variable is observed.
    """
    _variable_type = "concept"
    interpretable: bool = True


class OpaqueVariable(Variable):
    """A non-interpretable variable of the PGM.

    Unlike ``ConceptVariable``, it carries no interpretable semantics.
    """
    _variable_type = "opaque"
    interpretable: bool = False


# Readability alias: a synonym for ConceptVariable, used to emphasise that
# the node is generated by the model rather than supplied as input.
EndogenousVariable = ConceptVariable
