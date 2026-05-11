"""Variable spec objects — metadata describing each node of the PGM."""
from __future__ import annotations

import copy
from typing import List, Optional, Type, Union

import pyro.distributions as dist


def param_dim(distribution: Optional[Type[dist.Distribution]], size: int) -> int:
    """Return the number of scalar parameters a CPD must produce (§2.4)."""
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
    """Abstract spec for one node of the PGM. Not a Pyro primitive."""

    _variable_type: str = ""

    def __new__(
        cls,
        concepts: Union[str, List[str]],
        distribution=None,
        size: Union[int, List[int]] = 1,
        dist_kwargs: Optional[Union[dict, List[Optional[dict]]]] = None,
    ):
        if isinstance(concepts, str):
            return super().__new__(cls)
        if not isinstance(concepts, list) or not all(
            isinstance(n, str) for n in concepts
        ):
            raise TypeError(
                "`concepts` must be a string or a list of strings, "
                f"got {type(concepts).__name__}."
            )
        n = len(concepts)
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
            for i, name in enumerate(concepts)
        ]

    def __init__(
        self,
        concepts: Union[str, List[str]],
        distribution=None,
        size: Union[int, List[int]] = 1,
        dist_kwargs: Optional[Union[dict, List[Optional[dict]]]] = None,
    ):
        if not isinstance(concepts, str):
            return  # pragma: no cover (handled in __new__)
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive int, got {size!r}.")
        self.concept: str = concepts
        self.distribution = distribution
        self.size: int = size
        self.dist_kwargs: dict = dict(dist_kwargs) if dist_kwargs else {}
        self.metadata: dict = {"variable_type": self._variable_type}

    @property
    def name(self) -> str:
        return self.concept

    def __repr__(self) -> str:  # pragma: no cover
        d = self.distribution.__name__ if self.distribution is not None else "None"
        return (
            f"{type(self).__name__}(concept={self.concept!r}, "
            f"distribution={d}, size={self.size})"
        )


class ConceptVariable(Variable):
    """Interpretable, possibly supervised variable."""
    _variable_type = "concept"


class ExogenousVariable(Variable):
    """Non-interpretable input to the PGM; always a root and always observed."""
    _variable_type = "exogenous"


# Readability alias (§2.1).
EndogenousVariable = ConceptVariable
