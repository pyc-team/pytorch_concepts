"""Variable spec objects — metadata describing each node of the PGM."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type, Union

import pyro.distributions as dist


# ---------------------------------------------------------------------------
# Parameter-dimension lookup table.
#
# Maps each supported distribution class to a callable (size: int) -> int that
# returns the total number of scalar network outputs required to parameterise
# that distribution for a variable of the given size.
#
# Examples:
#   PARAM_DIM[dist.Normal](4)              → 8   (loc + scale, each dim-4)
#   PARAM_DIM[dist.MultivariateNormal](3)  → 9   (loc=3, tril_flat=6)
#   PARAM_DIM[dist.Bernoulli](1)           → 1
# ---------------------------------------------------------------------------
PARAM_DIM: Dict[Type[dist.Distribution], Callable[[int], int]] = {
    dist.Bernoulli:          lambda size: size,
    dist.Categorical:        lambda size: size,
    dist.OneHotCategorical:  lambda size: size,
    dist.Normal:             lambda size: 2 * size,
    dist.MultivariateNormal: lambda size: size + size * (size + 1) // 2,
    dist.Delta:              lambda size: size,
}


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


class Variable(ABC):
    """Abstract random variable.

    Holds the node name (``name``), its distribution family (``distribution``), its dimensionality (``size``),
    and any extra distribution kwargs. Does not own parameters and is not a
    Pyro primitive — it is consumed by ``ParametricCPD`` and
    ``ProbabilisticModel`` to build the actual stochastic function.

    Passing a list of names to the constructor returns a list of independent
    ``Variable`` instances (one per name); ``distribution``, ``size`` and
    ``dist_kwargs`` may then be scalars (broadcast) or per-name lists.

    Concrete subclasses must implement :attr:`variable_type`.
    """

    @property
    @abstractmethod
    def variable_type(self) -> str:
        """Short string tag identifying the variable kind (e.g. ``'concept'``, ``'opaque'``).

        Defined by each concrete subclass; not set on the abstract base.
        """

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
            "variable_type": self.variable_type,
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

    @property
    def variable_type(self) -> str:
        return "concept"


class OpaqueVariable(Variable):
    """A non-interpretable variable of the PGM.

    Unlike ``ConceptVariable``, it carries no interpretable semantics.
    """

    @property
    def variable_type(self) -> str:
        return "opaque"
