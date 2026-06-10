"""
This script defines the abstract base class ``Variable``
and its concrete subclasses ``ConceptVariable`` and ``EmbeddingVariable``, 
which represent random variables in a Probabilistic Graphical Model.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributions as dist

from .....distributions.delta import Delta


# ---------------------------------------------------------------------------
# Parameter-dimension lookup table.
#
# Maps each supported distribution class to a callable (size: int) -> int that
# returns the total number of scalar network outputs required to parameterise
# that distribution for a variable of the given size.
# ---------------------------------------------------------------------------
PARAM_DIM: Dict[Type[dist.Distribution], Callable[[int], int]] = {
    Delta:                   lambda size: size,
    dist.Bernoulli:          lambda size: size,
    dist.Categorical:        lambda size: size,
    dist.OneHotCategorical:  lambda size: size,
    dist.Normal:             lambda size: 2 * size,
    dist.MultivariateNormal: lambda size: size + size * (size + 1) // 2,
}


def _broadcast(value, n: int, name: str):
    """Return a list of length ``n``: broadcast scalar or check list length.
    
    This is used to construct multiple independent variables with a single constructor call.
    """
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

    Holds the node name (``name``), its distribution family (``distribution``),
    its event ``shape``, and any extra distribution kwargs.  ``size`` is a
    read-only property equal to ``math.prod(shape)``.

    Passing a list of names to the constructor returns a list of independent
    ``Variable`` instances (one per name); ``distribution``, ``shape``, and
    ``dist_kwargs`` may then be a single value (broadcast) or a per-name list.

    Concrete subclasses must implement :attr:`variable_type`.
    """

    @property
    @abstractmethod
    def variable_type(self) -> str:
        """Short string tag identifying the variable kind.

        Defined by each concrete subclass; not set on the abstract base.
        """

    def __new__(
        cls,
        names: Union[str, List[str]],
        distribution=None,
        shape: Union[int, Tuple, "torch.Size", List] = None,
        dist_kwargs: Optional[Union[dict, List[Optional[dict]]]] = None,
        size: Optional[Union[int, List[int]]] = None,
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
        shapes = _broadcast(shape, n, "shape")
        sizes = _broadcast(size, n, "size")
        kwargs_list = _broadcast(dist_kwargs, n, "dist_kwargs")
        return [
            cls(
                name,
                distribution=dists[i],
                shape=shapes[i],
                size=sizes[i],
                dist_kwargs=copy.deepcopy(kwargs_list[i]),
            )
            for i, name in enumerate(names)
        ]

    def __init__(
        self,
        names: Union[str, List[str]],
        distribution=None,
        shape: Union[int, Tuple, "torch.Size"] = None,
        dist_kwargs: Optional[Union[dict, List[Optional[dict]]]] = None,
        size: Optional[Union[int, List[int]]] = None,
    ):
        if not isinstance(names, str):
            return
        if shape is not None and size is not None:
            raise ValueError(
                f"{type(self).__name__}({names!r}): `shape` and `size` are mutually "
                "exclusive — provide one or the other, not both."
            )
        if size is not None:
            if not isinstance(size, int) or size <= 0:
                raise ValueError(
                    f"{type(self).__name__}({names!r}): `size` must be a positive int, "
                    f"got {size!r}."
                )
            shape = torch.Size([size])
        elif shape is None:
            shape = torch.Size([1])  # default
        elif isinstance(shape, int):
            shape = torch.Size([shape])
        else:
            shape = torch.Size(shape)
        if len(shape) == 0:
            raise ValueError("shape must be non-empty.")
        if any(s <= 0 for s in shape):
            raise ValueError(
                f"{type(self).__name__}({names!r}): all shape dimensions must be "
                f"positive, got {tuple(shape)}."
            )
        self.name: str = names
        if distribution is None:
            raise ValueError(
                f"{type(self).__name__}({names!r}): `distribution` is required. "
                "Pass an explicit distribution (e.g. dist.Normal, dist.Bernoulli, "
                "or dist.Delta)."
            )
        self.distribution = distribution
        self._shape: torch.Size = shape
        self.dist_kwargs: dict = dict(dist_kwargs) if dist_kwargs else {}
        self.metadata: dict = {
            "variable_type": self.variable_type,
        }

    @property
    def shape(self) -> torch.Size:
        """Event shape as a :class:`torch.Size`, e.g. ``torch.Size([4])`` or ``torch.Size([3, 4])``."""
        return self._shape

    @property
    def size(self) -> int:
        """Total number of scalar elements: ``math.prod(self.shape)``."""
        return math.prod(self._shape)

    def __repr__(self) -> str:
        s = (
            f"{type(self).__name__}(name={self.name!r}, "
            f"distribution={self.distribution.__name__}, shape={tuple(self.shape)}"
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


class EmbeddingVariable(Variable):
    """A non-interpretable embedding variable.

    May be observed, latent, or deterministic (via ``dist.Delta``); the engine
    decides on a per-call basis whether the variable is observed.
    """

    @property
    def variable_type(self) -> str:
        return "embedding"


