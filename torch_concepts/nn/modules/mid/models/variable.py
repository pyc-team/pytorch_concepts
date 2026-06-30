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
from functools import partial

import torch
import torch.nn as nn
import torch.distributions as dist

from .....distributions.delta import Delta


# ---------------------------------------------------------------------------
# Per-parameter dimension lookup table.
# ---------------------------------------------------------------------------
PARAM_DIM: Dict[Type[dist.Distribution], Dict[str, Callable[[int], int]]] = {
    Delta:                         {"value": lambda size: size},
    dist.Bernoulli:                {"probs": lambda size: size, "logits": lambda size: size},
    dist.RelaxedBernoulli:         {"probs": lambda size: size, "logits": lambda size: size},
    dist.Categorical:              {"probs": lambda size: size, "logits": lambda size: size},
    dist.OneHotCategorical:        {"probs": lambda size: size, "logits": lambda size: size},
    dist.RelaxedOneHotCategorical: {"probs": lambda size: size, "logits": lambda size: size},
    dist.Normal:                   {"loc": lambda size: size, "scale": lambda size: size},
    dist.MultivariateNormal:       {"loc": lambda size: size,
                                    "scale_tril": lambda size: size * (size + 1) // 2},
}

_DEFAULT_DISTRIBUTIONS = {
    'binary': dist.Bernoulli,
    'categorical': dist.OneHotCategorical,
    'continuous': dist.Normal,
}

_DEFAULT_DIST_KWARGS = {
    dist.RelaxedBernoulli: {'temperature': 0.5},
    dist.RelaxedOneHotCategorical: {'temperature': 0.5},
}

# Per-parameter activation mapping a raw network output to a valid distribution
# parameter, keyed by distribution family and then by the parameter name the
# CPD produced.
DEFAULT_ACTIVATIONS = {
    Delta:                         {"value": lambda x: x},
    dist.Bernoulli:                {"probs": lambda x: x, "logits": torch.sigmoid},
    dist.RelaxedBernoulli:         {"probs": lambda x: x, "logits": torch.sigmoid},
    dist.Categorical:              {"probs": lambda x: x, "logits": partial(torch.softmax, dim=-1)},
    dist.OneHotCategorical:        {"probs": lambda x: x, "logits": partial(torch.softmax, dim=-1)},
    dist.RelaxedOneHotCategorical: {"probs": lambda x: x, "logits": partial(torch.softmax, dim=-1)},
    dist.Normal:                   {"loc": lambda x: x, "scale": lambda x: x},
    dist.MultivariateNormal:       {"loc": lambda x: x, "scale_tril": lambda x: x},
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
        members: Optional[List[str]] = None,
    ):
        if isinstance(names, str):
            # Single variable — possibly a plate of named ``members``.
            return super().__new__(cls)
        if members is not None:
            raise TypeError(
                "`members` is only valid with a single (string) name — it makes that "
                "variable a plate of named members. Pass a list of names to create "
                "several independent variables instead."
            )
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
        members: Optional[List[str]] = None,
    ):
        if not isinstance(names, str):
            return
        self.name: str = names

        if members is not None:
            # Plate: a single variable holding several named members. ``size`` is
            # the per-member size (default 1); the total event width is
            # ``len(members) * member_size``, stacked on the last dimension.
            if shape is not None:
                raise ValueError(
                    f"{type(self).__name__}({names!r}): `members` and `shape` are mutually "
                    "exclusive — use `size` for the per-member size."
                )
            if (not isinstance(members, (list, tuple)) or not members
                    or not all(isinstance(m, str) for m in members)):
                raise ValueError(
                    f"{type(self).__name__}({names!r}): `members` must be a non-empty "
                    "list of strings."
                )
            if len(set(members)) != len(members):
                raise ValueError(
                    f"{type(self).__name__}({names!r}): duplicate member names in {members}."
                )
            member_size = 1 if size is None else size
            if not isinstance(member_size, int) or member_size <= 0:
                raise ValueError(
                    f"{type(self).__name__}({names!r}): per-member `size` must be a "
                    f"positive int, got {size!r}."
                )
            self._is_plate: bool = True
            self.members: List[str] = list(members)
            self.member_size: int = member_size
            total = len(self.members) * member_size
            # Per-member addressing slices a contiguous block of every parameter,
            # so each parameter must be laid out one-scalar-per-event-element
            # (probs/logits, loc, scale, value). MultivariateNormal's scale_tril
            # is triangular (size*(size+1)/2), so its members aren't sliceable —
            # model those as separate variables instead.
            if distribution in PARAM_DIM and not all(
                fn(total) == total for fn in PARAM_DIM[distribution].values()
            ):
                raise ValueError(
                    f"{type(self).__name__}({names!r}): plate `members` need a distribution "
                    f"with per-element parameters; {distribution.__name__} has a "
                    "non-per-element parameter (e.g. MultivariateNormal's scale_tril). "
                    "Model these members as separate variables instead."
                )
            shape = torch.Size([total])
        else:
            # Ordinary variable: one member coinciding with the variable name.
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
            self._is_plate: bool = False
            self.members = [self.name]
            self.member_size = math.prod(shape)

        if distribution is None:
            raise ValueError(
                f"{type(self).__name__}({names!r}): `distribution` is required. "
                "Pass an explicit distribution (e.g. dist.Normal, dist.Bernoulli, "
                "or dist.Delta)."
            )
        self.distribution = distribution
        self._shape: torch.Size = shape
        # Column span of each member within the event (last) dimension.
        self._column: Dict[str, slice] = {
            m: slice(i * self.member_size, (i + 1) * self.member_size)
            for i, m in enumerate(self.members)
        }
        self.dist_kwargs: dict = dict(dist_kwargs) if dist_kwargs else {}
        self.metadata: dict = {
            "variable_type": self.variable_type,
        }
        # Set on a member view returned by ``member()``; points back to the plate.
        self._plate: Optional["Variable"] = None

    @property
    def is_plate(self) -> bool:
        """Whether this variable was created with explicit named members."""
        return self._is_plate

    @property
    def plate(self) -> "Variable":
        """The plate this variable belongs to.

        For a member handle (from :meth:`member`) this is the owning plate; for an
        ordinary variable or a plate itself it is the variable. Graph code uses
        ``p.plate.name`` to find the node an edge from ``p`` originates at.
        """
        return self._plate if self._plate is not None else self

    def column_of(self, member: str) -> slice:
        """Column span of ``member`` within this variable's event (last) dimension."""
        return self._column[member]

    def member(self, name: str) -> "Variable":
        """A handle to a single member, usable as a parent (an edge to that member only).

        A child can then depend on just this member of the plate; the engine
        slices the member's column out of the plate's output. The handle carries
        the member's name, per-member size and the plate's distribution, plus a
        back-reference to the owning plate so the graph routes the edge from it.
        """
        if name not in self._column:
            raise KeyError(
                f"{type(self).__name__}({self.name!r}) has no member {name!r}; "
                f"members are {self.members}."
            )
        view = type(self)(
            name,
            distribution=self.distribution,
            size=self.member_size,
            dist_kwargs=copy.deepcopy(self.dist_kwargs),
        )
        view._plate = self
        return view

    @property
    def shape(self) -> torch.Size:
        """Event shape as a :class:`torch.Size`, e.g. ``torch.Size([4])`` or ``torch.Size([3, 4])``."""
        return self._shape

    @property
    def size(self) -> int:
        """Total number of scalar elements: ``math.prod(self.shape)``."""
        return math.prod(self._shape)

    @property
    def param_sizes(self) -> Dict[str, int]:
        """Per-parameter output sizes for this variable's distribution.

        Maps each distribution-parameter name (e.g. ``"loc"``/``"scale"`` for
        ``Normal``, ``"probs"``/``"logits"`` for ``Bernoulli``) to the true
        number of scalar network outputs needed to produce it. Most equal
        :attr:`size` (one scalar per event element); the exceptions are encoded
        in :data:`PARAM_DIM` — e.g. ``MultivariateNormal``'s ``scale_tril``
        needs ``size * (size + 1) // 2`` lower-triangular Cholesky entries.

        Raises
        ------
        ValueError
            If the distribution family has no :data:`PARAM_DIM` entry.
        """
        if self.distribution not in PARAM_DIM:
            raise ValueError(
                f"{type(self).__name__}({self.name!r}): distribution "
                f"{self.distribution.__name__} has no PARAM_DIM entry; cannot "
                "resolve per-parameter sizes."
            )
        return {
            param: fn(self.size)
            for param, fn in PARAM_DIM[self.distribution].items()
        }

    def __repr__(self) -> str:
        s = (
            f"{type(self).__name__}(name={self.name!r}, "
            f"distribution={self.distribution.__name__}, shape={tuple(self.shape)}"
        )
        # Show members only when they differ from the variable name (a plate).
        if self.members != [self.name]:
            s += f", members={self.members}"
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


