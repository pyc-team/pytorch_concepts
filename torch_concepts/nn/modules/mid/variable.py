"""
This script defines the abstract base class ``Variable``
and its concrete subclasses ``ConceptVariable`` and ``EmbeddingVariable``, 
which represent random variables in a Probabilistic Graphical Model.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributions as dist

from ....distributions.delta import Delta
from .distributions import spec_for


# Semantic concept type -> distribution family. A high-level *policy* (which
# family should a "binary" concept get?) that subclasses override.
_DEFAULT_DISTRIBUTIONS = {
    'binary': dist.Bernoulli,
    'categorical': dist.OneHotCategorical,
    'continuous': dist.Normal,
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
    its event ``shape``, and any extra distribution kwargs.

    Passing a list of names to the constructor returns a list of independent
    ``Variable`` instances (one per name); ``distribution``, ``shape``, and
    ``dist_kwargs`` may then be a single value (broadcast) or a per-name list.

    Concrete subclasses must implement :attr:`variable_type`.

    Parameters
    ----------
    names : str or list of str
        A single name builds one variable. A **list** of names builds one
        independent variable per name and returns them as a list; the remaining
        arguments are then either a single value (broadcast to every name) or a
        per-name list of the same length.
    distribution : type
        The distribution family (e.g. ``dist.Bernoulli``, ``dist.Normal``,
        ``Delta``). Required — there is no default. Determines which parameters
        a CPD must produce for this variable, how engines propagate and sample
        it, and whether belief propagation can enumerate it.
    shape : int or tuple of int or torch.Size, optional
        Event shape of a single realisation, e.g. ``(n_concepts, emb_dim)``.
        Mutually exclusive with ``size``; defaults to ``(1,)``. Not allowed
        together with ``members``.
    dist_kwargs : dict, optional
        Extra keyword arguments forwarded to the distribution constructor
        (e.g. ``{'temperature': 0.5}`` for the relaxed families).
    size : int, optional
        Shorthand for ``shape=(size,)``. When ``members`` is given this is
        instead the **per-member** size (default ``1``), and the total event
        width becomes ``len(members) * size``.
    members : list of str, optional
        Turn this into a **plate**: one variable whose event stacks the named
        members along the last dimension, each still addressable by its own name
        for queries, evidence and interventions. Only valid with a single
        (string) ``names``, mutually exclusive with ``shape``, and requires a
        registered family whose parameters are one-scalar-per-element (so
        ``MultivariateNormal`` is rejected).
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

        # A variable's family must be one the registry knows. This is the single
        # gate every variable passes through.
        if distribution is None:
            raise ValueError(
                f"{type(self).__name__}({names!r}): `distribution` is required. "
                "Pass an explicit distribution (e.g. dist.Normal, dist.Bernoulli, "
                "or dist.Delta)."
            )
        spec = spec_for(distribution, f"{type(self).__name__}({names!r})")

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
            if not spec.is_per_element:
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

        self.distribution = distribution
        self._shape: torch.Size = shape
        # Dictionary mapping member name -> slice corresponding to that member.
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
        ordinary variable or a plate itself it is the variable. 
        """
        return self._plate if self._plate is not None else self

    def column_of(self, member: str) -> slice:
        """The slice of the event dimension corresponding to a member."""
        return self._column[member]

    def member(self, name: str) -> "Variable":
        """A handle to a single member.

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

    # FIXME: there is a tricky for loop here. Maybe the same can be done without looping.
    def get_slice(self, labels: Union[str, List[str]]) -> Union[slice, List[int]]:
        """Flattened indices for member(s) in this variable's event dimension.

        Parameter-agnostic: the indices address event columns, which are the
        same for every distribution parameter (probs/logits, loc, scale, ...).
        """
        if isinstance(labels, str):
            labels = [labels]

        indices = []
        for label in labels:
            if label not in self._column:
                raise ValueError(f"Label '{label}' not found in members {self.members}")
            s = self._column[label]
            indices.extend(range(s.start, s.stop))

        return indices

    def select(
        self, params: Dict[str, torch.Tensor], name: str
    ) -> Dict[str, torch.Tensor]:
        """Distribution params for ``name``: the whole tensor for this variable's own
        name, or a member's column slice (a view, no copy)."""
        if name == self.name:
            return params
        columns = self.column_of(name)
        return {key: value[..., columns] for key, value in params.items()}

    def select_value(self, value: torch.Tensor, name: str) -> torch.Tensor:
        """Realised value for ``name``: the whole value, or a member's column slice."""
        if name == self.name:
            return value
        return value[..., self.column_of(name)]

    def clamp_members(
        self, value: torch.Tensor, observed: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Clamp individually-observed members to their observed values.

        Used for *partial observation* of a plate: given the whole stacked
        ``value``, overwrite the columns of the observed members with their
        observed tensors, leaving the unobserved members untouched. ``observed``
        maps member name -> observed tensor. Returns a new tensor (the input is
        not mutated); a no-op when ``observed`` is empty.

        The clone is required: the caller caches this value and reuses it as a
        parent input for downstream factors, so writing the observed columns
        in place would corrupt that cache and break autograd on the tensor the
        CPD produced.
        """
        if not observed:
            return value
        value = value.clone()
        for member, obs in observed.items():
            columns = self.column_of(member)
            slot = value[..., columns]
            value[..., columns] = obs.to(value.dtype).reshape(slot.shape)
        return value

    @property
    def param_sizes(self) -> Dict[str, int]:
        """Per-parameter output sizes for this variable's distribution.

        Maps each distribution-parameter name (e.g. ``"loc"``/``"scale"`` for
        ``Normal``, ``"probs"``/``"logits"`` for ``Bernoulli``) to the true
        number of scalar network outputs needed to produce it. Most equal
        :attr:`size` (one scalar per event element); the exceptions are encoded
        in the family's :class:`~.distributions.DistributionSpec` — e.g.
        ``MultivariateNormal``'s ``scale_tril`` needs ``size * (size + 1) // 2``
        lower-triangular Cholesky entries.

        Raises
        ------
        ValueError
            If the distribution family is not in the spec registry.
        """
        spec = spec_for(
            self.distribution, f"{type(self).__name__}({self.name!r})"
        )
        return {param: fn(self.size) for param, fn in spec.param_sizes.items()}

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


