"""
Abstract base class ``Variable`` and its concrete 
subclasses ``ConceptVariable`` and ``EmbeddingVariable``, 
which represent random variables in a Probabilistic Graphical Model.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributions as dist

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

        # A variable IS a plate: ``members`` names them, and by default it has
        # exactly one, coinciding with the variable's own name. There is no
        # second kind of variable, so there is no second branch below.
        if members is None:
            members = [self.name]
        else:
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

        # The event of ONE member: ``size`` gives its width, ``shape`` its full
        # (possibly multi-dimensional) event. Together with ``members`` this is
        # the variable's entire structural state — ``shape``, ``size`` and
        # ``member_size`` are all derived from it.
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
            member_shape = torch.Size([size])
        elif shape is None:
            member_shape = torch.Size([1])
        else:
            member_shape = torch.Size(
                [shape] if isinstance(shape, int) else shape
            )
            if len(member_shape) == 0:
                raise ValueError("shape must be non-empty.")
            if any(d <= 0 for d in member_shape):
                raise ValueError(
                    f"{type(self).__name__}({names!r}): all shape dimensions must be "
                    f"positive, got {tuple(member_shape)}."
                )

        self.members: List[str] = list(members)
        self._member_shape: torch.Size = member_shape

        # A plate sizes one parametrization for all k members at once, from the
        # variable's total event size — which splits per member only when each
        # parameter is one scalar per event element. MultivariateNormal's
        # triangular scale_tril is not, so its members get no shared head.
        if len(self.members) > 1 and not spec.is_per_element:
            raise ValueError(
                f"{type(self).__name__}({names!r}): plate `members` need a distribution "
                f"with per-element parameters; {distribution.__name__} has a "
                "non-per-element parameter (e.g. MultivariateNormal's scale_tril). "
                "Model these members as separate variables instead."
            )

        self.distribution = distribution
        self.dist_kwargs: dict = dict(dist_kwargs) if dist_kwargs else {}
        self.metadata: dict = {
            "variable_type": self.variable_type,
        }
        # Set on a member view returned by ``member()``; points back to the plate.
        self._plate: Optional["Variable"] = None

    @property
    def is_plate(self) -> bool:
        """Whether the members were named explicitly rather than defaulted.

        Note this is **not** ``n_members > 1``: a one-member plate is a real
        thing (the high level builds one for a lone concept), and its member
        carries its own name, distinct from the variable's.
        """
        return self.members != [self.name]

    @property
    def plate(self) -> "Variable":
        """The plate this variable belongs to.

        For a member handle (from :meth:`member`) this is the owning plate; an
        ordinary variable and a plate are their own, so ``v.plate is v`` is the
        test for "not a member handle".
        """
        return self._plate if self._plate is not None else self

    @property
    def member_shape(self) -> torch.Size:
        """Event shape of a *single* member.

        ``(member_size,)`` for a plate; the variable's own :attr:`shape` for an
        ordinary variable, which is its own single member. Together with
        :attr:`n_members` this fully describes the canonical member layout
        ``(*leading, n_members, *member_shape)``.
        """
        return self._member_shape

    @property
    def n_members(self) -> int:
        """Number of named members: ``k`` for a plate, ``1`` otherwise."""
        return len(self.members)

    @property
    def member_axis(self) -> int:
        """Negative index of the member axis in the canonical member layout.

        ``-1`` when a member is a scalar event, ``-2`` for the usual
        ``(*leading, k, m)`` plate, and further left for a multi-dimensional
        member event. Negative so it is independent of how many leading
        (batch-like) dimensions a caller uses.
        """
        return - 1 - len(self._member_shape)

    def param_trailing_shape(self, param: Optional[str] = None) -> Tuple[int, ...]:
        """Trailing shape of one tensor in the canonical member layout.

        ``(n_members, *member_shape)`` for a realisation or an ordinary
        parameter. A parameter declaring ``param_event_ndims`` carries extra
        rank on top of the member event — ``MultivariateNormal``'s
        ``scale_tril`` is an ``(n, n)`` matrix where the member event is
        ``(n,)`` — so those axes are appended.
        """
        extra = 0
        if param is not None:
            context = f"{type(self).__name__}({self.name!r})"
            spec = spec_for(self.distribution, context)
            # Check the parameter is valid for this distribution.
            spec.check_param(param, self.distribution, context)
            extra = spec.param_event_ndims.get(param, 0)
        member = tuple(self._member_shape)
        return (self.n_members, *member, *member[len(member) - extra:])

    def _fit(self, tensor: torch.Tensor, trailing: Tuple[int, ...]) -> torch.Tensor:
        """Reshape ``tensor`` so its last axes are exactly ``trailing``.

        Examples:
        (8, 3, 4)   -> (8, 3, 4)      case 1: no change
        (8, 12)     -> (8, 3, 4)      case 3: reshape of the last axis
        (12,)       -> (3, 4)         case 3: no leading dim, reshape of the last axis
        (2, 5, 12)  -> (2, 5, 3, 4)   case 3: arbitrary leading dims, reshape of the last axis
        (8, 7)      -> ValueError: cannot read a tensor of shape (8, 7) as (*leading, 3, 4)
        """
        n = len(trailing)
        # Already fitted. Strict ``>`` keeps a non-empty leading shape when both
        # readings are possible, which is the convention every engine expects.
        if tensor.dim() > n and tuple(tensor.shape[-n:]) == trailing:
            return tensor
        # The event is ranked; only the member axis is missing.
        if trailing[0] == 1 and tensor.dim() >= n - 1 and tuple(tensor.shape[-(n - 1):]) == trailing[1:]:
            return tensor.unsqueeze(tensor.dim() - (n - 1))
        # Flat: peel the shortest non-empty suffix that makes up one event.
        target = math.prod(trailing)
        seen, split = 1, tensor.dim()
        while split > 0:
            split -= 1
            seen *= tensor.shape[split]
            if seen >= target:
                break
        if seen != target and target == 1:
            split, seen = tensor.dim(), 1  # a width-1 event squeezed off
        if seen != target:
            raise ValueError(
                f"{type(self).__name__}({self.name!r}): cannot read a tensor of shape "
                f"{tuple(tensor.shape)} as (*leading, {', '.join(map(str, trailing))})."
            )
        return tensor.reshape(*tensor.shape[:split], *trailing)

    def _require_member_layout(
        self, tensor: torch.Tensor, param: Optional[str] = None
    ) -> None:
        """Raise unless ``tensor`` is already in the member layout.

        For helpers that treat the last axis as *one member's* event: handed a
        flat ``(*leading, size)`` row instead, they would silently mix members
        (a categorical plate's softmax taken over every member's classes) or
        address the wrong axis. Unlike :meth:`to_member`, nothing is inferred —
        the caller converts explicitly.
        """
        trailing = self.param_trailing_shape(param)
        n = len(trailing)
        if tensor.dim() >= n and tuple(tensor.shape[-n:]) == trailing:
            return
        what = "a value" if param is None else f"parameter {param!r}"
        read = "to_member(tensor)" if param is None else f"to_member(tensor, {param!r})"
        raise ValueError(
            f"{type(self).__name__}({self.name!r}): expected {what} in member layout "
            f"(*leading, {', '.join(map(str, trailing))}), got shape "
            f"{tuple(tensor.shape)}. Read a flat (*leading, size) tensor with "
            f"`variable.{read}` first."
        )

    def to_member(self, tensor: torch.Tensor, param: Optional[str] = None) -> torch.Tensor:
        """Read ``tensor`` into the canonical ``(*leading, n_members, *member_shape)`` layout."""
        return self._fit(tensor, self.param_trailing_shape(param))

    def to_event(
        self, tensor: torch.Tensor, param: Optional[str] = None
    ) -> torch.Tensor:
        """Inverse of :meth:`to_member`: fold the member axis back into the event.

        PLATE c: 3 members x 4 states,  shape=(12,)
        (8, 3, 4)      -> (8, 12)
        (3, 4)         -> (12,)          empty leading
        (2, 5, 3, 4)   -> (2, 5, 12)     arbitrary leading
        (8, 12)        -> (8, 12)        already folded: no-op
        """
        trailing = self.param_trailing_shape(param)
        axis = tensor.dim() - len(trailing)
        if axis < 0 or tuple(tensor.shape[axis:]) != tuple(trailing):
            return tensor  # already folded: idempotent, like ``to_member``
        return tensor.flatten(axis, axis + 1)

    def leading_of(
        self, tensor: torch.Tensor, member: Optional[str] = None
    ) -> torch.Size:
        """Return the leading shape of ``tensor``."""
        trailing = (1 if member is not None else self.n_members, *self._member_shape)
        fitted = self._fit(tensor, trailing)
        return fitted.shape[: fitted.dim() - len(trailing)]
    
    def as_event(
        self, tensor: torch.Tensor, param: Optional[str] = None
    ) -> torch.Tensor:
        """Read ``tensor`` in whatever layout it has and return it in event layout."""
        return self.to_event(self.to_member(tensor, param), param)

    def to_flat(self, tensor: torch.Tensor) -> torch.Tensor:
        """A realisation as one flat row: ``(*leading, size)``.

        The layout realisations are reported in, so that a variable with a
        multi-dimensional event sits on the same annotated axis as every other.
        :meth:`to_event` keeps the event's own rank instead, which is what a
        parametrization module and a reported *parameter* want.
        """
        leading = tensor.shape[: tensor.dim() + self.member_axis]
        return tensor.reshape(*leading, self.size)

    def index_of(self, member: str) -> int:
        """Position of ``member`` along the member axis."""
        try:
            return self.members.index(member)
        except ValueError:
            raise KeyError(
                f"{type(self).__name__}({self.name!r}) has no member {member!r}; "
                f"members are {self.members}."
            ) from None

    def flat_columns(self, members: Union[str, List[str]]) -> List[int]:
        """Column indices of ``members`` in the *flat* ``(*leading, size)`` event.

        The flat layout is what leaves the mid level — the annotated output axis,
        and the ``[B, F]` tensors the low-level intervention modules index — so
        this is a boundary helper rather than the way members are addressed
        internally (that is ``tensor.select(member_axis, index_of(name))``).
        """
        if isinstance(members, str):
            members = [members]
        width = self.member_size
        return [
            i * width + offset
            for i in map(self.index_of, members)
            for offset in range(width)
        ]

    def member(self, name: str) -> "Variable":
        """A handle to a single member.

        The handle carries the member's name, per-member size and the plate's distribution, 
        plus a back-reference to the owning plate so the graph routes the edge from it.
        """
        self.index_of(name)  # rejects a name that is not one of ours
        view = type(self)(
            name,
            distribution=self.distribution,
            size=self.member_size,
            dist_kwargs=copy.deepcopy(self.dist_kwargs),
        )
        # Set the back-reference to the owning plate.
        view._plate = self
        return view

    @property
    def shape(self) -> torch.Size:
        """Event shape of the whole variable: the members stacked on one axis."""
        first, *rest = self._member_shape
        return torch.Size([self.n_members * first, *rest])

    @property
    def member_size(self) -> int:
        """Number of scalar elements in *one* member's event."""
        return math.prod(self._member_shape)

    @property
    def size(self) -> int:
        """Total number of scalar elements: ``math.prod(self.shape)``."""
        return self.n_members * self.member_size

    def member_of(
        self, tensor: torch.Tensor, member: str, param: Optional[str] = None
    ) -> torch.Tensor:
        """One member's slice of ``tensor``, in event layout — a view, no copy.

        ``tensor`` may be in either layout; it is read into the member layout
        first, so a caller holding a flat row and one holding a cached value
        get the same answer.
        """
        trailing = self.param_trailing_shape(param)
        fitted = self._fit(tensor, trailing)
        return fitted.select(fitted.dim() - len(trailing), self.index_of(member))

    def clamp_members(
        self, value: torch.Tensor, observed: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Overwrite observed members with their observed values.

        Evidence may cover only some of a variable's members, so this splices
        the observed ones into a value the model produced for all of them. Both
        sides are in member layout, so it is one masked write over the member
        axis rather than a per-member column assignment.

        The result is a fresh tensor: the caller caches ``value`` and reuses it
        as a parent input downstream, so writing in place would corrupt that
        cache and break autograd on the tensor the CPD produced.

        Raises
        ------
        ValueError
            If ``value`` is not in member layout. A flat ``(*leading, size)``
            row would put the member axis on a batch axis and overwrite rows
            instead of members.
        """
        self._require_member_layout(value)
        if not observed:
            return value
        stacked = value.clone()
        axis = value.dim() + self.member_axis
        for member, obs in observed.items():
            index = self.index_of(member)
            slot = stacked.select(axis, index)
            slot.copy_(obs.to(value.dtype).reshape(slot.shape))
        return stacked

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


