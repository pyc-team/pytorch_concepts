"""
Annotated tensor for concept-based neural networks.

This module provides :class:`AnnotatedTensor`, a lightweight wrapper around a
:class:`torch.Tensor` that carries an :class:`~torch_concepts.Annotations`
for one of its axes (axis 1 by default), enabling label-based column slicing
and annotation-preserving tensor operations.
"""
from typing import Optional, Union, Dict

import torch

from torch_concepts.annotations import Annotations


def _as_contiguous_slice(indices):
    """Turn an ascending run of consecutive indices into an equivalent ``slice``.

    Advanced (list) indexing always copies, while a ``slice`` returns a view. The
    labels a caller selects are usually one concept, or several adjacent ones, so
    this makes the common label lookup free — which matters when the annotated
    tensor is the single storage for a whole inference result and per-variable
    access is expected to be a view into it.
    """
    if not isinstance(indices, list) or not indices:
        return indices
    start, stop = indices[0], indices[0] + len(indices)
    return slice(start, stop) if indices == list(range(start, stop)) else indices


class AnnotatedTensor:
    """
    A tensor annotated along one axis (axis 1 by default).

    Wraps a :class:`torch.Tensor` together with an :class:`Annotations`
    that describes the semantics of the annotated axis.  Supports:

    * **Label-based slicing** — select columns by concept name::

        sliced = t["cat", "dog"]   # keeps only 'cat' and 'dog' columns
        sliced = t[["cat", "dog"]] # same via list syntax

    * **Annotation-preserving operations** — any tensor operation that leaves
      the size of axis 1 unchanged automatically returns a new
      ``AnnotatedTensor`` carrying the same (or a subset) annotation::

        t.sum(dim=0)    # aggregation over batch → still annotated on axis 1
        t.mean(dim=-1)  # aggregation over last axis → still annotated on axis 1
        t.reshape(8, 3, -1)  # reshape that keeps axis-1 size → still annotated

    * **Transparent tensor proxy** — all tensor attributes and methods not
      defined on this class (``shape``, ``dtype``, ``.detach()``, ``.to()``,
      …) are forwarded to the underlying tensor via ``__getattr__``.

    * **``torch.*`` function protocol** — module-level functions such as
      ``torch.sum(t, dim=0)`` also propagate the annotation when axis 1 is
      unchanged.

    Args:
        data: The underlying tensor. Must have at least 2 dimensions when
            ``axis == 1``, and at least 1 when annotating the last axis.
        annotation: Annotation for the annotated axis. ``annotation.size``
            must equal ``data.shape[axis]``.
        axis: Which axis the annotation describes. ``1`` (the default) is the
            classic concept axis of a ``(batch, concepts)`` tensor. ``-1``
            annotates the **last** axis, which is what the mid level uses so a
            tensor may carry any number of leading (batch) dimensions
            ``(*leading, concepts)``. For a 2-D tensor the two are the same
            axis, so ``-1`` is a strict generalisation.

    Raises:
        ValueError: If ``data`` has too few dimensions, ``axis`` is neither
            ``1`` nor ``-1``, or the annotation size does not match
            ``data.shape[axis]``.

    Example:
        >>> import torch
        >>> from torch_concepts import Annotations
        >>> from torch_concepts.tensor import AnnotatedTensor
        >>>
        >>> ann = Annotations(labels=["cat", "dog", "bird"])
        >>> t = AnnotatedTensor(torch.rand(4, 3), ann)
        >>>
        >>> # Label-based slicing
        >>> sliced = t["cat", "dog"]
        >>> sliced.annotation.labels
        ['cat', 'dog']
        >>> sliced.tensor.shape
        torch.Size([4, 2])
    """

    def __init__(self, data: torch.Tensor, annotation: Annotations, axis: int = 1):
        if axis not in (1, -1):
            raise ValueError(
                f"AnnotatedTensor: `axis` must be 1 or -1, got {axis!r}."
            )
        min_dim = 2 if axis == 1 else 1
        if data.dim() < min_dim:
            raise ValueError(
                f"AnnotatedTensor with axis={axis} requires a tensor with at least "
                f"{min_dim} dimensions, got ndim={data.dim()}."
            )
        ann_size = annotation.size
        if data.shape[axis] != ann_size:
            raise ValueError(
                f"Annotation size ({ann_size}) must match tensor axis-{axis} size "
                f"({data.shape[axis]})."
            )
        # Use object.__setattr__ to bypass any future __setattr__ overrides
        object.__setattr__(self, '_data', data)
        object.__setattr__(self, '_annotation', annotation)
        object.__setattr__(self, '_axis', axis)

    # ------------------------------------------------------------------ #
    # Core properties                                                      #
    # ------------------------------------------------------------------ #

    @property
    def tensor(self) -> torch.Tensor:
        """The underlying :class:`torch.Tensor`."""
        return self._data

    @property
    def annotation(self) -> Annotations:
        """The :class:`Annotations` describing the annotated axis."""
        return self._annotation

    @property
    def axis(self) -> int:
        """Which axis the annotation describes (``1`` or ``-1``)."""
        return self._axis

    def _index(self, indices) -> tuple:
        """Index tuple selecting ``indices`` along the annotated axis."""
        if self._axis == -1:
            return (Ellipsis, indices)
        return (slice(None), indices)

    @property
    def device(self) -> torch.device:
        """Device of the underlying tensor.

        Defined on the class (not just proxied via ``__getattr__``) so that
        frameworks which detect movable batch elements by looking for ``to`` /
        ``device`` on the type (e.g. PyTorch Lightning's ``TransferableDataType``)
        recognise an :class:`AnnotatedTensor` and move it with the rest of the batch.
        """
        return self._data.device

    def to(self, *args, **kwargs) -> 'AnnotatedTensor':
        """Move/cast the underlying tensor, preserving the annotation.

        Mirrors :meth:`torch.Tensor.to` and returns a new ``AnnotatedTensor``
        wrapping the moved/cast data. Defined on the class so batch-transfer
        machinery (e.g. Lightning) treats this as a transferable element.
        """
        return AnnotatedTensor(self._data.to(*args, **kwargs), self._annotation, self._axis)

    # ------------------------------------------------------------------ #
    # Label-based slicing                                                  #
    # ------------------------------------------------------------------ #

    def __getitem__(self, key):
        """
        Select columns by label name or fall back to regular tensor indexing.

        Label-based access (all-string key):
            ``t["cat"]``            – single column  → ``AnnotatedTensor``
            ``t["cat", "dog"]``     – multiple cols  → ``AnnotatedTensor``
            ``t[["cat", "dog"]]``   – list syntax    → ``AnnotatedTensor``

        Any other key is forwarded to the underlying tensor; the annotation is
        preserved when the result's axis-1 size equals the original.
        """
        # Normalise to tuple-of-strings when possible
        if isinstance(key, str):
            key = (key,)
        elif isinstance(key, list) and key and all(isinstance(k, str) for k in key):
            key = tuple(key)

        if isinstance(key, tuple) and key and all(isinstance(k, str) for k in key):
            labels = self._resolve_labels(key)
            indices = _as_contiguous_slice(self._annotation.get_slice(labels))
            new_ann = self._annotation.subset(labels)
            return AnnotatedTensor(
                self._data[self._index(indices)], new_ann, self._axis
            )

        # Regular tensor indexing; re-wrap if axis 1 is unchanged
        return self._wrap(self._data[key])

    #: Metadata key naming the group a label belongs to. A key that is not
    #: itself a label is looked up here, so a producer that splits one source
    #: into several labels (the mid level expanding a plate into its members)
    #: keeps the source's name addressable.
    GROUP_KEY = 'variable'

    def _resolve_labels(self, keys) -> list:
        """Expand each key to labels: itself, or the group of labels it names.

        Keys are resolved left to right and concatenated, so
        ``t['g', 'other']`` returns ``g``'s whole group followed by ``other``.
        A key that is neither a label nor a group is passed through unchanged so
        the annotation raises its own "label not found" error.
        """
        labels: list = []
        groups = None
        known = self._annotation.label_to_index
        for key in keys:
            if key in known:
                labels.append(key)
                continue
            if groups is None:
                groups = self._annotation.groupby_metadata(self.GROUP_KEY)
            labels.extend(groups.get(key, [key]))
        return labels

    # ------------------------------------------------------------------ #
    # Merging                                                              #
    # ------------------------------------------------------------------ #

    def union_with(self, *others: 'AnnotatedTensor') -> 'AnnotatedTensor':
        """
        Concatenate this tensor with one or more ``AnnotatedTensor`` instances
        along axis 1 (the annotated axis), merging their annotations.

        All tensors must share the same shape on every axis **except** axis 1.
        The merged annotation is built by chaining
        :meth:`~torch_concepts.Annotations.union_with`: labels that already
        appear on the left are not duplicated; metadata is merged with
        left-wins semantics.

        Args:
            *others: One or more :class:`AnnotatedTensor` instances to merge in.

        Returns:
            A new :class:`AnnotatedTensor` whose underlying tensor is the
            concatenation along axis 1 and whose annotation is the union of
            all input annotations.

        Raises:
            TypeError:  If any element of *others* is not an
                        :class:`AnnotatedTensor`.
            ValueError: If any tensor's non-axis-1 shape differs from this
                        tensor's non-axis-1 shape.

        Example:
            >>> ann_a = Annotations(labels=["cat", "dog"])
            >>> ann_b = Annotations(labels=["bird", "fish"])
            >>> a = AnnotatedTensor(torch.rand(4, 2), ann_a)
            >>> b = AnnotatedTensor(torch.rand(4, 2), ann_b)
            >>> merged = a.union_with(b)
            >>> merged.annotation.labels
            ['cat', 'dog', 'bird', 'fish']
            >>> merged.tensor.shape
            torch.Size([4, 4])
        """
        ax = self._axis

        def _rest(shape):
            """Shape with the annotated axis removed."""
            dims = list(shape)
            del dims[ax]
            return dims

        for i, other in enumerate(others):
            if not isinstance(other, AnnotatedTensor):
                raise TypeError(
                    f"union_with expects AnnotatedTensor arguments, "
                    f"got {type(other).__name__} at position {i + 1}."
                )
            if other._axis != ax:
                raise ValueError(
                    f"union_with: annotated axis mismatch at position {i + 1}: "
                    f"this tensor annotates axis {ax}, the other annotates {other._axis}."
                )
            if _rest(self._data.shape) != _rest(other._data.shape):
                expected = list(self._data.shape)
                expected[ax] = '*'
                raise ValueError(
                    f"Shape mismatch at position {i + 1}: expected "
                    f"non-axis-{ax} shape {expected}, got {list(other._data.shape)}."
                )

        seen = set(self._annotation.labels)
        pieces = [self._data]
        merged_ann = self._annotation
        for other in others:
            new_labels = [l for l in other._annotation.labels if l not in seen]
            seen.update(new_labels)
            if new_labels:
                indices = _as_contiguous_slice(
                    other._annotation.get_slice(new_labels)
                )
                pieces.append(other._data[other._index(indices)])
            merged_ann = merged_ann.union_with(other._annotation)

        return AnnotatedTensor(torch.cat(pieces, dim=ax), merged_ann, ax)

    # ------------------------------------------------------------------ #
    # Type-based splitting                                                 #
    # ------------------------------------------------------------------ #

    def split_by_type(
        self, 
        concept_type: Optional[str] = None
    ) -> Union['AnnotatedTensor', Dict[str, 'AnnotatedTensor']]:
        """
        If ``concept_type`` is given, return the sub-tensor of concepts of ``concept_type``.
        
        If ``concept_type`` is ``None``, split this tensor into a 
        dictionary of :class:`AnnotatedTensor` instances, one per concept type present.
        The keys are the type strings ``'binary'`` / ``'categorical'`` /
        ``'continuous'`` (only the non-empty ones); each value is an
        :class:`AnnotatedTensor` containing only the columns of that type, with a
        correspondingly subsetted :class:`Annotations`.

        Example:
            >>> ann = Annotations(labels=['a', 'b', 'c'], cardinalities=[1, 3, 1])
            >>> t = AnnotatedTensor(torch.rand(4, 5), ann)
            >>> d = t.split_by_type()
            >>> d['binary'].annotation.labels
            ['a', 'c']
            >>> d['categorical'].annotation.labels
            ['b']
        """
        if concept_type is None:
            return {
                t: self[labels]
                for t, labels in self._annotation.labels_by_type.items()
            }
        return self[self._annotation.labels_by_type.get(concept_type, [])]

    # ------------------------------------------------------------------ #
    # torch.* function protocol                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Intercept ``torch.*`` calls and strip annotations.

        When an ``AnnotatedTensor`` is passed to any ``torch.*`` function or
        ``nn.Module``, the annotation is silently dropped and the result is a
        plain :class:`torch.Tensor`.  This prevents surprising behaviour inside
        module ``forward`` methods and avoids conflicts with operations that
        change the meaning of axis 1.

        Annotation-preserving behaviour is still available through direct
        method calls on the instance (e.g. ``t.sum(dim=0, keepdim=True)``).
        """
        if kwargs is None:
            kwargs = {}

        def _unwrap(a):
            if isinstance(a, AnnotatedTensor):
                return a._data
            if isinstance(a, (list, tuple)):
                unwrapped = [_unwrap(x) for x in a]
                return type(a)(unwrapped)
            return a

        new_args = tuple(_unwrap(a) for a in args)
        new_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}

        return func(*new_args, **new_kwargs)

    # ------------------------------------------------------------------ #
    # Transparent tensor proxy                                             #
    # ------------------------------------------------------------------ #

    def __getattr__(self, name: str):
        """
        Forward attribute lookups not found on this class to the underlying
        tensor. Callable attributes are wrapped so that any result that still
        has the same axis-1 size is returned as an :class:`AnnotatedTensor`.
        """
        attr = getattr(self._data, name)
        if callable(attr):
            def _wrapper(*args, **kwargs):
                return self._wrap(attr(*args, **kwargs))
            return _wrapper
        return attr

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _wrap(self, result) -> Union['AnnotatedTensor', torch.Tensor]:
        """Re-wrap *result* when it is a tensor whose annotated axis is unchanged."""
        min_dim = 2 if self._axis == 1 else 1
        if (
            isinstance(result, torch.Tensor)
            and result.dim() >= min_dim
            and result.shape[self._axis] == self._data.shape[self._axis]
        ):
            return AnnotatedTensor(result, self._annotation, self._axis)
        return result

    # ------------------------------------------------------------------ #
    # Standard dunder helpers                                              #
    # ------------------------------------------------------------------ #

    # All dunder methods below are looked up directly on the type by Python,
    # so __getattr__ is never triggered for them. Every tensor-returning dunder
    # goes through _wrap so the annotation is preserved when shape allows it.

    # --- Comparison ---
    def __gt__(self, other): return self._wrap(self._data.__gt__(other))
    def __lt__(self, other): return self._wrap(self._data.__lt__(other))
    def __ge__(self, other): return self._wrap(self._data.__ge__(other))
    def __le__(self, other): return self._wrap(self._data.__le__(other))
    def __eq__(self, other): return self._wrap(self._data.__eq__(other))
    def __ne__(self, other): return self._wrap(self._data.__ne__(other))

    # --- Arithmetic (binary) ---
    def __add__(self, other):       return self._wrap(self._data.__add__(other))
    def __radd__(self, other):      return self._wrap(self._data.__radd__(other))
    def __sub__(self, other):       return self._wrap(self._data.__sub__(other))
    def __rsub__(self, other):      return self._wrap(self._data.__rsub__(other))
    def __mul__(self, other):       return self._wrap(self._data.__mul__(other))
    def __rmul__(self, other):      return self._wrap(self._data.__rmul__(other))
    def __truediv__(self, other):   return self._wrap(self._data.__truediv__(other))
    def __rtruediv__(self, other):  return self._wrap(self._data.__rtruediv__(other))
    def __floordiv__(self, other):  return self._wrap(self._data.__floordiv__(other))
    def __rfloordiv__(self, other): return self._wrap(self._data.__rfloordiv__(other))
    def __mod__(self, other):       return self._wrap(self._data.__mod__(other))
    def __rmod__(self, other):      return self._wrap(self._data.__rmod__(other))
    def __pow__(self, other):       return self._wrap(self._data.__pow__(other))
    def __rpow__(self, other):      return self._wrap(self._data.__rpow__(other))
    def __matmul__(self, other):    return self._wrap(self._data.__matmul__(other))
    def __rmatmul__(self, other):   return self._wrap(self._data.__rmatmul__(other))

    # --- Unary ---
    def __neg__(self):    return self._wrap(self._data.__neg__())
    def __pos__(self):    return self._wrap(self._data.__pos__())
    def __abs__(self):    return self._wrap(self._data.__abs__())
    def __invert__(self): return self._wrap(self._data.__invert__())

    # --- Bitwise ---
    def __and__(self, other):  return self._wrap(self._data.__and__(other))
    def __rand__(self, other): return self._wrap(self._data.__rand__(other))
    def __or__(self, other):   return self._wrap(self._data.__or__(other))
    def __ror__(self, other):  return self._wrap(self._data.__ror__(other))
    def __xor__(self, other):  return self._wrap(self._data.__xor__(other))
    def __rxor__(self, other): return self._wrap(self._data.__rxor__(other))

    # --- In-place arithmetic (mutate _data, return self to keep annotation) ---
    def __iadd__(self, other):       self._data.__iadd__(other);       return self
    def __isub__(self, other):       self._data.__isub__(other);       return self
    def __imul__(self, other):       self._data.__imul__(other);       return self
    def __itruediv__(self, other):   self._data.__itruediv__(other);   return self
    def __ifloordiv__(self, other):  self._data.__ifloordiv__(other);  return self
    def __imod__(self, other):       self._data.__imod__(other);       return self
    def __ipow__(self, other):       self._data.__ipow__(other);       return self
    def __iand__(self, other):       self._data.__iand__(other);       return self
    def __ior__(self, other):        self._data.__ior__(other);        return self
    def __ixor__(self, other):       self._data.__ixor__(other);       return self

    # --- Scalar conversion (return plain Python scalar, no wrapping) ---
    def __bool__(self):  return self._data.__bool__()
    def __int__(self):   return self._data.__int__()
    def __float__(self): return self._data.__float__()
    def __index__(self): return self._data.__index__()

    def __repr__(self) -> str:
        return (
            repr(self._data)
            + f"\n# annotations(axis={self._axis}): {self._annotation.labels}"
        )

    def __len__(self) -> int:
        return self._data.shape[0]

    def __contains__(self, key) -> bool:
        """``'cat' in t`` tests for a *label*; anything else tests tensor values.

        Label membership is the question callers actually ask of an annotated
        tensor, and it mirrors ``__getitem__``, which also treats a string key as
        a label.
        """
        if isinstance(key, str):
            return (
                key in self._annotation.label_to_index
                or key in self._annotation.groupby_metadata(self.GROUP_KEY)
            )
        return key in self._data
