"""
Annotated tensor for concept-based neural networks.

This module provides :class:`AnnotatedTensor`, a lightweight wrapper around a
:class:`torch.Tensor` that carries an :class:`~torch_concepts.AxisAnnotation`
for its second axis (axis 1), enabling label-based column slicing and
annotation-preserving tensor operations.
"""
from typing import Optional, Union, Dict

import torch

from torch_concepts.annotations import AxisAnnotation


class AnnotatedTensor:
    """
    A tensor annotated along its second axis (axis 1).

    Wraps a :class:`torch.Tensor` together with an :class:`AxisAnnotation`
    that describes the semantics of axis 1.  Supports:

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
        data: The underlying tensor. Must have at least 2 dimensions.
        annotation: Annotation for axis 1. ``annotation.shape`` must equal
            ``data.shape[1]``.

    Raises:
        ValueError: If ``data.dim() < 2`` or the annotation size does not
            match ``data.shape[1]``.

    Example:
        >>> import torch
        >>> from torch_concepts import AxisAnnotation
        >>> from torch_concepts.tensor import AnnotatedTensor
        >>>
        >>> ann = AxisAnnotation(labels=["cat", "dog", "bird"])
        >>> t = AnnotatedTensor(torch.rand(4, 3), ann)
        >>>
        >>> # Label-based slicing
        >>> sliced = t["cat", "dog"]
        >>> sliced.annotation.labels   # ['cat', 'dog']
        >>> sliced.tensor.shape        # torch.Size([4, 2])
        >>>
        >>> # Aggregation over axis 0 preserves the annotation
        >>> col_means = t.mean(dim=0)
        >>> col_means.annotation.labels  # ['cat', 'dog', 'bird']
        >>>
        >>> # torch module-level functions work too
        >>> summed = torch.sum(t, dim=0)
        >>> summed.annotation.labels  # ['cat', 'dog', 'bird']
    """

    def __init__(self, data: torch.Tensor, annotation: AxisAnnotation):
        if data.dim() < 2:
            raise ValueError(
                "AnnotatedTensor requires a tensor with at least 2 dimensions, "
                f"got ndim={data.dim()}."
            )
        ann_size = annotation.shape
        if data.shape[1] != ann_size:
            raise ValueError(
                f"Annotation size ({ann_size}) must match tensor axis-1 size "
                f"({data.shape[1]})."
            )
        # Use object.__setattr__ to bypass any future __setattr__ overrides
        object.__setattr__(self, '_data', data)
        object.__setattr__(self, '_annotation', annotation)

    # ------------------------------------------------------------------ #
    # Core properties                                                      #
    # ------------------------------------------------------------------ #

    @property
    def tensor(self) -> torch.Tensor:
        """The underlying :class:`torch.Tensor`."""
        return self._data

    @property
    def annotation(self) -> AxisAnnotation:
        """The :class:`AxisAnnotation` describing axis 1."""
        return self._annotation

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
        return AnnotatedTensor(self._data.to(*args, **kwargs), self._annotation)

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
            labels = list(key)
            indices = self._annotation.get_slice(labels)
            new_ann = self._annotation.subset(labels)
            return AnnotatedTensor(self._data[:, indices], new_ann)

        # Regular tensor indexing; re-wrap if axis 1 is unchanged
        return self._wrap(self._data[key])

    # ------------------------------------------------------------------ #
    # Merging                                                              #
    # ------------------------------------------------------------------ #

    def union_with(self, *others: 'AnnotatedTensor') -> 'AnnotatedTensor':
        """
        Concatenate this tensor with one or more ``AnnotatedTensor`` instances
        along axis 1 (the annotated axis), merging their annotations.

        All tensors must share the same shape on every axis **except** axis 1.
        The merged annotation is built by chaining
        :meth:`~torch_concepts.AxisAnnotation.union_with`: labels that already
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
            >>> ann_a = AxisAnnotation(labels=["cat", "dog"])
            >>> ann_b = AxisAnnotation(labels=["bird", "fish"])
            >>> a = AnnotatedTensor(torch.rand(4, 2), ann_a)
            >>> b = AnnotatedTensor(torch.rand(4, 2), ann_b)
            >>> merged = a.union_with(b)
            >>> merged.annotation.labels   # ['cat', 'dog', 'bird', 'fish']
            >>> merged.tensor.shape        # torch.Size([4, 4])
        """
        all_tensors = [self, *others]

        for i, other in enumerate(others):
            if not isinstance(other, AnnotatedTensor):
                raise TypeError(
                    f"union_with expects AnnotatedTensor arguments, "
                    f"got {type(other).__name__} at position {i + 1}."
                )
            self_rest = list(self._data.shape[:1]) + list(self._data.shape[2:])
            other_rest = list(other._data.shape[:1]) + list(other._data.shape[2:])
            if self_rest != other_rest:
                raise ValueError(
                    f"Shape mismatch at position {i + 1}: expected "
                    f"non-axis-1 shape {list(self._data.shape[:1]) + ['*'] + list(self._data.shape[2:])}, "
                    f"got {list(other._data.shape)}."
                )

        seen = set(self._annotation.labels)
        pieces = [self._data]
        merged_ann = self._annotation
        for other in others:
            new_labels = [l for l in other._annotation.labels if l not in seen]
            seen.update(new_labels)
            if new_labels:
                pieces.append(other._data[:, other._annotation.get_slice(new_labels)])
            merged_ann = merged_ann.union_with(other._annotation)

        return AnnotatedTensor(torch.cat(pieces, dim=1), merged_ann)

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
        correspondingly subsetted :class:`AxisAnnotation`.

        Example:
            >>> ann = AxisAnnotation(labels=['a', 'b', 'c'], cardinalities=[1, 3, 1])
            >>> t = AnnotatedTensor(torch.rand(4, 5), ann)
            >>> d = t.split_by_type()
            >>> d['binary'].annotation.labels       # ['a', 'c']
            >>> d['categorical'].annotation.labels  # ['b']
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
        """Re-wrap *result* when it is a tensor whose axis-1 size is unchanged."""
        if (
            isinstance(result, torch.Tensor)
            and result.dim() >= 2
            and result.shape[1] == self._data.shape[1]
        ):
            return AnnotatedTensor(result, self._annotation)
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
        return repr(self._data) + f"\n# annotations(axis=1): {self._annotation.labels}"

    def __len__(self) -> int:
        return self._data.shape[0]
