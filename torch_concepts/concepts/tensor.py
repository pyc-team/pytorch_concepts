import numpy as np
import torch

import pandas as pd
from typing import List, Tuple, Union, Optional, Set

from torch.nested._internal.nested_tensor import NestedTensor

from torch import Tensor
from pandas import DataFrame
import networkx as nx
import torch_geometric as pyg


from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.concepts.utils import _check_tensors


class AnnotatedTensor(torch.Tensor):
    """
    AnnotatedTensor is a subclass of torch.Tensor with semantic annotations.

    Attributes:
        data (torch.Tensor): Data tensor.
        annotations (Annotations): Annotations object containing semantic labels
            for each annotated dimension.
    """

    def __new__(
            cls,
            data: Union[torch.Tensor, NestedTensor, List[torch.Tensor]],
            annotations: Annotations = None,
            *args,
            **kwargs,
    ) -> 'AnnotatedTensor':
        # detect type and eventually convert
        if isinstance(data, list):
            _check_tensors(data)
            dtype = data[0].dtype
            device = data[0].device
            data = torch.nested.nested_tensor(data, dtype=dtype, device=device)

        instance = torch.Tensor._make_subclass(cls, data)

        if data.is_nested:
            instance.B = data[0].shape[0]
            instance.C = len(data.unbind())  # Number of constituent tensors
            instance.trailing_shape = data[0].shape[2:]
        else:
            instance.B = data.shape[0]
            instance.C = data.shape[1]
            instance.trailing_shape = data.shape[2:]

        annotations = cls._maybe_auto_annotate(instance, annotations)
        instance.annotations = cls._check_annotations(instance, annotations)

        # Preserve requires_grad from the input data without calling requires_grad_()
        # Direct assignment keeps the tensor as a non-leaf, allowing gradient flow
        instance.requires_grad = data.requires_grad

        return instance

    # --------- Basic props ----------
    @property
    def shape(self) -> Tuple[int, int, *Tuple[int, ...]]:
        """Logical shape: (B, [c1, c2, c3, ...], *trailing_shape)."""
        if self.data.is_nested:
            sizes = [t.shape[1] for t in self.data.unbind()]
            return (self.B, sizes, *self.trailing_shape)
        else:
            return super().shape

    def size(self) -> Tuple[int, ...]:
        """Sizes of dim=1 for each field: (c_1, c_2, ..., c_C)."""
        return self.shape

    def __repr__(self) -> str:
        return self.data.__repr__()

    # TODO: add annotations

    @property
    def data(self) -> torch.Tensor:
        """Read-only access to the internal NestedTensor (outer=C, leaves (B, c_i, *rest))."""
        return self.as_subclass(torch.Tensor)

    def _unary_dispatch(self, torch_fn, inplace: bool = False) -> "AnnotatedTensor":
        """
        Dispatch unary operations, handling both regular and nested tensors.

        For nested tensors, tries the operation on the nested tensor directly,
        falling back to mapping over constituent tensors if not supported.
        For regular tensors, applies the operation normally.
        """
        if self.data.is_nested:
            try:
                out = torch_fn(self.data)
                return self._wrap_result(out)
            except Exception as e:
                msg = str(e)
                backend_missing = isinstance(e, (NotImplementedError, RuntimeError, TypeError)) and (
                        "NestedTensor" in msg or "backend" in msg or "NotImplemented" in msg
                )
                if not backend_missing:
                    raise
                if inplace:
                    raise RuntimeError("In-place ops not supported for nested tensors.")
                # Fallback: map over constituent tensors
                return AnnotatedTensor([torch_fn(t) for t in self.data.unbind()], annotations=self.annotations)
        else:
            # Regular tensor: apply operation normally
            out = torch_fn(self.data)
            return self._wrap_result(out)

    def _binary_dispatch(self, other, torch_fn, inplace: bool = False) -> "AnnotatedTensor":
        """
        Dispatch binary operations, handling both regular and nested tensors.

        For nested tensors, tries the operation on the nested tensor directly,
        falling back to mapping over constituent tensors if not supported.
        For regular tensors, applies the operation normally.
        """
        if self.data.is_nested:
            if isinstance(other, AnnotatedTensor):
                # Both are AnnotatedTensors
                try:
                    out = torch_fn(self.data, other.data)
                    return self._wrap_result(out)
                except Exception as e:
                    msg = str(e)
                    backend_missing = isinstance(e, (NotImplementedError, RuntimeError, TypeError)) and (
                            "NestedTensor" in msg or "backend" in msg or "NotImplemented" in msg
                    )
                    if not backend_missing:
                        raise
                    if inplace:
                        raise RuntimeError("In-place ops not supported for nested tensors.")
                    # Fallback: map over constituent tensors
                    return AnnotatedTensor(
                        [torch_fn(a, b) for a, b in zip(self.data.unbind(), other.data.unbind())],
                        annotations=self.annotations
                    )
            else:
                # self is AnnotatedTensor, other is scalar or regular tensor
                try:
                    out = torch_fn(self.data, other)
                    return self._wrap_result(out)
                except Exception as e:
                    msg = str(e)
                    backend_missing = isinstance(e, (NotImplementedError, RuntimeError, TypeError)) and (
                            "NestedTensor" in msg or "backend" in msg or "NotImplemented" in msg
                    )
                    if not backend_missing:
                        raise
                    if inplace:
                        raise RuntimeError("In-place ops not supported for nested tensors.")
                    # Fallback: map over constituent tensors
                    return AnnotatedTensor(
                        [torch_fn(a, other) for a in self.data.unbind()],
                        annotations=self.annotations
                    )
        else:
            # Regular tensor: apply operation normally
            if isinstance(other, AnnotatedTensor):
                out = torch_fn(self.data, other.data)
            else:
                out = torch_fn(self.data, other)
            return self._wrap_result(out)

    def _set_shape_attrs(self, tensor):
        """Set B, C, trailing_shape attributes based on tensor shape."""
        if tensor.is_nested:
            # Access underlying data directly to avoid recursion through __getitem__
            first_constituent = list(tensor.unbind())[0]
            tensor.B = first_constituent.shape[0]
            tensor.C = len(tensor.unbind())
            tensor.trailing_shape = first_constituent.shape[2:]
        else:
            tensor.B = tensor.shape[0] if tensor.ndim > 0 else 1
            tensor.C = tensor.shape[1] if tensor.ndim > 1 else 1
            tensor.trailing_shape = tuple(tensor.shape[2:]) if tensor.ndim > 2 else ()

    def _wrap_result(self, result, annotations=None):
        """
        Wrap a tensor result back into an AnnotatedTensor, preserving annotations.

        This method converts the result into an AnnotatedTensor subclass without calling __new__,
        which preserves the autograd graph and allows gradients to flow properly.

        Args:
            result: The tensor result to wrap.
            annotations: Optional annotations to use. If None, copies from self.
        """
        if isinstance(result, torch.Tensor):
            # If already an AnnotatedTensor with attributes, just return it
            if isinstance(result, AnnotatedTensor) and hasattr(result, 'annotations'):
                return result

            # Convert to AnnotatedTensor subclass without breaking autograd
            if not isinstance(result, AnnotatedTensor):
                wrapped = result.as_subclass(AnnotatedTensor)
            else:
                wrapped = result

            # Set shape attributes
            self._set_shape_attrs(wrapped)

            # Set annotations
            if annotations is not None:
                wrapped.annotations = annotations
            elif hasattr(self, 'annotations'):
                wrapped.annotations = self.annotations
            else:
                wrapped.annotations = AnnotatedTensor._maybe_auto_annotate(wrapped, None)

            return wrapped
        return result

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Handle torch function dispatch for both regular and nested tensors.

        For nested tensors, unwraps AnnotatedTensor to get the underlying data,
        calls the function, and wraps the result. Falls back to mapping over
        constituent tensors if the function is not supported for NestedTensor.

        For regular tensors, uses standard torch.Tensor behavior.
        """
        if kwargs is None:
            kwargs = {}

        # Check if any of the args are AnnotatedTensors with nested data
        has_nested = any(
            isinstance(arg, AnnotatedTensor) and arg.data.is_nested
            for arg in args
        )

        if has_nested:
            def unwrap(x):
                return x.data if isinstance(x, AnnotatedTensor) else x

            uargs = tuple(unwrap(a) for a in args)
            ukw = {k: unwrap(v) for k, v in kwargs.items()}

            # Find first AnnotatedTensor instance for wrapping logic
            first_at = next((a for a in args if isinstance(a, AnnotatedTensor)), None)

            try:
                out = func(*uargs, **ukw)
                if first_at is not None:
                    return first_at._wrap_result(out)
                return out
            except Exception as e:
                # Fallback: map over leaves for unary/binary elementwise ops
                msg = str(e)
                backend_missing = isinstance(e, (NotImplementedError, RuntimeError, TypeError)) and (
                        "NestedTensor" in msg or "backend" in msg or "NotImplemented" in msg
                )
                if not backend_missing:
                    raise

                # unary: func(self)
                if len(args) >= 1 and isinstance(args[0], AnnotatedTensor) and all(
                        not isinstance(a, AnnotatedTensor) for a in args[1:]
                ):
                    return AnnotatedTensor(
                        [func(t, *args[1:], **kwargs) for t in args[0].data.unbind()],
                        annotations=args[0].annotations
                    )

                # binary: func(self, other)
                if len(args) >= 2:
                    a0, a1 = args[0], args[1]
                    if isinstance(a0, AnnotatedTensor) and isinstance(a1, AnnotatedTensor):
                        return AnnotatedTensor(
                            [func(x, y) for x, y in zip(a0.data.unbind(), a1.data.unbind())],
                            annotations=a0.annotations
                        )
                    if isinstance(a0, AnnotatedTensor):
                        return AnnotatedTensor(
                            [func(x, a1) for x in a0.data.unbind()],
                            annotations=a0.annotations
                        )
                    if isinstance(a1, AnnotatedTensor):
                        return AnnotatedTensor(
                            [func(a0, y) for y in a1.data.unbind()],
                            annotations=a1.annotations
                        )

                raise

        else:
            # Regular tensor: use standard torch.Tensor behavior
            result = super().__torch_function__(func, types, args, kwargs)

            # Preserve annotations for element-wise operations
            if isinstance(result, torch.Tensor):
                first_at = next((a for a in args if isinstance(a, AnnotatedTensor)), None)
                if first_at is not None and hasattr(first_at, 'annotations'):
                    # Check if this is an element-wise operation (shape preserved)
                    if result.shape == first_at.shape:
                        # Wrap with annotations
                        return first_at._wrap_result(result)

            return result

    # ---------- Python operator overloads (thin wrappers) ----------
    def __add__(self, other):
        return self._binary_dispatch(other, torch.add)

    def __sub__(self, other):
        return self._binary_dispatch(other, torch.sub)

    def __mul__(self, other):
        return self._binary_dispatch(other, torch.mul)

    def __truediv__(self, other):
        return self._binary_dispatch(other, torch.div)

    def __pow__(self, other):
        return self._binary_dispatch(other, torch.pow)

    def __radd__(self, other):
        return self._binary_dispatch(other, torch.add)

    def __rsub__(self, other):
        return self._binary_dispatch(other, lambda a, b: torch.sub(b, a))

    def __rmul__(self, other):
        return self._binary_dispatch(other, torch.mul)

    def __rtruediv__(self, other):
        return self._binary_dispatch(other, lambda a, b: torch.div(b, a))

    def __rpow__(self, other):
        return self._binary_dispatch(other, torch.pow)

    def __neg__(self):
        return self._unary_dispatch(torch.neg)

    def __abs__(self):
        return self._unary_dispatch(torch.abs)

    @staticmethod
    def _maybe_auto_annotate(
            instance: Union[torch.Tensor, NestedTensor],
            annotations: Annotations = None,
    ) -> Annotations:
        """
        Automatically annotate the first dimension (axis 1) if not already annotated.

        Args:
            annotations: Existing Annotations object or None.
        Returns:
            Annotations object with axis 1 annotated as labels if normal annotations, or labels and states if nested.
        """
        if annotations is None:
            if instance.data.is_nested:
                cardinalities = tuple(instance.shape[1])  # sizes of dim=1 for each field
                annotations = Annotations({
                    1: AxisAnnotation(labels=tuple(f"concept_{i}" for i in range(instance.C)),
                                      cardinalities=cardinalities),
                })
            else:
                annotations = Annotations({
                    1: AxisAnnotation(labels=tuple(f"concept_{i}" for i in range(instance.C))),
                })
        return annotations

    @staticmethod
    def _check_annotations(
            instance: Union[torch.Tensor, NestedTensor],
            annotations: Annotations = None,
    ) -> Annotations:
        """
        Check and validate annotations for the tensor.

        Args:
            tensor: The tensor to annotate
            annotations: Annotations object or None

        Returns:
            Annotations object (possibly empty if None provided)
        """

        if not isinstance(annotations, Annotations):
            raise ValueError(
                f'Expected annotations to be an Annotations object. '
                f'Instead, we were given {type(annotations)}.'
            )

        # Validate that all annotated axes are within tensor dimensions
        for axis in annotations.annotated_axes:
            if axis < 0 or axis >= len(instance.shape):
                raise ValueError(
                    f"Annotation axis {axis} is out of range for "
                    f"tensor with shape {instance.shape}."
                )

            # Validate that axis annotation shape matches tensor shape
            axis_annotation = annotations[axis]
            expected_size = instance.shape[axis]

            if instance.data.is_nested and axis_annotation.is_nested:
                if axis_annotation.cardinalities != tuple(expected_size):
                    raise ValueError(
                        f'For dimension at axis {axis} we were given an '
                        f'annotation with cardinalities {axis_annotation.cardinalities}. '
                        f'However, we expected cardinalities {expected_size}.'
                    )
            else:
                if axis_annotation.shape != expected_size:
                    raise ValueError(
                        f'For dimension at axis {axis} we were given an '
                        f'annotation with shape {axis_annotation.shape}. '
                        f'However, we expected shape {expected_size} as the '
                        f'tensor has shape {instance.shape}.'
                    )

        return annotations

    def __str__(self):
        """
        Returns a string representation of the AnnotatedTensor.
        """
        return (
            f"AnnotatedTensor of shape {self.shape}, dtype {self.dtype}, and "
            f"annotations {self.annotations}."
        )

    def annotated_axis(self) -> List[int]:
        """Get list of annotated axes."""
        return self.annotations.annotated_axes

    def concat_concepts(self) -> torch.Tensor:
        """
        Concatenate all fields along channel/feature dim (dim=1).
        Works for any leaf rank >=2 as long as dims >=2 match across fields.
        Result shape: (B, sum_i c_i, *trailing_shape)
        """
        return torch.cat(list(self.data.unbind()), dim=1)

    def _apply_to_nested_or_regular(self, operation, *args, **kwargs):
        """
        Apply an operation to nested tensor (per constituent) or regular tensor.

        Args:
            operation: Function to apply (takes tensor, *args, **kwargs).
            *args, **kwargs: Arguments to pass to the operation.

        Returns:
            Resulting tensor (nested or regular).
        """
        if self.data.is_nested:
            result_list = [operation(t, *args, **kwargs) for t in self.data.unbind()]
            return torch.nested.nested_tensor(result_list)
        else:
            return operation(self.data, *args, **kwargs)

    def extract_by_annotations(
            self,
            target_annotations: List[Union[int, str]],
            target_axis: int = None,
    ) -> 'AnnotatedTensor':
        """
        Extract a subset of elements from the AnnotatedTensor by label names or indices.

        Args:
            target_annotations: List of label names or indices to extract.
            target_axis: Axis to extract from. If None, uses last annotated axis.

        Returns:
            AnnotatedTensor: Extracted AnnotatedTensor with updated annotations.

        Behavior for nested tensors:
            - If extracting from axis=1 (nested axis) with 1 index: returns regular tensor
            - If extracting from axis=1 (nested axis) with >1 indices: returns nested tensor
            - If extracting from other axes: preserves nested structure
        """
        if self.annotations.num_annotated_axes == 0:
            raise ValueError(
                'Cannot extract by annotations for AnnotatedTensor without '
                'any dimensions annotated.'
            )

        if target_axis is None:
            target_axis = self.annotated_axis()[-1]

        if not self.annotations.has_axis(target_axis):
            raise ValueError(
                f"Axis {target_axis} is not annotated in this AnnotatedTensor."
            )

        # Get indices for extraction
        axis_labels = self.annotations.get_axis_labels(target_axis)
        indices = []

        for annotation_name in target_annotations:
            if isinstance(annotation_name, str):
                try:
                    idx = self.annotations.get_index(target_axis, annotation_name)
                    indices.append(idx)
                except ValueError:
                    raise ValueError(
                        f"Annotation '{annotation_name}' was not found in "
                        f"axis {target_axis} labels: {axis_labels}."
                    )
            else:
                indices.append(annotation_name)

        # Handle nested tensors specially when extracting from axis=1
        if self.data.is_nested and target_axis == 1:
            constituents = self.data.unbind()

            if len(indices) == 1:
                # Single field extraction: return regular tensor
                extracted_data = constituents[indices[0]]
            else:
                # Multiple fields: return nested tensor with selected constituents
                extracted_data = torch.nested.nested_tensor(
                    [constituents[i] for i in indices],
                    dtype=self.dtype,
                    device=self.device
                )
        else:
            # Regular tensor or extracting from non-nested axis
            index_tensor = torch.tensor(indices, device=self.device)
            extracted_data = self._apply_to_nested_or_regular(
                lambda t: t.index_select(dim=target_axis, index=index_tensor)
            )

        # Create new annotations with extracted labels
        new_annotations = Annotations({})
        for axis in self.annotated_axis():
            if axis == target_axis:
                extracted_labels = tuple(axis_labels[i] for i in indices)
                axis_ann = self.annotations[axis]

                if axis_ann.is_nested:
                    # Handle nested annotations
                    if len(indices) == 1:
                        # Single extraction from nested: annotations become non-nested
                        # Use the states from the extracted field
                        new_axis_annotation = AxisAnnotation(
                            labels=axis_ann.states[indices[0]],
                            graph=axis_ann.graph,
                            metadata=axis_ann.metadata,
                        )
                    else:
                        # Multiple extractions: keep nested structure
                        new_axis_annotation = AxisAnnotation(
                            labels=extracted_labels,
                            states=tuple(axis_ann.states[i] for i in indices),
                            cardinalities=tuple(axis_ann.cardinalities[i] for i in indices),
                            graph=axis_ann.graph,
                            metadata=axis_ann.metadata,
                        )
                else:
                    new_axis_annotation = AxisAnnotation(
                        labels=extracted_labels,
                        graph=axis_ann.graph,
                        metadata=axis_ann.metadata,
                    )

                new_annotations.annotate_axis(new_axis_annotation, axis)
            else:
                new_annotations.annotate_axis(self.annotations[axis], axis)

        return self._wrap_result(extracted_data, annotations=new_annotations)

    def view(self, *shape, annotations: Annotations = None):
        """
        View the tensor with a new shape and optionally update annotations.

        Args:
            shape: New shape for the view.
            annotations: Optional new Annotations object.

        Note: For nested tensors, view is not supported and will raise an error.
        """
        if self.data.is_nested:
            raise RuntimeError("view() is not supported for nested tensors")

        new_tensor = self.data.view(*shape)
        return self._wrap_result(new_tensor, annotations=annotations or Annotations({}))

    def reshape(self, *shape, annotations: Annotations = None):
        """
        Reshape the tensor and optionally update annotations.

        Args:
            shape: New shape for the tensor.
            annotations: Optional new Annotations object.

        Note: For nested tensors, reshape is not supported and will raise an error.
        """
        if self.data.is_nested:
            raise RuntimeError("reshape() is not supported for nested tensors")

        new_tensor = self.data.reshape(*shape)
        return self._wrap_result(new_tensor, annotations=annotations or Annotations({}))

    def _normalize_dim(self, dim):
        """Normalize a dimension index to handle negative indexing."""
        ndim = self.data.ndim if not self.data.is_nested else self.data[0].ndim
        return dim if dim >= 0 else ndim + dim

    def _check_axis1_protection(self, *dims):
        """Check if any dimension is axis=1 (concept/field dimension) and raise error."""
        normalized_dims = [self._normalize_dim(d) for d in dims]
        if 1 in normalized_dims:
            raise ValueError(
                "Cannot operate on axis=1 (concept/field dimension). "
                "This dimension represents variable-sized concepts/fields and "
                "the operation would be ambiguous."
            )

    def transpose(self, dim0, dim1):
        """
        Transpose two dimensions of the tensor and swap their annotations.

        Args:
            dim0: First dimension.
            dim1: Second dimension.

        Note: For nested tensors, transpose is applied to each constituent tensor.
        Note: Cannot transpose axis=1 (concept/field dimension) as it's ambiguous for nested tensors.
        """
        self._check_axis1_protection(dim0, dim1)

        new_tensor = self._apply_to_nested_or_regular(lambda t: t.transpose(dim0, dim1))

        # Create new annotations with swapped axes
        new_annotations = Annotations({})
        for axis in self.annotated_axis():
            if axis == dim0:
                new_annotations.annotate_axis(self.annotations[axis], dim1)
            elif axis == dim1:
                new_annotations.annotate_axis(self.annotations[axis], dim0)
            else:
                new_annotations.annotate_axis(self.annotations[axis], axis)

        return self._wrap_result(new_tensor, annotations=new_annotations)

    def permute(self, *dims):
        """
        Permute the dimensions of the tensor and remap annotations accordingly.

        Args:
            dims: Desired ordering of dimensions.

        Note: For nested tensors, permute is applied to each constituent tensor.
        Note: Cannot move axis=1 (concept/field dimension) as it's ambiguous for nested tensors.
        """
        # Check if axis 1 is being moved to a different position
        if len(dims) > 1 and dims[1] != 1:
            raise ValueError(
                "Cannot permute axis=1 (concept/field dimension) to a different position. "
                "This dimension represents variable-sized concepts/fields and "
                "moving it would be ambiguous."
            )

        new_tensor = self._apply_to_nested_or_regular(lambda t: t.permute(*dims))

        # Create new annotations with permuted axes
        new_annotations = Annotations({})
        for old_axis in self.annotated_axis():
            new_axis = dims.index(old_axis) if old_axis in dims else None
            if new_axis is not None:
                new_annotations.annotate_axis(self.annotations[old_axis], new_axis)

        return self._wrap_result(new_tensor, annotations=new_annotations)

    def _adjust_annotations_for_removed_dim(self, removed_dims):
        """Create new annotations after dimensions have been removed."""
        new_annotations = Annotations({})
        for axis in self.annotated_axis():
            # Count how many dims before this one were removed
            offset = sum(1 for d in removed_dims if d < axis)
            if axis not in removed_dims:
                new_annotations.annotate_axis(self.annotations[axis], axis - offset)
        return new_annotations

    def _adjust_annotations_for_added_dim(self, inserted_dim):
        """Create new annotations after a dimension has been added."""
        new_annotations = Annotations({})
        for axis in self.annotated_axis():
            if axis < inserted_dim:
                new_annotations.annotate_axis(self.annotations[axis], axis)
            else:
                new_annotations.annotate_axis(self.annotations[axis], axis + 1)
        return new_annotations

    def squeeze(self, dim=None):
        """
        Squeeze the tensor and adjust annotations for removed dimensions.

        Args:
            dim: Dimension to squeeze, or None to squeeze all size-1 dimensions.

        Note: For nested tensors, squeeze is applied to each constituent tensor.
        Note: Cannot squeeze axis=1 (concept/field dimension).
        """
        if dim is not None:
            self._check_axis1_protection(dim)

        new_tensor = self._apply_to_nested_or_regular(
            lambda t: t.squeeze(dim) if dim is not None else t.squeeze()
        )

        # Handle annotations
        if dim is not None:
            old_shape = self.data.shape if not self.data.is_nested else self.data[0].shape
            normalized_dim = self._normalize_dim(dim)
            new_annotations = self._adjust_annotations_for_removed_dim([normalized_dim])
        else:
            old_shape = self.data.shape if not self.data.is_nested else self.data[0].shape
            squeezed_dims = [i for i, s in enumerate(old_shape) if s == 1]
            new_annotations = self._adjust_annotations_for_removed_dim(squeezed_dims)

        return self._wrap_result(new_tensor, annotations=new_annotations)

    def unsqueeze(self, dim):
        """
        Unsqueeze the tensor and adjust annotations for the new dimension.

        Args:
            dim: Position where the new dimension will be inserted.

        Note: For nested tensors, unsqueeze is applied to each constituent tensor.
        Note: Cannot unsqueeze at axis=1 (would displace concept/field dimension).
        """
        # For unsqueeze, normalize considering the new dimension
        ndim = self.data.ndim if not self.data.is_nested else self.data[0].ndim
        normalized_dim = dim if dim >= 0 else ndim + 1 + dim

        if normalized_dim == 1:
            raise ValueError(
                "Cannot unsqueeze at axis=1 (would displace concept/field dimension). "
                "This dimension represents variable-sized concepts/fields and "
                "must remain at position 1."
            )

        new_tensor = self._apply_to_nested_or_regular(lambda t: t.unsqueeze(dim))
        new_annotations = self._adjust_annotations_for_added_dim(normalized_dim)

        return self._wrap_result(new_tensor, annotations=new_annotations)

    def ravel(self):
        """
        Flatten the tensor to 1D and clear all annotations.

        Note: For nested tensors, ravel is not supported and will raise an error.
        """
        if self.data.is_nested:
            raise RuntimeError("ravel() is not supported for nested tensors")

        return self._wrap_result(self.data.ravel(), annotations=Annotations({}))

    def _slice_nested_tensor(self, key):
        """Apply slicing to nested tensor and return result."""
        constituent_tensors = list(self.data.unbind())
        sliced_constituents = [t[key] for t in constituent_tensors]

        # Try to reconstruct as nested tensor
        if all(isinstance(t, torch.Tensor) and t.ndim >= 2 for t in sliced_constituents):
            return torch.nested.nested_tensor(sliced_constituents)

        # Try to stack
        if all(isinstance(t, torch.Tensor) and t.ndim >= 1 for t in sliced_constituents):
            try:
                return torch.stack(sliced_constituents)
            except:
                pass

        # Return single element if only one, otherwise fail
        if len(sliced_constituents) == 1 and isinstance(sliced_constituents[0], torch.Tensor):
            return sliced_constituents[0]

        # Return scalar or raise
        if len(sliced_constituents) == 1:
            return sliced_constituents[0]
        raise ValueError("Cannot create AnnotatedTensor from sliced nested tensor")

    def _slice_axis_annotation(self, axis_ann, idx):
        """Slice annotation labels for a given index."""
        if isinstance(idx, int):
            return None  # Dimension removed

        axis_labels = axis_ann.labels

        if isinstance(idx, slice):
            sliced_labels = axis_labels[idx]
            if axis_ann.is_nested:
                return AxisAnnotation(
                    labels=sliced_labels,
                    states=axis_ann.states[idx],
                    cardinalities=axis_ann.cardinalities[idx],
                    graph=axis_ann.graph,
                    metadata=axis_ann.metadata,
                )
            else:
                return AxisAnnotation(
                    labels=sliced_labels,
                    graph=axis_ann.graph,
                    metadata=axis_ann.metadata,
                )

        elif isinstance(idx, (list, torch.Tensor, np.ndarray)):
            # Convert to list
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            elif isinstance(idx, np.ndarray):
                idx = idx.tolist()

            selected_labels = tuple(axis_labels[i] for i in idx)
            if axis_ann.is_nested:
                return AxisAnnotation(
                    labels=selected_labels,
                    states=tuple(axis_ann.states[i] for i in idx),
                    cardinalities=tuple(axis_ann.cardinalities[i] for i in idx),
                    graph=axis_ann.graph,
                    metadata=axis_ann.metadata,
                )
            else:
                return AxisAnnotation(
                    labels=selected_labels,
                    graph=axis_ann.graph,
                    metadata=axis_ann.metadata,
                )

        return None

    def __getitem__(self, key):
        """
        Slice the tensor and update annotations accordingly.

        Supports both regular and nested tensors, preserving gradient flow and annotations.

        Args:
            key: Indexing key (int, slice, tuple, etc.).

        For nested tensors:
        - Indexing at dim=0 (batch) returns a nested tensor with updated B
        - Indexing at other dims is applied to each constituent tensor
        """
        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Apply slicing
        if self.data.is_nested:
            sliced_tensor = self._slice_nested_tensor(key)
        else:
            sliced_tensor = self.data[key]

        # Return scalar if not a tensor
        if not isinstance(sliced_tensor, torch.Tensor):
            return sliced_tensor

        # Identify removed dimensions
        removed_dims = {i for i, idx in enumerate(key) if isinstance(idx, int)}

        # Create new annotations
        new_annotations = Annotations({})
        for axis in self.annotated_axis():
            if axis >= len(key):
                # Axis not affected - adjust for removed dims
                offset = sum(1 for d in removed_dims if d < axis)
                new_annotations.annotate_axis(self.annotations[axis], axis - offset)
            else:
                # Apply slicing to annotation
                new_axis_ann = self._slice_axis_annotation(self.annotations[axis], key[axis])
                if new_axis_ann is not None:
                    offset = sum(1 for d in removed_dims if d < axis)
                    new_annotations.annotate_axis(new_axis_ann, axis - offset)

        return self._wrap_result(sliced_tensor, annotations=new_annotations)


class AnnotatedAdjacencyMatrix(AnnotatedTensor):
    """
    Adjacency matrix with semantic annotations for rows and columns.

    This class extends AnnotatedTensor to provide specialized functionality for
    graph structures, particularly adjacency matrices where rows and columns
    represent nodes with meaningful names.

    The adjacency matrix A has shape (n_nodes, n_nodes) where:
        A[i, j] = weight if there's an edge from node i to node j, else 0

    Attributes:
        node_names: Names of nodes (from annotations)
        n_nodes: Number of nodes in the graph
        is_directed: Whether the graph is directed (default: True)
        loc: Label-based indexer (like pandas DataFrame.loc)
        iloc: Integer position-based indexer (like pandas DataFrame.iloc)

    Args:
        data (Tensor): Adjacency matrix of shape (n_nodes, n_nodes)
        annotations (Union[Annotations, List[str]]): Either an Annotations object with
            axis 0 and 1 annotated, or a list of node names that will be used for both axes.
            If a list is provided, it will be converted to an Annotations object.
        is_directed (bool, optional): Whether graph is directed, default True
    """
    # TODO: check whether we can extend from networkx.DiGraph and pyg
    def __new__(
            cls,
            data: Tensor,
            annotations: Union[Annotations, List[str]] = None,
            is_directed: bool = True,
    ):
        """Create new AnnotatedAdjacencyMatrix instance."""
        # Validate shape
        if data.dim() != 2:
            raise ValueError(f"Adjacency matrix must be 2D, got {data.dim()}D")
        if data.shape[0] != data.shape[1]:
            raise ValueError(
                f"Adjacency matrix must be square, got shape {data.shape}"
            )

        # Convert list of node names to Annotations object if needed
        if isinstance(annotations, list):
            # Check if it's a list of lists (old API: [row_names, col_names])
            if len(annotations) == 2 and isinstance(annotations[0], (list, tuple)):
                # Old API: [row_names, col_names]
                row_labels = tuple(annotations[0])
                col_labels = tuple(annotations[1])
                annotations = Annotations({
                    0: AxisAnnotation(labels=row_labels),
                    1: AxisAnnotation(labels=col_labels)
                })
            else:
                # Single list of node names, use for both axes
                node_labels = tuple(annotations)
                annotations = Annotations({
                    0: AxisAnnotation(labels=node_labels),
                    1: AxisAnnotation(labels=node_labels)
                })
        elif annotations is None:
            # Auto-annotate both axes with default node names
            n_nodes = data.shape[0]
            node_labels = tuple(f"node_{i}" for i in range(n_nodes))
            annotations = Annotations({
                0: AxisAnnotation(labels=node_labels),
                1: AxisAnnotation(labels=node_labels)
            })

        # Create AnnotatedTensor instance
        obj = super().__new__(cls, data, annotations)

        # Add graph-specific attributes
        # TODO: is this needed?
        obj.is_directed = is_directed

        return obj

    @property
    def node_names(self) -> List[str]:
        """Get list of node names from annotations."""
        # Get node names from axis 0 annotations
        if hasattr(self, 'annotations') and 0 in self.annotations.annotated_axes:
            return list(self.annotations[0].labels)
        return []

    @property
    def n_nodes(self) -> int:
        """Get number of nodes in the graph."""
        return self.shape[0]

    def dense_to_sparse(self, threshold: float = 0.0) -> Tuple[Tensor, Tensor]:
        """
        Convert dense adjacency matrix to sparse edge representation (COO format).

        This is similar to PyTorch Geometric's dense_to_sparse function.

        Args:
            threshold: Minimum value to consider as an edge (default: 0.0)

        Returns:
            edge_index: Tensor of shape (2, num_edges) with source and target indices
            edge_weight: Tensor of shape (num_edges,) with edge weights

        Example:
            >>> edge_index, edge_weight = graph.dense_to_sparse()
            >>> print(edge_index.shape)  # torch.Size([2, num_edges])
            >>> print(edge_weight.shape)  # torch.Size([num_edges])
        """
        return dense_to_sparse(self, threshold=threshold)

    def to_networkx(self) -> nx.DiGraph:
        """
        Convert to NetworkX directed graph.

        Returns:
            nx.DiGraph: NetworkX directed graph with node and edge attributes

        Example:
            >>> nx_graph = graph.to_networkx()
            >>> print(list(nx_graph.nodes()))  # Node names
            >>> print(list(nx_graph.edges()))  # Edges
        """
        return to_networkx_graph(self)

    def get_root_nodes(self) -> List[str]:
        """
        Get nodes with no incoming edges (root nodes).

        Returns:
            List of root node names

        Example:
            >>> roots = graph.get_root_nodes()
            >>> print(roots)  # ['input_node']
        """
        return get_root_nodes(self)

    def get_leaf_nodes(self) -> List[str]:
        """
        Get nodes with no outgoing edges (leaf nodes).

        Returns:
            List of leaf node names

        Example:
            >>> leaves = graph.get_leaf_nodes()
            >>> print(leaves)  # ['output_node']
        """
        return get_leaf_nodes(self)

    def topological_sort(self) -> List[str]:
        """
        Compute topological ordering of nodes.

        Only valid for directed acyclic graphs (DAGs).

        Returns:
            List of node names in topological order

        Raises:
            nx.NetworkXError: If graph contains cycles

        Example:
            >>> ordered = graph.topological_sort()
            >>> print(ordered)  # ['A', 'B', 'C']
        """
        return topological_sort(self)

    def get_predecessors(self, node: Union[str, int]) -> List[str]:
        """
        Get all predecessors (parents) of a node.

        Args:
            node: Node name (str) or index (int)

        Returns:
            List of predecessor node names

        Example:
            >>> preds = graph.get_predecessors('C')
            >>> print(preds)  # ['A', 'B']
        """
        return get_predecessors(self, node)

    def get_successors(self, node: Union[str, int]) -> List[str]:
        """
        Get all successors (children) of a node.

        Args:
            node: Node name (str) or index (int)

        Returns:
            List of successor node names

        Example:
            >>> succs = graph.get_successors('A')
            >>> print(succs)  # ['B', 'C']
        """
        return get_successors(self, node)

    def get_ancestors(self, node: Union[str, int]) -> Set[str]:
        """
        Get all ancestors of a node (recursive predecessors).

        Args:
            node: Node name (str) or index (int)

        Returns:
            Set of ancestor node names

        Example:
            >>> ancestors = graph.get_ancestors('D')
            >>> print(ancestors)  # {'A', 'B', 'C'}
        """
        return get_ancestors(self, node)

    def get_descendants(self, node: Union[str, int]) -> Set[str]:
        """
        Get all descendants of a node (recursive successors).

        Args:
            node: Node name (str) or index (int)

        Returns:
            Set of descendant node names

        Example:
            >>> descendants = graph.get_descendants('A')
            >>> print(descendants)  # {'B', 'C', 'D'}
        """
        return get_descendants(self, node)

    def is_directed_acyclic(self) -> bool:
        """
        Check if the graph is a directed acyclic graph (DAG).

        Returns:
            True if graph is a DAG, False otherwise
        """
        return is_directed_acyclic(self)

    def is_dag(self) -> bool:
        """
        Check if the graph is a directed acyclic graph (DAG).

        Alias for is_directed_acyclic() for convenience.

        Returns:
            True if graph is a DAG, False otherwise
        """
        return self.is_directed_acyclic()

    def get_edge_weight(self, source: Union[str, int], target: Union[str, int]) -> float:
        """
        Get the weight of an edge.

        Args:
            source: Source node name or index
            target: Target node name or index

        Returns:
            Edge weight, or 0.0 if no edge exists
        """
        source_idx = self._node_to_index(source)
        target_idx = self._node_to_index(target)
        return self[source_idx, target_idx].item()

    def has_edge(self, source: Union[str, int], target: Union[str, int], threshold: float = 0.0) -> bool:
        """
        Check if an edge exists between two nodes.

        Args:
            source: Source node name or index
            target: Target node name or index
            threshold: Minimum weight to consider as edge

        Returns:
            True if edge exists, False otherwise
        """
        weight = self.get_edge_weight(source, target)
        return abs(weight) > threshold

    def _node_to_index(self, node: Union[str, int]) -> int:
        """Convert node name or index to index."""
        if isinstance(node, int):
            if node < 0 or node >= self.n_nodes:
                raise IndexError(f"Node index {node} out of range [0, {self.n_nodes})")
            return node
        elif isinstance(node, str):
            if node not in self.node_names:
                raise ValueError(f"Node '{node}' not found in graph")
            return self.node_names.index(node)
        else:
            raise TypeError(f"Node must be str or int, got {type(node)}")

    def get_by_nodes(
            self,
            rows: Union[str, List[str]],
            cols: Union[str, List[str]]
    ) -> Tensor:
        """
        Get graph values by node names.

        Args:
            rows: Node name(s) for rows - single string or list of strings
            cols: Node name(s) for columns - single string or list of strings

        Returns:
            Tensor with the requested values

        Example:
            >>> graph.get_by_nodes('A', 'B')  # Single edge weight
            >>> graph.get_by_nodes('A', ['B', 'C'])  # Multiple edges from A
            >>> graph.get_by_nodes(['A', 'B'], ['C', 'D'])  # 2x2 subgraph
        """
        # Convert names to indices
        if isinstance(rows, str):
            row_indices = self._node_to_index(rows)
        else:
            row_indices = [self._node_to_index(r) for r in rows]

        if isinstance(cols, str):
            col_indices = self._node_to_index(cols)
        else:
            col_indices = [self._node_to_index(c) for c in cols]

        # Handle list indexing for 2D submatrix
        if isinstance(row_indices, list) and isinstance(col_indices, list):
            row_tensor = torch.tensor(row_indices).unsqueeze(1)
            col_tensor = torch.tensor(col_indices).unsqueeze(0)
            return self.data[row_tensor, col_tensor]
        else:
            return self.data[row_indices, col_indices]

    def get_by_index(
            self,
            rows: Union[int, List[int]],
            cols: Union[int, List[int]]
    ) -> Tensor:
        """
        Get graph values by integer indices.

        Args:
            rows: Row index/indices - single int or list of ints
            cols: Column index/indices - single int or list of ints

        Returns:
            Tensor with the requested values

        Example:
            >>> graph.get_by_index(0, 1)  # Single edge weight
            >>> graph.get_by_index(0, [1, 2])  # Multiple edges from node 0
            >>> graph.get_by_index([0, 1], [2, 3])  # 2x2 subgraph
        """
        # Handle list indexing for 2D submatrix
        if isinstance(rows, list) and isinstance(cols, list):
            row_tensor = torch.tensor(rows).unsqueeze(1)
            col_tensor = torch.tensor(cols).unsqueeze(0)
            return self.data[row_tensor, col_tensor]
        else:
            return self.data[rows, cols]

    def to_pandas(self) -> DataFrame:
        """
        Convert adjacency matrix to pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame representation of the adjacency matrix
        """
        import pandas as pd
        df = pd.DataFrame(
            self.data.cpu().numpy(),
            index=self.node_names,
            columns=self.node_names
        )
        return df


def dense_to_sparse(
        adj_matrix: Union[AnnotatedAdjacencyMatrix, Tensor],
        threshold: float = 0.0
) -> Tuple[Tensor, Tensor]:
    """
    Convert dense adjacency matrix to sparse COO format (edge list).

    Uses PyTorch Geometric's native dense_to_sparse function if available,
    otherwise falls back to manual implementation.

    Args:
        adj_matrix: Dense adjacency matrix of shape (n_nodes, n_nodes)
        threshold: Minimum absolute value to consider as an edge (only used in fallback)

    Returns:
        edge_index: Tensor of shape (2, num_edges) with [source_indices, target_indices]
        edge_weight: Tensor of shape (num_edges,) with edge weights

    Example:
        >>> adj = torch.tensor([[0., 1., 0.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> edge_index, edge_weight = dense_to_sparse(adj)
        >>> print(edge_index)
        tensor([[0, 1],
                [1, 2]])
        >>> print(edge_weight)
        tensor([1., 1.])
    """
    # Convert AnnotatedAdjacencyMatrix to regular tensor if needed
    if isinstance(adj_matrix, AnnotatedTensor):
        adj_tensor = adj_matrix.as_subclass(Tensor)
    else:
        adj_tensor = adj_matrix

    return pyg.utils.dense_to_sparse(adj_tensor)


def to_networkx_graph(
        adj_matrix: Union[AnnotatedAdjacencyMatrix, Tensor],
        node_names: Optional[List[str]] = None,
        threshold: float = 0.0
) -> nx.DiGraph:
    """
    Convert adjacency matrix to NetworkX directed graph.

    Uses NetworkX's native from_numpy_array function for conversion.

    Args:
        adj_matrix: Adjacency matrix (dense)
        node_names: Optional node names. If adj_matrix is AnnotatedAdjacencyMatrix,
                   uses its node_names. Otherwise uses integer indices.
        threshold: Minimum absolute value to consider as an edge

    Returns:
        nx.DiGraph: NetworkX directed graph

    Example:
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> G = to_networkx_graph(adj, node_names=['A', 'B', 'C'])
        >>> print(list(G.nodes()))  # ['A', 'B', 'C']
        >>> print(list(G.edges()))  # [('A', 'B'), ('A', 'C'), ('B', 'C')]
    """
    # Extract node names if AnnotatedAdjacencyMatrix
    if isinstance(adj_matrix, AnnotatedAdjacencyMatrix):
        if node_names is None:
            node_names = adj_matrix.node_names
        adj_tensor = adj_matrix.as_subclass(Tensor)
    else:
        adj_tensor = adj_matrix
        if node_names is None:
            node_names = list(range(adj_tensor.shape[0]))

    # Apply threshold if needed
    if threshold > 0.0:
        adj_tensor = adj_tensor.clone()
        adj_tensor[torch.abs(adj_tensor) <= threshold] = 0.0

    # Convert to numpy for NetworkX
    adj_numpy = adj_tensor.detach().cpu().numpy()

    # Use NetworkX's native conversion
    # from_numpy_array creates a graph from adjacency matrix
    G = nx.from_numpy_array(
        adj_numpy,
        create_using=nx.DiGraph
    )

    # Relabel nodes with custom names if provided
    if node_names != list(range(len(node_names))):
        mapping = {i: name for i, name in enumerate(node_names)}
        G = nx.relabel_nodes(G, mapping)

    return G


def get_root_nodes(
        adj_matrix: Union[AnnotatedAdjacencyMatrix, Tensor, nx.DiGraph],
        node_names: Optional[List[str]] = None
) -> List[str]:
    """
    Get nodes with no incoming edges (in-degree = 0).

    Args:
        adj_matrix: Adjacency matrix or NetworkX graph
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        List of root node names

    Example:
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> roots = get_root_nodes(adj, node_names=['A', 'B', 'C'])
        >>> print(roots)  # ['A']
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
    else:
        if isinstance(adj_matrix, AnnotatedAdjacencyMatrix):
            node_names = adj_matrix.annotations.get_axis_labels(axis=1)

        G = to_networkx_graph(adj_matrix, node_names=node_names)

    return [node for node, degree in G.in_degree() if degree == 0]


def get_leaf_nodes(
        adj_matrix: Union[AnnotatedAdjacencyMatrix, Tensor, nx.DiGraph],
        node_names: Optional[List[str]] = None
) -> List[str]:
    """
    Get nodes with no outgoing edges (out-degree = 0).

    Args:
        adj_matrix: Adjacency matrix or NetworkX graph
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        List of leaf node names

    Example:
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> leaves = get_leaf_nodes(adj, node_names=['A', 'B', 'C'])
        >>> print(leaves)  # ['C']
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
    else:
        if isinstance(adj_matrix, AnnotatedAdjacencyMatrix):
            node_names = adj_matrix.annotations.get_axis_labels(axis=1)

        G = to_networkx_graph(adj_matrix, node_names=node_names)

    return [node for node, degree in G.out_degree() if degree == 0]


def topological_sort(
        adj_matrix: Union[AnnotatedAdjacencyMatrix, Tensor, nx.DiGraph],
        node_names: Optional[List[str]] = None
) -> List[str]:
    """
    Compute topological ordering of nodes (only for DAGs).

    Uses NetworkX's native topological_sort function.

    Args:
        adj_matrix: Adjacency matrix or NetworkX graph
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        List of node names in topological order

    Raises:
        nx.NetworkXError: If graph contains cycles

    Example:
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> ordered = topological_sort(adj, node_names=['A', 'B', 'C'])
        >>> print(ordered)  # ['A', 'B', 'C']
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
    else:
        if isinstance(adj_matrix, AnnotatedAdjacencyMatrix):
            node_names = adj_matrix.annotations.get_axis_labels(axis=1)

        G = to_networkx_graph(adj_matrix, node_names=node_names)

    # Use NetworkX's native implementation
    return list(nx.topological_sort(G))


def get_predecessors(
        adj_matrix: Union[AnnotatedAdjacencyMatrix, Tensor, nx.DiGraph],
        node: Union[str, int],
        node_names: Optional[List[str]] = None
) -> List[str]:
    """
    Get immediate predecessors (parents) of a node.

    Uses NetworkX's native predecessors method.

    Args:
        adj_matrix: Adjacency matrix or NetworkX graph
        node: Node name (str) or index (int)
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        List of predecessor node names

    Example:
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> preds = get_predecessors(adj, 'C', node_names=['A', 'B', 'C'])
        >>> print(preds)  # ['A', 'B']
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
        if isinstance(node, int) and node_names:
            node = node_names[node]
    else:
        if isinstance(adj_matrix, AnnotatedAdjacencyMatrix):
            node_names = adj_matrix.annotations.get_axis_labels(axis=1)

        G = to_networkx_graph(adj_matrix, node_names=node_names)
        if isinstance(node, int):
            node = node_names[node]

    # Use NetworkX's native implementation
    return list(G.predecessors(node))


def get_successors(
        adj_matrix: Union[AnnotatedAdjacencyMatrix, Tensor, nx.DiGraph],
        node: Union[str, int],
        node_names: Optional[List[str]] = None
) -> List[str]:
    """
    Get immediate successors (children) of a node.

    Uses NetworkX's native successors method.

    Args:
        adj_matrix: Adjacency matrix or NetworkX graph
        node: Node name (str) or index (int)
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        List of successor node names

    Example:
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> succs = get_successors(adj, 'A', node_names=['A', 'B', 'C'])
        >>> print(succs)  # ['B', 'C']
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
        if isinstance(node, int) and node_names:
            node = node_names[node]
    else:
        if isinstance(adj_matrix, AnnotatedAdjacencyMatrix):
            node_names = adj_matrix.annotations.get_axis_labels(axis=1)

        G = to_networkx_graph(adj_matrix, node_names=node_names)
        if isinstance(node, int):
            node = node_names[node]

    # Use NetworkX's native implementation
    return list(G.successors(node))


def get_ancestors(
        adj_matrix: Union[AnnotatedAdjacencyMatrix, Tensor, nx.DiGraph],
        node: Union[str, int],
        node_names: Optional[List[str]] = None
) -> Set[str]:
    """
    Get all ancestors of a node (transitive predecessors).

    Uses NetworkX's native ancestors function.

    Args:
        adj_matrix: Adjacency matrix or NetworkX graph
        node: Node name (str) or index (int)
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        Set of ancestor node names

    Example:
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> ancestors = get_ancestors(adj, 'C', node_names=['A', 'B', 'C'])
        >>> print(ancestors)  # {'A', 'B'}
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
        if isinstance(node, int) and node_names:
            node = node_names[node]
    else:
        if isinstance(adj_matrix, AnnotatedAdjacencyMatrix):
            node_names = adj_matrix.annotations.get_axis_labels(axis=1)

        G = to_networkx_graph(adj_matrix, node_names=node_names)
        if isinstance(node, int):
            node = node_names[node]

    # Use NetworkX's native implementation
    return nx.ancestors(G, node)


def get_descendants(
        adj_matrix: Union[AnnotatedAdjacencyMatrix, Tensor, nx.DiGraph],
        node: Union[str, int],
        node_names: Optional[List[str]] = None
) -> Set[str]:
    """
    Get all descendants of a node (transitive successors).

    Uses NetworkX's native descendants function.

    Args:
        adj_matrix: Adjacency matrix or NetworkX graph
        node: Node name (str) or index (int)
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        Set of descendant node names

    Example:
        >>> adj = torch.tensor([[0., 1., 1.],
        ...                     [0., 0., 1.],
        ...                     [0., 0., 0.]])
        >>> descendants = get_descendants(adj, 'A', node_names=['A', 'B', 'C'])
        >>> print(descendants)  # {'B', 'C'}
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
        if isinstance(node, int) and node_names:
            node = node_names[node]
    else:
        if isinstance(adj_matrix, AnnotatedAdjacencyMatrix):
            node_names = adj_matrix.annotations.get_axis_labels(axis=1)

        G = to_networkx_graph(adj_matrix, node_names=node_names)
        if isinstance(node, int):
            node = node_names[node]

    # Use NetworkX's native implementation
    return nx.descendants(G, node)


def is_directed_acyclic(
        adj_matrix: Union[AnnotatedAdjacencyMatrix, Tensor, nx.DiGraph],
        node_names: Optional[List[str]] = None
) -> bool:
    """
    Check if the graph is a directed acyclic graph (DAG).

    Uses NetworkX's native is_directed_acyclic_graph function.

    Args:
        adj_matrix: Adjacency matrix or NetworkX graph
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        True if graph is a DAG, False otherwise

    Example:
        >>> adj = torch.tensor([[0., 1., 0.],
        ...                     [0., 0., 1.],
        ...                     [1., 0., 0.]])  # Contains cycle
        >>> print(is_directed_acyclic(adj))  # False
    """
    if isinstance(adj_matrix, nx.DiGraph):
        G = adj_matrix
    else:
        if isinstance(adj_matrix, AnnotatedAdjacencyMatrix):
            node_names = adj_matrix.annotations.get_axis_labels(axis=1)

        G = to_networkx_graph(adj_matrix, node_names=node_names)

    # Use NetworkX's native implementation
    return nx.is_directed_acyclic_graph(G)


def is_dag(
        adj_matrix: Union[AnnotatedAdjacencyMatrix, Tensor, nx.DiGraph],
        node_names: Optional[List[str]] = None
) -> bool:
    """
    Check if the graph is a directed acyclic graph (DAG).

    Alias for is_directed_acyclic() for convenience.

    Args:
        adj_matrix: Adjacency matrix or NetworkX graph
        node_names: Optional node names (only needed if adj_matrix is Tensor)

    Returns:
        True if graph is a DAG, False otherwise
    """
    return is_directed_acyclic(adj_matrix, node_names=node_names)
