"""
Concept annotations for tensors.

This module provides annotation structures for concept-based tensors, allowing
semantic labeling of tensor dimensions and their components. It supports both
simple (flat) and nested (hierarchical) concept structures.
"""

import warnings
import torch

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional, Any, Sequence


@dataclass
class AxisAnnotation:
    """
    Annotations for a single axis of a tensor.

    This class provides semantic labeling for one dimension of a tensor,
    supporting both simple binary concepts and nested multi-state concepts.

    Attributes:
        labels (list[str]): Ordered, unique labels for this axis.
        states (Optional[list[list[str]]]): State labels for each concept (if nested).
        cardinalities (Optional[list[int]]): Cardinality of each concept.
        metadata (Optional[Dict[str, Dict]]): Additional metadata for each label.
        is_nested (bool): Whether this axis has nested/hierarchical structure.

    Args:
        labels: List of concept names for this axis.
        states: Optional list of state lists for nested concepts.
        cardinalities: Optional list of cardinalities per concept.
        metadata: Optional metadata dictionary keyed by label names.

    Example:
        >>> from torch_concepts import AxisAnnotation
        >>>
        >>> # Simple binary concepts
        >>> axis_binary = AxisAnnotation(
        ...     labels=['has_wheels', 'has_windows', 'is_red']
        ... )
        >>> print(axis_binary.labels)  # ['has_wheels', 'has_windows', 'is_red']
        >>> print(axis_binary.is_nested)  # False
        >>> print(axis_binary.cardinalities)  # [1, 1, 1] - binary concepts
        >>>
        >>> # Nested concepts with explicit states
        >>> axis_nested = AxisAnnotation(
        ...     labels=['color', 'shape'],
        ...     states=[['red', 'green', 'blue'], ['circle', 'square']],
        ... )
        >>> print(axis_nested.labels)  # ['color', 'shape']
        >>> print(axis_nested.is_nested)  # True
        >>> print(axis_nested.cardinalities)  # [3, 2]
        >>> print(axis_nested.states[0])  # ['red', 'green', 'blue']
        >>>
        >>> # With cardinalities only (auto-generates state labels)
        >>> axis_cards = AxisAnnotation(
        ...     labels=['size', 'material'],
        ...     cardinalities=[3, 4]  # 3 sizes, 4 materials
        ... )
        >>> print(axis_cards.cardinalities)  # [3, 4]
        >>> print(axis_cards.states[0])  # ['0', '1', '2']
        >>>
        >>> # Access methods
        >>> idx = axis_binary.get_index('has_wheels')
        >>> print(idx)  # 0
        >>> label = axis_binary.get_label(1)
        >>> print(label)  # 'has_windows'
    """
    labels: List[str]
    states: Optional[List[List[str]]] = field(default=None)
    cardinalities: Optional[List[int]] = field(default=None)
    metadata: Optional[Dict[str, Dict]] = field(default=None)

    def __setattr__(self, key, value):
        # Allow first assignment or initialization
        if key == 'metadata':
            super().__setattr__(key, value)
            return
        if key in self.__dict__ and self.__dict__[key] is not None:
            raise AttributeError(f"'{key}' is write-once and already set")
        super().__setattr__(key, value)

    def __post_init__(self):
        """Validate consistency, infer is_nested and eventually states, and cardinalities."""
        # Initialize states and cardinalities based on what's provided
        if self.states is not None and self.cardinalities is None:
            # Infer cardinalities from states
            self.cardinalities = [len(state_tuple) for state_tuple in self.states]
        elif self.states is None and self.cardinalities is not None:
            # Generate default state labels from cardinalities
            self.states = [
                [str(i) for i in range(card)] if card > 1 else ['0']
                for card in self.cardinalities
            ]
        elif self.states is None and self.cardinalities is None:
            # Neither provided - assume binary
            warnings.warn(
                "Annotations: neither 'states' nor 'cardinalities' provided; "
                "assuming all concepts are binary."
            )
            self.cardinalities = [1 for _ in self.labels]
            self.states = [['0'] for _ in self.labels]
        else:
            # Both provided - use as-is for now, will validate below
            pass

        # Validate consistency now that both are populated
        if len(self.states) != len(self.labels):
            raise ValueError(
                f"Number of state tuples ({len(self.states)}) must match "
                f"number of labels ({len(self.labels)})"
            )
        if len(self.cardinalities) != len(self.labels):
            raise ValueError(
                f"Number of cardinalities ({len(self.cardinalities)}) must match "
                f"number of labels ({len(self.labels)})"
            )

        # Verify states length matches cardinalities
        # does not break with tuple cardinalities
        inferred_cardinalities = [len(state_tuple) for state_tuple in self.states]
        if list(self.cardinalities) != inferred_cardinalities:
            raise ValueError(
                f"Provided cardinalities {self.cardinalities} don't match "
                f"inferred cardinalities {inferred_cardinalities} from states"
            )

        # Determine is_nested from cardinalities
        # FIXME: should we consider nested also mix of scalars and bernoulli?
        is_nested = any(card > 1 for card in self.cardinalities)

        object.__setattr__(self, 'is_nested', is_nested)

        # Consistency checks on metadata
        if self.metadata is not None:
            if not isinstance(self.metadata, dict):
                raise ValueError("metadata must be a dictionary")
            # Only validate if metadata is non-empty
            if self.metadata:
                for label in self.labels:
                    if label not in self.metadata:
                        raise ValueError(f"Metadata missing for label {label!r}")

    @property
    def shape(self) -> Union[int, Tuple[int, ...]]:
        """
        Return the size of this axis.
        For non-nested: int (number of labels)
        For nested: tuple of ints (cardinalities)
        """
        if self.is_nested:
            return sum(self.cardinalities)
        return len(self.labels)
    
    def has_metadata(self, key) -> bool:
        """Check if metadata contains a specific key for all labels."""
        if self.metadata is None:
            return False
        return all(key in self.metadata.get(label, {}) for label in self.labels)

    def groupby_metadata(self, key, layout: str='labels') -> dict:
        """Check if metadata contains a specific key for all labels."""
        if self.metadata is None:
            return {}
        result = {}
        for label in self.labels:
            meta = self.metadata.get(label, {})
            if key in meta:
                group = meta[key]
                if group not in result:
                    result[group] = []
                if layout == 'labels':
                    result[group].append(label)
                elif layout == 'indices':
                    result[group].append(self.get_index(label))
                else:
                    raise ValueError(f"Unknown layout {layout}")
        return result

    def __len__(self) -> int:
        """Return number of labels in this axis."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Union[str, Dict[str, Union[str, Tuple[str, ...]]]]:
        """
        Get label or states at index.
        For non-nested: returns labels[idx] (str)
        For nested: returns dict {'label': label, 'states': state_tuple}
        """
        if not (0 <= idx < len(self.labels)):
            raise IndexError(f"Index {idx} out of range")

        return self.labels[idx]

    def get_index(self, label: str) -> int:
        """Get index of a label in this axis."""
        try:
            return self.labels.index(label)
        except ValueError:
            raise ValueError(f"Label {label!r} not found in labels {self.labels}")

    def get_label(self, idx: int) -> str:
        """Get label at given index in this axis."""
        if not (0 <= idx < len(self.labels)):
            raise IndexError(f"Index {idx} out of range with {len(self.labels)} labels")
        return self.labels[idx]

    def get_total_cardinality(self) -> Optional[int]:
        """Get total cardinality for nested axis, or None if not nested."""
        if self.is_nested:
            if self.cardinalities is not None:
                return sum(self.cardinalities)
            else:
                raise ValueError("Cardinalities are not defined for this nested axis")
        else:
            return len(self.labels)

    def get_endogenous_idx(self, labels: List[str]) -> List[int]:
        """Get endogenous (logit-level) indices for a list of concept labels.
        
        This method returns the flattened tensor indices where the logits/values
        for the specified concepts appear, accounting for each concept's cardinality.
        
        Args:
            labels: List of concept label names to get indices for.
            
        Returns:
            List of endogenous indices in the flattened tensor, in the order 
            corresponding to the input labels.
            
        Raises:
            ValueError: If any label is not found in the axis labels.
            
        Example:
            >>> # Concepts: ['color', 'shape', 'size'] with cardinalities [3, 2, 1]
            >>> # Flattened tensor has 6 positions: [c0, c1, c2, s0, s1, sz]
            >>> axis = AxisAnnotation(
            ...     labels=['color', 'shape', 'size'],
            ...     cardinalities=[3, 2, 1]
            ... )
            >>> axis.get_endogenous_idx(['color', 'size'])
            [0, 1, 2, 5]  # color takes positions 0-2, size takes position 5
        """
        endogenous_indices = []
        cum_idx = [0] + list(torch.cumsum(torch.tensor(self.cardinalities), dim=0).tolist())
        
        for label in labels:
            # Validate label exists
            try:
                concept_idx = self.get_index(label)
            except ValueError:
                raise ValueError(f"Label '{label}' not found in axis labels {self.labels}")
            
            # Get the range of endogenous indices for this concept
            start_idx = cum_idx[concept_idx]
            end_idx = cum_idx[concept_idx + 1]
            endogenous_indices.extend(range(start_idx, end_idx))
        
        return endogenous_indices

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary with all attributes, converting DataFrame to dict format.
        """
        result = {
            'labels': list(self.labels),
            'is_nested': self.is_nested,
            'states': [list(s) for s in self.states] if self.states else None,
            'cardinalities': list(self.cardinalities) if self.cardinalities else None,
            'metadata': self.metadata,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AxisAnnotation':
        """
        Create AxisAnnotation from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with serialized AxisAnnotation data.

        Returns
        -------
        AxisAnnotation
            Reconstructed AxisAnnotation object.
        """
        # Keep as lists (native format)
        labels = data['labels']
        states = [list(s) for s in data['states']] if data.get('states') else None
        cardinalities = data['cardinalities']

        return cls(
            labels=labels,
            states=states,
            cardinalities=cardinalities,
            metadata=data.get('metadata'),
        )

    def subset(self, keep_labels: Sequence[str]) -> "AxisAnnotation":
        """
        Return a new AxisAnnotation restricted to `keep_labels`
        (order follows the order in `keep_labels`).

        Raises
        ------
        ValueError if any requested label is missing.
        """
        # 1) validate + map to indices, preserving requested order
        label_set = set(self.labels)
        missing = [lab for lab in keep_labels if lab not in label_set]
        if missing:
            raise ValueError(f"Unknown labels for subset: {missing}")

        idxs = [self.get_index(lab) for lab in keep_labels]

        # 2) slice labels / states / cardinalities
        new_labels = [self.labels[i] for i in idxs]

        if self.states is not None:
            new_states = [self.states[i] for i in idxs]
            new_cards = [len(s) for s in new_states]
        else:
            new_states = None
            new_cards = None

        # 3) slice metadata (if present)
        new_metadata = None
        if self.metadata is not None:
            new_metadata = {lab: self.metadata[lab] for lab in keep_labels}

        # 4) build a fresh object
        return AxisAnnotation(
            labels=new_labels,
            states=new_states,
            cardinalities=new_cards,
            metadata=new_metadata,
        )

    # --- AxisAnnotation: add a tiny union helper (non-nested kept non-nested) ---
    def union_with(self, other: "AxisAnnotation") -> "AxisAnnotation":
        left = list(self.labels)
        right_only = [l for l in other.labels if l not in set(left)]
        labels = left + right_only
        # keep it simple: stay non-nested; merge metadata left-win
        meta = None
        if self.metadata or other.metadata:
            meta = {}
            if self.metadata: meta.update(self.metadata)
            if other.metadata:
                for k, v in other.metadata.items():
                    if k not in meta:
                        meta[k] = v
        return AxisAnnotation(labels=labels, states=None, cardinalities=None, metadata=meta)


class Annotations:
    """
    Multi-axis annotation container for concept tensors.

    This class manages annotations for multiple tensor dimensions, providing
    a unified interface for working with concept-based tensors that may have
    different semantic meanings along different axes.

    Attributes:
        _axis_annotations (Dict[int, AxisAnnotation]): Map from axis index to annotation.

    Args:
        axis_annotations: Either a list of AxisAnnotations (indexed 0, 1, 2, ...)
                         or a dict mapping axis numbers to AxisAnnotations.

    Example:
        >>> from torch_concepts import Annotations, AxisAnnotation
        >>>
        >>> # Create annotations for a concept tensor
        >>> # Axis 0: batch (typically not annotated)
        >>> # Axis 1: concepts
        >>> concept_ann = AxisAnnotation(
        ...     labels=['color', 'shape', 'size'],
        ...     cardinalities=[3, 2, 1]  # 3 colors, 2 shapes, 1 binary size
        ... )
        >>>
        >>> # Create annotations object
        >>> annotations = Annotations({1: concept_ann})
        >>>
        >>> # Access concept labels
        >>> print(annotations.get_axis_labels(1))  # ['color', 'shape', 'size']
        >>>
        >>> # Get index of a concept
        >>> idx = annotations.get_index(1, 'color')
        >>> print(idx)  # 0
        >>>
        >>> # Check if axis is nested
        >>> print(annotations.is_axis_nested(1))  # True
        >>>
        >>> # Get cardinalities
        >>> print(annotations.get_axis_cardinalities(1))  # [3, 2, 1]
        >>>
        >>> # Access via indexing
        >>> print(annotations[1].labels)  # ['color', 'shape', 'size']
        >>>
        >>> # Multiple axes example
        >>> task_ann = AxisAnnotation(labels=['task1', 'task2', 'task3'])
        >>> multi_ann = Annotations({
        ...     1: concept_ann,
        ...     2: task_ann
        ... })
        >>> print(multi_ann.annotated_axes)  # (1, 2)
    """

    def __init__(self, axis_annotations: Optional[Union[List, Dict[int, AxisAnnotation]]] = None):
        """
        Initialize Annotations container.

        Args:
            axis_annotations: Either a list or dict of AxisAnnotation objects.
        """

        if axis_annotations is None:
            self._axis_annotations = {}
        else:
            if isinstance(axis_annotations, list):
                # assume list corresponds to axes 0, 1, 2, ...
                self._axis_annotations = {}
                for axis, ann in enumerate(axis_annotations):
                    assert axis >= 0, "Axis must be non-negative"
                    self._axis_annotations[axis] = ann
            else:
                # Validate that axis numbers in annotations match dict keys
                self._axis_annotations = deepcopy(axis_annotations)

    def annotate_axis(self, axis_annotation: AxisAnnotation, axis: int) -> None:
        """
        Add or update annotation for an axis.
        """
        assert axis >= 0, "Axis must be non-negative"
        self._axis_annotations[axis] = axis_annotation

    # ------------------------------ Introspection ------------------------------ #
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get shape of the annotated tensor based on annotations."""
        shape = []
        max_axis = max(self._axis_annotations.keys(), default=-1)
        for axis in range(max_axis + 1):
            if axis in self._axis_annotations:
                shape.append(self._axis_annotations[axis].shape)
            else:
                shape.append(-1)  # Unknown size for unannotated axes
        return tuple(shape)

    @property
    def num_annotated_axes(self) -> int:
        """Number of annotated axes."""
        return len(self._axis_annotations)

    @property
    def annotated_axes(self) -> Tuple[int, ...]:
        """Tuple of annotated axis numbers (sorted)."""
        return tuple(sorted(self._axis_annotations.keys()))

    def has_axis(self, axis: int) -> bool:
        """Check if an axis is annotated."""
        return axis in self._axis_annotations

    def get_axis_annotation(self, axis: int) -> AxisAnnotation:
        """Get annotation for a specific axis."""
        if axis not in self._axis_annotations:
            raise ValueError(f"Axis {axis} is not annotated")
        return self._axis_annotations[axis]

    def get_axis_labels(self, axis: int) -> List[str]:
        """Get ordered labels for an axis."""
        return self.get_axis_annotation(axis).labels

    def get_axis_cardinalities(self, axis: int) -> Optional[List[int]]:
        """Get cardinalities for an axis (if nested), or None."""
        return self.get_axis_annotation(axis).cardinalities

    def is_axis_nested(self, axis: int) -> bool:
        """Check if an axis has nested structure."""
        return self.get_axis_annotation(axis).is_nested

    def get_index(self, axis: int, label: str) -> int:
        """Get index of a label within an axis."""
        return self.get_axis_annotation(axis).get_index(label)

    def get_label(self, axis: int, idx: int) -> str:
        """Get label at index within an axis."""
        return self.get_axis_annotation(axis).get_label(idx)

    def get_states(self, axis: int) -> Optional[List[List[str]]]:
        """Get states for a nested axis, or None."""
        return self.get_axis_annotation(axis).states

    def get_label_states(self, axis: int, label: str) -> List[str]:
        """Get states of a concept in a nested axis."""
        ann = self.get_axis_annotation(axis)
        if ann.states is None:
            raise ValueError(f"Axis {axis} has no states defined")
        idx = ann.get_index(label)
        return ann.states[idx]

    def get_label_state(self, axis: int, label: str, idx: int) -> str:
        """Get states of a concept in a nested axis."""
        ann = self.get_axis_annotation(axis)
        if ann.states is None:
            raise ValueError(f"Axis {axis} has no states defined")
        idx_label = ann.get_index(label)
        state = ann.states[idx_label][idx]
        return state

    def get_state_index(self, axis: int, label: str, state: str) -> int:
        """Get index of a state label for a concept in a nested axis."""
        ann = self.get_axis_annotation(axis)
        if ann.states is None:
            raise ValueError(f"Axis {axis} has no states defined")
        idx_label = ann.get_index(label)
        try:
            return ann.states[idx_label].index(state)
        except ValueError:
            raise ValueError(f"State {state!r} not found for concept {label!r} in axis {axis}")

    def __getitem__(self, axis: int) -> AxisAnnotation:
        """
        Get annotations for an axis (list-like indexing).
        ann[0] returns AxisAnnotation for axis 0
        ann[0][2] returns label at index 2 of axis 0
        ann[1][2][0] returns first state of concept at index 2 of axis 1
        """
        return self.get_axis_annotation(axis)

    def __setitem__(self, axis: int, annotation: AxisAnnotation) -> None:
        """Set annotation for an axis."""
        self.annotate_axis(annotation, axis)

    def __delitem__(self, axis: int) -> None:
        """Remove annotation for an axis."""
        if axis not in self._axis_annotations:
            raise KeyError(f"Axis {axis} is not annotated")
        del self._axis_annotations[axis]

    def __contains__(self, axis: int) -> bool:
        """Check if an axis is annotated."""
        return axis in self._axis_annotations

    def __len__(self) -> int:
        """Return number of annotated axes."""
        return len(self._axis_annotations)

    def __iter__(self):
        """Iterate over axis numbers."""
        return iter(self._axis_annotations)

    def keys(self):
        """Return axis numbers (dict-like interface)."""
        return self._axis_annotations.keys()

    def values(self):
        """Return AxisAnnotation objects (dict-like interface)."""
        return self._axis_annotations.values()

    def items(self):
        """Return (axis, AxisAnnotation) pairs (dict-like interface)."""
        return self._axis_annotations.items()

    @property
    def axis_annotations(self) -> Dict[int, AxisAnnotation]:
        """Access to the underlying axis annotations dictionary."""
        return self._axis_annotations

    def __repr__(self) -> str:
        """String representation."""
        if not self._axis_annotations:
            return "Annotations({})"

        parts = []
        for axis in sorted(self._axis_annotations.keys()):
            ann = self._axis_annotations[axis]
            if ann.is_nested:
                parts.append(f"axis{axis}={ann.labels} (nested, cards={ann.cardinalities})")
            else:
                parts.append(f"axis{axis}={ann.labels}")
        return f"Annotations({', '.join(parts)})"

    def select(self, axis: int, keep_labels: Sequence[str]) -> "Annotations":
        """
        Return a new Annotations where only `keep_labels` are kept on `axis`.
        Other axes are unchanged.
        """
        if axis not in self._axis_annotations:
            raise ValueError(f"Axis {axis} is not annotated")

        new_map = deepcopy(self._axis_annotations)
        new_map[axis] = new_map[axis].subset(keep_labels)
        return Annotations(new_map)

    def select_many(self, labels_by_axis: Dict[int, Sequence[str]]) -> "Annotations":
        """
        Return a new Annotations applying independent label filters per axis.
        """
        new_map = deepcopy(self._axis_annotations)
        for ax, labs in labels_by_axis.items():
            if ax not in new_map:
                raise ValueError(f"Axis {ax} is not annotated")
            new_map[ax] = new_map[ax].subset(labs)
        return Annotations(new_map)

    # --- Annotations: union join that allows overlapping labels on the join axis ---
    def join_union(self, other: "Annotations", axis: int) -> "Annotations":
        if axis not in self._axis_annotations or axis not in other._axis_annotations:
            raise ValueError(f"Both annotations must include axis {axis} to join")

        # non-join axes must match exactly
        all_axes = set(self._axis_annotations.keys()).union(other._axis_annotations.keys())
        for ax in all_axes:
            if ax == axis:
                continue
            if ax not in self._axis_annotations or ax not in other._axis_annotations:
                raise ValueError(f"Axis {ax} missing on one side while joining on axis {axis}")
            if self._axis_annotations[ax].to_dict() != other._axis_annotations[ax].to_dict():
                raise ValueError(f"Non-join axis {ax} differs between annotations")

        joined = deepcopy(self._axis_annotations)
        joined[axis] = self._axis_annotations[axis].union_with(other._axis_annotations[axis])
        return Annotations(joined)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary with axis annotations.
        """
        return {
            'axis_annotations': {
                str(axis): ann.to_dict() for axis, ann in self._axis_annotations.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Annotations':
        """
        Create Annotations from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with serialized Annotations data.

        Returns
        -------
        Annotations
            Reconstructed Annotations object.
        """
        axis_annotations = {}
        if 'axis_annotations' in data:
            for axis_str, ann_data in data['axis_annotations'].items():
                axis = int(axis_str)
                axis_annotations[axis] = AxisAnnotation.from_dict(ann_data)
        return cls(axis_annotations=axis_annotations)
