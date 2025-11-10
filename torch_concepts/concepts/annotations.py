import warnings
import torch

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional, Any, Sequence


@dataclass
class AxisAnnotation:
    """
    Annotations for a single axis of a tensor.

    Attributes
    ----------
    axis : int
        The tensor dimension this annotates (0 for batch, 1 for concept, etc.)
    labels : tuple[str, ...]
        Ordered, unique labels for this axis
    is_nested : bool
        Whether this axis has nested structure (inferred from states if present)
    cardinalities : Optional[tuple[int, ...]]
        IF NESTED, the cardinality of each component (inferred from states)
    states : Optional[tuple[tuple[str, ...], ...]]
        IF NESTED, state labels for each component. None for non-nested.
    """
    labels: Tuple[str, ...]
    states: Optional[Tuple[Tuple[str, ...], ...]] = field(default=None)
    cardinalities: Optional[Tuple[int, ...]] = field(default=None)
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
        # Case 1: both states and cardinalities are provided
        if self.states is not None and self.cardinalities is not None:
            # Validate states length and cardinality length matches labels length
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
            # check states length matches cardinalities
            inferred_cardinalities = tuple(len(state_tuple) for state_tuple in self.states)
            if self.cardinalities != inferred_cardinalities:
                raise ValueError(
                    f"Provided cardinalities {self.cardinalities} don't match "
                    f"inferred cardinalities {inferred_cardinalities} from states"
                )
            cardinalities = self.cardinalities
            states = self.states

        # Case 2: only states are provided (no cardinalities)
        elif self.states is not None and self.cardinalities is None:
            # Validate states length matches labels length
            if len(self.states) != len(self.labels):
                raise ValueError(
                    f"Number of state tuples ({len(self.states)}) must match "
                    f"number of labels ({len(self.labels)})"
                )
            cardinalities = tuple(len(state_tuple) for state_tuple in self.states)
            states = self.states

        # Case 3: only cardinalities provided (no states)
        elif self.states is None and self.cardinalities is not None:
            # Validate cardinalities length matches labels length
            if len(self.cardinalities) != len(self.labels):
                raise ValueError(
                    f"Number of cardinalities ({len(self.cardinalities)}) must match "
                    f"number of labels ({len(self.labels)})"
                )
            # Generate default state labels '0', '1', '2', etc.
            cardinalities = self.cardinalities
            states = tuple(tuple(str(i) for i in range(card)) if card > 1 else ('0', '1')
                           for card in self.cardinalities)

        # Case 4: neither states nor cardinalities provided
        else:
            warnings.warn("Annotations: neither 'states' nor 'cardinalities' provided; "
                         "assuming all concepts are binary.")
            cardinalities = tuple(1 for _ in self.labels)
            states = tuple(('0', '1') for _ in self.labels)

        # Eventually convert categorical with card=2 to bernoulli (card=1)
        cardinalities = tuple(card if card > 1 else 1 for card in cardinalities)
        # Determine is_nested from cardinalities
        is_nested = any(card > 1 for card in cardinalities)

        object.__setattr__(self, 'cardinalities', cardinalities)
        object.__setattr__(self, 'states', states)
        object.__setattr__(self, 'is_nested', is_nested)

        # consistency checks on metadata
        if self.metadata is not None:
            if not isinstance(self.metadata, dict):
                raise ValueError("metadata must be a dictionary")
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

        if self.is_nested and self.states is not None:
            return self.states[idx]
        else:
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
        # Convert lists back to tuples
        labels = tuple(data['labels'])
        states = tuple(tuple(s) for s in data['states']) if data.get('states') else None
        cardinalities = tuple(data['cardinalities']) if data.get('cardinalities') else None

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
        new_labels = tuple(self.labels[i] for i in idxs)

        if self.states is not None:
            new_states = tuple(self.states[i] for i in idxs)
            new_cards = tuple(len(s) for s in new_states)
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
        left = tuple(self.labels)
        right_only = tuple(l for l in other.labels if l not in set(left))
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
    """

    def __init__(self, axis_annotations: Optional[Union[List, Dict[int, AxisAnnotation]]] = None):
        """
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

    def get_axis_labels(self, axis: int) -> Tuple[str, ...]:
        """Get ordered labels for an axis."""
        return self.get_axis_annotation(axis).labels

    def get_axis_cardinalities(self, axis: int) -> Optional[Tuple[int, ...]]:
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

    def get_states(self, axis: int) -> Optional[Tuple[Tuple[str, ...], ...]]:
        """Get states for a nested axis, or None."""
        return self.get_axis_annotation(axis).states

    def get_label_states(self, axis: int, label: str) -> Tuple[str, ...]:
        """Get states of a concept in a nested axis."""
        ann = self.get_axis_annotation(axis)
        if ann.states is None:
            raise ValueError(f"Axis {axis} has no states defined")
        idx = ann.get_index(label)
        return ann.states[idx]

    def get_label_state(self, axis: int, label: str, idx: int) -> Tuple[str, ...]:
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

    # ---------------------- Backward compatibility ---------------------- #
    # @property
    # def concept_names(self) -> Tuple[str, ...]:
    #     """Get concept names (assumes concept axis = 1). For backward compatibility."""
    #     if 1 not in self._axis_annotations:
    #         raise ValueError("Concept axis (1) is not annotated")
    #     return self.labels_for_axis(1)

    def __getitem__(self, axis: int) -> AxisAnnotation:
        """
        Get annotations for an axis (list-like indexing).
        ann[0] returns AxisAnnotation for axis 0
        ann[0][2] returns label at index 2 of axis 0
        ann[1][2][0] returns first state of concept at index 2 of axis 1
        """
        return self.get_axis_annotation(axis)

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

