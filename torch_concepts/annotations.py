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
from functools import cached_property
from typing import Dict, List, Tuple, Type, Union, Optional, Any, Sequence


@dataclass(frozen=True)
class Concept:
    """Read-only, per-concept view over a single column of an :class:`AxisAnnotation`.

    Groups one concept's properties into a single named object so callers can write
    ``axis.concept('color').cardinality`` instead of the index-dance
    ``int(axis.cardinalities[axis.get_index('color')])``. It is a *view*: the values
    are read from the owning ``AxisAnnotation``'s parallel lists at construction
    time (no duplicated storage), so the lists remain the canonical representation.

    Attributes:
        name (str): Concept label.
        index (int): Concept-level index within the axis.
        cardinality (int): Number of states (1 for binary/continuous scalars).
        type (Optional[str]): Concept type, e.g. ``'discrete'`` / ``'continuous'``.
        distribution (Optional[type]): Distribution class (e.g. ``Bernoulli``), or
            ``None`` if not yet assigned.
        dist_kwargs (Optional[dict]): Distribution kwargs (still read from metadata).
        states (Optional[List[str]]): State labels for this concept.
        slice (slice): Column span of this concept in the flattened (logit) tensor.
        metadata (dict): Raw per-concept metadata (escape hatch for extra keys).
    """
    name: str
    index: int
    cardinality: int
    type: Optional[str]
    distribution: Optional[Type]
    dist_kwargs: Optional[dict]
    states: Optional[List[str]]
    slice: slice
    metadata: dict = field(default_factory=dict)

    @property
    def is_continuous(self) -> bool:
        """Whether this concept is continuous (``type == 'continuous'``)."""
        return self.type == 'continuous'

    @property
    def is_binary(self) -> bool:
        """Whether this concept is a single binary value (discrete, cardinality 1)."""
        return not self.is_continuous and self.cardinality == 1

    @property
    def is_categorical(self) -> bool:
        """Whether this concept is a multi-state discrete (categorical) variable."""
        return not self.is_continuous and self.cardinality > 1


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
    types: Optional[List[str]] = field(default=None)  # e.g., 'discrete' or 'continuous' # TODO: make consistent
    distributions: Optional[List[Type]] = field(default=None)  # distribution class per concept
    metadata: Optional[Dict[str, Dict]] = field(default=None)

    def __setattr__(self, key, value):
        # `metadata` and `distributions` may change after construction — a
        # concept's distribution can be swapped between experiments — so they are
        # freely reassignable. The structural fields (labels, states,
        # cardinalities, types) remain write-once.
        if key in ('metadata', 'distributions'):
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

        # Validate optional per-concept lists line up with the labels.
        if self.types is not None and len(self.types) != len(self.labels):
            raise ValueError(
                f"Number of types ({len(self.types)}) must match "
                f"number of labels ({len(self.labels)})"
            )
        if self.distributions is not None and len(self.distributions) != len(self.labels):
            raise ValueError(
                f"Number of distributions ({len(self.distributions)}) must match "
                f"number of labels ({len(self.labels)})"
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

    def __getitem__(self, key: Union[int, str]) -> Union[str, "Concept"]:
        """
        Index by position or by name.
        - ``axis[int]`` returns the label at that index (``str``).
        - ``axis[str]`` returns the :class:`Concept` view for that label.
        """
        if isinstance(key, str):
            return self.concept(key)
        if not (0 <= key < len(self.labels)):
            raise IndexError(f"Index {key} out of range")
        return self.labels[key]

    @cached_property
    def label_to_index(self) -> Dict[str, int]:
        """Precomputed mapping from concept name to concept-level index.
        
        Provides O(1) lookup for concept indices, useful for efficient
        concept extraction operations.
        
        Example:
            >>> axis = AxisAnnotation(labels=['color', 'shape', 'size'])
            >>> axis.label_to_index['shape']
            1
        """
        return {name: i for i, name in enumerate(self.labels)}
    
    def get_index(self, label: str) -> int:
        """Get index of a label in this axis."""
        try:
            return self.label_to_index[label]
        except KeyError:
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

    # =========================================================================
    # Cached index properties for efficient tensor slicing
    # =========================================================================
    
    @cached_property
    def cumulative_cardinalities(self) -> List[int]:
        """Precomputed cumulative cardinalities for O(1) slicing.
        
        Returns a list where cumulative_cardinalities[i] is the starting
        position of concept i in the flattened tensor, and 
        cumulative_cardinalities[i+1] is the ending position (exclusive).
        
        Example:
            >>> axis = AxisAnnotation(labels=['color', 'shape', 'size'], cardinalities=[3, 2, 1])
            >>> axis.cumulative_cardinalities
            [0, 3, 5, 6]  # color: 0-3, shape: 3-5, size: 5-6
        """
        cum = [0]
        for c in self.cardinalities:
            cum.append(cum[-1] + c)
        return cum
    
    @cached_property
    def concept_slices(self) -> Dict[str, slice]:
        """Precomputed mapping from concept name to slice in flattened tensor.
        
        Example:
            >>> axis = AxisAnnotation(labels=['color', 'shape', 'size'], cardinalities=[3, 2, 1])
            >>> axis.concept_slices['color']
            slice(0, 3)
            >>> tensor[:, axis.concept_slices['shape']]  # Get shape logits
        """
        cum = self.cumulative_cardinalities
        return {name: slice(cum[i], cum[i+1]) 
                for i, name in enumerate(self.labels)}
    
    @cached_property
    def labels_by_type(self) -> Dict[str, List[str]]:
        """Precomputed mapping from type name to the ordered list of labels of that type.

        Returns ``{}`` when ``types`` is ``None``.

        Example:
            >>> axis = AxisAnnotation(
            ...     labels=['a', 'b', 'c'],
            ...     types=['discrete', 'continuous', 'discrete'],
            ... )
            >>> axis.labels_by_type
            {'discrete': ['a', 'c'], 'continuous': ['b']}
        """
        if self.types is None:
            return {}
        groups: Dict[str, List[str]] = {}
        for label, t in zip(self.labels, self.types):
            groups.setdefault(t, []).append(label)
        return groups

    @cached_property
    def type_groups(self) -> Dict[str, Dict[str, List]]:
        """Precomputed type-based groupings at both concept and logit levels.
        
        Returns a dict with keys 'binary', 'categorical', 'continuous', each
        containing:
            - 'labels': list of concept names
            - 'concept_idx': list of concept-level indices
            - 'logits_idx': list of logit-level indices
        
        Example:
            >>> axis = AxisAnnotation(
            ...     labels=['size', 'color', 'temp'],
            ...     cardinalities=[1, 3, 1],
            ...     metadata={
            ...         'size': {'type': 'discrete'},
            ...         'color': {'type': 'discrete'},
            ...         'temp': {'type': 'continuous'}
            ...     }
            ... )
            >>> axis.type_groups['binary']['labels']  # ['size']
            >>> axis.type_groups['categorical']['logits_idx']  # [1, 2, 3]
        """
        cum = self.cumulative_cardinalities
        
        groups = {
            'binary': {'labels': [], 'concept_idx': [], 'logits_idx': []},
            'categorical': {'labels': [], 'concept_idx': [], 'logits_idx': []},
            'continuous': {'labels': [], 'concept_idx': [], 'logits_idx': []},
        }
        
        for i, label in enumerate(self.labels):
            card = self.cardinalities[i]
            concept_type = self._type_of(i)

            # Classify into binary/categorical/continuous
            if concept_type == 'continuous':
                group_key = 'continuous'
            elif card == 1:
                group_key = 'binary'
            else:  # discrete with card > 1
                group_key = 'categorical'

            # Store at concept level
            groups[group_key]['labels'].append(label)
            groups[group_key]['concept_idx'].append(i)

            # Store at logit level (all positions for this concept)
            groups[group_key]['logits_idx'].extend(range(cum[i], cum[i+1]))

        return groups

    def _type_of(self, index: int) -> str:
        """Resolve a concept's type by index.

        Prefers the first-class ``types`` field, falls back to
        ``metadata[label]['type']``, then defaults to ``'discrete'``.
        """
        if self.types is not None:
            return self.types[index]
        label = self.labels[index]
        if self.metadata and label in self.metadata:
            return self.metadata[label].get('type', 'discrete')
        return 'discrete'

    def concept(self, name: str) -> "Concept":
        """Return a read-only :class:`Concept` view for ``name``.

        Groups the concept's per-column properties (cardinality, type,
        distribution, states, logit slice) into one object, so callers can write
        ``axis.concept('color').cardinality`` instead of the index-dance over the
        parallel lists. Built fresh on each call so it always reflects the current
        (mutable) ``distributions`` / ``metadata``.
        """
        i = self.get_index(name)
        meta = (self.metadata.get(name, {}) if self.metadata else {}) or {}
        return Concept(
            name=name,
            index=i,
            cardinality=int(self.cardinalities[i]),
            type=self._type_of(i),
            distribution=self.distributions[i] if self.distributions is not None else None,
            dist_kwargs=meta.get('dist_kwargs'),
            states=self.states[i] if self.states is not None else None,
            slice=self.concept_slices[name],
            metadata=meta,
        )

    @property
    def concepts(self) -> List["Concept"]:
        """All concepts as :class:`Concept` views, in axis order.

        Views are read from the canonical parallel lists (no duplicated storage);
        useful for one-pass iteration, e.g.
        ``[c.distribution for c in axis.concepts]``. Not cached, so it reflects the
        current (mutable) ``distributions`` / ``metadata``.
        """
        return [self.concept(name) for name in self.labels]

    def slice_tensor(self, tensor: torch.Tensor, concepts: List[str]) -> torch.Tensor:
        """Extract and concatenate columns for specified concepts.
        
        Args:
            tensor: Input tensor of shape (batch, total_logits)
            concepts: List of concept names to extract, in desired output order
            
        Returns:
            Tensor with columns for specified concepts concatenated
            
        Example:
            >>> # Reorder from topological to annotation order
            >>> reordered = axis.slice_tensor(predictions, axis.labels)
        """
        pieces = [tensor[:, self.concept_slices[c]] for c in concepts]
        return torch.cat(pieces, dim=1)
    
    def get_slice(self, labels: Union[str, List[str]]) -> Union[slice, List[int]]:
        """Get slice or indices for concept(s) in the flattened tensor.
        
        Unified method for accessing concept positions:
        - Single concept name → returns slice object for tensor indexing
        - List of concept names → returns flattened list of indices
        
        Uses precomputed concept_slices for O(1) per-concept lookup.
        
        Args:
            labels: Single concept name (str) or list of concept names.
            
        Returns:
            - slice: If labels is a single string
            - List[int]: If labels is a list of strings
            
        Raises:
            ValueError: If any label is not found in the axis labels.
            
        Example:
            >>> axis = AxisAnnotation(
            ...     labels=['color', 'shape', 'size'],
            ...     cardinalities=[3, 2, 1]
            ... )
            >>> # Single concept → slice
            >>> axis.get_slice('color')
            slice(0, 3, None)
            >>> tensor[:, axis.get_slice('color')]  # slicing
            
            >>> # Multiple concepts → flattened indices
            >>> axis.get_slice(['color', 'size'])
            [0, 1, 2, 5]  # color takes 0-2, size takes 5
        """
        slices = self.concept_slices  # Use cached property
        
        # Single concept → return slice directly
        if isinstance(labels, str):
            if labels not in slices:
                raise ValueError(f"Label '{labels}' not found in axis labels {self.labels}")
            return slices[labels]
        
        # Multiple concepts → return flattened indices
        logits_indices = []
        for label in labels:
            if label not in slices:
                raise ValueError(f"Label '{label}' not found in axis labels {self.labels}")
            s = slices[label]
            logits_indices.extend(range(s.start, s.stop))
        
        return logits_indices

    def get_logits_idx(self, labels: List[str]) -> List[int]:
        """Alias for get_slice(labels) when labels is a list.
        
        Deprecated: Use get_slice() instead.
        """
        return self.get_slice(labels)

    @classmethod
    def empty(
            cls,
            n: int,
            cardinalities: Optional[Union[int, List[int]]] = None,
            types: Optional[Union[str, List[str]]] = None
    ) -> "AxisAnnotation":
        """Create an AxisAnnotation with *n* anonymous binary labels ``c_0 … c_{n-1}``.

        Args:
            n: Number of labels.

        Returns:
            A new :class:`AxisAnnotation` with labels ``['c_0', 'c_1', 'c_2', 'c_3']``.

        Example:
            >>> axis = AxisAnnotation.empty(4)
            >>> axis.labels   # ['c_0', 'c_1', 'c_2', 'c_3']
        """
        cardinalities = [cardinalities] * n if isinstance(cardinalities, int) else cardinalities
        types = [types] * n if isinstance(types, str) else types  # broadcast single str to list
        return cls(
            labels=[f"c_{i}" for i in range(n)],
            cardinalities=cardinalities,
            types=types
        )

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
            'types': list(self.types) if self.types else None,
            'distributions': list(self.distributions) if self.distributions else None,
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
            types=data.get('types'),
            distributions=data.get('distributions'),
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

        # 2) slice labels / states / cardinalities / types
        new_labels = [self.labels[i] for i in idxs]

        if self.states is not None:
            new_states = [self.states[i] for i in idxs]
            new_cards = [len(s) for s in new_states]
        else:
            new_states = None
            new_cards = None

        # Materialise the *resolved* type per kept concept (field → metadata →
        # default) so the subset is self-describing even when the source carried
        # types only in metadata.
        new_types = [self._type_of(i) for i in idxs]
        new_distributions = (
            [self.distributions[i] for i in idxs] if self.distributions is not None else None
        )

        # 3) slice metadata (if present)
        new_metadata = None
        if self.metadata is not None:
            new_metadata = {lab: self.metadata[lab] for lab in keep_labels}

        # 4) build a fresh object
        return AxisAnnotation(
            labels=new_labels,
            states=new_states,
            cardinalities=new_cards,
            types=new_types,
            distributions=new_distributions,
            metadata=new_metadata,
        )

    def union_with(self, other: "AxisAnnotation") -> "AxisAnnotation":
        left = list(self.labels)
        right_only = [l for l in other.labels if l not in set(left)]
        labels = left + right_only
        # merge types: left types + right-only types (left-wins for overlap)
        new_types = None
        if self.types is not None or other.types is not None:
            left_types = self.types or ['discrete'] * len(self.labels)
            right_types = other.types or ['discrete'] * len(other.labels)
            right_only_types = [
                right_types[other.labels.index(l)]
                for l in right_only
            ]
            new_types = left_types + right_only_types
        # merge distributions: left + right-only (left-wins for overlap)
        new_distributions = None
        if self.distributions is not None or other.distributions is not None:
            left_dists = self.distributions or [None] * len(self.labels)
            right_dists = other.distributions or [None] * len(other.labels)
            right_only_dists = [
                right_dists[other.labels.index(l)]
                for l in right_only
            ]
            new_distributions = left_dists + right_only_dists
        # merge metadata left-wins
        meta = None
        if self.metadata or other.metadata:
            meta = {}
            if self.metadata: meta.update(self.metadata)
            if other.metadata:
                for k, v in other.metadata.items():
                    if k not in meta:
                        meta[k] = v
        return AxisAnnotation(
            labels=labels, states=None, cardinalities=None,
            types=new_types, distributions=new_distributions, metadata=meta,
        )


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
