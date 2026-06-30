"""
Concept annotations for tensors.

This module provides annotation structures for concept-based tensors, allowing
semantic labeling of a tensor's concept axis and its components. It supports both
simple (flat) and nested (hierarchical) concept structures.
"""

import warnings
import torch

from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Tuple, Union, Optional, Any, Sequence


#: The canonical concept-type vocabulary. A concept is exactly one of these.
_CONCEPT_TYPES = ('binary', 'categorical', 'continuous')


@dataclass(frozen=True)
class Concept:
    """Read-only, per-concept view over a single column of an :class:`Annotations`.

    Groups one concept's properties into a single named object so callers can write
    ``annotations.concept('color').cardinality`` instead of the index-dance
    ``int(annotations.cardinalities[annotations.get_index('color')])``. It is a *view*:
    the values are read from the owning ``Annotations``' parallel lists at construction
    time (no duplicated storage), so the lists remain the canonical representation.

    Attributes:
        name (str): Concept label.
        index (int): Concept-level index within the axis.
        cardinality (int): Number of states (1 for binary/continuous scalars).
        type (str): Concept type, one of ``'binary'`` / ``'categorical'`` / ``'continuous'``.
        states (Optional[List[str]]): State labels for this concept.
        slice (slice): Column span of this concept in the flattened (logit) tensor.
        metadata (dict): Raw per-concept metadata (escape hatch for extra keys).
    """
    name: str
    index: int
    cardinality: int
    type: str
    states: Optional[List[str]]
    slice: slice
    metadata: dict = field(default_factory=dict)

    @property
    def is_continuous(self) -> bool:
        """Whether this concept is continuous."""
        return self.type == 'continuous'

    @property
    def is_binary(self) -> bool:
        """Whether this concept is a single binary value (cardinality 1)."""
        return self.type == 'binary'

    @property
    def is_categorical(self) -> bool:
        """Whether this concept is a multi-state (categorical) variable."""
        return self.type == 'categorical'


@dataclass
class Annotations:
    """
    Annotations for the concept axis of a tensor.

    This class provides semantic labeling for the concept dimension (axis 1) of a
    tensor, supporting both simple binary concepts and nested multi-state concepts.
    Axis 0 is the (unannotated) batch dimension.

    Attributes:
        labels (list[str]): Ordered, unique concept labels.
        states (Optional[list[list[str]]]): State labels for each concept (if nested).
        cardinalities (Optional[list[int]]): Cardinality of each concept.
        types (Optional[list[str]]): ``'binary'`` / ``'categorical'`` / ``'continuous'`` per concept.
        metadata (Optional[Dict[str, Dict]]): Additional metadata for each label.
        is_nested (bool): Whether the axis has nested/hierarchical structure.

    Args:
        labels: List of concept names.
        states: Optional list of state lists for nested concepts.
        cardinalities: Optional list of cardinalities per concept.
        types: Optional concept types per concept.
        metadata: Optional metadata dictionary keyed by label names.

    Example:
        >>> from torch_concepts import Annotations
        >>>
        >>> # Simple binary concepts
        >>> ann_binary = Annotations(
        ...     labels=['has_wheels', 'has_windows', 'is_red']
        ... )
        >>> print(ann_binary.labels)
        ['has_wheels', 'has_windows', 'is_red']
        >>> print(ann_binary.is_nested)
        False
        >>> print(ann_binary.cardinalities)
        [1, 1, 1]
        >>> print(ann_binary.shape)
        (-1, 3)
        >>>
        >>> # Nested concepts with explicit states
        >>> ann_nested = Annotations(
        ...     labels=['color', 'shape'],
        ...     states=[['red', 'green', 'blue'], ['circle', 'square']],
        ... )
        >>> print(ann_nested.labels)
        ['color', 'shape']
        >>> print(ann_nested.is_nested)
        True
        >>> print(ann_nested.cardinalities)
        [3, 2]
        >>> print(ann_nested.states[0])
        ['red', 'green', 'blue']
        >>> print(ann_nested.shape)
        (-1, 5)
        >>>
        >>> # With cardinalities only (auto-generates state labels)
        >>> ann_cards = Annotations(
        ...     labels=['size', 'material'],
        ...     cardinalities=[3, 4]
        ... )
        >>> print(ann_cards.cardinalities)
        [3, 4]
        >>> print(ann_cards.states[0])
        ['0', '1', '2']
        >>>
        >>> # Access methods
        >>> idx = ann_binary.get_index('has_wheels')
        >>> print(idx)
        0
        >>> label = ann_binary.get_label(1)
        >>> print(label)
        has_windows
    """
    labels: List[str]
    states: Optional[List[List[str]]] = field(default=None)
    cardinalities: Optional[List[int]] = field(default=None)
    types: Optional[List[str]] = field(default=None)  # 'binary' | 'categorical' | 'continuous'
    metadata: Optional[Dict[str, Dict]] = field(default=None)
    # Concept-space annotation: each concept occupies a single integer-coded
    # column regardless of its type (so all cardinalities are 1). This describes
    # a ground-truth concept tensor, not the model's logit space. When True the
    # ``categorical requires cardinality > 1`` invariant is relaxed (a categorical
    # concept's column holds an integer class index). Build one from a normal
    # (logit-space) annotation via :meth:`to_concept_space`.
    concept_space: bool = field(default=False)

    def __setattr__(self, key, value):
        # `metadata` may change after construction, so it is
        # freely reassignable. The structural fields (labels, states,
        # cardinalities, types) remain write-once.
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

        # Validate optional per-concept lists line up with the labels.
        if self.types is not None and len(self.types) != len(self.labels):
            raise ValueError(
                f"Number of types ({len(self.types)}) must match "
                f"number of labels ({len(self.labels)})"
            )

        # Canonicalise the concept type. It is one of 'binary' / 'categorical' /
        # 'continuous'. When omitted, default discrete concepts from cardinality
        # (binary if card==1 else categorical); 'continuous' must be declared
        # explicitly (it cannot be inferred). Then enforce the type<->cardinality
        # invariant so the two never drift.
        if self.types is None:
            resolved_types = [
                'binary' if card == 1 else 'categorical' for card in self.cardinalities
            ]
        else:
            resolved_types = list(self.types)
        for label, t, card in zip(self.labels, resolved_types, self.cardinalities):
            if t not in _CONCEPT_TYPES:
                raise ValueError(
                    f"Concept {label!r}: type must be one of {_CONCEPT_TYPES}, got {t!r}."
                )
            if t == 'binary' and card != 1:
                raise ValueError(
                    f"Concept {label!r}: 'binary' requires cardinality 1, got {card}."
                )
            # In concept-space every concept is a single integer-coded column, so a
            # categorical concept legitimately has cardinality 1 (the column holds a
            # class index); only enforce the >1 invariant in logit-space annotations.
            if t == 'categorical' and card <= 1 and not self.concept_space:
                raise ValueError(
                    f"Concept {label!r}: 'categorical' requires cardinality > 1, got {card}."
                )
        object.__setattr__(self, 'types', resolved_types)

        # Determine is_nested from cardinalities
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
    def size(self) -> int:
        """Flattened concept dimension: ``sum(cardinalities)``.

        Equals ``len(labels)`` when non-nested (all cardinalities are 1). This is the
        size of axis 1 of the annotated (logit-space) tensor.
        """
        return sum(self.cardinalities)

    @property
    def shape(self) -> Tuple[int, int]:
        """Annotated tensor shape ``(B, sum(cardinalities))``.

        Axis 0 is the unknown batch dimension B, returned as ``-1``; axis 1 is the
        flattened concept dimension :attr:`size`.
        """
        return (-1, self.size)

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
        """Return number of labels."""
        return len(self.labels)

    def __getitem__(self, key: Union[int, str]) -> Union[str, "Concept"]:
        """
        Index by position or by name.
        - ``annotations[int]`` returns the label at that index (``str``).
        - ``annotations[str]`` returns the :class:`Concept` view for that label.
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
            >>> ann = Annotations(labels=['color', 'shape', 'size'])
            >>> ann.label_to_index['shape']
            1
        """
        return {name: i for i, name in enumerate(self.labels)}

    def get_index(self, label: str) -> int:
        """Get index of a label."""
        try:
            return self.label_to_index[label]
        except KeyError:
            raise ValueError(f"Label {label!r} not found in labels {self.labels}")

    def get_label(self, idx: int) -> str:
        """Get label at given index."""
        if not (0 <= idx < len(self.labels)):
            raise IndexError(f"Index {idx} out of range with {len(self.labels)} labels")
        return self.labels[idx]

    def get_total_cardinality(self) -> Optional[int]:
        """Get total cardinality for nested annotations, or number of labels otherwise."""
        if self.is_nested:
            if self.cardinalities is not None:
                return sum(self.cardinalities)
            else:
                raise ValueError("Cardinalities are not defined for this nested annotation")
        else:
            return len(self.labels)

    # =========================================================================
    # State navigation
    # =========================================================================

    def get_label_states(self, label: str) -> List[str]:
        """Get the ordered state labels of a concept."""
        return self.states[self.get_index(label)]

    def get_label_state(self, label: str, idx: int) -> str:
        """Get the state label at position ``idx`` of a concept."""
        return self.states[self.get_index(label)][idx]

    def get_state_index(self, label: str, state: str) -> int:
        """Get the index of a state label for a concept."""
        states = self.states[self.get_index(label)]
        try:
            return states.index(state)
        except ValueError:
            raise ValueError(f"State {state!r} not found for concept {label!r}")

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
            >>> ann = Annotations(labels=['color', 'shape', 'size'], cardinalities=[3, 2, 1])
            >>> ann.cumulative_cardinalities
            [0, 3, 5, 6]
        """
        cum = [0]
        for c in self.cardinalities:
            cum.append(cum[-1] + c)
        return cum

    @cached_property
    def concept_slices(self) -> Dict[str, slice]:
        """Precomputed mapping from concept name to slice in flattened tensor.

        Example:
            >>> ann = Annotations(labels=['color', 'shape', 'size'], cardinalities=[3, 2, 1])
            >>> ann.concept_slices['color']
            slice(0, 3, None)
        """
        cum = self.cumulative_cardinalities
        return {name: slice(cum[i], cum[i+1])
                for i, name in enumerate(self.labels)}

    @cached_property
    def labels_by_type(self) -> Dict[str, List[str]]:
        """Mapping from concept type to the ordered list of labels of that type.

        Only non-empty types are included. Derived from :attr:`type_groups` so the
        grouping logic lives in one place.

        Example:
            >>> ann = Annotations(
            ...     labels=['a', 'b', 'c'],
            ...     cardinalities=[1, 1, 3],
            ... )
            >>> ann.labels_by_type
            {'binary': ['a', 'b'], 'categorical': ['c']}
        """
        return {t: g['labels'] for t, g in self.type_groups.items() if g['labels']}

    @cached_property
    def type_groups(self) -> Dict[str, Dict[str, List]]:
        """Precomputed type-based groupings at both concept and logit levels.

        Returns a dict with keys 'binary', 'categorical', 'continuous', each
        containing:
            - 'labels': list of concept names
            - 'concept_idx': list of concept-level indices
            - 'logits_idx': list of logit-level indices

        Example:
            >>> ann = Annotations(
            ...     labels=['size', 'color', 'temp'],
            ...     cardinalities=[1, 3, 1],
            ...     types=['binary', 'categorical', 'continuous'],
            ... )
            >>> ann.type_groups['binary']['labels']
            ['size']
            >>> ann.type_groups['categorical']['logits_idx']
            [1, 2, 3]
        """
        cum = self.cumulative_cardinalities

        groups = {t: {'labels': [], 'concept_idx': [], 'logits_idx': []}
                  for t in _CONCEPT_TYPES}

        for i, label in enumerate(self.labels):
            group = groups[self.types[i]]
            group['labels'].append(label)
            group['concept_idx'].append(i)
            group['logits_idx'].extend(range(cum[i], cum[i + 1]))

        return groups

    def concept(self, name: str) -> "Concept":
        """Return a read-only :class:`Concept` view for ``name``.

        Groups the concept's per-column properties (cardinality, type, states, logit slice)
        into one object, so callers can write ``annotations.concept('color').cardinality``
        instead of the index-dance over the parallel lists.
        Built fresh on each call so it always reflects the current (mutable) ``metadata``.
        """
        i = self.get_index(name)
        meta = (self.metadata.get(name, {}) if self.metadata else {}) or {}
        return Concept(
            name=name,
            index=i,
            cardinality=int(self.cardinalities[i]),
            type=self.types[i],
            states=self.states[i] if self.states is not None else None,
            slice=self.concept_slices[name],
            metadata=meta,
        )

    @property
    def concepts(self) -> List["Concept"]:
        """All concepts as :class:`Concept` views, in axis order.

        Views are read from the canonical parallel lists (no duplicated storage);
        useful for one-pass iteration. Not cached, so it reflects the
        current (mutable) ``metadata``.
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
            >>> import torch
            >>> ann = Annotations(labels=['color', 'shape'], cardinalities=[3, 2])
            >>> predictions = torch.rand(4, 5)
            >>> reordered = ann.slice_tensor(predictions, ann.labels)
            >>> reordered.shape
            torch.Size([4, 5])
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
            >>> ann = Annotations(
            ...     labels=['color', 'shape', 'size'],
            ...     cardinalities=[3, 2, 1]
            ... )
            >>> # Single concept → slice
            >>> ann.get_slice('color')
            slice(0, 3, None)
            >>> # Multiple concepts → flattened indices
            >>> ann.get_slice(['color', 'size'])
            [0, 1, 2, 5]
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
    ) -> "Annotations":
        """Create an Annotations with *n* anonymous binary labels ``c_0 … c_{n-1}``.

        Args:
            n: Number of labels.

        Returns:
            A new :class:`Annotations` with labels ``['c_0', 'c_1', 'c_2', 'c_3']``.

        Example:
            >>> ann = Annotations.empty(4)
            >>> ann.labels
            ['c_0', 'c_1', 'c_2', 'c_3']
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
            'metadata': self.metadata,
        }
        return result

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
        # Keep as lists (native format)
        labels = data['labels']
        states = [list(s) for s in data['states']] if data.get('states') else None
        cardinalities = data['cardinalities']

        return cls(
            labels=labels,
            states=states,
            cardinalities=cardinalities,
            types=data.get('types'),
            metadata=data.get('metadata'),
        )

    def subset(self, keep_labels: Sequence[str]) -> "Annotations":
        """
        Return a new Annotations restricted to `keep_labels`
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

        new_types = [self.types[i] for i in idxs]

        # 3) slice metadata (if present)
        new_metadata = None
        if self.metadata is not None:
            new_metadata = {lab: self.metadata[lab] for lab in keep_labels}

        # 4) build a fresh object
        return Annotations(
            labels=new_labels,
            states=new_states,
            cardinalities=new_cards,
            types=new_types,
            metadata=new_metadata,
            concept_space=self.concept_space,
        )

    def to_concept_space(self) -> "Annotations":
        """Return a concept-space view: one integer-coded column per concept.

        Cardinalities collapse to 1 (each concept becomes a single column) while
        labels and types are preserved, so a ground-truth concept tensor of shape
        ``(batch, n_concepts)`` (integer class indices for categorical concepts,
        0/1 for binary) can be wrapped as an :class:`~torch_concepts.tensor.AnnotatedTensor`.
        The result's :attr:`size` equals the number of concepts, and
        label-based slicing / :meth:`labels_by_type` operate per concept.

        Returns ``self`` unchanged if this annotation is already concept-space.
        """
        if self.concept_space:
            return self
        return Annotations(
            labels=list(self.labels),
            cardinalities=[1] * len(self.labels),
            types=list(self.types),
            metadata=self.metadata,
            concept_space=True,
        )

    def union_with(self, other: "Annotations") -> "Annotations":
        left = list(self.labels)
        right_only = [l for l in other.labels if l not in set(left)]
        labels = left + right_only

        def _merge(left_values, right_values):
            """Left values + right-only values (left wins for overlapping labels)."""
            return list(left_values) + [
                right_values[other.labels.index(l)] for l in right_only
            ]

        # ``states`` / ``types`` are always populated after construction, so we
        # carry them through directly (cardinalities re-infer from states). This
        # keeps categorical concepts' cardinalities intact through a union.
        new_states = _merge(self.states, other.states)
        new_types = _merge(self.types, other.types)

        # merge metadata left-wins
        meta = None
        if self.metadata or other.metadata:
            meta = {}
            if self.metadata: meta.update(self.metadata)
            if other.metadata:
                for k, v in other.metadata.items():
                    if k not in meta:
                        meta[k] = v
        return Annotations(
            labels=labels, states=new_states, cardinalities=None,
            types=new_types, metadata=meta,
        )
