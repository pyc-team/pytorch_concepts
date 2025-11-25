"""
Comprehensive tests for torch_concepts.annotations to increase coverage.
"""
import pytest
import torch
from torch_concepts.annotations import AxisAnnotation, Annotations


class TestAxisAnnotationMetadata:
    """Tests for AxisAnnotation metadata functionality."""

    def test_has_metadata_returns_false_when_none(self):
        """Test has_metadata returns False when metadata is None."""
        axis = AxisAnnotation(labels=['a', 'b', 'c'])
        assert not axis.has_metadata('distribution')

    def test_has_metadata_returns_true_when_all_have_key(self):
        """Test has_metadata returns True when all labels have the key."""
        axis = AxisAnnotation(
            labels=['a', 'b'],
            metadata={
                'a': {'distribution': 'Bernoulli'},
                'b': {'distribution': 'Bernoulli'}
            }
        )
        assert axis.has_metadata('distribution')

    def test_has_metadata_returns_false_when_some_missing(self):
        """Test has_metadata returns False when some labels lack the key."""
        axis = AxisAnnotation(
            labels=['a', 'b', 'c'],
            metadata={
                'a': {'distribution': 'Bernoulli'},
                'b': {'distribution': 'Bernoulli'},
                'c': {}  # Missing 'distribution'
            }
        )
        assert not axis.has_metadata('distribution')

    def test_groupby_metadata_with_labels_layout(self):
        """Test groupby_metadata with labels layout."""
        axis = AxisAnnotation(
            labels=['red', 'green', 'blue', 'circle', 'square'],
            metadata={
                'red': {'type': 'color'},
                'green': {'type': 'color'},
                'blue': {'type': 'color'},
                'circle': {'type': 'shape'},
                'square': {'type': 'shape'}
            }
        )

        groups = axis.groupby_metadata('type', layout='labels')
        assert 'color' in groups
        assert 'shape' in groups
        assert set(groups['color']) == {'red', 'green', 'blue'}
        assert set(groups['shape']) == {'circle', 'square'}

    def test_groupby_metadata_with_indices_layout(self):
        """Test groupby_metadata with indices layout."""
        axis = AxisAnnotation(
            labels=['a', 'b', 'c'],
            metadata={
                'a': {'group': 'first'},
                'b': {'group': 'second'},
                'c': {'group': 'first'}
            }
        )

        groups = axis.groupby_metadata('group', layout='indices')
        assert groups['first'] == [0, 2]
        assert groups['second'] == [1]

    def test_groupby_metadata_invalid_layout(self):
        """Test groupby_metadata raises error on invalid layout."""
        axis = AxisAnnotation(
            labels=['a', 'b'],
            metadata={'a': {'type': 'x'}, 'b': {'type': 'x'}}
        )

        with pytest.raises(ValueError, match="Unknown layout"):
            axis.groupby_metadata('type', layout='invalid')

    def test_groupby_metadata_returns_empty_when_none(self):
        """Test groupby_metadata returns empty dict when metadata is None."""
        axis = AxisAnnotation(labels=['a', 'b'])
        groups = axis.groupby_metadata('type')
        assert groups == {}

    def test_groupby_metadata_skips_missing_keys(self):
        """Test groupby_metadata skips labels without the requested key."""
        axis = AxisAnnotation(
            labels=['a', 'b', 'c'],
            metadata={
                'a': {'type': 'x'},
                'b': {},  # Missing 'type'
                'c': {'type': 'y'}
            }
        )

        groups = axis.groupby_metadata('type', layout='labels')
        assert 'x' in groups
        assert 'y' in groups
        assert 'b' not in groups.get('x', [])
        assert 'b' not in groups.get('y', [])


class TestAxisAnnotationCardinalities:
    """Tests for AxisAnnotation cardinality handling."""

    def test_states_infer_cardinalities(self):
        """Test that cardinalities are inferred from states."""
        axis = AxisAnnotation(
            labels=['color', 'size'],
            states=[['red', 'blue'], ['small', 'medium', 'large']]
        )

        assert axis.cardinalities == [2, 3]
        assert axis.is_nested

    def test_cardinalities_generate_states(self):
        """Test that states are generated from cardinalities."""
        axis = AxisAnnotation(
            labels=['a', 'b'],
            cardinalities=[3, 2]
        )

        assert axis.states == [['0', '1', '2'], ['0', '1']]
        assert axis.is_nested

    def test_binary_default_when_neither_provided(self):
        """Test binary assumption when neither states nor cardinalities provided."""
        with pytest.warns(UserWarning, match="assuming all concepts are binary"):
            axis = AxisAnnotation(labels=['a', 'b', 'c'])

        assert axis.cardinalities == [1, 1, 1]
        assert axis.states == [['0'], ['0'], ['0']]
        assert not axis.is_nested

    def test_cardinality_of_one_not_nested(self):
        """Test that cardinality of 1 means not nested."""
        axis = AxisAnnotation(
            labels=['a', 'b'],
            cardinalities=[1, 1]
        )

        assert not axis.is_nested

    def test_mixed_cardinalities_is_nested(self):
        """Test that any cardinality > 1 makes it nested."""
        axis = AxisAnnotation(
            labels=['a', 'b', 'c'],
            cardinalities=[1, 3, 1]
        )

        assert axis.is_nested

    def test_get_total_cardinality_nested(self):
        """Test get_total_cardinality for nested axis."""
        axis = AxisAnnotation(
            labels=['a', 'b'],
            cardinalities=[2, 3]
        )

        assert axis.get_total_cardinality() == 5

    def test_get_total_cardinality_not_nested(self):
        """Test get_total_cardinality for non-nested axis."""
        axis = AxisAnnotation(
            labels=['a', 'b', 'c'],
            cardinalities=[1, 1, 1]
        )

        assert axis.get_total_cardinality() == 3


class TestAxisAnnotationValidation:
    """Tests for AxisAnnotation validation and error handling."""

    def test_mismatched_states_length_raises_error(self):
        """Test that mismatched states length raises ValueError."""
        with pytest.raises(ValueError, match="Number of state tuples"):
            AxisAnnotation(
                labels=['a', 'b'],
                states=[['x', 'y'], ['p', 'q'], ['extra']]  # 3 states for 2 labels
            )

    def test_mismatched_cardinalities_length_raises_error(self):
        """Test that mismatched cardinalities length raises ValueError."""
        with pytest.raises(ValueError, match="Number of state tuples"):
            AxisAnnotation(
                labels=['a', 'b'],
                cardinalities=[2, 3, 4]  # 3 cardinalities for 2 labels
            )

    def test_inconsistent_states_cardinalities_raises_error(self):
        """Test that inconsistent states and cardinalities raises ValueError."""
        with pytest.raises(ValueError, match="don't match inferred cardinalities"):
            AxisAnnotation(
                labels=['a', 'b'],
                states=[['x', 'y'], ['p', 'q', 'r']],  # [2, 3]
                cardinalities=[2, 2]  # Mismatch: should be [2, 3]
            )

    def test_metadata_not_dict_raises_error(self):
        """Test that non-dict metadata raises ValueError."""
        with pytest.raises(ValueError, match="metadata must be a dictionary"):
            AxisAnnotation(
                labels=['a', 'b'],
                metadata=['not', 'a', 'dict']
            )

    def test_metadata_missing_label_raises_error(self):
        """Test that metadata missing a label raises ValueError."""
        with pytest.raises(ValueError, match="Metadata missing for label"):
            AxisAnnotation(
                labels=['a', 'b', 'c'],
                metadata={
                    'a': {},
                    'b': {}
                    # Missing 'c'
                }
            )

    def test_get_index_invalid_label_raises_error(self):
        """Test that get_index with invalid label raises ValueError."""
        axis = AxisAnnotation(labels=['a', 'b', 'c'])

        with pytest.raises(ValueError, match="not found in labels"):
            axis.get_index('invalid')

    def test_get_label_invalid_index_raises_error(self):
        """Test that get_label with invalid index raises IndexError."""
        axis = AxisAnnotation(labels=['a', 'b', 'c'])

        with pytest.raises(IndexError, match="out of range"):
            axis.get_label(10)

    def test_get_label_negative_index_raises_error(self):
        """Test that get_label with negative index raises IndexError."""
        axis = AxisAnnotation(labels=['a', 'b', 'c'])

        with pytest.raises(IndexError, match="out of range"):
            axis.get_label(-1)

    def test_getitem_invalid_index_raises_error(self):
        """Test that __getitem__ with invalid index raises IndexError."""
        axis = AxisAnnotation(labels=['a', 'b'])

        with pytest.raises(IndexError, match="out of range"):
            _ = axis[5]


class TestAxisAnnotationSerialization:
    """Tests for AxisAnnotation serialization."""

    def test_to_dict_simple(self):
        """Test to_dict for simple axis."""
        axis = AxisAnnotation(
            labels=['a', 'b'],
            cardinalities=[1, 1]
        )

        d = axis.to_dict()
        assert d['labels'] == ['a', 'b']
        assert d['cardinalities'] == [1, 1]
        assert d['is_nested'] == False

    def test_to_dict_nested_with_metadata(self):
        """Test to_dict for nested axis with metadata."""
        axis = AxisAnnotation(
            labels=['color', 'size'],
            states=[['red', 'blue'], ['small', 'large']],
            metadata={
                'color': {'type': 'visual'},
                'size': {'type': 'physical'}
            }
        )

        d = axis.to_dict()
        assert d['labels'] == ['color', 'size']
        assert d['states'] == [['red', 'blue'], ['small', 'large']]
        assert d['cardinalities'] == [2, 2]
        assert d['is_nested'] == True
        assert d['metadata'] == {
            'color': {'type': 'visual'},
            'size': {'type': 'physical'}
        }

    def test_from_dict_simple(self):
        """Test from_dict for simple axis."""
        data = {
            'labels': ['a', 'b', 'c'],
            'cardinalities': [1, 1, 1],
            'states': [['0'], ['0'], ['0']],
            'is_nested': False,
            'metadata': None
        }

        axis = AxisAnnotation.from_dict(data)
        assert axis.labels == ['a', 'b', 'c']
        assert axis.cardinalities == [1, 1, 1]
        assert not axis.is_nested

    def test_from_dict_nested(self):
        """Test from_dict for nested axis."""
        data = {
            'labels': ['x', 'y'],
            'cardinalities': [2, 3],
            'states': [['a', 'b'], ['p', 'q', 'r']],
            'is_nested': True,
            'metadata': None
        }

        axis = AxisAnnotation.from_dict(data)
        assert axis.labels == ['x', 'y']
        assert axis.cardinalities == [2, 3]
        assert axis.is_nested
        assert axis.states == [['a', 'b'], ['p', 'q', 'r']]


class TestAxisAnnotationShape:
    """Tests for AxisAnnotation shape property."""

    def test_shape_not_nested(self):
        """Test shape property for non-nested axis."""
        axis = AxisAnnotation(
            labels=['a', 'b', 'c'],
            cardinalities=[1, 1, 1]
        )

        assert axis.shape == 3

    def test_shape_nested(self):
        """Test shape property for nested axis."""
        axis = AxisAnnotation(
            labels=['a', 'b'],
            cardinalities=[2, 3]
        )

        assert axis.shape == 5  # Sum of cardinalities


class TestAxisAnnotationImmutability:
    """Tests for AxisAnnotation write-once behavior."""

    def test_cannot_modify_labels_after_init(self):
        """Test that labels cannot be modified after initialization."""
        axis = AxisAnnotation(labels=['a', 'b'])

        with pytest.raises(AttributeError, match="write-once"):
            axis.labels = ['x', 'y']

    def test_cannot_modify_states_after_init(self):
        """Test that states cannot be modified after initialization."""
        axis = AxisAnnotation(
            labels=['a', 'b'],
            states=[['x'], ['y']]
        )

        with pytest.raises(AttributeError, match="write-once"):
            axis.states = [['p'], ['q']]

    def test_cannot_modify_cardinalities_after_init(self):
        """Test that cardinalities cannot be modified after initialization."""
        axis = AxisAnnotation(
            labels=['a', 'b'],
            cardinalities=[2, 3]
        )

        with pytest.raises(AttributeError, match="write-once"):
            axis.cardinalities = [4, 5]

    def test_metadata_can_be_set(self):
        """Test that metadata can be set (special case)."""
        axis = AxisAnnotation(labels=['a', 'b'])

        # Metadata can be set even after init
        axis.metadata = {'a': {}, 'b': {}}
        assert axis.metadata is not None


class TestAnnotationsComprehensive:
    """Comprehensive tests for Annotations class."""

    def test_annotations_with_single_axis(self):
        """Test Annotations with a single axis."""
        axis = AxisAnnotation(labels=['a', 'b', 'c'])
        annotations = Annotations(axis_annotations={1: axis})

        assert annotations.get_axis_annotation(1) == axis
        assert len(annotations.get_axis_labels(1)) == 3

    def test_annotations_shape_property(self):
        """Test Annotations shape property."""
        axis = AxisAnnotation(
            labels=['a', 'b'],
            cardinalities=[2, 3]
        )
        annotations = Annotations(axis_annotations={1: axis})

        assert annotations.shape == (-1, 5)

    def test_annotations_to_dict_and_back(self):
        """Test Annotations serialization round-trip."""
        axis = AxisAnnotation(
            labels=['x', 'y', 'z'],
            cardinalities=[1, 2, 1],
            metadata={
                'x': {'type': 'binary'},
                'y': {'type': 'categorical'},
                'z': {'type': 'binary'}
            }
        )
        annotations = Annotations(axis_annotations={1: axis})

        # Serialize
        data = annotations.to_dict()

        # Deserialize
        annotations2 = Annotations.from_dict(data)

        assert annotations2.get_axis_labels(1) == ['x', 'y', 'z']
        assert annotations2.get_axis_cardinalities(1) == [1, 2, 1]
        assert annotations2.get_axis_annotation(1).shape == 4
