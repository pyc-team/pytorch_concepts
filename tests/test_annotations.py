"""
Comprehensive tests for torch_concepts/annotations.py

This test suite covers the single, merged `Annotations` class: initialization,
validation, properties, and methods.
"""
import unittest
import warnings
import pytest
import torch
from torch_concepts.annotations import Annotations, Concept
from torch_concepts.tensor import AnnotatedTensor


class TestAnnotationsBasics(unittest.TestCase):
    """Test suite for the Annotations class (former AxisAnnotation)."""

    def test_binary_concepts_initialization(self):
        """Test initialization of binary concepts (non-nested)."""
        axis = Annotations(labels=['has_wheels', 'has_windows', 'is_red'])

        self.assertEqual(axis.labels, ['has_wheels', 'has_windows', 'is_red'])
        self.assertFalse(axis.is_nested)
        self.assertEqual(axis.cardinalities, [1, 1, 1])
        self.assertEqual(len(axis), 3)
        self.assertEqual(axis.shape, (-1, 3))
        self.assertEqual(axis.size, 3)

    def test_nested_concepts_with_states(self):
        """Test initialization of nested concepts with explicit states."""
        axis = Annotations(
            labels=['color', 'shape', 'size'],
            states=[['red', 'green', 'blue'], ['circle', 'square', 'triangle'], ['small', 'large']]
        )

        self.assertEqual(axis.labels, ['color', 'shape', 'size'])
        self.assertTrue(axis.is_nested)
        self.assertEqual(axis.cardinalities, [3, 3, 2])  # When only states provided, cardinality is length of states
        self.assertEqual(axis.states, [['red', 'green', 'blue'], ['circle', 'square', 'triangle'], ['small', 'large']])
        self.assertEqual(axis.shape, (-1, 8))  # 3 + 3 + 2
        self.assertEqual(axis.size, 8)

    def test_nested_concepts_with_cardinalities(self):
        """Test initialization of nested concepts with only cardinalities."""
        axis = Annotations(
            labels=['size', 'material'],
            cardinalities=[3, 4]
        )

        self.assertEqual(axis.labels, ['size', 'material'])
        self.assertTrue(axis.is_nested)
        self.assertEqual(axis.cardinalities, [3, 4])
        # Auto-generated states
        self.assertEqual(axis.states[0], ['0', '1', '2'])
        self.assertEqual(axis.states[1], ['0', '1', '2', '3'])

    def test_states_and_cardinalities_consistency(self):
        """Test that states and cardinalities are validated for consistency."""
        # Valid: states match cardinalities
        axis = Annotations(
            labels=['color',],
            states=(('red', 'green', 'blue'),),
            cardinalities=[3,]
        )
        self.assertEqual(axis.cardinalities, [3,])

        # Invalid: cardinalities don't match states
        with self.assertRaises(ValueError) as context:
            Annotations(
                labels=['color',],
                states=(('red', 'green', 'blue'),),
                cardinalities=[2,]
            )
        self.assertIn("don't match", str(context.exception))

    def test_invalid_states_length(self):
        """Test error when states length doesn't match labels length."""
        with self.assertRaises(ValueError) as context:
            Annotations(
                labels=['color', 'shape'],
                states=(('red', 'green', 'blue'),)  # Missing state tuple for 'shape'
            )
        self.assertIn("must match", str(context.exception))

    def test_invalid_cardinalities_length(self):
        """Test error when cardinalities length doesn't match labels length."""
        with self.assertRaises(ValueError) as context:
            Annotations(
                labels=['color', 'shape'],
                cardinalities=[3,]  # Missing cardinality for 'shape'
            )
        self.assertIn("must match", str(context.exception))

    def test_no_states_no_cardinalities_warning(self):
        """Test warning when neither states nor cardinalities provided."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            axis = Annotations(labels=['concept1', 'concept2'])

            self.assertEqual(len(w), 1)
            self.assertIn("binary", str(w[0].message))
            self.assertEqual(axis.cardinalities, [1, 1])

    def test_get_index_and_label(self):
        """Test get_index and get_label methods."""
        axis = Annotations(labels=['a', 'b', 'c'])

        self.assertEqual(axis.get_index('a'), 0)
        self.assertEqual(axis.get_index('b'), 1)
        self.assertEqual(axis.get_index('c'), 2)

        self.assertEqual(axis.get_label(0), 'a')
        self.assertEqual(axis.get_label(1), 'b')
        self.assertEqual(axis.get_label(2), 'c')

        # Test invalid label
        with self.assertRaises(ValueError):
            axis.get_index('d')

        # Test invalid index
        with self.assertRaises(IndexError):
            axis.get_label(5)

    def test_getitem(self):
        """Test __getitem__ method with int keys (returns label)."""
        axis = Annotations(labels=['a', 'b', 'c'])

        self.assertEqual(axis[0], 'a')
        self.assertEqual(axis[1], 'b')
        self.assertEqual(axis[2], 'c')

        with self.assertRaises(IndexError):
            _ = axis[5]

    def test_get_total_cardinality(self):
        """Test get_total_cardinality method."""
        axis_nested = Annotations(
            labels=['color', 'shape'],
            cardinalities=[3, 2]
        )
        self.assertEqual(axis_nested.get_total_cardinality(), 5)

        axis_flat = Annotations(labels=['a', 'b', 'c'])
        self.assertEqual(axis_flat.get_total_cardinality(), 3)

    def test_metadata(self):
        """Test metadata handling."""
        metadata = {
            'color': {'type': 'discrete', 'group': 'appearance'},
            'shape': {'type': 'discrete', 'group': 'geometry'}
        }
        axis = Annotations(
            labels=['color', 'shape'],
            cardinalities=[3, 2],
            metadata=metadata
        )

        self.assertEqual(axis.metadata['color']['type'], 'discrete')
        self.assertEqual(axis.metadata['shape']['group'], 'geometry')

    def test_metadata_missing_label(self):
        """Test error when metadata is missing a label."""
        metadata = {'color': {'type': 'discrete'}}

        with self.assertRaises(ValueError) as context:
            Annotations(
                labels=['color', 'shape'],
                cardinalities=[3, 2],
                metadata=metadata
            )
        self.assertIn("Metadata missing", str(context.exception))

    def test_groupby_metadata(self):
        """Test groupby_metadata method."""
        metadata = {
            'color': {'type': 'discrete', 'group': 'appearance'},
            'shape': {'type': 'discrete', 'group': 'geometry'},
            'size': {'type': 'continuous', 'group': 'geometry'}
        }
        axis = Annotations(
            labels=['color', 'shape', 'size'],
            metadata=metadata
        )

        # Group by 'group' key
        groups = axis.groupby_metadata('group', layout='labels')
        self.assertEqual(set(groups['appearance']), {'color'})
        self.assertEqual(set(groups['geometry']), {'shape', 'size'})

        # Group by indices
        groups_idx = axis.groupby_metadata('group', layout='indices')
        self.assertEqual(groups_idx['appearance'], [0])
        self.assertEqual(set(groups_idx['geometry']), {1, 2})

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        axis = Annotations(
            labels=['color', 'shape'],
            states=[['red', 'green', 'blue'], ['circle', 'square', 'triangle']],
            metadata={'color': {'type': 'discrete'}, 'shape': {'type': 'discrete'}}
        )

        # Serialize
        data = axis.to_dict()
        self.assertEqual(data['labels'], ['color', 'shape'])
        self.assertNotIn('axis_annotations', data)

        # Deserialize
        axis_restored = Annotations.from_dict(data)
        self.assertEqual(axis_restored.labels, axis.labels)
        self.assertEqual(axis_restored.states, axis.states)
        self.assertEqual(axis_restored.cardinalities, axis.cardinalities)

    def test_repr(self):
        """Test __repr__ method (dataclass repr)."""
        axis = Annotations(labels=['a', 'b'])
        repr_str = repr(axis)
        self.assertIn('Annotations', repr_str)
        self.assertIn('a', repr_str)

    def test_str(self):
        """Test __str__ method."""
        axis = Annotations(labels=['concept1', 'concept2'])
        str_output = str(axis)
        self.assertIsInstance(str_output, str)
        self.assertIn('concept1', str_output)


class TestAnnotationsEdgeCases(unittest.TestCase):
    """Test edge cases for Annotations."""

    def test_single_label(self):
        """Test with single label."""
        axis = Annotations(labels=['single',])
        self.assertEqual(len(axis), 1)
        self.assertEqual(axis[0], 'single')

    def test_many_labels(self):
        """Test with many labels."""
        labels = tuple(f'label_{i}' for i in range(100))
        axis = Annotations(labels=labels)
        self.assertEqual(len(axis), 100)

    def test_large_cardinality(self):
        """Test with large cardinality."""
        axis = Annotations(
            labels=['concept',],
            cardinalities=[1000,]
        )
        self.assertEqual(axis.cardinalities[0], 1000)
        self.assertEqual(len(axis.states[0]), 1000)

    def test_mixed_cardinalities(self):
        """Test with mixed cardinalities (binary and multi-class)."""
        axis = Annotations(
            labels=['binary', 'ternary', 'quad', 'many'],
            cardinalities=[1, 3, 4, 10]
        )
        self.assertEqual(axis.cardinalities, [1, 3, 4, 10])

    def test_get_label_negative_index(self):
        """Test get_label with negative index."""
        axis = Annotations(labels=['a', 'b', 'c'])
        # Negative indexing might not be supported
        with self.assertRaises((IndexError, ValueError)):
            axis.get_label(-1)

    def test_duplicate_labels_warning(self):
        """Test warning or error with duplicate labels."""
        # Depending on implementation, this might raise or warn
        try:
            axis = Annotations(labels=['a', 'b', 'a'])
            # If no error, check behavior
            self.assertEqual(len(axis.labels), 3)
        except ValueError:
            pass  # Expected if duplicates not allowed

    def test_empty_metadata(self):
        """Test with empty metadata dict."""
        axis = Annotations(
            labels=['a', 'b'],
            metadata={}
        )
        # Should work or raise error
        self.assertEqual(len(axis.labels), 2)

    def test_special_characters_in_labels(self):
        """Test labels with special characters."""
        axis = Annotations(labels=['label-1', 'label_2', 'label.3', 'label@4'])
        self.assertEqual(len(axis), 4)

    def test_unicode_labels(self):
        """Test labels with unicode characters."""
        axis = Annotations(labels=['色彩', 'форма', '🎨'])
        self.assertEqual(len(axis), 3)

    def test_very_long_label_names(self):
        """Test with very long label names."""
        long_label = 'a' * 1000
        axis = Annotations(labels=[long_label, 'short'])
        self.assertEqual(axis[0], long_label)


class TestAnnotationsMetadata:
    """Tests for Annotations metadata functionality."""

    def test_has_metadata_returns_false_when_none(self):
        """Test has_metadata returns False when metadata is None."""
        axis = Annotations(labels=['a', 'b', 'c'])
        assert not axis.has_metadata('distribution')

    def test_has_metadata_returns_true_when_all_have_key(self):
        """Test has_metadata returns True when all labels have the key."""
        axis = Annotations(
            labels=['a', 'b'],
            metadata={
                'a': {'distribution': 'Bernoulli'},
                'b': {'distribution': 'Bernoulli'}
            }
        )
        assert axis.has_metadata('distribution')

    def test_has_metadata_returns_false_when_some_missing(self):
        """Test has_metadata returns False when some labels lack the key."""
        axis = Annotations(
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
        axis = Annotations(
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
        axis = Annotations(
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
        axis = Annotations(
            labels=['a', 'b'],
            metadata={'a': {'type': 'x'}, 'b': {'type': 'x'}}
        )

        with pytest.raises(ValueError, match="Unknown layout"):
            axis.groupby_metadata('type', layout='invalid')

    def test_groupby_metadata_returns_empty_when_none(self):
        """Test groupby_metadata returns empty dict when metadata is None."""
        axis = Annotations(labels=['a', 'b'])
        groups = axis.groupby_metadata('type')
        assert groups == {}

    def test_groupby_metadata_skips_missing_keys(self):
        """Test groupby_metadata skips labels without the requested key."""
        axis = Annotations(
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


class TestAnnotationsCardinalities:
    """Tests for Annotations cardinality handling."""

    def test_states_infer_cardinalities(self):
        """Test that cardinalities are inferred from states."""
        axis = Annotations(
            labels=['color', 'size'],
            states=[['red', 'blue'], ['small', 'medium', 'large']]
        )

        assert axis.cardinalities == [2, 3]
        assert axis.is_nested

    def test_cardinalities_generate_states(self):
        """Test that states are generated from cardinalities."""
        axis = Annotations(
            labels=['a', 'b'],
            cardinalities=[3, 2]
        )

        assert axis.states == [['0', '1', '2'], ['0', '1']]
        assert axis.is_nested

    def test_binary_default_when_neither_provided(self):
        """Test binary assumption when neither states nor cardinalities provided."""
        with pytest.warns(UserWarning, match="assuming all concepts are binary"):
            axis = Annotations(labels=['a', 'b', 'c'])

        assert axis.cardinalities == [1, 1, 1]
        assert axis.states == [['0'], ['0'], ['0']]
        assert not axis.is_nested

    def test_cardinality_of_one_not_nested(self):
        """Test that cardinality of 1 means not nested."""
        axis = Annotations(
            labels=['a', 'b'],
            cardinalities=[1, 1]
        )

        assert not axis.is_nested

    def test_mixed_cardinalities_is_nested(self):
        """Test that any cardinality > 1 makes it nested."""
        axis = Annotations(
            labels=['a', 'b', 'c'],
            cardinalities=[1, 3, 1]
        )

        assert axis.is_nested

    def test_get_total_cardinality_nested(self):
        """Test get_total_cardinality for nested axis."""
        axis = Annotations(
            labels=['a', 'b'],
            cardinalities=[2, 3]
        )

        assert axis.get_total_cardinality() == 5

    def test_get_total_cardinality_not_nested(self):
        """Test get_total_cardinality for non-nested axis."""
        axis = Annotations(
            labels=['a', 'b', 'c'],
            cardinalities=[1, 1, 1]
        )

        assert axis.get_total_cardinality() == 3


class TestAnnotationsValidation:
    """Tests for Annotations validation and error handling."""

    def test_mismatched_states_length_raises_error(self):
        """Test that mismatched states length raises ValueError."""
        with pytest.raises(ValueError, match="Number of state tuples"):
            Annotations(
                labels=['a', 'b'],
                states=[['x', 'y'], ['p', 'q'], ['extra']]  # 3 states for 2 labels
            )

    def test_mismatched_cardinalities_length_raises_error(self):
        """Test that mismatched cardinalities length raises ValueError."""
        with pytest.raises(ValueError, match="Number of state tuples"):
            Annotations(
                labels=['a', 'b'],
                cardinalities=[2, 3, 4]  # 3 cardinalities for 2 labels
            )

    def test_inconsistent_states_cardinalities_raises_error(self):
        """Test that inconsistent states and cardinalities raises ValueError."""
        with pytest.raises(ValueError, match="don't match inferred cardinalities"):
            Annotations(
                labels=['a', 'b'],
                states=[['x', 'y'], ['p', 'q', 'r']],  # [2, 3]
                cardinalities=[2, 2]  # Mismatch: should be [2, 3]
            )

    def test_metadata_not_dict_raises_error(self):
        """Test that non-dict metadata raises ValueError."""
        with pytest.raises(ValueError, match="metadata must be a dictionary"):
            Annotations(
                labels=['a', 'b'],
                metadata=['not', 'a', 'dict']
            )

    def test_metadata_missing_label_raises_error(self):
        """Test that metadata missing a label raises ValueError."""
        with pytest.raises(ValueError, match="Metadata missing for label"):
            Annotations(
                labels=['a', 'b', 'c'],
                metadata={
                    'a': {},
                    'b': {}
                    # Missing 'c'
                }
            )

    def test_get_index_invalid_label_raises_error(self):
        """Test that get_index with invalid label raises ValueError."""
        axis = Annotations(labels=['a', 'b', 'c'])

        with pytest.raises(ValueError, match="not found in labels"):
            axis.get_index('invalid')

    def test_get_label_invalid_index_raises_error(self):
        """Test that get_label with invalid index raises IndexError."""
        axis = Annotations(labels=['a', 'b', 'c'])

        with pytest.raises(IndexError, match="out of range"):
            axis.get_label(10)

    def test_get_label_negative_index_raises_error(self):
        """Test that get_label with negative index raises IndexError."""
        axis = Annotations(labels=['a', 'b', 'c'])

        with pytest.raises(IndexError, match="out of range"):
            axis.get_label(-1)

    def test_getitem_invalid_index_raises_error(self):
        """Test that __getitem__ with invalid index raises IndexError."""
        axis = Annotations(labels=['a', 'b'])

        with pytest.raises(IndexError, match="out of range"):
            _ = axis[5]


class TestAnnotationsSerialization:
    """Tests for Annotations serialization."""

    def test_to_dict_simple(self):
        """Test to_dict for simple axis."""
        axis = Annotations(
            labels=['a', 'b'],
            cardinalities=[1, 1]
        )

        d = axis.to_dict()
        assert d['labels'] == ['a', 'b']
        assert d['cardinalities'] == [1, 1]
        assert d['is_nested'] == False

    def test_to_dict_nested_with_metadata(self):
        """Test to_dict for nested axis with metadata."""
        axis = Annotations(
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

        axis = Annotations.from_dict(data)
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

        axis = Annotations.from_dict(data)
        assert axis.labels == ['x', 'y']
        assert axis.cardinalities == [2, 3]
        assert axis.is_nested
        assert axis.states == [['a', 'b'], ['p', 'q', 'r']]


class TestAnnotationsShape:
    """Tests for Annotations shape/size property."""

    def test_shape_not_nested(self):
        """Test shape property for non-nested axis."""
        axis = Annotations(
            labels=['a', 'b', 'c'],
            cardinalities=[1, 1, 1]
        )

        assert axis.shape == (-1, 3)
        assert axis.size == 3

    def test_shape_nested(self):
        """Test shape property for nested axis."""
        axis = Annotations(
            labels=['a', 'b'],
            cardinalities=[2, 3]
        )

        assert axis.shape == (-1, 5)  # Sum of cardinalities
        assert axis.size == 5


class TestAnnotationsImmutability:
    """Tests for Annotations write-once behavior."""

    def test_cannot_modify_labels_after_init(self):
        """Test that labels cannot be modified after initialization."""
        axis = Annotations(labels=['a', 'b'])

        with pytest.raises(AttributeError, match="write-once"):
            axis.labels = ['x', 'y']

    def test_cannot_modify_states_after_init(self):
        """Test that states cannot be modified after initialization."""
        axis = Annotations(
            labels=['a', 'b'],
            states=[['x'], ['y']]
        )

        with pytest.raises(AttributeError, match="write-once"):
            axis.states = [['p'], ['q']]

    def test_cannot_modify_cardinalities_after_init(self):
        """Test that cardinalities cannot be modified after initialization."""
        axis = Annotations(
            labels=['a', 'b'],
            cardinalities=[2, 3]
        )

        with pytest.raises(AttributeError, match="write-once"):
            axis.cardinalities = [4, 5]

    def test_metadata_can_be_set(self):
        """Test that metadata can be set (special case)."""
        axis = Annotations(labels=['a', 'b'])

        # Metadata can be set even after init
        axis.metadata = {'a': {}, 'b': {}}
        assert axis.metadata is not None


class TestAnnotationsExtended:
    """Extended tests for Annotations class to improve coverage."""

    def test_cardinality_mismatch_with_states(self):
        """Test that mismatched cardinalities and states raise error."""
        with pytest.raises(ValueError, match="don't match inferred cardinalities"):
            Annotations(
                labels=['a', 'b'],
                states=[['x', 'y'], ['p', 'q', 'r']],
                cardinalities=[2, 2]  # Should be [2, 3] based on states
            )

    def test_metadata_validation_non_dict(self):
        """Test that non-dict metadata raises error."""
        with pytest.raises(ValueError, match="metadata must be a dictionary"):
            Annotations(
                labels=['a', 'b'],
                metadata="invalid"  # Should be dict
            )

    def test_metadata_validation_missing_label(self):
        """Test that metadata missing a label raises error."""
        with pytest.raises(ValueError, match="Metadata missing for label"):
            Annotations(
                labels=['a', 'b', 'c'],
                metadata={'a': {}, 'b': {}}  # Missing 'c'
            )

    def test_has_metadata_with_key(self):
        """Test has_metadata method with specific key."""
        axis = Annotations(
            labels=['a', 'b'],
            metadata={'a': {'type': 'binary'}, 'b': {'type': 'binary'}}
        )
        assert axis.has_metadata('type') is True
        assert axis.has_metadata('missing_key') is False

    def test_has_metadata_none(self):
        """Test has_metadata when metadata is None."""
        axis = Annotations(labels=['a', 'b'])
        assert axis.has_metadata('any_key') is False

    def test_groupby_metadata_labels_layout(self):
        """Test groupby_metadata with labels layout."""
        axis = Annotations(
            labels=['a', 'b', 'c', 'd'],
            metadata={
                'a': {'group': 'A'},
                'b': {'group': 'A'},
                'c': {'group': 'B'},
                'd': {'group': 'B'}
            }
        )
        result = axis.groupby_metadata('group', layout='labels')
        assert result == {'A': ['a', 'b'], 'B': ['c', 'd']}

    def test_groupby_metadata_indices_layout(self):
        """Test groupby_metadata with indices layout."""
        axis = Annotations(
            labels=['a', 'b', 'c'],
            metadata={
                'a': {'group': 'X'},
                'b': {'group': 'Y'},
                'c': {'group': 'X'}
            }
        )
        result = axis.groupby_metadata('group', layout='indices')
        assert result == {'X': [0, 2], 'Y': [1]}

    def test_groupby_metadata_invalid_layout(self):
        """Test groupby_metadata with invalid layout raises error."""
        axis = Annotations(
            labels=['a', 'b'],
            metadata={'a': {'g': '1'}, 'b': {'g': '2'}}
        )
        with pytest.raises(ValueError, match="Unknown layout"):
            axis.groupby_metadata('g', layout='invalid')

    def test_groupby_metadata_none(self):
        """Test groupby_metadata when metadata is None."""
        axis = Annotations(labels=['a', 'b'])
        result = axis.groupby_metadata('any_key')
        assert result == {}

    def test_get_index_not_found(self):
        """Test get_index with non-existent label."""
        axis = Annotations(labels=['a', 'b', 'c'])
        with pytest.raises(ValueError, match="Label 'z' not found"):
            axis.get_index('z')

    def test_get_label_out_of_range(self):
        """Test get_label with out-of-range index."""
        axis = Annotations(labels=['a', 'b'])
        with pytest.raises(IndexError, match="Index 5 out of range"):
            axis.get_label(5)

    def test_getitem_out_of_range(self):
        """Test __getitem__ with out-of-range index."""
        axis = Annotations(labels=['a', 'b'])
        with pytest.raises(IndexError, match="Index 10 out of range"):
            _ = axis[10]

    def test_get_total_cardinality_nested(self):
        """Test get_total_cardinality for nested axis."""
        axis = Annotations(
            labels=['a', 'b', 'c'],
            cardinalities=[2, 3, 4]
        )
        assert axis.get_total_cardinality() == 9

    def test_get_total_cardinality_not_nested(self):
        """Test get_total_cardinality for non-nested axis."""
        axis = Annotations(labels=['a', 'b', 'c'])
        assert axis.get_total_cardinality() == 3

    def test_to_dict_with_all_fields(self):
        """Test to_dict with all fields populated."""
        axis = Annotations(
            labels=['a', 'b'],
            states=[['0', '1'], ['x', 'y', 'z']],
            metadata={'a': {'type': 'binary'}, 'b': {'type': 'categorical'}}
        )
        result = axis.to_dict()

        assert result['labels'] == ['a', 'b']
        assert result['states'] == [['0', '1'], ['x', 'y', 'z']]
        assert result['cardinalities'] == [2, 3]
        assert result['is_nested'] is True
        assert result['metadata'] == {'a': {'type': 'binary'}, 'b': {'type': 'categorical'}}

    def test_from_dict_reconstruction(self):
        """Test from_dict reconstructs Annotations correctly."""
        original = Annotations(
            labels=['x', 'y'],
            cardinalities=[2, 3],
            metadata={'x': {'info': 'test'}, 'y': {'info': 'test2'}}
        )

        data = original.to_dict()
        reconstructed = Annotations.from_dict(data)

        assert reconstructed.labels == original.labels
        assert reconstructed.cardinalities == original.cardinalities
        assert reconstructed.is_nested == original.is_nested
        assert reconstructed.metadata == original.metadata

    def test_subset_basic(self):
        """Test subset method with valid labels."""
        axis = Annotations(
            labels=['a', 'b', 'c', 'd'],
            cardinalities=[1, 2, 3, 1]
        )

        subset = axis.subset(['b', 'd'])

        assert subset.labels == ['b', 'd']
        assert subset.cardinalities == [2, 1]

    def test_subset_with_metadata(self):
        """Test subset preserves metadata."""
        axis = Annotations(
            labels=['a', 'b', 'c'],
            metadata={'a': {'x': 1}, 'b': {'x': 2}, 'c': {'x': 3}}
        )

        subset = axis.subset(['a', 'c'])

        assert subset.labels == ['a', 'c']
        assert subset.metadata == {'a': {'x': 1}, 'c': {'x': 3}}

    def test_subset_missing_labels(self):
        """Test subset with non-existent labels raises error."""
        axis = Annotations(labels=['a', 'b', 'c'])

        with pytest.raises(ValueError, match="Unknown labels for subset"):
            axis.subset(['a', 'z'])

    def test_subset_preserves_order(self):
        """Test subset preserves the requested label order."""
        axis = Annotations(labels=['a', 'b', 'c', 'd'])

        subset = axis.subset(['d', 'b', 'a'])

        assert subset.labels == ['d', 'b', 'a']

    def test_union_with_no_overlap(self):
        """Test union_with with no overlapping labels."""
        axis1 = Annotations(labels=['a', 'b'])
        axis2 = Annotations(labels=['c', 'd'])

        union = axis1.union_with(axis2)

        assert union.labels == ['a', 'b', 'c', 'd']

    def test_union_with_overlap(self):
        """Test union_with with overlapping labels."""
        axis1 = Annotations(labels=['a', 'b', 'c'])
        axis2 = Annotations(labels=['b', 'c', 'd'])

        union = axis1.union_with(axis2)

        assert union.labels == ['a', 'b', 'c', 'd']

    def test_union_with_metadata_merge(self):
        """Test union_with merges metadata with left-win."""
        axis1 = Annotations(
            labels=['a', 'b'],
            metadata={'a': {'x': 1}, 'b': {'x': 2}}
        )
        axis2 = Annotations(
            labels=['b', 'c'],
            metadata={'b': {'x': 999}, 'c': {'x': 3}}
        )

        union = axis1.union_with(axis2)

        # Left-win: 'b' should keep metadata from axis1
        assert union.metadata['a'] == {'x': 1}
        assert union.metadata['b'] == {'x': 2}
        assert union.metadata['c'] == {'x': 3}

    def test_write_once_labels_attribute(self):
        """Test that labels attribute is write-once."""
        axis = Annotations(labels=['a', 'b'])

        with pytest.raises(AttributeError, match="write-once and already set"):
            axis.labels = ['x', 'y']

    def test_write_once_states_attribute(self):
        """Test that states attribute is write-once."""
        axis = Annotations(labels=['a', 'b'], cardinalities=[2, 3])

        with pytest.raises(AttributeError, match="write-once and already set"):
            axis.states = [['0', '1'], ['0', '1', '2']]

    def test_metadata_can_be_modified(self):
        """Test that metadata can be modified after creation."""
        axis = Annotations(labels=['a', 'b'])

        # Metadata is not write-once, so this should work
        axis.metadata = {'a': {'test': 1}, 'b': {'test': 2}}
        assert axis.metadata is not None


class TestAnnotationsComprehensive:
    """Comprehensive tests for the Annotations class."""

    def test_single_axis(self):
        """Test a single Annotations object."""
        annotations = Annotations(labels=['a', 'b', 'c'])

        assert len(annotations.labels) == 3

    def test_shape_property(self):
        """Test Annotations shape property."""
        annotations = Annotations(
            labels=['a', 'b'],
            cardinalities=[2, 3]
        )

        assert annotations.shape == (-1, 5)

    def test_to_dict_and_back(self):
        """Test Annotations serialization round-trip."""
        annotations = Annotations(
            labels=['x', 'y', 'z'],
            cardinalities=[1, 2, 1],
            metadata={
                'x': {'type': 'binary'},
                'y': {'type': 'categorical'},
                'z': {'type': 'binary'}
            }
        )

        # Serialize
        data = annotations.to_dict()

        # Deserialize
        annotations2 = Annotations.from_dict(data)

        assert annotations2.labels == ['x', 'y', 'z']
        assert annotations2.cardinalities == [1, 2, 1]
        assert annotations2.size == 4


class TestAnnotationsCachedUtilities(unittest.TestCase):
    """Test suite for Annotations cached index utilities."""

    def setUp(self):
        """Set up test fixtures with various concept configurations."""
        import torch
        self.torch = torch

        # Mixed types: binary, categorical, continuous
        self.mixed_axis = Annotations(
            labels=['is_big', 'color', 'shape', 'temperature'],
            cardinalities=[1, 3, 2, 1],
            types=['binary', 'categorical', 'categorical', 'continuous'],
        )

        # All binary
        self.binary_axis = Annotations(
            labels=['a', 'b', 'c'],
            cardinalities=[1, 1, 1],
            types=['binary', 'binary', 'binary'],
        )

        # All categorical
        self.categorical_axis = Annotations(
            labels=['x', 'y'],
            cardinalities=[3, 4],
            types=['categorical', 'categorical'],
        )

    # =========================================================================
    # size tests (cached, so it must not leak between instances)
    # =========================================================================

    def test_size_is_sum_of_cardinalities(self):
        """size sums the cardinalities, and repeats agree (it is cached)."""
        self.assertEqual(self.mixed_axis.size, 7)
        self.assertEqual(self.mixed_axis.size, 7)
        self.assertEqual(self.binary_axis.size, 3)
        self.assertEqual(self.categorical_axis.size, 7)

    def test_size_cache_is_per_instance(self):
        """A derived annotation computes its own size, not the parent's."""
        parent_size = self.mixed_axis.size  # populate the parent's cache first
        sub = self.mixed_axis.subset(['color', 'temperature'])
        self.assertEqual(sub.size, 4)  # 3 + 1
        self.assertEqual(self.mixed_axis.size, parent_size)

        concept_space = self.mixed_axis.to_concept_space()
        self.assertEqual(concept_space.size, 4)  # one column per concept
        self.assertEqual(self.mixed_axis.size, parent_size)

    # =========================================================================
    # cumulative_cardinalities tests
    # =========================================================================

    def test_cumulative_cardinalities_mixed(self):
        """Test cumulative_cardinalities with mixed cardinalities."""
        cum = self.mixed_axis.cumulative_cardinalities
        # [1, 3, 2, 1] -> [0, 1, 4, 6, 7]
        self.assertEqual(cum, [0, 1, 4, 6, 7])
        self.assertEqual(len(cum), len(self.mixed_axis.labels) + 1)

    def test_cumulative_cardinalities_binary(self):
        """Test cumulative_cardinalities with all binary concepts."""
        cum = self.binary_axis.cumulative_cardinalities
        # [1, 1, 1] -> [0, 1, 2, 3]
        self.assertEqual(cum, [0, 1, 2, 3])

    def test_cumulative_cardinalities_categorical(self):
        """Test cumulative_cardinalities with categorical concepts."""
        cum = self.categorical_axis.cumulative_cardinalities
        # [3, 4] -> [0, 3, 7]
        self.assertEqual(cum, [0, 3, 7])

    def test_cumulative_cardinalities_is_cached(self):
        """Test that cumulative_cardinalities is cached (same object returned)."""
        first_call = self.mixed_axis.cumulative_cardinalities
        second_call = self.mixed_axis.cumulative_cardinalities
        self.assertIs(first_call, second_call)

    # =========================================================================
    # concept_slices tests
    # =========================================================================

    def test_concept_slices_mixed(self):
        """Test concept_slices returns correct slice objects."""
        slices = self.mixed_axis.concept_slices

        self.assertEqual(slices['is_big'], slice(0, 1))
        self.assertEqual(slices['color'], slice(1, 4))
        self.assertEqual(slices['shape'], slice(4, 6))
        self.assertEqual(slices['temperature'], slice(6, 7))

    def test_concept_slices_all_keys_present(self):
        """Test all concept labels are keys in concept_slices."""
        slices = self.mixed_axis.concept_slices
        for label in self.mixed_axis.labels:
            self.assertIn(label, slices)

    def test_concept_slices_tensor_indexing(self):
        """Test concept_slices can be used for tensor indexing."""
        tensor = self.torch.arange(7).unsqueeze(0).float()  # [[0,1,2,3,4,5,6]]
        slices = self.mixed_axis.concept_slices

        color_logits = tensor[:, slices['color']]
        self.assertTrue(self.torch.equal(color_logits, self.torch.tensor([[1., 2., 3.]])))

    def test_concept_slices_is_cached(self):
        """Test that concept_slices is cached."""
        first_call = self.mixed_axis.concept_slices
        second_call = self.mixed_axis.concept_slices
        self.assertIs(first_call, second_call)

    # =========================================================================
    # type_groups tests
    # =========================================================================

    def test_type_groups_structure(self):
        """Test type_groups returns correct structure."""
        groups = self.mixed_axis.type_groups

        # Should have all three keys
        self.assertIn('binary', groups)
        self.assertIn('categorical', groups)
        self.assertIn('continuous', groups)

        # Each group should have labels, concept_idx, logits_idx
        for group_type in ['binary', 'categorical', 'continuous']:
            self.assertIn('labels', groups[group_type])
            self.assertIn('concept_idx', groups[group_type])
            self.assertIn('logits_idx', groups[group_type])

    def test_type_groups_mixed_labels(self):
        """Test type_groups correctly categorizes labels."""
        groups = self.mixed_axis.type_groups

        self.assertEqual(groups['binary']['labels'], ['is_big'])
        self.assertEqual(groups['categorical']['labels'], ['color', 'shape'])
        self.assertEqual(groups['continuous']['labels'], ['temperature'])

    def test_type_groups_mixed_concept_idx(self):
        """Test type_groups returns correct concept-level indices."""
        groups = self.mixed_axis.type_groups

        # is_big is at index 0
        self.assertEqual(groups['binary']['concept_idx'], [0])
        # color at 1, shape at 2
        self.assertEqual(groups['categorical']['concept_idx'], [1, 2])
        # temperature at 3
        self.assertEqual(groups['continuous']['concept_idx'], [3])

    def test_type_groups_mixed_logits_idx(self):
        """Test type_groups returns correct logit-level indices."""
        groups = self.mixed_axis.type_groups

        # is_big: positions 0
        self.assertEqual(groups['binary']['logits_idx'], [0])
        # color: 1,2,3; shape: 4,5
        self.assertEqual(groups['categorical']['logits_idx'], [1, 2, 3, 4, 5])
        # temperature: 6
        self.assertEqual(groups['continuous']['logits_idx'], [6])

    def test_type_groups_all_binary(self):
        """Test type_groups with all binary concepts."""
        groups = self.binary_axis.type_groups

        self.assertEqual(groups['binary']['labels'], ['a', 'b', 'c'])
        self.assertEqual(groups['categorical']['labels'], [])
        self.assertEqual(groups['continuous']['labels'], [])
        self.assertEqual(groups['binary']['logits_idx'], [0, 1, 2])

    def test_type_groups_all_categorical(self):
        """Test type_groups with all categorical concepts."""
        groups = self.categorical_axis.type_groups

        self.assertEqual(groups['binary']['labels'], [])
        self.assertEqual(groups['categorical']['labels'], ['x', 'y'])
        self.assertEqual(groups['categorical']['logits_idx'], [0, 1, 2, 3, 4, 5, 6])

    def test_type_groups_is_cached(self):
        """Test that type_groups is cached."""
        first_call = self.mixed_axis.type_groups
        second_call = self.mixed_axis.type_groups
        self.assertIs(first_call, second_call)

    # =========================================================================
    # get_slice tests
    # =========================================================================

    def test_get_slice_single_concept_returns_slice(self):
        """Test get_slice with single concept returns slice object."""
        result = self.mixed_axis.get_slice('color')
        self.assertIsInstance(result, slice)
        self.assertEqual(result, slice(1, 4))

    def test_get_slice_multiple_concepts_returns_list(self):
        """Test get_slice with multiple concepts returns list of indices."""
        result = self.mixed_axis.get_slice(['is_big', 'temperature'])
        self.assertIsInstance(result, list)
        self.assertEqual(result, [0, 6])

    def test_get_slice_non_contiguous_concepts(self):
        """Test get_slice with non-contiguous concepts."""
        result = self.mixed_axis.get_slice(['color', 'temperature'])
        self.assertEqual(result, [1, 2, 3, 6])

    def test_get_slice_order_matters(self):
        """Test get_slice respects input order for multiple concepts."""
        result1 = self.mixed_axis.get_slice(['is_big', 'color'])
        result2 = self.mixed_axis.get_slice(['color', 'is_big'])

        self.assertEqual(result1, [0, 1, 2, 3])
        self.assertEqual(result2, [1, 2, 3, 0])

    def test_get_slice_single_invalid_label(self):
        """Test get_slice raises error for invalid single label."""
        with self.assertRaises(ValueError) as context:
            self.mixed_axis.get_slice('nonexistent')
        self.assertIn('not found', str(context.exception))

    def test_get_slice_list_invalid_label(self):
        """Test get_slice raises error for invalid label in list."""
        with self.assertRaises(ValueError) as context:
            self.mixed_axis.get_slice(['is_big', 'nonexistent'])
        self.assertIn('not found', str(context.exception))

    def test_get_slice_empty_list(self):
        """Test get_slice with empty list returns empty list."""
        result = self.mixed_axis.get_slice([])
        self.assertEqual(result, [])

    def test_get_slice_tensor_indexing_single(self):
        """Test get_slice result can be used for tensor indexing (single)."""
        tensor = self.torch.arange(7).unsqueeze(0).float()
        s = self.mixed_axis.get_slice('shape')
        result = tensor[:, s]
        self.assertTrue(self.torch.equal(result, self.torch.tensor([[4., 5.]])))

    def test_get_slice_tensor_indexing_multiple(self):
        """Test get_slice result can be used for tensor indexing (multiple)."""
        tensor = self.torch.arange(7).unsqueeze(0).float()
        idx = self.mixed_axis.get_slice(['is_big', 'temperature'])
        result = tensor[:, idx]
        self.assertTrue(self.torch.equal(result, self.torch.tensor([[0., 6.]])))

    # =========================================================================
    # slice_tensor tests
    # =========================================================================

    def test_slice_tensor_basic(self):
        """Test slice_tensor extracts correct columns."""
        tensor = self.torch.arange(7).unsqueeze(0).float()
        result = self.mixed_axis.slice_tensor(tensor, ['color'])
        self.assertTrue(self.torch.equal(result, self.torch.tensor([[1., 2., 3.]])))

    def test_slice_tensor_multiple(self):
        """Test slice_tensor with multiple concepts."""
        tensor = self.torch.arange(7).unsqueeze(0).float()
        result = self.mixed_axis.slice_tensor(tensor, ['is_big', 'temperature'])
        self.assertTrue(self.torch.equal(result, self.torch.tensor([[0., 6.]])))

    def test_slice_tensor_reorder(self):
        """Test slice_tensor reorders columns."""
        tensor = self.torch.arange(7).unsqueeze(0).float()
        result = self.mixed_axis.slice_tensor(tensor, ['temperature', 'is_big'])
        self.assertTrue(self.torch.equal(result, self.torch.tensor([[6., 0.]])))

    def test_slice_tensor_all_concepts_same_order(self):
        """Test slice_tensor with all concepts in original order."""
        tensor = self.torch.arange(7).unsqueeze(0).float()
        result = self.mixed_axis.slice_tensor(tensor, self.mixed_axis.labels)
        self.assertTrue(self.torch.equal(result, tensor))

    def test_slice_tensor_all_concepts_reversed(self):
        """Test slice_tensor with all concepts in reversed order."""
        tensor = self.torch.arange(7).unsqueeze(0).float()
        reversed_labels = list(reversed(self.mixed_axis.labels))
        result = self.mixed_axis.slice_tensor(tensor, reversed_labels)
        expected = self.torch.tensor([[6., 4., 5., 1., 2., 3., 0.]])
        self.assertTrue(self.torch.equal(result, expected))

    def test_slice_tensor_batch_dimension(self):
        """Test slice_tensor preserves batch dimension."""
        tensor = self.torch.arange(14).reshape(2, 7).float()
        result = self.mixed_axis.slice_tensor(tensor, ['color'])
        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(self.torch.equal(result[0], self.torch.tensor([1., 2., 3.])))
        self.assertTrue(self.torch.equal(result[1], self.torch.tensor([8., 9., 10.])))

    # =========================================================================
    # get_logits_idx backward compatibility tests
    # =========================================================================

    def test_get_logits_idx_is_alias_for_get_slice(self):
        """Test get_logits_idx works as alias for get_slice with list."""
        result1 = self.mixed_axis.get_logits_idx(['color', 'shape'])
        result2 = self.mixed_axis.get_slice(['color', 'shape'])
        self.assertEqual(result1, result2)

    def test_get_logits_idx_single_concept_list(self):
        """Test get_logits_idx with single-element list."""
        result = self.mixed_axis.get_logits_idx(['temperature'])
        self.assertEqual(result, [6])

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_single_concept_axis(self):
        """Test utilities work with single concept axis."""
        axis = Annotations(
            labels=['only_one'],
            cardinalities=[5],
            metadata={'only_one': {'type': 'discrete'}}
        )

        self.assertEqual(axis.cumulative_cardinalities, [0, 5])
        self.assertEqual(axis.concept_slices, {'only_one': slice(0, 5)})
        self.assertEqual(axis.type_groups['categorical']['labels'], ['only_one'])
        self.assertEqual(axis.get_slice('only_one'), slice(0, 5))

    def test_large_cardinalities(self):
        """Test utilities work with large cardinalities."""
        axis = Annotations(
            labels=['big_concept'],
            cardinalities=[1000],
            metadata={'big_concept': {'type': 'discrete'}}
        )

        self.assertEqual(axis.cumulative_cardinalities, [0, 1000])
        self.assertEqual(axis.get_slice('big_concept'), slice(0, 1000))

    def test_many_concepts(self):
        """Test utilities work with many concepts."""
        n = 50
        labels = [f'c{i}' for i in range(n)]
        cardinalities = [(i % 5) + 1 for i in range(n)]  # 1-5 cardinality
        metadata = {label: {'type': 'discrete'} for label in labels}

        axis = Annotations(
            labels=labels,
            cardinalities=cardinalities,
            metadata=metadata
        )

        cum = axis.cumulative_cardinalities
        self.assertEqual(len(cum), n + 1)
        self.assertEqual(cum[0], 0)
        self.assertEqual(cum[-1], sum(cardinalities))

        slices = axis.concept_slices
        self.assertEqual(len(slices), n)


class TestConceptView(unittest.TestCase):
    """Tests for the per-concept `Concept` view."""

    def setUp(self):
        self.axis = Annotations(
            labels=['size', 'color', 'temp'],
            cardinalities=[1, 3, 1],
            types=['binary', 'categorical', 'continuous'],
        )

    def test_concept_basic_fields(self):
        c = self.axis.concept('color')
        self.assertIsInstance(c, Concept)
        self.assertEqual(c.name, 'color')
        self.assertEqual(c.index, 1)
        self.assertEqual(c.cardinality, 3)
        self.assertEqual(c.type, 'categorical')
        self.assertEqual(c.slice, slice(1, 4))

    def test_concept_type_predicates(self):
        self.assertTrue(self.axis.concept('size').is_binary)
        self.assertFalse(self.axis.concept('size').is_categorical)
        self.assertTrue(self.axis.concept('color').is_categorical)
        self.assertTrue(self.axis.concept('temp').is_continuous)

    def test_concepts_property_order(self):
        names = [c.name for c in self.axis.concepts]
        self.assertEqual(names, ['size', 'color', 'temp'])

    def test_getitem_str_returns_concept(self):
        self.assertIsInstance(self.axis['color'], Concept)
        self.assertEqual(self.axis['color'].cardinality, 3)

    def test_getitem_int_returns_label(self):
        self.assertEqual(self.axis[0], 'size')


class TestAnnotationsExtraCoverage(unittest.TestCase):
    """Extra tests targeting uncovered lines in annotations.py."""

    def setUp(self):
        self.basic_axis = Annotations(
            labels=['a', 'b', 'c'],
            cardinalities=[1, 3, 1],
            types=['binary', 'categorical', 'binary'],
        )

    # ------------------------------------------------------------------
    # Annotations validation edge cases
    # ------------------------------------------------------------------

    def test_cardinalities_length_mismatch_raises(self):
        """Extra cardinalities raises ValueError."""
        with self.assertRaises(ValueError):
            Annotations(
                labels=['a', 'b'],
                cardinalities=[1, 2, 3],  # too many
            )

    def test_types_length_mismatch_raises(self):
        """Mismatched types length raises ValueError."""
        with self.assertRaises(ValueError):
            Annotations(
                labels=['a', 'b'],
                cardinalities=[1, 1],
                types=['binary'],  # too short
            )

    def test_invalid_concept_type_raises(self):
        """Unknown type string raises ValueError."""
        with self.assertRaises(ValueError):
            Annotations(
                labels=['a'],
                cardinalities=[1],
                types=['invalid_type'],
            )

    def test_binary_with_cardinality_gt_1_raises(self):
        """binary type with cardinality > 1 raises ValueError."""
        with self.assertRaises(ValueError):
            Annotations(
                labels=['a'],
                cardinalities=[3],
                types=['binary'],
            )

    # ------------------------------------------------------------------
    # Annotations.subset
    # ------------------------------------------------------------------

    def test_subset_no_states_preserves_types(self):
        """subset on annotation without explicit states works."""
        axis = Annotations(labels=['x', 'y'], cardinalities=[1, 1], types=['binary', 'binary'])
        sub = axis.subset(['x'])
        self.assertEqual(sub.labels, ['x'])

    def test_subset_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            self.basic_axis.subset(['nonexistent'])

    # ------------------------------------------------------------------
    # Annotations.to_concept_space
    # ------------------------------------------------------------------

    def test_to_concept_space_already_concept_space(self):
        cs = Annotations(labels=['a'], cardinalities=[1], types=['binary'], concept_space=True)
        result = cs.to_concept_space()
        self.assertIs(result, cs)

    def test_to_concept_space_converts(self):
        axis = Annotations(labels=['c1', 'c2'], cardinalities=[3, 2], types=['categorical', 'categorical'])
        cs = axis.to_concept_space()
        self.assertTrue(cs.concept_space)
        self.assertEqual(cs.cardinalities, [1, 1])

    # ------------------------------------------------------------------
    # Annotations.union_with
    # ------------------------------------------------------------------

    def test_union_with_merges_labels(self):
        a = Annotations(labels=['x'], cardinalities=[1], types=['binary'])
        b = Annotations(labels=['y'], cardinalities=[3], types=['categorical'])
        merged = a.union_with(b)
        self.assertIn('x', merged.labels)
        self.assertIn('y', merged.labels)

    def test_union_with_deduplicates(self):
        a = Annotations(labels=['x', 'y'], cardinalities=[1, 1], types=['binary', 'binary'])
        b = Annotations(labels=['y', 'z'], cardinalities=[1, 1], types=['binary', 'binary'])
        merged = a.union_with(b)
        self.assertEqual(merged.labels.count('y'), 1)

    # ------------------------------------------------------------------
    # Annotations accessors
    # ------------------------------------------------------------------

    def test_shape(self):
        # axis-1 size is 5 (1 + 3 + 1)
        shape = self.basic_axis.shape
        self.assertEqual(shape, (-1, 5))
        self.assertEqual(self.basic_axis.size, 5)

    def test_labels(self):
        self.assertEqual(self.basic_axis.labels, ['a', 'b', 'c'])

    def test_cardinalities(self):
        self.assertEqual(self.basic_axis.cardinalities, [1, 3, 1])

    def test_is_nested(self):
        self.assertTrue(self.basic_axis.is_nested)

    def test_get_index(self):
        self.assertEqual(self.basic_axis.get_index('b'), 1)

    def test_get_label(self):
        self.assertEqual(self.basic_axis.get_label(0), 'a')

    def test_states(self):
        self.assertIsNotNone(self.basic_axis.states)

    def test_get_label_states(self):
        states = self.basic_axis.get_label_states('b')
        self.assertEqual(len(states), 3)  # cardinality 3

    def test_get_label_state(self):
        state = self.basic_axis.get_label_state('b', 0)
        self.assertEqual(state, '0')

    def test_get_state_index(self):
        idx = self.basic_axis.get_state_index('b', '0')
        self.assertEqual(idx, 0)

    def test_get_state_index_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.basic_axis.get_state_index('b', 'invalid_state')

    def test_len(self):
        self.assertEqual(len(self.basic_axis), 3)


class TestAnnotatedTensorCoverage(unittest.TestCase):
    """Tests targeting uncovered lines in tensor.py."""

    def setUp(self):
        import torch as _torch
        self.torch = _torch
        self.ann = Annotations(labels=['a', 'b', 'c'])
        from torch_concepts.tensor import AnnotatedTensor
        self.AnnotatedTensor = AnnotatedTensor
        self.t = AnnotatedTensor(_torch.rand(4, 3), self.ann)

    def test_init_1d_tensor_raises_for_axis_1(self):
        # axis=1 needs 2+ dims for that axis to exist at all.
        with self.assertRaises(ValueError):
            self.AnnotatedTensor(self.torch.rand(3), self.ann, axis=1)

    def test_init_1d_tensor_allowed_on_last_axis(self):
        # The default axis=-1 annotates the last axis, which a 1-D tensor has.
        t = self.AnnotatedTensor(self.torch.rand(3), self.ann)
        self.assertEqual(t.annotation.labels, ['a', 'b', 'c'])

    def test_init_mismatched_size_raises(self):
        with self.assertRaises(ValueError):
            self.AnnotatedTensor(self.torch.rand(4, 5), self.ann)

    def test_device_property(self):
        self.assertEqual(self.t.device, self.torch.device('cpu'))

    def test_to_method_returns_annotated_tensor(self):
        moved = self.t.to(self.torch.float64)
        self.assertIsInstance(moved, self.AnnotatedTensor)
        self.assertEqual(moved.tensor.dtype, self.torch.float64)

    def test_getitem_list_syntax(self):
        result = self.t[['a', 'b']]
        self.assertIsInstance(result, self.AnnotatedTensor)
        self.assertEqual(result.annotation.labels, ['a', 'b'])

    def test_getitem_fallback_index_keeps_annotation_on_last_axis(self):
        # Under the default axis=-1, indexing a row of (4, 3) leaves a (3,)
        # whose last axis is still the annotated one, so the labels survive.
        result = self.t[0]
        self.assertIsInstance(result, self.AnnotatedTensor)
        self.assertEqual(result.annotation.labels, ['a', 'b', 'c'])

    def test_getitem_fallback_index_drops_annotation_for_axis_1(self):
        # With axis=1 the same index drops to 1-D, so axis 1 no longer exists.
        t = self.AnnotatedTensor(self.torch.rand(4, 3), self.ann, axis=1)
        self.assertNotIsInstance(t[0], self.AnnotatedTensor)

    def test_union_with_type_error(self):
        with self.assertRaises(TypeError):
            self.t.union_with(self.torch.rand(4, 2))

    def test_union_with_shape_mismatch(self):
        ann2 = Annotations(labels=['x', 'y'])
        t2 = self.AnnotatedTensor(self.torch.rand(5, 2), ann2)  # batch=5 vs 4
        with self.assertRaises(ValueError):
            self.t.union_with(t2)

    def test_union_with_deduplicates_overlap(self):
        torch = self.torch
        ann2 = Annotations(labels=['b', 'c', 'd'])  # 'b','c' overlap
        ann1 = Annotations(labels=['a', 'b', 'c'])
        from torch_concepts.tensor import AnnotatedTensor as AT
        t1 = AT(torch.rand(4, 3), ann1)
        t2 = AT(torch.rand(4, 3), ann2)
        merged = t1.union_with(t2)
        # 'd' added; 'b','c' not duplicated
        self.assertIn('d', merged.annotation.labels)
        self.assertEqual(merged.annotation.labels.count('b'), 1)

    def test_split_by_type_no_arg_returns_dict(self):
        torch = self.torch
        ann = Annotations(
            labels=['x', 'y'],
            cardinalities=[1, 1],
            types=['binary', 'binary'],
        )
        from torch_concepts.tensor import AnnotatedTensor as AT
        t = AT(torch.rand(3, 2), ann)
        result = t.split_by_type()
        self.assertIsInstance(result, dict)

    def test_split_by_type_specific_type(self):
        torch = self.torch
        ann = Annotations(
            labels=['x', 'y'],
            cardinalities=[1, 1],
            types=['binary', 'binary'],
        )
        from torch_concepts.tensor import AnnotatedTensor as AT
        t = AT(torch.rand(3, 2), ann)
        result = t.split_by_type('binary')
        self.assertIsInstance(result, AT)

    def test_torch_function_passthrough(self):
        torch = self.torch
        result = torch.sum(self.t, dim=0)
        self.assertIsInstance(result, torch.Tensor)

    def test_repr_contains_annotation(self):
        r = repr(self.t)
        self.assertIn('a', r)

    def test_len(self):
        self.assertEqual(len(self.t), 4)

    def test_arithmetic_ops(self):
        result = self.t + self.t
        # same axis-1 size → still annotated
        self.assertIsInstance(result, self.AnnotatedTensor)

    def test_wrap_dimension_change(self):
        # sum over axis 1 changes axis-1 size → returns plain tensor
        result = self.t.sum(dim=1)
        self.assertNotIsInstance(result, self.AnnotatedTensor)


if __name__ == '__main__':
    unittest.main()


class TestAnnotatedTensorLastAxis(unittest.TestCase):
    """``axis=-1`` annotates the last axis, so a tensor may carry any number of
    leading dimensions. For a 2-D tensor it coincides with the default axis 1."""

    def setUp(self):
        self.ann = Annotations(labels=['a', 'b', 'c'], cardinalities=[1, 2, 1])

    def test_matches_axis_1_for_a_2d_tensor(self):
        data = torch.arange(24.0).reshape(6, 4)
        default = AnnotatedTensor(data, self.ann)
        last = AnnotatedTensor(data, self.ann, axis=-1)
        self.assertTrue(torch.equal(default['b'].tensor, last['b'].tensor))

    def test_slices_the_last_axis_with_leading_dims(self):
        t = AnnotatedTensor(torch.arange(48.0).reshape(2, 3, 2, 4), self.ann, axis=-1)
        self.assertEqual(tuple(t['b'].tensor.shape), (2, 3, 2, 2))
        self.assertTrue(torch.equal(t['b'].tensor, t.tensor[..., 1:3]))

    def test_rejects_a_size_mismatch_on_the_last_axis(self):
        with self.assertRaises(ValueError):
            AnnotatedTensor(torch.zeros(2, 3, 5), self.ann, axis=-1)

    def test_rejects_an_unsupported_axis(self):
        with self.assertRaises(ValueError):
            AnnotatedTensor(torch.zeros(2, 4), self.ann, axis=0)

    def test_contiguous_labels_slice_without_copying(self):
        t = AnnotatedTensor(torch.arange(48.0).reshape(2, 3, 2, 4), self.ann, axis=-1)
        storage = t.tensor.untyped_storage().data_ptr()
        # A single label and an adjacent run both become plain slices (views).
        self.assertEqual(t['b'].tensor.untyped_storage().data_ptr(), storage)
        self.assertEqual(t['a', 'b'].tensor.untyped_storage().data_ptr(), storage)

    def test_label_membership(self):
        t = AnnotatedTensor(torch.zeros(2, 3, 4), self.ann, axis=-1)
        self.assertIn('b', t)
        self.assertNotIn('zzz', t)

    def test_group_metadata_resolves_to_its_labels(self):
        ann = Annotations(
            labels=['m1', 'm2'],
            cardinalities=[1, 1],
            metadata={'m1': {'variable': 'g'}, 'm2': {'variable': 'g'}},
        )
        t = AnnotatedTensor(torch.arange(12.0).reshape(2, 3, 2), ann, axis=-1)
        # 'g' is not a label, but it names the group both labels came from.
        self.assertIn('g', t)
        self.assertTrue(torch.equal(t['g'].tensor, t.tensor))
        self.assertEqual(t['g'].annotation.labels, ['m1', 'm2'])

    def test_union_with_concatenates_on_the_last_axis(self):
        left = AnnotatedTensor(
            torch.zeros(2, 3, 2), Annotations(labels=['a'], cardinalities=[2]), axis=-1
        )
        right = AnnotatedTensor(
            torch.ones(2, 3, 1), Annotations(labels=['b'], cardinalities=[1]), axis=-1
        )
        merged = left.union_with(right)
        self.assertEqual(tuple(merged.tensor.shape), (2, 3, 3))
        self.assertEqual(merged.annotation.labels, ['a', 'b'])

    def test_requires_explicit_axis_for_more_than_2d(self):
        # The default is only well-defined where axis 1 and -1 coincide (<=2-D).
        # A 3-D+ tensor must name the axis, so concepts can never silently land
        # on the embedding axis of a (batch, concept, embedding) tensor.
        with self.assertRaises(ValueError):
            AnnotatedTensor(torch.zeros(2, 3, 4), self.ann)

    def test_explicit_axis_allows_more_than_2d(self):
        for axis in (1, -1):
            t = AnnotatedTensor(torch.zeros(2, 4, 4), self.ann, axis=axis)
            self.assertEqual(t.axis, axis)

    def test_axis_1_slices_the_concept_axis_of_a_3d_tensor(self):
        # (batch, concepts, embedding): axis=1 addresses the concept axis while
        # the trailing embedding axis rides along untouched.
        ann = Annotations(labels=['p', 'q'])
        t = AnnotatedTensor(torch.randn(8, 2, 5), ann, axis=1)
        self.assertEqual(tuple(t['q'].tensor.shape), (8, 1, 5))
        self.assertTrue(torch.equal(t['q'].tensor, t.tensor[:, 1:2]))

    def test_annotation_survives_unsqueeze_and_expand(self):
        # Adding a leading dim keeps the last (annotated) axis, so the labels
        # survive — the mechanism the multiple-leading-dims examples rely on.
        t = AnnotatedTensor(torch.randn(5, 4), self.ann, axis=-1)
        u = t.unsqueeze(0)
        self.assertIsInstance(u, AnnotatedTensor)
        self.assertEqual(u.annotation.labels, ['a', 'b', 'c'])
        e = u.expand(3, 5, 4)
        self.assertIsInstance(e, AnnotatedTensor)
        self.assertEqual(tuple(e['b'].tensor.shape), (3, 5, 2))

    def test_reshape_expanding_leading_keeps_annotation(self):
        # (batch, K) -> (*leading, K): the ``_restore_leading`` round trip used by
        # the estimator/Pyro engines. Raising the rank keeps the last axis, so
        # the annotation must survive.
        ann = Annotations(labels=['a', 'b', 'c'])
        r = AnnotatedTensor(torch.randn(6, 3), ann, axis=-1).reshape(2, 3, 3)
        self.assertIsInstance(r, AnnotatedTensor)
        self.assertEqual(r.annotation.labels, ['a', 'b', 'c'])

    def test_reduction_dropping_an_axis_on_3d_drops_annotation(self):
        # The mislabel hazard: on a 3-D source a reduction can slide a different,
        # equal-sized axis into the annotated slot. A size-only match would keep
        # a wrongly-labelled tensor, so a drop in rank must drop the annotation.
        ann = Annotations(labels=['a', 'b', 'c'])
        # axis=1: sum(dim=1) removes the concept axis; the last (size-3) axis
        # would slide into position 1.
        self.assertNotIsInstance(
            AnnotatedTensor(torch.randn(7, 3, 3), ann, axis=1).sum(dim=1),
            AnnotatedTensor,
        )
        # axis=-1: sum(dim=-1) removes the concept axis; the middle (size-3) axis
        # would slide to the end.
        self.assertNotIsInstance(
            AnnotatedTensor(torch.randn(7, 3, 3), ann, axis=-1).sum(dim=-1),
            AnnotatedTensor,
        )

    def test_moving_the_annotated_axis_drops_annotation(self):
        # A move that shifts the annotated axis out of its slot returns a plain
        # tensor (re-annotate explicitly if it still applies).
        ann = Annotations(labels=['a', 'b', 'c'])
        t = AnnotatedTensor(torch.randn(7, 3, 3), ann, axis=-1)
        for moved in (t.transpose(1, 2), t.permute(0, 2, 1),
                      t.movedim(-1, 0), t.swapaxes(1, 2)):
            self.assertNotIsInstance(moved, AnnotatedTensor)
        self.assertNotIsInstance(t.mT, AnnotatedTensor)  # property, already plain

    def test_permuting_only_leading_axes_keeps_annotation(self):
        # A move that leaves the annotated (last) axis in place keeps the labels.
        ann = Annotations(labels=['a', 'b', 'c'])
        t = AnnotatedTensor(torch.randn(2, 5, 3), ann, axis=-1)
        moved = t.transpose(0, 1)  # swaps the two leading axes; last axis intact
        self.assertIsInstance(moved, AnnotatedTensor)
        self.assertEqual(moved.annotation.labels, ['a', 'b', 'c'])
        self.assertEqual(tuple(moved.tensor.shape), (5, 2, 3))
        self.assertTrue(torch.equal(moved.tensor, t.tensor.transpose(0, 1)))


class TestDerivedCaches(unittest.TestCase):
    """groupby_metadata / subset / label-slicing are memoised on the annotation.

    The structural fields they read are write-once, but ``metadata`` is not, so
    every cache derived from it must be dropped when it is reassigned.
    """

    def _annotation(self):
        return Annotations(
            labels=['a', 'b', 'c'],
            cardinalities=[1, 1, 1],
            types=['binary'] * 3,
            metadata={
                'a': {'variable': 'plate'},
                'b': {'variable': 'plate'},
                'c': {'variable': 'c'},
            },
        )

    # ------------------------------------------------------------ groupby --

    def test_groupby_metadata_is_memoised_and_correct(self):
        ann = self._annotation()
        first = ann.groupby_metadata('variable')
        self.assertEqual(first, {'plate': ['a', 'b'], 'c': ['c']})
        self.assertIs(ann.groupby_metadata('variable'), first)

    def test_groupby_metadata_keys_on_layout(self):
        ann = self._annotation()
        self.assertEqual(ann.groupby_metadata('variable', layout='labels'),
                         {'plate': ['a', 'b'], 'c': ['c']})
        self.assertEqual(ann.groupby_metadata('variable', layout='indices'),
                         {'plate': [0, 1], 'c': [2]})

    def test_groupby_metadata_invalidated_on_metadata_reassignment(self):
        ann = self._annotation()
        self.assertEqual(ann.groupby_metadata('variable'),
                         {'plate': ['a', 'b'], 'c': ['c']})
        ann.metadata = {'a': {'variable': 'x'},
                        'b': {'variable': 'y'},
                        'c': {'variable': 'y'}}
        self.assertEqual(ann.groupby_metadata('variable'),
                         {'x': ['a'], 'y': ['b', 'c']})

    # ------------------------------------------------------------- subset --

    def test_subset_is_memoised_and_correct(self):
        ann = self._annotation()
        first = ann.subset(['b', 'a'])
        self.assertEqual(first.labels, ['b', 'a'])
        self.assertIs(ann.subset(['b', 'a']), first)
        # a different request is a different entry, not the cached one
        self.assertEqual(ann.subset(['a', 'b']).labels, ['a', 'b'])
        self.assertEqual(ann.subset(['c']).labels, ['c'])

    def test_subset_still_rejects_unknown_labels(self):
        ann = self._annotation()
        with self.assertRaises(ValueError):
            ann.subset(['nope'])

    def test_subset_invalidated_on_metadata_reassignment(self):
        ann = self._annotation()
        self.assertEqual(ann.subset(['a']).metadata, {'a': {'variable': 'plate'}})
        ann.metadata = {'a': {'variable': 'x'}, 'b': {}, 'c': {}}
        self.assertEqual(ann.subset(['a']).metadata, {'a': {'variable': 'x'}})

    # ------------------------------------------- AnnotatedTensor slicing --

    def test_label_and_group_slicing_are_cached_but_track_the_data(self):
        ann = self._annotation()
        data = torch.arange(3).float().expand(4, 3)
        t = AnnotatedTensor(data, ann, axis=-1)

        # repeated access resolves to the same columns every time
        for _ in range(3):
            self.assertEqual(t['a'].annotation.labels, ['a'])
            self.assertTrue(torch.equal(t['a'].tensor, data[:, 0:1]))
            self.assertEqual(t['plate'].annotation.labels, ['a', 'b'])
            self.assertTrue(torch.equal(t['plate'].tensor, data[:, 0:2]))
            self.assertEqual(t['c', 'a'].annotation.labels, ['c', 'a'])

        # a second tensor sharing the annotation reuses the resolution but
        # must slice *its own* data
        other = AnnotatedTensor(torch.ones(4, 3), ann, axis=-1)
        self.assertTrue(torch.equal(other['plate'].tensor, torch.ones(4, 2)))
        self.assertTrue(torch.equal(t['plate'].tensor, data[:, 0:2]))

    def test_group_slicing_invalidated_on_metadata_reassignment(self):
        ann = self._annotation()
        data = torch.arange(3).float().expand(4, 3)
        t = AnnotatedTensor(data, ann, axis=-1)
        self.assertEqual(t['plate'].annotation.labels, ['a', 'b'])

        # 'plate' now covers all three labels
        ann.metadata = {l: {'variable': 'plate'} for l in ('a', 'b', 'c')}
        self.assertEqual(t['plate'].annotation.labels, ['a', 'b', 'c'])
        self.assertTrue(torch.equal(t['plate'].tensor, data))

    def test_unknown_key_still_raises_after_caching(self):
        ann = self._annotation()
        t = AnnotatedTensor(torch.randn(4, 3), ann, axis=-1)
        t['a']  # populate the cache first
        with self.assertRaises(ValueError):
            t['not_a_label']
