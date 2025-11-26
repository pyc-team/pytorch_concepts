"""
Comprehensive tests for torch_concepts/annotations.py

This test suite covers:
- AxisAnnotation: initialization, validation, properties, and methods
- Annotations: multi-axis annotation container functionality
"""
import unittest
import warnings
import pytest
from torch_concepts.annotations import AxisAnnotation, Annotations


class TestAxisAnnotation(unittest.TestCase):
    """Test suite for AxisAnnotation class."""

    def test_binary_concepts_initialization(self):
        """Test initialization of binary concepts (non-nested)."""
        axis = AxisAnnotation(labels=['has_wheels', 'has_windows', 'is_red'])

        self.assertEqual(axis.labels, ['has_wheels', 'has_windows', 'is_red'])
        self.assertFalse(axis.is_nested)
        self.assertEqual(axis.cardinalities, [1, 1, 1])
        self.assertEqual(len(axis), 3)
        self.assertEqual(axis.shape, 3)

    def test_nested_concepts_with_states(self):
        """Test initialization of nested concepts with explicit states."""
        axis = AxisAnnotation(
            labels=['color', 'shape', 'size'],
            states=[['red', 'green', 'blue'], ['circle', 'square', 'triangle'], ['small', 'large']]
        )

        self.assertEqual(axis.labels, ['color', 'shape', 'size'])
        self.assertTrue(axis.is_nested)
        self.assertEqual(axis.cardinalities, [3, 3, 2])  # When only states provided, cardinality is length of states
        self.assertEqual(axis.states, [['red', 'green', 'blue'], ['circle', 'square', 'triangle'], ['small', 'large']])
        self.assertEqual(axis.shape, 8)  # 3 + 3 + 2

    def test_nested_concepts_with_cardinalities(self):
        """Test initialization of nested concepts with only cardinalities."""
        axis = AxisAnnotation(
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
        axis = AxisAnnotation(
            labels=['color',],
            states=(('red', 'green', 'blue'),),
            cardinalities=[3,]
        )
        self.assertEqual(axis.cardinalities, [3,])

        # Invalid: cardinalities don't match states
        with self.assertRaises(ValueError) as context:
            AxisAnnotation(
                labels=['color',],
                states=(('red', 'green', 'blue'),),
                cardinalities=[2,]
            )
        self.assertIn("don't match", str(context.exception))

    def test_invalid_states_length(self):
        """Test error when states length doesn't match labels length."""
        with self.assertRaises(ValueError) as context:
            AxisAnnotation(
                labels=['color', 'shape'],
                states=(('red', 'green', 'blue'),)  # Missing state tuple for 'shape'
            )
        self.assertIn("must match", str(context.exception))

    def test_invalid_cardinalities_length(self):
        """Test error when cardinalities length doesn't match labels length."""
        with self.assertRaises(ValueError) as context:
            AxisAnnotation(
                labels=['color', 'shape'],
                cardinalities=[3,]  # Missing cardinality for 'shape'
            )
        self.assertIn("must match", str(context.exception))

    def test_no_states_no_cardinalities_warning(self):
        """Test warning when neither states nor cardinalities provided."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            axis = AxisAnnotation(labels=['concept1', 'concept2'])

            self.assertEqual(len(w), 1)
            self.assertIn("binary", str(w[0].message))
            self.assertEqual(axis.cardinalities, [1, 1])

    def test_get_index_and_label(self):
        """Test get_index and get_label methods."""
        axis = AxisAnnotation(labels=['a', 'b', 'c'])

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
        """Test __getitem__ method."""
        axis = AxisAnnotation(labels=['a', 'b', 'c'])

        self.assertEqual(axis[0], 'a')
        self.assertEqual(axis[1], 'b')
        self.assertEqual(axis[2], 'c')

        with self.assertRaises(IndexError):
            _ = axis[5]

    def test_get_total_cardinality(self):
        """Test get_total_cardinality method."""
        axis_nested = AxisAnnotation(
            labels=['color', 'shape'],
            cardinalities=[3, 2]
        )
        self.assertEqual(axis_nested.get_total_cardinality(), 5)

        axis_flat = AxisAnnotation(labels=['a', 'b', 'c'])
        self.assertEqual(axis_flat.get_total_cardinality(), 3)

    def test_metadata(self):
        """Test metadata handling."""
        metadata = {
            'color': {'type': 'discrete', 'group': 'appearance'},
            'shape': {'type': 'discrete', 'group': 'geometry'}
        }
        axis = AxisAnnotation(
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
            AxisAnnotation(
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
        axis = AxisAnnotation(
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
        axis = AxisAnnotation(
            labels=['color', 'shape'],
            states=[['red', 'green', 'blue'], ['circle', 'square', 'triangle']],
            metadata={'color': {'type': 'discrete'}, 'shape': {'type': 'discrete'}}
        )

        # Serialize
        data = axis.to_dict()
        self.assertEqual(data['labels'], ['color', 'shape'])

        # Deserialize
        axis_restored = AxisAnnotation.from_dict(data)
        self.assertEqual(axis_restored.labels, axis.labels)
        self.assertEqual(axis_restored.states, axis.states)
        self.assertEqual(axis_restored.cardinalities, axis.cardinalities)

    def test_repr(self):
        """Test __repr__ method."""
        axis = AxisAnnotation(labels=['a', 'b'])
        repr_str = repr(axis)
        self.assertIn('AxisAnnotation', repr_str)
        self.assertIn('a', repr_str)

    def test_str(self):
        """Test __str__ method."""
        axis = AxisAnnotation(labels=['concept1', 'concept2'])
        str_output = str(axis)
        self.assertIsInstance(str_output, str)
        self.assertIn('concept1', str_output)


class TestAnnotations(unittest.TestCase):
    """Test suite for Annotations class."""

    def test_initialization_empty(self):
        """Test initialization with no axes."""
        annotations = Annotations()
        self.assertEqual(len(annotations.axis_annotations), 0)

    def test_initialization_with_axes(self):
        """Test initialization with axis annotations."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        axis2 = AxisAnnotation(labels=['x', 'y', 'z'])

        annotations = Annotations(axis_annotations={1: axis1, 2: axis2})
        self.assertEqual(len(annotations.axis_annotations), 2)
        self.assertIn(1, annotations.axis_annotations)
        self.assertIn(2, annotations.axis_annotations)

    def test_getitem(self):
        """Test __getitem__ method."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        annotations = Annotations(axis_annotations={1: axis1})

        retrieved = annotations[1]
        self.assertEqual(retrieved, axis1)

    def test_setitem(self):
        """Test __setitem__ method."""
        annotations = Annotations()
        axis1 = AxisAnnotation(labels=['a', 'b'])

        annotations[1] = axis1
        self.assertEqual(annotations[1], axis1)

    def test_delitem(self):
        """Test __delitem__ method."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        annotations = Annotations(axis_annotations={1: axis1})

        del annotations[1]
        self.assertNotIn(1, annotations.axis_annotations)

    def test_contains(self):
        """Test __contains__ method."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        annotations = Annotations(axis_annotations={1: axis1})

        self.assertTrue(1 in annotations)
        self.assertFalse(2 in annotations)

    def test_len(self):
        """Test __len__ method."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        axis2 = AxisAnnotation(labels=['x', 'y'])
        annotations = Annotations(axis_annotations={1: axis1, 2: axis2})

        self.assertEqual(len(annotations), 2)

    def test_iter(self):
        """Test __iter__ method."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        axis2 = AxisAnnotation(labels=['x', 'y'])
        annotations = Annotations(axis_annotations={1: axis1, 2: axis2})

        keys = list(annotations)
        self.assertEqual(sorted(keys), [1, 2])

    def test_keys(self):
        """Test keys method."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        annotations = Annotations(axis_annotations={1: axis1})

        keys = list(annotations.keys())
        self.assertEqual(keys, [1])

    def test_values(self):
        """Test values method."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        annotations = Annotations(axis_annotations={1: axis1})

        values = list(annotations.values())
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], axis1)

    def test_items(self):
        """Test items method."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        annotations = Annotations(axis_annotations={1: axis1})

        items = list(annotations.items())
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0], (1, axis1))

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        axis2 = AxisAnnotation(labels=['x', 'y', 'z'])
        annotations = Annotations(axis_annotations={1: axis1, 2: axis2})

        # Serialize
        data = annotations.to_dict()
        self.assertIn('axis_annotations', data)

        # Deserialize
        annotations_restored = Annotations.from_dict(data)
        self.assertEqual(len(annotations_restored), len(annotations))

    def test_multiple_axes(self):
        """Test with multiple axis annotations."""
        axis0 = AxisAnnotation(labels=['batch',])
        axis1 = AxisAnnotation(labels=['color', 'shape'])
        axis2 = AxisAnnotation(labels=['x', 'y', 'z'])

        annotations = Annotations(axis_annotations={0: axis0, 1: axis1, 2: axis2})
        self.assertEqual(len(annotations), 3)

    def test_nested_concepts_in_annotations(self):
        """Test annotations with nested concepts."""
        axis = AxisAnnotation(
            labels=['color', 'shape'],
            cardinalities=[3, 4]
        )
        annotations = Annotations(axis_annotations={1: axis})

        self.assertTrue(annotations[1].is_nested)

    def test_repr(self):
        """Test __repr__ method."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        annotations = Annotations(axis_annotations={1: axis1})

        repr_str = repr(annotations)
        self.assertIsInstance(repr_str, str)
        self.assertIn('Annotations', repr_str)

    def test_str(self):
        """Test __str__ method."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        annotations = Annotations(axis_annotations={1: axis1})

        str_output = str(annotations)
        self.assertIsInstance(str_output, str)

    def test_empty_annotations_operations(self):
        """Test operations on empty annotations."""
        annotations = Annotations()

        self.assertEqual(len(annotations), 0)
        self.assertEqual(list(annotations.keys()), [])
        self.assertEqual(list(annotations.values()), [])


class TestAxisAnnotationEdgeCases(unittest.TestCase):
    """Test edge cases for AxisAnnotation."""

    def test_single_label(self):
        """Test with single label."""
        axis = AxisAnnotation(labels=['single',])
        self.assertEqual(len(axis), 1)
        self.assertEqual(axis[0], 'single')

    def test_many_labels(self):
        """Test with many labels."""
        labels = tuple(f'label_{i}' for i in range(100))
        axis = AxisAnnotation(labels=labels)
        self.assertEqual(len(axis), 100)

    def test_large_cardinality(self):
        """Test with large cardinality."""
        axis = AxisAnnotation(
            labels=['concept',],
            cardinalities=[1000,]
        )
        self.assertEqual(axis.cardinalities[0], 1000)
        self.assertEqual(len(axis.states[0]), 1000)

    def test_mixed_cardinalities(self):
        """Test with mixed cardinalities (binary and multi-class)."""
        axis = AxisAnnotation(
            labels=['binary', 'ternary', 'quad', 'many'],
            cardinalities=[1, 3, 4, 10]
        )
        self.assertEqual(axis.cardinalities, [1, 3, 4, 10])

    def test_get_label_negative_index(self):
        """Test get_label with negative index."""
        axis = AxisAnnotation(labels=['a', 'b', 'c'])
        # Negative indexing might not be supported
        with self.assertRaises((IndexError, ValueError)):
            axis.get_label(-1)

    def test_duplicate_labels_warning(self):
        """Test warning or error with duplicate labels."""
        # Depending on implementation, this might raise or warn
        try:
            axis = AxisAnnotation(labels=['a', 'b', 'a'])
            # If no error, check behavior
            self.assertEqual(len(axis.labels), 3)
        except ValueError:
            pass  # Expected if duplicates not allowed

    def test_empty_metadata(self):
        """Test with empty metadata dict."""
        axis = AxisAnnotation(
            labels=['a', 'b'],
            metadata={}
        )
        # Should work or raise error
        self.assertEqual(len(axis.labels), 2)

    def test_special_characters_in_labels(self):
        """Test labels with special characters."""
        axis = AxisAnnotation(labels=['label-1', 'label_2', 'label.3', 'label@4'])
        self.assertEqual(len(axis), 4)

    def test_unicode_labels(self):
        """Test labels with unicode characters."""
        axis = AxisAnnotation(labels=['色彩', 'форма', '🎨'])
        self.assertEqual(len(axis), 3)

    def test_very_long_label_names(self):
        """Test with very long label names."""
        long_label = 'a' * 1000
        axis = AxisAnnotation(labels=[long_label, 'short'])
        self.assertEqual(axis[0], long_label)

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


class TestAxisAnnotationExtended:
    """Extended tests for AxisAnnotation class to improve coverage."""

    def test_cardinality_mismatch_with_states(self):
        """Test that mismatched cardinalities and states raise error."""
        with pytest.raises(ValueError, match="don't match inferred cardinalities"):
            AxisAnnotation(
                labels=['a', 'b'],
                states=[['x', 'y'], ['p', 'q', 'r']],
                cardinalities=[2, 2]  # Should be [2, 3] based on states
            )

    def test_metadata_validation_non_dict(self):
        """Test that non-dict metadata raises error."""
        with pytest.raises(ValueError, match="metadata must be a dictionary"):
            AxisAnnotation(
                labels=['a', 'b'],
                metadata="invalid"  # Should be dict
            )

    def test_metadata_validation_missing_label(self):
        """Test that metadata missing a label raises error."""
        with pytest.raises(ValueError, match="Metadata missing for label"):
            AxisAnnotation(
                labels=['a', 'b', 'c'],
                metadata={'a': {}, 'b': {}}  # Missing 'c'
            )

    def test_has_metadata_with_key(self):
        """Test has_metadata method with specific key."""
        axis = AxisAnnotation(
            labels=['a', 'b'],
            metadata={'a': {'type': 'binary'}, 'b': {'type': 'binary'}}
        )
        assert axis.has_metadata('type') is True
        assert axis.has_metadata('missing_key') is False

    def test_has_metadata_none(self):
        """Test has_metadata when metadata is None."""
        axis = AxisAnnotation(labels=['a', 'b'])
        assert axis.has_metadata('any_key') is False

    def test_groupby_metadata_labels_layout(self):
        """Test groupby_metadata with labels layout."""
        axis = AxisAnnotation(
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
        axis = AxisAnnotation(
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
        axis = AxisAnnotation(
            labels=['a', 'b'],
            metadata={'a': {'g': '1'}, 'b': {'g': '2'}}
        )
        with pytest.raises(ValueError, match="Unknown layout"):
            axis.groupby_metadata('g', layout='invalid')

    def test_groupby_metadata_none(self):
        """Test groupby_metadata when metadata is None."""
        axis = AxisAnnotation(labels=['a', 'b'])
        result = axis.groupby_metadata('any_key')
        assert result == {}

    def test_get_index_not_found(self):
        """Test get_index with non-existent label."""
        axis = AxisAnnotation(labels=['a', 'b', 'c'])
        with pytest.raises(ValueError, match="Label 'z' not found"):
            axis.get_index('z')

    def test_get_label_out_of_range(self):
        """Test get_label with out-of-range index."""
        axis = AxisAnnotation(labels=['a', 'b'])
        with pytest.raises(IndexError, match="Index 5 out of range"):
            axis.get_label(5)

    def test_getitem_out_of_range(self):
        """Test __getitem__ with out-of-range index."""
        axis = AxisAnnotation(labels=['a', 'b'])
        with pytest.raises(IndexError, match="Index 10 out of range"):
            _ = axis[10]

    def test_get_total_cardinality_nested(self):
        """Test get_total_cardinality for nested axis."""
        axis = AxisAnnotation(
            labels=['a', 'b', 'c'],
            cardinalities=[2, 3, 4]
        )
        assert axis.get_total_cardinality() == 9

    def test_get_total_cardinality_not_nested(self):
        """Test get_total_cardinality for non-nested axis."""
        axis = AxisAnnotation(labels=['a', 'b', 'c'])
        assert axis.get_total_cardinality() == 3

    def test_to_dict_with_all_fields(self):
        """Test to_dict with all fields populated."""
        axis = AxisAnnotation(
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
        """Test from_dict reconstructs AxisAnnotation correctly."""
        original = AxisAnnotation(
            labels=['x', 'y'],
            cardinalities=[2, 3],
            metadata={'x': {'info': 'test'}, 'y': {'info': 'test2'}}
        )

        data = original.to_dict()
        reconstructed = AxisAnnotation.from_dict(data)

        assert reconstructed.labels == original.labels
        assert reconstructed.cardinalities == original.cardinalities
        assert reconstructed.is_nested == original.is_nested
        assert reconstructed.metadata == original.metadata

    def test_subset_basic(self):
        """Test subset method with valid labels."""
        axis = AxisAnnotation(
            labels=['a', 'b', 'c', 'd'],
            cardinalities=[1, 2, 3, 1]
        )

        subset = axis.subset(['b', 'd'])

        assert subset.labels == ['b', 'd']
        assert subset.cardinalities == [2, 1]

    def test_subset_with_metadata(self):
        """Test subset preserves metadata."""
        axis = AxisAnnotation(
            labels=['a', 'b', 'c'],
            metadata={'a': {'x': 1}, 'b': {'x': 2}, 'c': {'x': 3}}
        )

        subset = axis.subset(['a', 'c'])

        assert subset.labels == ['a', 'c']
        assert subset.metadata == {'a': {'x': 1}, 'c': {'x': 3}}

    def test_subset_missing_labels(self):
        """Test subset with non-existent labels raises error."""
        axis = AxisAnnotation(labels=['a', 'b', 'c'])

        with pytest.raises(ValueError, match="Unknown labels for subset"):
            axis.subset(['a', 'z'])

    def test_subset_preserves_order(self):
        """Test subset preserves the requested label order."""
        axis = AxisAnnotation(labels=['a', 'b', 'c', 'd'])

        subset = axis.subset(['d', 'b', 'a'])

        assert subset.labels == ['d', 'b', 'a']

    def test_union_with_no_overlap(self):
        """Test union_with with no overlapping labels."""
        axis1 = AxisAnnotation(labels=['a', 'b'])
        axis2 = AxisAnnotation(labels=['c', 'd'])

        union = axis1.union_with(axis2)

        assert union.labels == ['a', 'b', 'c', 'd']

    def test_union_with_overlap(self):
        """Test union_with with overlapping labels."""
        axis1 = AxisAnnotation(labels=['a', 'b', 'c'])
        axis2 = AxisAnnotation(labels=['b', 'c', 'd'])

        union = axis1.union_with(axis2)

        assert union.labels == ['a', 'b', 'c', 'd']

    def test_union_with_metadata_merge(self):
        """Test union_with merges metadata with left-win."""
        axis1 = AxisAnnotation(
            labels=['a', 'b'],
            metadata={'a': {'x': 1}, 'b': {'x': 2}}
        )
        axis2 = AxisAnnotation(
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
        axis = AxisAnnotation(labels=['a', 'b'])

        with pytest.raises(AttributeError, match="write-once and already set"):
            axis.labels = ['x', 'y']

    def test_write_once_states_attribute(self):
        """Test that states attribute is write-once."""
        axis = AxisAnnotation(labels=['a', 'b'], cardinalities=[2, 3])

        with pytest.raises(AttributeError, match="write-once and already set"):
            axis.states = [['0', '1'], ['0', '1', '2']]

    def test_metadata_can_be_modified(self):
        """Test that metadata can be modified after creation."""
        axis = AxisAnnotation(labels=['a', 'b'])

        # Metadata is not write-once, so this should work
        axis.metadata = {'a': {'test': 1}, 'b': {'test': 2}}
        assert axis.metadata is not None


class TestAnnotationsExtended:
    """Extended tests for Annotations class to improve coverage."""

    def test_annotations_with_dict_input(self):
        """Test Annotations with dict input."""
        axis0 = AxisAnnotation(labels=['batch'])
        axis1 = AxisAnnotation(labels=['a', 'b', 'c'])

        annotations = Annotations({0: axis0, 1: axis1})

        assert 0 in annotations._axis_annotations
        assert 1 in annotations._axis_annotations

    def test_annotations_with_list_input(self):
        """Test Annotations with list input."""
        axis0 = AxisAnnotation(labels=['a', 'b'])
        axis1 = AxisAnnotation(labels=['x', 'y', 'z'])

        annotations = Annotations([axis0, axis1])

        assert len(annotations._axis_annotations) == 2
        assert annotations._axis_annotations[0].labels == ['a', 'b']
        assert annotations._axis_annotations[1].labels == ['x', 'y', 'z']

    def test_annotations_getitem(self):
        """Test Annotations __getitem__ method."""
        axis = AxisAnnotation(labels=['a', 'b', 'c'])
        annotations = Annotations({1: axis})

        retrieved = annotations[1]
        assert retrieved.labels == ['a', 'b', 'c']

    def test_annotations_setitem(self):
        """Test Annotations __setitem__ method."""
        annotations = Annotations({})
        axis = AxisAnnotation(labels=['x', 'y'])

        annotations[2] = axis

        assert annotations[2].labels == ['x', 'y']

    def test_annotations_len(self):
        """Test Annotations __len__ method."""
        axis0 = AxisAnnotation(labels=['a'])
        axis1 = AxisAnnotation(labels=['b'])
        axis2 = AxisAnnotation(labels=['c'])

        annotations = Annotations({0: axis0, 1: axis1, 2: axis2})

        assert len(annotations) == 3

    def test_annotations_iter(self):
        """Test Annotations __iter__ method."""
        axis0 = AxisAnnotation(labels=['a'])
        axis1 = AxisAnnotation(labels=['b'])

        annotations = Annotations({0: axis0, 1: axis1})

        axes = list(annotations)
        assert len(axes) == 2

    def test_annotations_contains(self):
        """Test Annotations __contains__ method."""
        axis = AxisAnnotation(labels=['a', 'b'])
        annotations = Annotations({1: axis})

        assert 1 in annotations
        assert 0 not in annotations
        assert 5 not in annotations


if __name__ == '__main__':
    unittest.main()
