"""
Comprehensive tests for torch_concepts/annotations.py

This test suite covers:
- AxisAnnotation: initialization, validation, properties, and methods
- Annotations: multi-axis annotation container functionality
"""
import unittest
import warnings
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


if __name__ == '__main__':
    unittest.main()
