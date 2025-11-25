"""Extended tests for torch_concepts.annotations module to improve coverage."""

import pytest
import torch
from torch_concepts.annotations import AxisAnnotation, Annotations


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

