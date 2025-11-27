import unittest
import torch
from torch import nn
import pytest
import torch
import numpy as np
import pandas as pd
from torch_concepts.data.utils import (
    ensure_list,
    files_exist,
    parse_tensor,
    convert_precision,
    resolve_size,
    colorize,
    affine_transform,
    transform_images,
    assign_random_values,
)
import tempfile
import os

import numpy as np
from torch_concepts.data.utils import (
    assign_values_based_on_intervals,
    colorize_and_transform,
)


class TestEnsureList(unittest.TestCase):
    """Test suite for ensure_list utility function."""

    def test_list_remains_list(self):
        """Test that a list remains unchanged."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])
        
    def test_tuple_converts_to_list(self):
        """Test that a tuple is converted to list."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list((1, 2, 3))
        self.assertEqual(result, [1, 2, 3])
        self.assertIsInstance(result, list)
        
    def test_single_value_wraps_in_list(self):
        """Test that a single value is wrapped in a list."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list(5)
        self.assertEqual(result, [5])
        
        result = ensure_list(3.14)
        self.assertEqual(result, [3.14])
        
    def test_string_wraps_in_list(self):
        """Test that a string is wrapped (not converted to list of chars)."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list('hello')
        self.assertEqual(result, ['hello'])
        self.assertEqual(len(result), 1)
        
    def test_set_converts_to_list(self):
        """Test that a set is converted to list."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list({1, 2, 3})
        self.assertEqual(set(result), {1, 2, 3})
        self.assertIsInstance(result, list)
        
    def test_range_converts_to_list(self):
        """Test that a range is converted to list."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list(range(5))
        self.assertEqual(result, [0, 1, 2, 3, 4])
        
    def test_generator_converts_to_list(self):
        """Test that a generator is consumed and converted to list."""
        from torch_concepts.data.utils import ensure_list
        
        gen = (x * 2 for x in range(3))
        result = ensure_list(gen)
        self.assertEqual(result, [0, 2, 4])
        
    def test_numpy_array_converts_to_list(self):
        """Test that a numpy array is converted to list."""
        from torch_concepts.data.utils import ensure_list
        import numpy as np
        
        arr = np.array([1, 2, 3])
        result = ensure_list(arr)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result, list)
        
    def test_torch_tensor_converts_to_list(self):
        """Test that a torch tensor is converted to list."""
        from torch_concepts.data.utils import ensure_list
        
        tensor = torch.tensor([1, 2, 3])
        result = ensure_list(tensor)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result, list)
        
    def test_none_wraps_in_list(self):
        """Test that None is wrapped in a list."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list(None)
        self.assertEqual(result, [None])
        
    def test_nested_list_preserved(self):
        """Test that nested lists are preserved."""
        from torch_concepts.data.utils import ensure_list
        
        nested = [[1, 2], [3, 4]]
        result = ensure_list(nested)
        self.assertEqual(result, [[1, 2], [3, 4]])
        
    def test_dict_raises_error(self):
        """Test that a dict raises TypeError with helpful message."""
        from torch_concepts.data.utils import ensure_list
        
        with self.assertRaises(TypeError) as context:
            ensure_list({'a': 1, 'b': 2})
        
        self.assertIn('Cannot convert dict to list', str(context.exception))
        self.assertIn('keys', str(context.exception))
        self.assertIn('values', str(context.exception))
        
    def test_empty_list_remains_empty(self):
        """Test that an empty list remains empty."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list([])
        self.assertEqual(result, [])
        
    def test_empty_tuple_converts_to_empty_list(self):
        """Test that an empty tuple converts to empty list."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list(())
        self.assertEqual(result, [])


class TestEnsureList:
    """Test ensure_list function."""

    def test_list_input(self):
        """Test that lists remain unchanged."""
        result = ensure_list([1, 2, 3])
        assert result == [1, 2, 3]

    def test_tuple_input(self):
        """Test tuple conversion to list."""
        result = ensure_list((1, 2, 3))
        assert result == [1, 2, 3]

    def test_single_value(self):
        """Test single value wrapping."""
        result = ensure_list(5)
        assert result == [5]

    def test_string_input(self):
        """Test that strings are wrapped, not split."""
        result = ensure_list("hello")
        assert result == ["hello"]

    def test_dict_raises_error(self):
        """Test that dict conversion raises TypeError."""
        with pytest.raises(TypeError, match="Cannot convert dict to list"):
            ensure_list({'a': 1, 'b': 2})

    def test_set_input(self):
        """Test set conversion to list."""
        result = ensure_list({1, 2, 3})
        assert set(result) == {1, 2, 3}

    def test_numpy_array(self):
        """Test numpy array conversion."""
        arr = np.array([1, 2, 3])
        result = ensure_list(arr)
        assert result == [1, 2, 3]


class TestFilesExist:
    """Test files_exist function."""

    def test_existing_files(self):
        """Test with existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.txt")
            file2 = os.path.join(tmpdir, "file2.txt")

            with open(file1, 'w') as f:
                f.write("test")
            with open(file2, 'w') as f:
                f.write("test")

            assert files_exist([file1, file2]) is True

    def test_nonexistent_file(self):
        """Test with non-existent file."""
        result = files_exist(["/nonexistent/file.txt"])
        assert result is False

    def test_mixed_files(self):
        """Test with mix of existing and non-existent files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing = os.path.join(tmpdir, "exists.txt")
            with open(existing, 'w') as f:
                f.write("test")

            nonexisting = os.path.join(tmpdir, "does_not_exist.txt")
            assert files_exist([existing, nonexisting]) is False

    def test_empty_list(self):
        """Test with empty list (vacuous truth)."""
        assert files_exist([]) is True


class TestParseTensor:
    """Test parse_tensor function."""

    def test_numpy_input(self):
        """Test numpy array conversion."""
        arr = np.array([[1, 2], [3, 4]])
        result = parse_tensor(arr, "test", 32)
        assert isinstance(result, torch.Tensor)
        # Note: precision might not change dtype automatically
        assert result.shape == (2, 2)

    def test_dataframe_input(self):
        """Test pandas DataFrame conversion."""
        df = pd.DataFrame([[1, 2], [3, 4]])
        result = parse_tensor(df, "test", 32)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2)

    def test_tensor_input(self):
        """Test tensor passthrough with precision conversion."""
        tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
        result = parse_tensor(tensor, "test", 32)
        # Check it's still a tensor
        assert isinstance(result, torch.Tensor)

    def test_invalid_input(self):
        """Test invalid input type raises error."""
        with pytest.raises(AssertionError):
            parse_tensor([1, 2, 3], "test", 32)


class TestConvertPrecision:
    """Test convert_precision function."""

    def test_float32(self):
        """Test conversion to float32."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.float64)
        result = convert_precision(tensor, "float32")
        assert result.dtype == torch.float32

    def test_float64(self):
        """Test conversion to float64."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
        result = convert_precision(tensor, "float64")
        assert result.dtype == torch.float64

    def test_float16(self):
        """Test conversion to float16."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
        result = convert_precision(tensor, "float16")
        assert result.dtype == torch.float16

    def test_no_change(self):
        """Test when precision doesn't change."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
        result = convert_precision(tensor, "unknown")
        assert result.dtype == torch.float32


class TestResolveSize:
    """Test resolve_size function."""

    def test_fractional_size(self):
        """Test fractional size conversion."""
        result = resolve_size(0.2, 100)
        assert result == 20

    def test_absolute_size(self):
        """Test absolute size passthrough."""
        result = resolve_size(50, 100)
        assert result == 50

    def test_zero_fraction(self):
        """Test zero fraction."""
        result = resolve_size(0.0, 100)
        assert result == 0

    def test_one_fraction(self):
        """Test full fraction."""
        result = resolve_size(1.0, 100)
        assert result == 100

    def test_invalid_fraction(self):
        """Test invalid fractional size raises error."""
        with pytest.raises(ValueError, match="Fractional size must be in"):
            resolve_size(1.5, 100)

        with pytest.raises(ValueError, match="Fractional size must be in"):
            resolve_size(-0.1, 100)

    def test_negative_absolute(self):
        """Test negative absolute size raises error."""
        with pytest.raises(ValueError, match="Absolute size must be non-negative"):
            resolve_size(-10, 100)

    def test_invalid_type(self):
        """Test invalid type raises error."""
        with pytest.raises(TypeError, match="Size must be int or float"):
            resolve_size("10", 100)


class TestColorize:
    """Test colorize function."""

    def test_red_channel(self):
        """Test colorization to red channel."""
        images = torch.ones(2, 28, 28)
        colors = torch.tensor([0, 0])  # Red
        result = colorize(images, colors)

        assert result.shape == (2, 3, 28, 28)
        assert torch.all(result[:, 0, :, :] == 1)  # Red channel
        assert torch.all(result[:, 1, :, :] == 0)  # Green channel
        assert torch.all(result[:, 2, :, :] == 0)  # Blue channel

    def test_green_channel(self):
        """Test colorization to green channel."""
        images = torch.ones(2, 28, 28)
        colors = torch.tensor([1, 1])  # Green
        result = colorize(images, colors)

        assert result.shape == (2, 3, 28, 28)
        assert torch.all(result[:, 1, :, :] == 1)  # Green channel
        assert torch.all(result[:, 0, :, :] == 0)  # Red channel
        assert torch.all(result[:, 2, :, :] == 0)  # Blue channel

    def test_blue_channel(self):
        """Test colorization to blue channel."""
        images = torch.ones(2, 28, 28)
        colors = torch.tensor([2, 2])  # Blue
        result = colorize(images, colors)

        assert result.shape == (2, 3, 28, 28)
        assert torch.all(result[:, 2, :, :] == 1)  # Blue channel
        assert torch.all(result[:, 0, :, :] == 0)  # Red channel
        assert torch.all(result[:, 1, :, :] == 0)  # Green channel

    def test_mixed_colors(self):
        """Test colorization with different colors."""
        images = torch.ones(3, 28, 28)
        colors = torch.tensor([0, 1, 2])  # Red, Green, Blue
        result = colorize(images, colors)

        assert result.shape == (3, 3, 28, 28)
        assert torch.all(result[0, 0, :, :] == 1)  # First image in red
        assert torch.all(result[1, 1, :, :] == 1)  # Second image in green
        assert torch.all(result[2, 2, :, :] == 1)  # Third image in blue

    def test_invalid_colors(self):
        """Test that invalid colors raise assertion error."""
        images = torch.ones(2, 28, 28)
        colors = torch.tensor([0, 3])  # 3 is invalid

        with pytest.raises((AssertionError, IndexError)):
            colorize(images, colors)


class TestAffineTransform:
    """Test affine_transform function."""

    def test_rotation(self):
        """Test rotation transformation."""
        images = torch.randn(5, 28, 28)
        degrees = torch.tensor([0.0, 90.0, 180.0, 270.0, 45.0])
        scales = torch.ones(5)

        result = affine_transform(images, degrees, scales)
        assert result.shape == (5, 1, 28, 28)

    def test_scaling(self):
        """Test scaling transformation."""
        images = torch.randn(5, 28, 28)
        degrees = torch.zeros(5)
        scales = torch.tensor([0.5, 1.0, 1.5, 2.0, 0.8])

        result = affine_transform(images, degrees, scales)
        assert result.shape == (5, 1, 28, 28)

    def test_rgb_images(self):
        """Test with RGB images."""
        images = torch.randn(5, 3, 28, 28)
        degrees = torch.zeros(5)
        scales = torch.ones(5)

        result = affine_transform(images, degrees, scales)
        assert result.shape == (5, 3, 28, 28)

    def test_none_degrees(self):
        """Test with None degrees (should default to 0)."""
        images = torch.randn(5, 28, 28)
        scales = torch.ones(5)

        result = affine_transform(images, None, scales)
        assert result.shape == (5, 1, 28, 28)

    def test_none_scales(self):
        """Test with None scales (should default to 1)."""
        images = torch.randn(5, 28, 28)
        degrees = torch.zeros(5)

        result = affine_transform(images, degrees, None)
        assert result.shape == (5, 1, 28, 28)

    def test_batching(self):
        """Test batching with large number of images."""
        images = torch.randn(10, 28, 28)
        degrees = torch.zeros(10)
        scales = torch.ones(10)

        result = affine_transform(images, degrees, scales, batch_size=3)
        assert result.shape == (10, 1, 28, 28)


class TestTransformImages:
    """Test transform_images function."""

    def test_colorize_transformation(self):
        """Test colorize transformation."""
        images = torch.ones(3, 28, 28)
        colors = torch.tensor([0, 1, 2])

        result = transform_images(images, ['colorize'], colors=colors)
        assert result.shape == (3, 3, 28, 28)

    def test_affine_transformation(self):
        """Test affine transformation."""
        images = torch.randn(3, 28, 28)
        degrees = torch.zeros(3)
        scales = torch.ones(3)

        result = transform_images(images, ['affine'], degrees=degrees, scales=scales)
        assert result.shape == (3, 1, 28, 28)

    def test_combined_transformations(self):
        """Test multiple transformations in sequence."""
        images = torch.ones(3, 28, 28)
        colors = torch.tensor([0, 1, 2])
        degrees = torch.zeros(3)
        scales = torch.ones(3)

        result = transform_images(
            images,
            ['colorize', 'affine'],
            colors=colors,
            degrees=degrees,
            scales=scales
        )
        assert result.shape == (3, 3, 28, 28)

    def test_missing_colors(self):
        """Test that missing colors for colorize raises error."""
        images = torch.ones(3, 28, 28)

        with pytest.raises(ValueError, match="Colors must be provided"):
            transform_images(images, ['colorize'])

    def test_unknown_transformation(self):
        """Test unknown transformation raises error."""
        images = torch.randn(3, 28, 28)

        with pytest.raises(ValueError, match="Unknown transformation"):
            transform_images(images, ['invalid_transform'])


class TestAssignRandomValues:
    """Test assign_random_values function."""

    def test_basic_binary(self):
        """Test basic binary random assignment."""
        concept = torch.arange(10)
        result = assign_random_values(concept, random_prob=[0.5, 0.5], values=[0, 1])

        assert result.shape == (10,)
        assert torch.all((result == 0) | (result == 1))

    def test_deterministic(self):
        """Test deterministic assignment."""
        torch.manual_seed(42)
        concept = torch.zeros(100)
        result = assign_random_values(concept, random_prob=[1.0, 0.0], values=[0, 1])

        assert torch.all(result == 0)

    def test_multi_value(self):
        """Test with multiple values."""
        concept = torch.arange(10)
        result = assign_random_values(
            concept,
            random_prob=[0.33, 0.33, 0.34],
            values=[0, 1, 2]
        )

        assert result.shape == (10,)
        assert torch.all((result == 0) | (result == 1) | (result == 2))

    def test_invalid_shape(self):
        """Test that non-1D tensor raises error."""
        concept = torch.zeros(10, 2)

        with pytest.raises(AssertionError, match="concepts must be a 1D tensor"):
            assign_random_values(concept)

    def test_empty_prob(self):
        """Test that empty probability raises error."""
        concept = torch.zeros(10)

        with pytest.raises(AssertionError, match="random_prob must not be empty"):
            assign_random_values(concept, random_prob=[], values=[])

    def test_mismatched_lengths(self):
        """Test that mismatched prob and values raises error."""
        concept = torch.zeros(10)

        with pytest.raises(AssertionError, match="random_prob must have the same length"):
            assign_random_values(concept, random_prob=[0.5, 0.5], values=[0])

    def test_invalid_probabilities(self):
        """Test that invalid probabilities raise error."""
        concept = torch.zeros(10)

        with pytest.raises(AssertionError, match="random_prob must be between 0 and 1"):
            assign_random_values(concept, random_prob=[-0.1, 1.1], values=[0, 1])

    def test_probabilities_not_sum_to_one(self):
        """Test that probabilities not summing to 1 raise error."""
        concept = torch.zeros(10)

        with pytest.raises(AssertionError, match="random_prob must sum to 1"):
            assign_random_values(concept, random_prob=[0.3, 0.3], values=[0, 1])


class TestAssignValuesBasedOnIntervals:
    """Test assign_values_based_on_intervals function."""

    def test_basic_intervals(self):
        """Test basic interval assignment."""
        concept = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        intervals = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
        values = [[0], [1], [2]]

        result = assign_values_based_on_intervals(concept, intervals, values)

        assert result.shape == (10,)
        assert torch.all(result[:3] == 0)
        assert torch.all(result[3:6] == 1)
        assert torch.all(result[6:] == 2)

    def test_multiple_values_per_interval(self):
        """Test intervals with multiple possible output values."""
        torch.manual_seed(42)
        concept = torch.tensor([0, 1, 2, 3, 4, 5])
        intervals = [[0, 1, 2], [3, 4, 5]]
        values = [[0, 1], [2, 3]]

        result = assign_values_based_on_intervals(concept, intervals, values)

        assert result.shape == (6,)
        # First 3 should be 0 or 1
        assert torch.all((result[:3] == 0) | (result[:3] == 1))
        # Last 3 should be 2 or 3
        assert torch.all((result[3:] == 2) | (result[3:] == 3))

    def test_single_element_intervals(self):
        """Test with single element intervals."""
        concept = torch.tensor([0, 1, 2])
        intervals = [[0], [1], [2]]
        values = [[10], [20], [30]]

        result = assign_values_based_on_intervals(concept, intervals, values)

        assert result[0] == 10
        assert result[1] == 20
        assert result[2] == 30

    def test_non_contiguous_concept_values(self):
        """Test with non-contiguous concept values."""
        concept = torch.tensor([1, 5, 9, 1, 5, 9])
        intervals = [[1, 5], [9]]
        values = [[0], [1]]

        result = assign_values_based_on_intervals(concept, intervals, values)

        assert torch.sum(result == 0) == 4
        assert torch.sum(result == 1) == 2

    def test_invalid_concept_shape(self):
        """Test that 2D concept tensor raises error."""
        concept = torch.zeros(10, 2)
        intervals = [[0], [1]]
        values = [[0], [1]]

        with pytest.raises(AssertionError, match="concepts must be a 1D tensor"):
            assign_values_based_on_intervals(concept, intervals, values)

    def test_mismatched_intervals_values_length(self):
        """Test that mismatched intervals and values lengths raise error."""
        concept = torch.tensor([0, 1, 2])
        intervals = [[0, 1], [2]]
        values = [[0]]  # Only 1 value list, but 2 intervals

        with pytest.raises(AssertionError, match="intervals and values must have the same length"):
            assign_values_based_on_intervals(concept, intervals, values)

    def test_overlapping_intervals(self):
        """Test that overlapping intervals raise error."""
        concept = torch.tensor([0, 1, 2, 3])
        intervals = [[0, 1], [1, 2]]  # 1 appears in both
        values = [[0], [1]]

        with pytest.raises(AssertionError, match="input intervals must not overlap"):
            assign_values_based_on_intervals(concept, intervals, values)

    def test_empty_interval(self):
        """Test that empty interval raises error."""
        concept = torch.tensor([0, 1, 2])
        intervals = [[0, 1], []]  # Empty interval
        values = [[0], [1]]

        with pytest.raises(AssertionError, match="each entry in intervals must contain at least one value"):
            assign_values_based_on_intervals(concept, intervals, values)

    def test_empty_values(self):
        """Test that empty values list raises error."""
        concept = torch.tensor([0, 1, 2])
        intervals = [[0, 1], [2]]
        values = [[0], []]  # Empty values

        with pytest.raises(AssertionError, match="each entry in values must contain at least one value"):
            assign_values_based_on_intervals(concept, intervals, values)

    def test_large_dataset(self):
        """Test with larger dataset."""
        concept = torch.randint(0, 10, (1000,))
        intervals = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        values = [[0, 1], [2, 3]]

        result = assign_values_based_on_intervals(concept, intervals, values)

        assert result.shape == (1000,)
        # All values should be in [0, 1, 2, 3]
        assert torch.all((result >= 0) & (result <= 3))


class TestColorizeAndTransform:
    """Test colorize_and_transform function."""

    def test_random_mode_basic(self):
        """Test basic random coloring mode."""
        torch.manual_seed(42)
        data = torch.randn(100, 28, 28)
        targets = torch.randint(0, 10, (100,))

        training_kwargs = [{'random_prob': [0.5, 0.5], 'values': ['red', 'green']}]
        test_kwargs = [{'random_prob': [0.5, 0.5], 'values': ['red', 'green']}]

        embeddings, concepts, out_targets, coloring_mode = colorize_and_transform(
            data, targets,
            training_percentage=0.8,
            test_percentage=0.2,
            training_mode=['random'],
            test_mode=['random'],
            training_kwargs=training_kwargs,
            test_kwargs=test_kwargs
        )

        assert embeddings.shape == (100, 3, 28, 28)
        assert 'colors' in concepts
        assert len(out_targets) == 100
        assert len(coloring_mode) == 100
        assert coloring_mode.count('training') == 80
        assert coloring_mode.count('test') == 20

    def test_random_mode_uniform(self):
        """Test random coloring with uniform probability."""
        torch.manual_seed(42)
        data = torch.randn(50, 28, 28)
        targets = torch.randint(0, 10, (50,))

        training_kwargs = [{'random_prob': ['uniform'], 'values': ['red', 'green', 'blue']}]
        test_kwargs = [{'random_prob': ['uniform'], 'values': ['red', 'green', 'blue']}]

        embeddings, concepts, out_targets, coloring_mode = colorize_and_transform(
            data, targets,
            training_percentage=0.6,
            test_percentage=0.4,
            training_mode=['random'],
            test_mode=['random'],
            training_kwargs=training_kwargs,
            test_kwargs=test_kwargs
        )

        assert embeddings.shape == (50, 3, 28, 28)
        assert torch.all((concepts['colors'] >= 0) & (concepts['colors'] <= 2))
        assert coloring_mode.count('training') == 30
        assert coloring_mode.count('test') == 20

    def test_intervals_mode(self):
        """Test intervals coloring mode."""
        torch.manual_seed(42)
        data = torch.randn(100, 28, 28)
        # Ensure all digits 0-9 are present
        targets = torch.cat([torch.arange(10).repeat(10)])

        training_kwargs = [{
            'intervals': [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            'values': [['red'], ['blue']]
        }]
        test_kwargs = [{
            'intervals': [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            'values': [['green'], ['red']]
        }]

        embeddings, concepts, out_targets, coloring_mode = colorize_and_transform(
            data, targets,
            training_percentage=0.7,
            test_percentage=0.3,
            training_mode=['intervals'],
            test_mode=['intervals'],
            training_kwargs=training_kwargs,
            test_kwargs=test_kwargs
        )

        assert embeddings.shape == (100, 3, 28, 28)
        assert 'colors' in concepts
        assert len(out_targets) == 100

    def test_additional_concepts_random_mode(self):
        """Test additional_concepts_random mode."""
        torch.manual_seed(42)
        data = torch.randn(50, 28, 28)
        targets = torch.randint(0, 10, (50,))

        training_kwargs = [{
            'concepts_used': ['colors', 'scales', 'degrees'],
            'values': [['red', 'green'], [0.8, 1.2], [0.0, 45.0]],
            'random_prob': [['uniform'], ['uniform'], ['uniform']]
        }]
        test_kwargs = [{
            'concepts_used': ['colors', 'scales', 'degrees'],
            'values': [['blue', 'green'], [0.9, 1.1], [0.0, 90.0]],
            'random_prob': [['uniform'], ['uniform'], ['uniform']]
        }]

        embeddings, concepts, out_targets, coloring_mode = colorize_and_transform(
            data, targets,
            training_percentage=0.6,
            test_percentage=0.4,
            training_mode=['additional_concepts_random'],
            test_mode=['additional_concepts_random'],
            training_kwargs=training_kwargs,
            test_kwargs=test_kwargs
        )

        assert embeddings.shape == (50, 3, 28, 28)
        assert 'colors' in concepts
        assert 'scales' in concepts
        assert 'degrees' in concepts

    def test_additional_concepts_custom_mode(self):
        """Test additional_concepts_custom mode."""
        torch.manual_seed(42)
        data = torch.randn(50, 28, 28)
        targets = torch.randint(0, 10, (50,))

        training_kwargs = [{
            'concepts_used': ['colors', 'scales'],
            'values': [
                [['red', 'green'], ['blue']],
                [[0.8, 1.0], [1.2]]
            ]
        }]
        test_kwargs = [{
            'concepts_used': ['colors', 'scales'],
            'values': [
                [['red'], ['blue', 'green']],
                [[0.9], [1.1, 1.3]]
            ]
        }]

        embeddings, concepts, out_targets, coloring_mode = colorize_and_transform(
            data, targets,
            training_percentage=0.5,
            test_percentage=0.5,
            training_mode=['additional_concepts_custom'],
            test_mode=['additional_concepts_custom'],
            training_kwargs=training_kwargs,
            test_kwargs=test_kwargs
        )

        assert embeddings.shape == (50, 3, 28, 28)
        assert 'colors' in concepts
        assert 'scales' in concepts

    def test_additional_concepts_custom_with_clothing(self):
        """Test additional_concepts_custom mode with clothing concept."""
        torch.manual_seed(42)
        data = torch.randn(50, 28, 28)
        targets = torch.arange(10).repeat(5)  # All digits 0-9

        training_kwargs = [{
            'concepts_used': ['clothing', 'colors'],
            'values': [
                [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                [['red'], ['blue']]
            ]
        }]
        test_kwargs = [{
            'concepts_used': ['clothing', 'colors'],
            'values': [
                [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                [['green'], ['red']]
            ]
        }]

        embeddings, concepts, out_targets, coloring_mode = colorize_and_transform(
            data, targets,
            training_percentage=0.6,
            test_percentage=0.4,
            training_mode=['additional_concepts_custom'],
            test_mode=['additional_concepts_custom'],
            training_kwargs=training_kwargs,
            test_kwargs=test_kwargs
        )

        assert embeddings.shape == (50, 3, 28, 28)
        assert 'colors' in concepts
        assert 'clothing' not in concepts  # Clothing should be removed from concepts

    def test_invalid_percentage_sum(self):
        """Test that percentages not summing to 1 raise error."""
        data = torch.randn(10, 28, 28)
        targets = torch.randint(0, 10, (10,))

        with pytest.raises(AssertionError, match="training_percentage and test_percentage must sum to 1"):
            colorize_and_transform(
                data, targets,
                training_percentage=0.5,
                test_percentage=0.3  # Doesn't sum to 1
            )

    def test_random_mode_missing_keys(self):
        """Test that random mode with missing keys raises error."""
        data = torch.randn(10, 28, 28)
        targets = torch.randint(0, 10, (10,))

        training_kwargs = [{'random_prob': [0.5, 0.5]}]  # Missing 'values'

        with pytest.raises(ValueError, match="random coloring requires the following keys"):
            colorize_and_transform(
                data, targets,
                training_mode=['random'],
                test_mode=['random'],
                training_kwargs=training_kwargs,
                test_kwargs=[{'random_prob': [0.5, 0.5], 'values': ['red', 'green']}]
            )

    def test_random_mode_invalid_color(self):
        """Test that invalid color raises error."""
        data = torch.randn(10, 28, 28)
        targets = torch.randint(0, 10, (10,))

        training_kwargs = [{'random_prob': [0.5, 0.5], 'values': ['red', 'invalid_color']}]

        with pytest.raises(ValueError, match="All values must be one of"):
            colorize_and_transform(
                data, targets,
                training_mode=['random'],
                test_mode=['random'],
                training_kwargs=training_kwargs,
                test_kwargs=[{'random_prob': [0.5, 0.5], 'values': ['red', 'green']}]
            )

    def test_intervals_mode_missing_keys(self):
        """Test that intervals mode with missing keys raises error."""
        data = torch.randn(10, 28, 28)
        targets = torch.randint(0, 10, (10,))

        training_kwargs = [{'intervals': [[0, 1], [2, 3]]}]  # Missing 'values'

        with pytest.raises(ValueError, match="intervals coloring requires the following keys"):
            colorize_and_transform(
                data, targets,
                training_mode=['intervals'],
                test_mode=['intervals'],
                training_kwargs=training_kwargs,
                test_kwargs=[{'intervals': [[0, 1], [2, 3]], 'values': [['red'], ['blue']]}]
            )

    def test_intervals_mode_incomplete_coverage(self):
        """Test that intervals not covering all targets raise error."""
        data = torch.randn(10, 28, 28)
        targets = torch.arange(10)  # 0-9

        # Only covering 0-5, missing 6-9
        training_kwargs = [{
            'intervals': [[0, 1, 2], [3, 4, 5]],
            'values': [['red'], ['blue']]
        }]

        with pytest.raises(AssertionError, match="intervals must cover all target values"):
            colorize_and_transform(
                data, targets,
                training_mode=['intervals'],
                test_mode=['intervals'],
                training_kwargs=training_kwargs,
                test_kwargs=training_kwargs
            )

    def test_additional_concepts_random_missing_colors(self):
        """Test that additional_concepts_random without colors raises error."""
        data = torch.randn(10, 28, 28)
        targets = torch.randint(0, 10, (10,))

        training_kwargs = [{
            'concepts_used': ['scales', 'degrees'],  # Missing 'colors'
            'values': [[0.8, 1.2], [0.0, 45.0]],
            'random_prob': [['uniform'], ['uniform']]
        }]

        with pytest.raises(AssertionError, match="concepts_used must contain 'colors'"):
            colorize_and_transform(
                data, targets,
                training_mode=['additional_concepts_random'],
                test_mode=['additional_concepts_random'],
                training_kwargs=training_kwargs,
                test_kwargs=training_kwargs
            )

    def test_additional_concepts_random_with_clothing(self):
        """Test that additional_concepts_random with clothing raises error."""
        data = torch.randn(10, 28, 28)
        targets = torch.randint(0, 10, (10,))

        training_kwargs = [{
            'concepts_used': ['clothing', 'colors'],
            'values': [[0, 1], ['red', 'green']],
            'random_prob': [['uniform'], ['uniform']]
        }]

        with pytest.raises(AssertionError, match="'clothing' cannot be used"):
            colorize_and_transform(
                data, targets,
                training_mode=['additional_concepts_random'],
                test_mode=['additional_concepts_random'],
                training_kwargs=training_kwargs,
                test_kwargs=training_kwargs
            )

    def test_unknown_mode(self):
        """Test that unknown mode raises error."""
        data = torch.randn(10, 28, 28)
        targets = torch.randint(0, 10, (10,))

        with pytest.raises(ValueError, match="Unknown coloring mode"):
            colorize_and_transform(
                data, targets,
                training_mode=['unknown_mode'],
                test_mode=['random'],
                training_kwargs=[{}],
                test_kwargs=[{'random_prob': [0.5, 0.5], 'values': ['red', 'green']}]
            )

    def test_data_shuffling(self):
        """Test that data and targets are shuffled together."""
        torch.manual_seed(42)
        data = torch.arange(50).reshape(50, 1, 1).repeat(1, 28, 28).float()
        targets = torch.arange(50)

        training_kwargs = [{'random_prob': [0.5, 0.5], 'values': ['red', 'green']}]
        test_kwargs = [{'random_prob': [0.5, 0.5], 'values': ['red', 'green']}]

        embeddings, concepts, out_targets, coloring_mode = colorize_and_transform(
            data, targets,
            training_percentage=0.5,
            test_percentage=0.5,
            training_mode=['random'],
            test_mode=['random'],
            training_kwargs=training_kwargs,
            test_kwargs=test_kwargs
        )

        # Targets should be shuffled (not in original order)
        assert not torch.equal(out_targets, targets)


if __name__ == '__main__':
    unittest.main()
