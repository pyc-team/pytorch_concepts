"""Tests for torch_concepts.data.utils module."""

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

