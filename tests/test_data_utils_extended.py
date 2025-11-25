"""Extended tests for torch_concepts.data.utils module to improve coverage."""

import pytest
import torch
import numpy as np
from torch_concepts.data.utils import (
    assign_values_based_on_intervals,
    colorize_and_transform,
)


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

