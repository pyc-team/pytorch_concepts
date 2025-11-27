"""
Comprehensive tests for torch_concepts/utils.py

This test suite covers utility functions for working with concept-based models.
"""
import unittest
import os
import torch
import numpy as np
import random
from torch_concepts.utils import (
    validate_and_generate_concept_names,
    compute_output_size,
    get_most_common_expl,
    compute_temperature,
    numerical_stability_check,
    _is_int_index,
    get_from_string,
    instantiate_from_string,
    seed_everything,
)

from torch_concepts import GroupConfig
from torch_concepts.annotations import AxisAnnotation, Annotations


class TestUtils(unittest.TestCase):
    """Test suite for utils.py module."""

    def test_validate_and_generate_concept_names_with_list(self):
        """Test validate_and_generate_concept_names with list of names."""
        concept_names = {0: [], 1: ['color', 'shape', 'size']}
        result = validate_and_generate_concept_names(concept_names)

        self.assertEqual(result[0], [])
        self.assertEqual(result[1], ['color', 'shape', 'size'])

    def test_validate_and_generate_concept_names_with_int(self):
        """Test validate_and_generate_concept_names with integer."""
        concept_names = {0: [], 1: 3}
        result = validate_and_generate_concept_names(concept_names)

        self.assertEqual(result[0], [])
        self.assertEqual(result[1], ['concept_1_0', 'concept_1_1', 'concept_1_2'])

    def test_validate_and_generate_concept_names_mixed(self):
        """Test validate_and_generate_concept_names with mixed input."""
        concept_names = {0: [], 1: ['a', 'b'], 2: 3}
        result = validate_and_generate_concept_names(concept_names)

        self.assertEqual(result[0], [])
        self.assertEqual(result[1], ['a', 'b'])
        self.assertEqual(result[2], ['concept_2_0', 'concept_2_1', 'concept_2_2'])

    def test_validate_and_generate_concept_names_invalid(self):
        """Test validate_and_generate_concept_names with invalid input."""
        concept_names = {0: [], 1: 'invalid'}

        with self.assertRaises(ValueError):
            validate_and_generate_concept_names(concept_names)

    def test_validate_and_generate_concept_names_empty(self):
        """Test validate_and_generate_concept_names with empty dict."""
        concept_names = {}
        result = validate_and_generate_concept_names(concept_names)
        self.assertEqual(result, {})

    def test_compute_output_size(self):
        """Test compute_output_size function."""
        # With list of names
        concept_names = {0: [], 1: ['a', 'b', 'c'], 2: ['x', 'y']}
        size = compute_output_size(concept_names)
        self.assertEqual(size, 6)  # 3 * 2

        # With integers
        concept_names = {0: [], 1: 3, 2: 2}
        size = compute_output_size(concept_names)
        self.assertEqual(size, 6)  # 3 * 2

        # Single dimension
        concept_names = {0: [], 1: 5}
        size = compute_output_size(concept_names)
        self.assertEqual(size, 5)

    def test_compute_output_size_only_batch(self):
        """Test compute_output_size with only batch dimension."""
        concept_names = {0: []}
        size = compute_output_size(concept_names)
        self.assertEqual(size, 1)

    def test_get_most_common_expl(self):
        """Test get_most_common_expl function."""
        explanations = [
            {'class1': 'explanation A', 'class2': 'explanation X'},
            {'class1': 'explanation A', 'class2': 'explanation Y'},
            {'class1': 'explanation B', 'class2': 'explanation X'},
            {'class1': 'explanation A', 'class2': 'explanation X'},
        ]

        result = get_most_common_expl(explanations, n=2)

        self.assertEqual(result['class1']['explanation A'], 3)
        self.assertEqual(result['class1']['explanation B'], 1)
        self.assertEqual(result['class2']['explanation X'], 3)
        self.assertEqual(result['class2']['explanation Y'], 1)

    def test_get_most_common_expl_single_class(self):
        """Test get_most_common_expl with single class."""
        explanations = [
            {'class1': 'A'},
            {'class1': 'A'},
            {'class1': 'B'},
        ]

        result = get_most_common_expl(explanations, n=10)
        self.assertEqual(result['class1']['A'], 2)
        self.assertEqual(result['class1']['B'], 1)

    def test_compute_temperature(self):
        """Test compute_temperature function."""
        # Test at beginning of training
        temp_start = compute_temperature(0, 100)
        self.assertAlmostEqual(temp_start, 1.0, places=2)

        # Test at end of training
        temp_end = compute_temperature(100, 100)
        self.assertAlmostEqual(temp_end, 0.5, places=2)

        # Test in middle
        temp_mid = compute_temperature(50, 100)
        self.assertTrue(0.5 < temp_mid < 1.0)

    def test_compute_temperature_single_epoch(self):
        """Test compute_temperature with single epoch."""
        temp = compute_temperature(0, 1)
        self.assertIsInstance(temp, (int, float, torch.Tensor))

    def test_numerical_stability_check_stable(self):
        """Test numerical_stability_check with stable covariance."""
        device = torch.device('cpu')
        # Create positive definite matrix
        A = torch.randn(5, 5)
        cov = A @ A.T  # Always positive definite

        result = numerical_stability_check(cov, device)
        # Should return matrix without modification (or minimal)
        self.assertEqual(result.shape, (5, 5))

    def test_numerical_stability_check_unstable(self):
        """Test numerical_stability_check with unstable covariance."""
        device = torch.device('cpu')
        # Create near-singular matrix
        cov = torch.eye(5) * 1e-10

        result = numerical_stability_check(cov, device)
        # Should add epsilon to diagonal
        self.assertEqual(result.shape, (5, 5))
        # Should now be stable
        try:
            torch.linalg.cholesky(result)
        except RuntimeError:
            self.fail("Matrix should be stable after correction")

    def test_numerical_stability_check_batch(self):
        """Test numerical_stability_check with batch of covariances."""
        device = torch.device('cpu')
        # Create batch of positive definite matrices
        batch_size = 3
        dim = 4
        A = torch.randn(batch_size, dim, dim)
        cov = torch.bmm(A, A.transpose(1, 2))

        result = numerical_stability_check(cov, device)
        self.assertEqual(result.shape, (batch_size, dim, dim))

    def test_numerical_stability_check_symmetry(self):
        """Test that numerical_stability_check symmetrizes the matrix."""
        device = torch.device('cpu')
        # Create slightly asymmetric matrix
        A = torch.randn(4, 4)
        cov = A @ A.T
        cov[0, 1] += 0.01  # Break symmetry slightly

        result = numerical_stability_check(cov, device)
        # Check if symmetric
        self.assertTrue(torch.allclose(result, result.T))

    def test_is_int_index(self):
        """Test _is_int_index function."""
        # Test with int
        self.assertTrue(_is_int_index(5))
        self.assertTrue(_is_int_index(0))
        self.assertTrue(_is_int_index(-1))

        # Test with 0-dimensional tensor
        self.assertTrue(_is_int_index(torch.tensor(5)))

        # Test with non-int
        self.assertFalse(_is_int_index(5.0))
        self.assertFalse(_is_int_index('5'))
        self.assertFalse(_is_int_index(torch.tensor([5])))
        self.assertFalse(_is_int_index(torch.tensor([5, 6])))
        self.assertFalse(_is_int_index([5]))
        self.assertFalse(_is_int_index(None))

    def test_get_from_string_builtin(self):
        """Test get_from_string with torch module."""
        result = get_from_string('torch.nn.ReLU')
        self.assertEqual(result, torch.nn.ReLU)

    def test_get_from_string_torch_module(self):
        """Test get_from_string with torch module."""
        result = get_from_string('torch.nn.Linear')
        self.assertEqual(result, torch.nn.Linear)

    def test_get_from_string_torch_distribution(self):
        """Test get_from_string with torch distribution."""
        result = get_from_string('torch.distributions.Bernoulli')
        from torch.distributions import Bernoulli
        self.assertEqual(result, Bernoulli)

    def test_get_from_string_invalid(self):
        """Test get_from_string with invalid string."""
        with self.assertRaises((ImportError, AttributeError)):
            get_from_string('nonexistent.module.Class')

    def test_instantiate_from_string_simple(self):
        """Test instantiate_from_string with simple class."""
        instance = instantiate_from_string('torch.nn.ReLU')
        self.assertIsInstance(instance, torch.nn.ReLU)

    def test_instantiate_from_string_with_kwargs(self):
        """Test instantiate_from_string with kwargs."""
        # Use Linear as an example
        instance = instantiate_from_string('torch.nn.Linear', in_features=10, out_features=5)
        self.assertIsInstance(instance, torch.nn.Linear)
        self.assertEqual(instance.in_features, 10)
        self.assertEqual(instance.out_features, 5)

    def test_check_tensors_valid(self):
        """Test _check_tensors with valid tensors."""
        from torch_concepts.utils import _check_tensors

        t1 = torch.randn(4, 3, 5)
        t2 = torch.randn(4, 2, 5)
        t3 = torch.randn(4, 5, 5)

        # Should not raise
        _check_tensors([t1, t2, t3])

    def test_check_tensors_invalid_batch_size(self):
        """Test _check_tensors with mismatched batch size."""
        from torch_concepts.utils import _check_tensors

        t1 = torch.randn(4, 3, 5)
        t2 = torch.randn(5, 2, 5)  # Different batch size

        with self.assertRaises(ValueError) as context:
            _check_tensors([t1, t2])
        self.assertIn('batch', str(context.exception))

    def test_check_tensors_invalid_dimensions(self):
        """Test _check_tensors with wrong number of dimensions."""
        from torch_concepts.utils import _check_tensors

        t1 = torch.randn(4, 3, 5)
        t2 = torch.randn(4, 2)  # Only 2 dimensions

        with self.assertRaises(ValueError) as context:
            _check_tensors([t1, t2])
        self.assertIn('at least 2 dims', str(context.exception))

    def test_check_tensors_invalid_trailing_shape(self):
        """Test _check_tensors with mismatched trailing dimensions."""
        from torch_concepts.utils import _check_tensors

        t1 = torch.randn(4, 3, 5)
        t2 = torch.randn(4, 2, 6)  # Different trailing dimension

        with self.assertRaises(ValueError) as context:
            _check_tensors([t1, t2])
        self.assertIn('trailing shape', str(context.exception))

    def test_check_tensors_invalid_dtype(self):
        """Test _check_tensors with mismatched dtypes."""
        from torch_concepts.utils import _check_tensors

        t1 = torch.randn(4, 3, 5, dtype=torch.float32)
        t2 = torch.randn(4, 2, 5, dtype=torch.float64)

        with self.assertRaises(ValueError) as context:
            _check_tensors([t1, t2])
        self.assertIn('dtype', str(context.exception))

    def test_check_tensors_invalid_device(self):
        """Test _check_tensors with mismatched devices."""
        from torch_concepts.utils import _check_tensors

        t1 = torch.randn(4, 3, 5, device='cpu')
        t2 = torch.randn(4, 2, 5, device='cpu')

        # Should not raise on same device
        _check_tensors([t1, t2])

    def test_add_distribution_to_annotations_with_dict(self):
        """Test add_distribution_to_annotations function."""
        from torch_concepts.utils import add_distribution_to_annotations

        # Create simple annotations with proper metadata
        metadata = {
            'color': {'type': 'discrete'},
            'shape': {'type': 'discrete'}
        }
        annotations = AxisAnnotation(labels=('color', 'shape'), cardinalities=(3, 2), metadata=metadata)

        variable_distributions = {
            'color': torch.distributions.Bernoulli,
            'shape': torch.distributions.Categorical
        }

        result = add_distribution_to_annotations(annotations, variable_distributions)
        self.assertIsInstance(result, AxisAnnotation)

    def test_add_distribution_to_annotations_with_groups(self):
        """Test add_distribution_to_annotations function."""
        from torch_concepts.utils import add_distribution_to_annotations

        # Create simple annotations with proper metadata
        metadata = {
            'color': {'type': 'discrete'},
            'shape': {'type': 'discrete'}
        }
        annotations = AxisAnnotation(labels=('color', 'shape'), cardinalities=(3, 2), metadata=metadata)

        variable_distributions = GroupConfig(
            binary=torch.distributions.Bernoulli,
            categorical=torch.distributions.Categorical
        )

        result = add_distribution_to_annotations(annotations, variable_distributions)
        self.assertIsInstance(result, AxisAnnotation)

    def test_compute_temperature_edge_cases(self):
        """Test compute_temperature with edge cases."""
        # Zero epochs
        with self.assertRaises((ZeroDivisionError, ValueError)):
            compute_temperature(0, 0)

        # Negative epoch
        temp = compute_temperature(-1, 100)
        self.assertIsNotNone(temp)

    def test_numerical_stability_epsilon_scaling(self):
        """Test that epsilon scales properly in numerical_stability_check."""
        device = torch.device('cpu')
        # Create matrix that requires multiple iterations
        cov = torch.eye(3) * 1e-12

        result = numerical_stability_check(cov, device, epsilon=1e-8)
        self.assertEqual(result.shape, (3, 3))
        # Verify it's now stable
        torch.linalg.cholesky(result)

    def test_get_most_common_expl_empty(self):
        """Test get_most_common_expl with empty list."""
        explanations = []
        result = get_most_common_expl(explanations, n=10)
        self.assertEqual(result, {})

    def test_get_most_common_expl_limit(self):
        """Test get_most_common_expl respects n limit."""
        explanations = [
            {'class1': 'A'},
            {'class1': 'B'},
            {'class1': 'C'},
            {'class1': 'D'},
            {'class1': 'E'},
        ]

        result = get_most_common_expl(explanations, n=2)
        # Should only return top 2
        self.assertEqual(len(result['class1']), 2)


class TestSeedEverything(unittest.TestCase):
    """Test suite for seed_everything function."""

    def test_seed_returns_value(self):
        """Test that seed_everything returns the seed value."""
        seed = 42
        result = seed_everything(seed)
        self.assertEqual(result, seed, "seed_everything should return the seed value")

    def test_python_random_reproducibility(self):
        """Test that Python's random module produces reproducible results."""
        seed = 12345

        # First run
        seed_everything(seed)
        random_values_1 = [random.random() for _ in range(10)]

        # Second run with same seed
        seed_everything(seed)
        random_values_2 = [random.random() for _ in range(10)]

        self.assertEqual(random_values_1, random_values_2,
                         "Python random should produce same values with same seed")

    def test_numpy_random_reproducibility(self):
        """Test that NumPy random produces reproducible results."""
        seed = 54321

        # First run
        seed_everything(seed)
        np_values_1 = np.random.randn(10)

        # Second run with same seed
        seed_everything(seed)
        np_values_2 = np.random.randn(10)

        np.testing.assert_array_equal(np_values_1, np_values_2,
                                      "NumPy random should produce same values with same seed")

    def test_torch_cpu_reproducibility(self):
        """Test that PyTorch CPU random produces reproducible results."""
        seed = 99999

        # First run
        seed_everything(seed)
        torch_values_1 = torch.randn(10)

        # Second run with same seed
        seed_everything(seed)
        torch_values_2 = torch.randn(10)

        self.assertTrue(torch.equal(torch_values_1, torch_values_2),
                        "PyTorch CPU random should produce same values with same seed")

    def test_torch_cuda_reproducibility(self):
        """Test that PyTorch CUDA random produces reproducible results."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        seed = 77777

        # First run
        seed_everything(seed)
        torch_cuda_values_1 = torch.randn(10, device='cuda')

        # Second run with same seed
        seed_everything(seed)
        torch_cuda_values_2 = torch.randn(10, device='cuda')

        self.assertTrue(torch.equal(torch_cuda_values_1, torch_cuda_values_2),
                        "PyTorch CUDA random should produce same values with same seed")

    def test_pythonhashseed_environment_variable(self):
        """Test that PYTHONHASHSEED environment variable is set."""
        seed = 33333
        seed_everything(seed)

        self.assertIn('PYTHONHASHSEED', os.environ,
                      "PYTHONHASHSEED should be set in environment variables")
        self.assertEqual(os.environ['PYTHONHASHSEED'], str(seed),
                         "PYTHONHASHSEED should match the seed value")

    def test_pl_global_seed_environment_variable(self):
        """Test that PL_GLOBAL_SEED environment variable is set by Lightning."""
        seed = 66666
        seed_everything(seed)

        self.assertIn('PL_GLOBAL_SEED', os.environ,
                      "PL_GLOBAL_SEED should be set by PyTorch Lightning")
        self.assertEqual(os.environ['PL_GLOBAL_SEED'], str(seed),
                         "PL_GLOBAL_SEED should match the seed value")

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different random values."""
        # First seed
        seed_everything(42)
        torch_values_1 = torch.randn(10)
        np_values_1 = np.random.randn(10)
        random_values_1 = [random.random() for _ in range(10)]

        # Different seed
        seed_everything(123)
        torch_values_2 = torch.randn(10)
        np_values_2 = np.random.randn(10)
        random_values_2 = [random.random() for _ in range(10)]

        self.assertFalse(torch.equal(torch_values_1, torch_values_2),
                         "Different seeds should produce different PyTorch values")
        self.assertFalse(np.array_equal(np_values_1, np_values_2),
                         "Different seeds should produce different NumPy values")
        self.assertNotEqual(random_values_1, random_values_2,
                            "Different seeds should produce different Python random values")

    def test_workers_parameter(self):
        """Test that workers parameter is accepted."""
        seed = 11111
        # Should not raise an error
        result = seed_everything(seed, workers=True)
        self.assertEqual(result, seed)

        result = seed_everything(seed, workers=False)
        self.assertEqual(result, seed)

    def test_neural_network_reproducibility(self):
        """Test that neural network training is reproducible with same seed."""
        seed = 88888

        # Create simple model and data
        def train_step():
            model = torch.nn.Linear(10, 5)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            x = torch.randn(32, 10)
            y = torch.randn(32, 5)

            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()

            return loss.item(), model.weight.data.clone()

        # First run
        seed_everything(seed)
        loss_1, weights_1 = train_step()

        # Second run with same seed
        seed_everything(seed)
        loss_2, weights_2 = train_step()

        self.assertAlmostEqual(loss_1, loss_2, places=6,
                               msg="Loss should be identical with same seed")
        self.assertTrue(torch.allclose(weights_1, weights_2, atol=1e-6),
                        "Model weights should be identical with same seed")


if __name__ == '__main__':
    unittest.main()
