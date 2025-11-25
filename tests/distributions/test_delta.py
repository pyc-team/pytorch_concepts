"""
Comprehensive tests for torch_concepts/distributions/delta.py

This test suite covers the Delta (deterministic) distribution implementation.
"""
import unittest
import torch
from torch_concepts.distributions.delta import Delta


class TestDelta(unittest.TestCase):
    """Test suite for Delta distribution."""

    def test_initialization_with_list(self):
        """Test Delta initialization with list."""
        dist = Delta([1.0, 2.0, 3.0])
        self.assertEqual(dist.mean.tolist(), [1.0, 2.0, 3.0])

    def test_initialization_with_tensor(self):
        """Test Delta initialization with tensor."""
        value = torch.tensor([1.0, 2.0, 3.0])
        dist = Delta(value)
        self.assertTrue(torch.equal(dist.mean, value))

    def test_initialization_with_float(self):
        """Test Delta initialization with single float in list."""
        dist = Delta([5.0])
        self.assertEqual(dist.mean.item(), 5.0)

    def test_sample(self):
        """Test sampling from Delta distribution."""
        value = torch.tensor([1.0, 2.0, 3.0])
        dist = Delta(value)

        sample = dist.sample()
        self.assertTrue(torch.equal(sample, value))

        # Multiple samples should all be the same
        sample2 = dist.sample()
        self.assertTrue(torch.equal(sample2, value))

    def test_sample_with_shape(self):
        """Test sampling with sample_shape parameter."""
        value = torch.tensor([1.0, 2.0, 3.0])
        dist = Delta(value)

        # Note: Delta ignores sample_shape in current implementation
        sample = dist.sample(torch.Size([5, 2]))
        self.assertTrue(torch.equal(sample, value))

    def test_rsample(self):
        """Test reparameterized sampling from Delta distribution."""
        value = torch.tensor([1.0, 2.0, 3.0])
        dist = Delta(value)

        sample = dist.rsample()
        self.assertTrue(torch.equal(sample, value))

    def test_rsample_with_shape(self):
        """Test reparameterized sampling with sample_shape parameter."""
        value = torch.tensor([1.0, 2.0, 3.0])
        dist = Delta(value)

        # Note: Delta ignores sample_shape in current implementation
        sample = dist.rsample(torch.Size([3]))
        self.assertTrue(torch.equal(sample, value))

    def test_mean(self):
        """Test mean property of Delta distribution."""
        value = torch.tensor([5.0, 10.0, 15.0])
        dist = Delta(value)

        self.assertTrue(torch.equal(dist.mean, value))

    def test_log_prob(self):
        """Test log_prob method of Delta distribution."""
        value = torch.tensor([1.0, 2.0, 3.0])
        dist = Delta(value)

        # For Delta distribution, log_prob returns zeros
        test_value = torch.tensor([[1.0, 2.0, 3.0]])
        log_prob = dist.log_prob(test_value)
        self.assertTrue(torch.all(log_prob == 0))

    def test_log_prob_different_value(self):
        """Test log_prob with value different from distribution's value."""
        value = torch.tensor([1.0, 2.0, 3.0])
        dist = Delta(value)

        # Even for different values, current implementation returns 0
        test_value = torch.tensor([[5.0, 6.0, 7.0]])
        log_prob = dist.log_prob(test_value)
        self.assertTrue(torch.all(log_prob == 0))

    def test_log_prob_batch(self):
        """Test log_prob with batch of values."""
        value = torch.tensor([1.0, 2.0])
        dist = Delta(value)

        test_values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        log_prob = dist.log_prob(test_values)
        # The implementation returns zeros with shape based on event_shape
        # For this case it returns a scalar since event_shape is empty
        self.assertTrue(torch.all(log_prob == 0))

    def test_has_rsample(self):
        """Test has_rsample attribute."""
        dist = Delta([1.0, 2.0])
        self.assertFalse(dist.has_rsample)

    def test_arg_constraints(self):
        """Test arg_constraints attribute."""
        dist = Delta([1.0, 2.0])
        self.assertEqual(dist.arg_constraints, {})

    def test_support(self):
        """Test support attribute."""
        dist = Delta([1.0, 2.0])
        self.assertIsNone(dist.support)

    def test_repr(self):
        """Test __repr__ method."""
        value = torch.tensor([1.0, 2.0, 3.0, 4.0])
        dist = Delta(value)

        repr_str = repr(dist)
        self.assertIn('Delta', repr_str)
        self.assertIn('value_shape', repr_str)
        self.assertIn('4', repr_str)  # Shape dimension

    def test_immutability(self):
        """Test that original value is cloned and independent."""
        value = torch.tensor([1.0, 2.0, 3.0])
        dist = Delta(value)

        # Modify original value
        value[0] = 999.0

        # Distribution should still have original value
        self.assertEqual(dist.mean[0].item(), 1.0)

    def test_multidimensional(self):
        """Test Delta distribution with multidimensional tensors."""
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        dist = Delta(value)

        sample = dist.sample()
        self.assertTrue(torch.equal(sample, value))
        self.assertEqual(sample.shape, (2, 2))

    def test_3d_tensor(self):
        """Test Delta distribution with 3D tensor."""
        value = torch.randn(2, 3, 4)
        dist = Delta(value)

        sample = dist.sample()
        self.assertTrue(torch.equal(sample, value))
        self.assertEqual(sample.shape, (2, 3, 4))

    def test_scalar(self):
        """Test Delta distribution with scalar value."""
        value = torch.tensor(5.0)
        dist = Delta(value)

        sample = dist.sample()
        self.assertEqual(sample.item(), 5.0)

    def test_zero_value(self):
        """Test Delta distribution with zero value."""
        value = torch.tensor([0.0, 0.0, 0.0])
        dist = Delta(value)

        sample = dist.sample()
        self.assertTrue(torch.equal(sample, value))
        self.assertTrue(torch.all(sample == 0))

    def test_negative_values(self):
        """Test Delta distribution with negative values."""
        value = torch.tensor([-1.0, -2.0, -3.0])
        dist = Delta(value)

        sample = dist.sample()
        self.assertTrue(torch.equal(sample, value))
        self.assertEqual(dist.mean.tolist(), [-1.0, -2.0, -3.0])

    def test_large_values(self):
        """Test Delta distribution with large values."""
        value = torch.tensor([1e6, 1e7, 1e8])
        dist = Delta(value)

        sample = dist.sample()
        self.assertTrue(torch.equal(sample, value))

    def test_dtype_preservation(self):
        """Test that dtype is preserved."""
        value_float32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        dist_float32 = Delta(value_float32)
        self.assertEqual(dist_float32.mean.dtype, torch.float32)

        value_float64 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        dist_float64 = Delta(value_float64)
        self.assertEqual(dist_float64.mean.dtype, torch.float64)

    def test_batch_shape(self):
        """Test batch_shape attribute."""
        dist = Delta([1.0, 2.0])
        self.assertEqual(dist.batch_shape, torch.Size([]))

    def test_multiple_samples_consistency(self):
        """Test that multiple samples are consistent."""
        value = torch.randn(5, 3)
        dist = Delta(value)

        samples = [dist.sample() for _ in range(10)]
        for sample in samples:
            self.assertTrue(torch.equal(sample, value))

    def test_gradient_flow(self):
        """Test that gradients can flow through rsample."""
        value = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        dist = Delta(value)

        # rsample should return the value (which has gradients)
        sample = dist.rsample()
        # The sample should reference the same tensor
        loss = sample.sum()
        loss.backward()

        # Original value should have gradients
        self.assertIsNotNone(value.grad)


if __name__ == '__main__':
    unittest.main()
