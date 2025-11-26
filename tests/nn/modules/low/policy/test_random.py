"""
Comprehensive tests for torch_concepts.nn.modules.low.policy

Tests intervention policy modules (random, uncertainty, uniform).
"""
import unittest
import torch
from torch_concepts.nn.modules.low.policy.random import RandomPolicy


class TestRandomPolicy(unittest.TestCase):
    """Test RandomPolicy."""

    def test_initialization(self):
        """Test random policy initialization."""
        policy = RandomPolicy(out_features=10, scale=2.0)
        self.assertEqual(policy.out_features, 10)
        self.assertEqual(policy.scale, 2.0)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        policy = RandomPolicy(out_features=10, scale=1.0)
        endogenous = torch.randn(4, 10)
        output = policy(endogenous)
        self.assertEqual(output.shape, (4, 10))

    def test_random_values(self):
        """Test that output contains random values."""
        policy = RandomPolicy(out_features=10, scale=1.0)
        endogenous = torch.randn(4, 10)

        output1 = policy(endogenous)
        output2 = policy(endogenous)

        # Outputs should be different (random)
        self.assertFalse(torch.equal(output1, output2))

    def test_value_range(self):
        """Test that values are in expected range."""
        policy = RandomPolicy(out_features=10, scale=2.0)
        endogenous = torch.randn(100, 10)
        output = policy(endogenous)

        # Should be non-negative and scaled
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 2.0))

    def test_scale_effect(self):
        """Test that scale parameter affects output."""
        endogenous = torch.randn(100, 10)

        policy_small = RandomPolicy(out_features=10, scale=0.5)
        policy_large = RandomPolicy(out_features=10, scale=5.0)

        output_small = policy_small(endogenous)
        output_large = policy_large(endogenous)

        # Larger scale should produce larger values on average
        self.assertLess(output_small.mean(), output_large.mean())


if __name__ == '__main__':
    unittest.main()
