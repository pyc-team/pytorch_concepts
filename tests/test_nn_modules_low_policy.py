"""
Comprehensive tests for torch_concepts.nn.modules.low.policy

Tests intervention policy modules (random, uncertainty, uniform).
"""
import unittest
import torch
from torch_concepts.nn.modules.low.policy.random import RandomPolicy
from torch_concepts.nn.modules.low.policy.uncertainty import UncertaintyInterventionPolicy
from torch_concepts.nn.modules.low.policy.uniform import UniformPolicy


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
        logits = torch.randn(4, 10)
        output = policy(logits)
        self.assertEqual(output.shape, (4, 10))

    def test_random_values(self):
        """Test that output contains random values."""
        policy = RandomPolicy(out_features=10, scale=1.0)
        logits = torch.randn(4, 10)

        output1 = policy(logits)
        output2 = policy(logits)

        # Outputs should be different (random)
        self.assertFalse(torch.equal(output1, output2))

    def test_value_range(self):
        """Test that values are in expected range."""
        policy = RandomPolicy(out_features=10, scale=2.0)
        logits = torch.randn(100, 10)
        output = policy(logits)

        # Should be non-negative and scaled
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 2.0))

    def test_scale_effect(self):
        """Test that scale parameter affects output."""
        logits = torch.randn(100, 10)

        policy_small = RandomPolicy(out_features=10, scale=0.5)
        policy_large = RandomPolicy(out_features=10, scale=5.0)

        output_small = policy_small(logits)
        output_large = policy_large(logits)

        # Larger scale should produce larger values on average
        self.assertLess(output_small.mean(), output_large.mean())


class TestUncertaintyInterventionPolicy(unittest.TestCase):
    """Test UncertaintyInterventionPolicy."""

    def test_initialization(self):
        """Test uncertainty policy initialization."""
        policy = UncertaintyInterventionPolicy(out_features=10)
        self.assertEqual(policy.out_features, 10)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        policy = UncertaintyInterventionPolicy(out_features=10)
        logits = torch.randn(4, 10)
        output = policy(logits)
        self.assertEqual(output.shape, (4, 10))

    def test_uncertainty_measure(self):
        """Test that certainty is measured correctly (returns absolute values)."""
        policy = UncertaintyInterventionPolicy(out_features=10)

        # High certainty (logits far from 0)
        high_certainty = torch.tensor([[10.0, -10.0, 10.0, -10.0]])

        # Low certainty (logits near 0)
        low_certainty = torch.tensor([[0.1, -0.1, 0.2, -0.2]])

        certainty_high = policy(high_certainty)
        certainty_low = policy(low_certainty)

        # Implementation returns abs values, so high certainty inputs produce higher scores
        self.assertGreater(certainty_high.mean().item(), certainty_low.mean().item())

    def test_gradient_flow(self):
        """Test gradient flow through policy."""
        policy = UncertaintyInterventionPolicy(out_features=5)
        logits = torch.randn(2, 5, requires_grad=True)
        output = policy(logits)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(logits.grad)


class TestUniformPolicy(unittest.TestCase):
    """Test UniformPolicy."""

    def test_initialization(self):
        """Test uniform policy initialization."""
        policy = UniformPolicy(out_features=10)
        self.assertEqual(policy.out_features, 10)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        policy = UniformPolicy(out_features=10)
        logits = torch.randn(4, 10)
        output = policy(logits)
        self.assertEqual(output.shape, (4, 10))

    def test_uniform_values(self):
        """Test that output is uniform across concepts."""
        policy = UniformPolicy(out_features=10)
        logits = torch.randn(4, 10)
        output = policy(logits)

        # All values in each row should be equal
        for i in range(output.shape[0]):
            values = output[i]
            self.assertTrue(torch.allclose(values, values[0].expand_as(values)))

    def test_different_inputs_same_output(self):
        """Test that different inputs produce same uniform output."""
        policy = UniformPolicy(out_features=5)

        logits1 = torch.randn(2, 5)
        logits2 = torch.randn(2, 5)

        output1 = policy(logits1)
        output2 = policy(logits2)

        # Outputs should be same (uniform policy)
        self.assertTrue(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()
