"""
Comprehensive tests for torch_concepts.nn.modules.low.policy

Tests intervention policy modules (random, uncertainty, uniform).
"""
import unittest
import torch
from torch_concepts.nn.modules.low.policy.uncertainty import UncertaintyInterventionPolicy


class TestUncertaintyInterventionPolicy(unittest.TestCase):
    """Test UncertaintyInterventionPolicy."""

    def test_initialization(self):
        """Test uncertainty policy initialization."""
        policy = UncertaintyInterventionPolicy(out_features=10)
        self.assertEqual(policy.out_features, 10)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        policy = UncertaintyInterventionPolicy(out_features=10)
        endogenous = torch.randn(4, 10)
        output = policy(endogenous)
        self.assertEqual(output.shape, (4, 10))

    def test_uncertainty_measure(self):
        """Test that certainty is measured correctly (returns absolute values)."""
        policy = UncertaintyInterventionPolicy(out_features=10)

        # High certainty (endogenous far from 0)
        high_certainty = torch.tensor([[10.0, -10.0, 10.0, -10.0]])

        # Low certainty (endogenous near 0)
        low_certainty = torch.tensor([[0.1, -0.1, 0.2, -0.2]])

        certainty_high = policy(high_certainty)
        certainty_low = policy(low_certainty)

        # Implementation returns abs values, so high certainty inputs produce higher scores
        self.assertGreater(certainty_high.mean().item(), certainty_low.mean().item())

    def test_gradient_flow(self):
        """Test gradient flow through policy."""
        policy = UncertaintyInterventionPolicy(out_features=5)
        endogenous = torch.randn(2, 5, requires_grad=True)
        output = policy(endogenous)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(endogenous.grad)


if __name__ == '__main__':
    unittest.main()
