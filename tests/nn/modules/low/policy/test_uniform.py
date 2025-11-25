"""
Comprehensive tests for torch_concepts.nn.modules.low.policy

Tests intervention policy modules (random, uncertainty, uniform).
"""
import unittest
import torch
from torch_concepts.nn.modules.low.policy.uniform import UniformPolicy


class TestUniformPolicy(unittest.TestCase):
    """Test UniformPolicy."""

    def test_initialization(self):
        """Test uniform policy initialization."""
        policy = UniformPolicy(out_features=10)
        self.assertEqual(policy.out_features, 10)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        policy = UniformPolicy(out_features=10)
        endogenous = torch.randn(4, 10)
        output = policy(endogenous)
        self.assertEqual(output.shape, (4, 10))

    def test_uniform_values(self):
        """Test that output is uniform across concepts."""
        policy = UniformPolicy(out_features=10)
        endogenous = torch.randn(4, 10)
        output = policy(endogenous)

        # All values in each row should be equal
        for i in range(output.shape[0]):
            values = output[i]
            self.assertTrue(torch.allclose(values, values[0].expand_as(values)))

    def test_different_inputs_same_output(self):
        """Test that different inputs produce same uniform output."""
        policy = UniformPolicy(out_features=5)

        endogenous1 = torch.randn(2, 5)
        endogenous2 = torch.randn(2, 5)

        output1 = policy(endogenous1)
        output2 = policy(endogenous2)

        # Outputs should be same (uniform policy)
        self.assertTrue(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()
