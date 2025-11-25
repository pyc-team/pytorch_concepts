"""
Comprehensive tests for torch_concepts.nn.modules.low.predictors.call

Tests the CallableCC module with various callable functions.
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn import CallableCC


class TestCallableCCInitialization(unittest.TestCase):
    """Test CallableCC initialization."""

    def test_basic_initialization(self):
        """Test basic predictor initialization."""
        def simple_func(probs):
            return probs.sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func
        )
        self.assertTrue(predictor.use_bias)
        self.assertEqual(predictor.min_std, 1e-6)

    def test_initialization_without_bias(self):
        """Test predictor initialization without bias."""
        def simple_func(probs):
            return probs.mean(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func,
            use_bias=False
        )
        self.assertFalse(predictor.use_bias)

    def test_initialization_custom_bias_params(self):
        """Test initialization with custom bias parameters."""
        def simple_func(probs):
            return probs.sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func,
            init_bias_mean=1.0,
            init_bias_std=0.5,
            min_std=1e-5
        )
        self.assertAlmostEqual(predictor.bias_mean.item(), 1.0, places=5)
        self.assertEqual(predictor.min_std, 1e-5)

    def test_initialization_with_custom_activation(self):
        """Test initialization with custom activation function."""
        def simple_func(probs):
            return probs.sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func,
            in_activation=torch.sigmoid
        )
        self.assertTrue(predictor.use_bias)


class TestCallableCCForward(unittest.TestCase):
    """Test CallableCC forward pass."""

    def test_forward_simple_sum(self):
        """Test forward pass with simple sum function."""
        def sum_func(probs):
            return probs.sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=sum_func,
            use_bias=False
        )

        endogenous = torch.randn(4, 5)
        output = predictor(endogenous)

        self.assertEqual(output.shape, (4, 1))

    def test_forward_with_activation(self):
        """Test forward pass with input activation."""
        def sum_func(probs):
            return probs.sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=sum_func,
            in_activation=torch.sigmoid,
            use_bias=False
        )

        endogenous = torch.randn(4, 5)
        output = predictor(endogenous)

        # Verify output is sum of sigmoid(endogenous)
        expected = torch.sigmoid(endogenous).sum(dim=1, keepdim=True)
        torch.testing.assert_close(output, expected)

    def test_forward_quadratic_function(self):
        """Test forward pass with quadratic function (from docstring example)."""
        def quadratic_predictor(probs):
            c0, c1, c2 = probs[:, 0:1], probs[:, 1:2], probs[:, 2:3]
            output1 = 0.5*c0**2 + 1.0*c1**2 + 1.5*c2
            output2 = 2.0*c0 - 1.0*c1**2 + 0.5*c2**3
            return torch.cat([output1, output2], dim=1)

        predictor = CallableCC(
            func=quadratic_predictor,
            use_bias=False
        )

        batch_size = 32
        endogenous = torch.randn(batch_size, 3)
        output = predictor(endogenous)

        self.assertEqual(output.shape, (batch_size, 2))

    def test_forward_with_bias(self):
        """Test forward pass with stochastic bias."""
        def simple_func(probs):
            return probs.mean(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func,
            use_bias=True
        )

        endogenous = torch.randn(4, 5)

        # Run multiple times and check outputs are different (due to stochastic bias)
        output1 = predictor(endogenous)
        output2 = predictor(endogenous)

        self.assertEqual(output1.shape, (4, 1))
        self.assertEqual(output2.shape, (4, 1))
        # Due to stochastic sampling, outputs should be different
        self.assertFalse(torch.allclose(output1, output2))

    def test_forward_multi_output(self):
        """Test forward pass with multiple outputs."""
        def multi_output_func(probs):
            # Return 3 different aggregations
            sum_out = probs.sum(dim=1, keepdim=True)
            mean_out = probs.mean(dim=1, keepdim=True)
            max_out = probs.max(dim=1, keepdim=True)[0]
            return torch.cat([sum_out, mean_out, max_out], dim=1)

        predictor = CallableCC(
            func=multi_output_func,
            use_bias=False
        )

        endogenous = torch.randn(4, 5)
        output = predictor(endogenous)

        self.assertEqual(output.shape, (4, 3))

    def test_forward_with_kwargs(self):
        """Test forward pass with additional kwargs to callable."""
        def weighted_sum(probs, weights=None):
            if weights is None:
                weights = torch.ones(probs.shape[1])
            return (probs * weights).sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=weighted_sum,
            use_bias=False
        )

        endogenous = torch.randn(4, 5)
        weights = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])

        output = predictor(endogenous, weights=weights)
        self.assertEqual(output.shape, (4, 1))

    def test_forward_with_args(self):
        """Test forward pass with additional args to callable."""
        def parameterized_func(probs, scale):
            return probs.sum(dim=1, keepdim=True) * scale

        predictor = CallableCC(
            func=parameterized_func,
            use_bias=False
        )

        endogenous = torch.randn(4, 5)
        scale = 2.0

        output = predictor(endogenous, scale)
        self.assertEqual(output.shape, (4, 1))


class TestCallableCCGradients(unittest.TestCase):
    """Test gradient flow through CallableCC."""

    def test_gradient_flow(self):
        """Test gradient flow through predictor."""
        def simple_func(probs):
            return probs.sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func,
            use_bias=False
        )

        endogenous = torch.randn(2, 8, requires_grad=True)
        output = predictor(endogenous)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(endogenous.grad)
        self.assertEqual(endogenous.grad.shape, endogenous.shape)

    def test_gradient_flow_with_bias(self):
        """Test gradient flow with learnable bias parameters."""
        def simple_func(probs):
            return probs.mean(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func,
            use_bias=True
        )

        endogenous = torch.randn(4, 5, requires_grad=True)
        output = predictor(endogenous)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(endogenous.grad)
        self.assertIsNotNone(predictor.bias_mean.grad)
        self.assertIsNotNone(predictor.bias_raw_std.grad)

    def test_gradient_flow_quadratic(self):
        """Test gradient flow through quadratic function."""
        def quadratic_func(probs):
            return (probs ** 2).sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=quadratic_func,
            use_bias=False
        )

        endogenous = torch.randn(4, 5, requires_grad=True)
        output = predictor(endogenous)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(endogenous.grad)


class TestCallableCCBiasStd(unittest.TestCase):
    """Test bias standard deviation computation."""

    def test_bias_std_positive(self):
        """Test that bias std is always positive."""
        def simple_func(probs):
            return probs.sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func,
            use_bias=True
        )

        std = predictor._bias_std()
        self.assertGreater(std.item(), 0)

    def test_bias_std_minimum(self):
        """Test that bias std respects minimum floor."""
        def simple_func(probs):
            return probs.sum(dim=1, keepdim=True)

        min_std = 1e-4
        predictor = CallableCC(
            func=simple_func,
            use_bias=True,
            min_std=min_std
        )

        std = predictor._bias_std()
        self.assertGreaterEqual(std.item(), min_std)

    def test_bias_std_initialization(self):
        """Test bias std is initialized close to init_bias_std."""
        def simple_func(probs):
            return probs.sum(dim=1, keepdim=True)

        init_std = 0.1
        predictor = CallableCC(
            func=simple_func,
            use_bias=True,
            init_bias_std=init_std,
            min_std=1e-6
        )

        std = predictor._bias_std()
        # Should be close to init_std (within reasonable tolerance)
        self.assertAlmostEqual(std.item(), init_std, places=2)


class TestCallableCCEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_single_sample(self):
        """Test with single sample (batch size 1)."""
        def simple_func(probs):
            return probs.sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func,
            use_bias=False
        )

        endogenous = torch.randn(1, 5)
        output = predictor(endogenous)

        self.assertEqual(output.shape, (1, 1))

    def test_large_batch(self):
        """Test with large batch size."""
        def simple_func(probs):
            return probs.mean(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func,
            use_bias=False
        )

        batch_size = 1000
        endogenous = torch.randn(batch_size, 10)
        output = predictor(endogenous)

        self.assertEqual(output.shape, (batch_size, 1))

    def test_identity_function(self):
        """Test with identity function."""
        def identity_func(probs):
            return probs

        predictor = CallableCC(
            func=identity_func,
            use_bias=False
        )

        endogenous = torch.randn(4, 5)
        output = predictor(endogenous)

        # Output should equal input endogenous (with identity activation)
        torch.testing.assert_close(output, endogenous)

    def test_complex_function(self):
        """Test with complex mathematical function."""
        def complex_func(probs):
            # Combination of multiple operations
            linear = probs @ torch.randn(probs.shape[1], 3)
            activated = torch.tanh(linear)
            squared = activated ** 2
            return squared

        predictor = CallableCC(
            func=complex_func,
            use_bias=False
        )

        endogenous = torch.randn(4, 5)
        output = predictor(endogenous)

        self.assertEqual(output.shape, (4, 3))

    def test_deterministic_without_bias(self):
        """Test that output is deterministic when use_bias=False."""
        def simple_func(probs):
            return probs.sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func,
            use_bias=False
        )

        endogenous = torch.randn(4, 5)

        output1 = predictor(endogenous)
        output2 = predictor(endogenous)

        # Should be identical without bias
        torch.testing.assert_close(output1, output2)


class TestCallableCCDeviceCompatibility(unittest.TestCase):
    """Test device compatibility."""

    def test_cpu_device(self):
        """Test predictor works on CPU."""
        def simple_func(probs):
            return probs.sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func,
            use_bias=True
        )

        endogenous = torch.randn(4, 5)
        output = predictor(endogenous)

        self.assertEqual(output.device.type, 'cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device(self):
        """Test predictor works on CUDA."""
        def simple_func(probs):
            return probs.sum(dim=1, keepdim=True)

        predictor = CallableCC(
            func=simple_func,
            use_bias=True
        ).cuda()

        endogenous = torch.randn(4, 5).cuda()
        output = predictor(endogenous)

        self.assertEqual(output.device.type, 'cuda')


if __name__ == '__main__':
    unittest.main()
