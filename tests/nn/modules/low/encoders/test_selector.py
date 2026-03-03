"""
Comprehensive tests for torch_concepts.nn.modules.low.encoders

Tests all encoder modules (linear, exogenous, selector, stochastic).
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn.modules.low.encoders.selector import (
    CategoricalSelectorLatentToExogenous,
    SelectorLatentToExogenous,
)


class TestSelectorLatentToExogenous(unittest.TestCase):
    """Test SelectorLatentToExogenous."""

    def test_initialization(self):
        """Test selector initialization."""
        selector = SelectorLatentToExogenous(
            in_latent=64,
            out_concepts=5,
            memory_size=20,
            out_exogenous=8
        )
        self.assertEqual(selector.in_latent, 64)
        self.assertEqual(selector.out_concepts, 5)
        self.assertEqual(selector.memory_size, 20)
        self.assertEqual(selector.out_exogenous, 8)

    def test_forward_without_sampling(self):
        """Test forward pass without sampling (soft selection)."""
        selector = SelectorLatentToExogenous(
            in_latent=64,
            out_concepts=4,
            memory_size=10,
            out_exogenous=6
        )
        latent = torch.randn(2, 64)
        output = selector(latent=latent, sampling=False)
        self.assertEqual(output.shape, (2, 4, 6))

    def test_forward_with_sampling(self):
        """Test forward pass with sampling (Gumbel-softmax)."""
        selector = SelectorLatentToExogenous(
            in_latent=64,
            out_concepts=4,
            memory_size=10,
            out_exogenous=6
        )
        latent = torch.randn(2, 64)
        output = selector(latent=latent, sampling=True)
        self.assertEqual(output.shape, (2, 4, 6))

    def test_gradient_flow_soft(self):
        """Test gradient flow with soft selection."""
        selector = SelectorLatentToExogenous(
            in_latent=32,
            out_concepts=3,
            memory_size=8,
            out_exogenous=4
        )
        embeddings = torch.randn(2, 32, requires_grad=True)
        output = selector(latent=embeddings, sampling=False)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_gradient_flow_hard(self):
        """Test gradient flow with hard selection."""
        selector = SelectorLatentToExogenous(
            in_latent=32,
            out_concepts=3,
            memory_size=8,
            out_exogenous=4
        )
        embeddings = torch.randn(2, 32, requires_grad=True)
        output = selector(latent=embeddings, sampling=True)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_different_temperatures(self):
        """Test with different temperature values."""
        for temp in [0.1, 0.5, 1.0, 2.0]:
            selector = SelectorLatentToExogenous(
                in_latent=32,
                out_concepts=3,
                memory_size=8,
                out_exogenous=4,
                temperature=temp
            )
            self.assertEqual(selector.temperature, temp)
            embeddings = torch.randn(2, 32)
            output = selector(latent=embeddings, sampling=False)
            self.assertEqual(output.shape, (2, 3, 4))

    def test_memory_initialization(self):
        """Test memory bank initialization."""
        selector = SelectorLatentToExogenous(
            in_latent=32,
            out_concepts=5,
            memory_size=10,
            out_exogenous=8
        )
        # Check memory has correct shape
        self.assertEqual(selector.memory.weight.shape, (5, 80))  # out_concepts x (memory_size * embedding_size)

    def test_selector_network(self):
        """Test selector network structure."""
        selector = SelectorLatentToExogenous(
            in_latent=64,
            out_concepts=4,
            memory_size=10,
            out_exogenous=6
        )
        # Check selector is a Sequential module
        self.assertIsInstance(selector.selector, nn.Sequential)

    def test_batch_processing(self):
        """Test different batch sizes."""
        selector = SelectorLatentToExogenous(
            in_latent=32,
            out_concepts=3,
            memory_size=5,
            out_exogenous=4
        )
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 32)
            output = selector(latent=embeddings, sampling=False)
            self.assertEqual(output.shape, (batch_size, 3, 4))


class TestCategoricalSelectorLatentToExogenous(unittest.TestCase):
    """Test CategoricalSelectorLatentToExogenous."""

    def test_initialization(self):
        """Test selector initialization."""
        selector = CategoricalSelectorLatentToExogenous(
            in_latent=64,
            out_concepts=5,
            out_exogenous=8,
            selector_hidden_layers=2,
        )
        self.assertEqual(selector.in_latent, 64)
        self.assertEqual(selector.out_concepts, 5)
        self.assertEqual(selector.out_exogenous, 8)
        self.assertEqual(selector.selector_hidden_layers, 2)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        selector = CategoricalSelectorLatentToExogenous(
            in_latent=64,
            out_concepts=4,
            out_exogenous=6,
        )
        latent = torch.randn(2, 64)
        output = selector(latent=latent)
        self.assertEqual(output.shape, (2, 4, 6))

    def test_output_is_normalized_over_exogenous_dim(self):
        """Test output probabilities sum to 1 over exogenous dimension."""
        selector = CategoricalSelectorLatentToExogenous(
            in_latent=32,
            out_concepts=3,
            out_exogenous=5,
        )
        latent = torch.randn(3, 32)
        output = selector(latent=latent)

        sums = output.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_gradient_flow(self):
        """Test gradient flow through selector."""
        selector = CategoricalSelectorLatentToExogenous(
            in_latent=32,
            out_concepts=3,
            out_exogenous=4,
        )
        embeddings = torch.randn(2, 32, requires_grad=True)
        output = selector(latent=embeddings)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_hidden_layers_configuration(self):
        """Test configurable hidden layers in selector network."""
        selector_zero = CategoricalSelectorLatentToExogenous(
            in_latent=32,
            out_concepts=3,
            out_exogenous=4,
            selector_hidden_layers=0,
        )
        selector_two = CategoricalSelectorLatentToExogenous(
            in_latent=32,
            out_concepts=3,
            out_exogenous=4,
            selector_hidden_layers=2,
        )

        linear_zero = sum(isinstance(layer, nn.Linear) for layer in selector_zero.selector)
        linear_two = sum(isinstance(layer, nn.Linear) for layer in selector_two.selector)

        self.assertEqual(linear_zero, 1)
        self.assertEqual(linear_two, 3)

    def test_selector_hidden_layers_validation(self):
        """Test hidden layer argument validation."""
        with self.assertRaises(ValueError):
            CategoricalSelectorLatentToExogenous(
                in_latent=32,
                out_concepts=3,
                out_exogenous=4,
                selector_hidden_layers=-1,
            )

    def test_selector_network(self):
        """Test selector network structure."""
        selector = CategoricalSelectorLatentToExogenous(
            in_latent=64,
            out_concepts=4,
            out_exogenous=6,
        )
        self.assertIsInstance(selector.selector, nn.Sequential)

    def test_batch_processing(self):
        """Test different batch sizes."""
        selector = CategoricalSelectorLatentToExogenous(
            in_latent=32,
            out_concepts=3,
            out_exogenous=4,
        )
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 32)
            output = selector(latent=embeddings)
            self.assertEqual(output.shape, (batch_size, 3, 4))


if __name__ == '__main__':
    unittest.main()
