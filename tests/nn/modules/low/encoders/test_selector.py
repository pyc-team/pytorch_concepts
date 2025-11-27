"""
Comprehensive tests for torch_concepts.nn.modules.low.encoders

Tests all encoder modules (linear, exogenous, selector, stochastic).
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn.modules.low.encoders.selector import SelectorZU


class TestSelectorZU(unittest.TestCase):
    """Test SelectorZU."""

    def test_initialization(self):
        """Test selector initialization."""
        selector = SelectorZU(
            in_features=64,
            out_features=5,
            memory_size=20,
            exogenous_size=8
        )
        self.assertEqual(selector.in_features, 64)
        self.assertEqual(selector.out_features, 5)
        self.assertEqual(selector.memory_size, 20)
        self.assertEqual(selector.exogenous_size, 8)

    def test_forward_without_sampling(self):
        """Test forward pass without sampling (soft selection)."""
        selector = SelectorZU(
            in_features=64,
            out_features=4,
            memory_size=10,
            exogenous_size=6
        )
        latent = torch.randn(2, 64)
        output = selector(input=latent, sampling=False)
        self.assertEqual(output.shape, (2, 4, 6))

    def test_forward_with_sampling(self):
        """Test forward pass with sampling (Gumbel-softmax)."""
        selector = SelectorZU(
            in_features=64,
            out_features=4,
            memory_size=10,
            exogenous_size=6
        )
        latent = torch.randn(2, 64)
        output = selector(input=latent, sampling=True)
        self.assertEqual(output.shape, (2, 4, 6))

    def test_gradient_flow_soft(self):
        """Test gradient flow with soft selection."""
        selector = SelectorZU(
            in_features=32,
            out_features=3,
            memory_size=8,
            exogenous_size=4
        )
        embeddings = torch.randn(2, 32, requires_grad=True)
        output = selector(input=embeddings, sampling=False)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_gradient_flow_hard(self):
        """Test gradient flow with hard selection."""
        selector = SelectorZU(
            in_features=32,
            out_features=3,
            memory_size=8,
            exogenous_size=4
        )
        embeddings = torch.randn(2, 32, requires_grad=True)
        output = selector(input=embeddings, sampling=True)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_different_temperatures(self):
        """Test with different temperature values."""
        for temp in [0.1, 0.5, 1.0, 2.0]:
            selector = SelectorZU(
                in_features=32,
                out_features=3,
                memory_size=8,
                exogenous_size=4,
                temperature=temp
            )
            self.assertEqual(selector.temperature, temp)
            embeddings = torch.randn(2, 32)
            output = selector(input=embeddings, sampling=False)
            self.assertEqual(output.shape, (2, 3, 4))

    def test_memory_initialization(self):
        """Test memory bank initialization."""
        selector = SelectorZU(
            in_features=32,
            out_features=5,
            memory_size=10,
            exogenous_size=8
        )
        # Check memory has correct shape
        self.assertEqual(selector.memory.weight.shape, (5, 80))  # out_features x (memory_size * embedding_size)

    def test_selector_network(self):
        """Test selector network structure."""
        selector = SelectorZU(
            in_features=64,
            out_features=4,
            memory_size=10,
            exogenous_size=6
        )
        # Check selector is a Sequential module
        self.assertIsInstance(selector.selector, nn.Sequential)

    def test_batch_processing(self):
        """Test different batch sizes."""
        selector = SelectorZU(
            in_features=32,
            out_features=3,
            memory_size=5,
            exogenous_size=4
        )
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 32)
            output = selector(input=embeddings, sampling=False)
            self.assertEqual(output.shape, (batch_size, 3, 4))


if __name__ == '__main__':
    unittest.main()
