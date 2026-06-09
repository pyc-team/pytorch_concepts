"""
Tests for SelectorEmbeddingEncoder (formerly SelectorLatentToExogenous).
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.nn.modules.low.dense_layers import SelectorEmbeddingEncoder


class TestSelectorEmbeddingEncoder(unittest.TestCase):
    """Test SelectorEmbeddingEncoder."""

    def test_initialization(self):
        """Test selector initialization."""
        selector = SelectorEmbeddingEncoder(
            in_features=64,
            out_features=8,
            n_embeddings=5,
            memory_size=20,
        )
        self.assertEqual(selector.out_features, 8)
        self.assertEqual(selector.memory_size, 20)
        self.assertEqual(selector.memory.num_embeddings, 5)

    def test_forward_without_sampling(self):
        """Test forward pass without sampling (soft selection)."""
        selector = SelectorEmbeddingEncoder(
            in_features=64,
            out_features=6,
            n_embeddings=4,
            memory_size=10,
        )
        x = torch.randn(2, 64)
        output = selector(x, sampling=False)
        self.assertEqual(output.shape, (2, 4, 6))

    def test_forward_with_sampling(self):
        """Test forward pass with sampling (Gumbel-softmax)."""
        selector = SelectorEmbeddingEncoder(
            in_features=64,
            out_features=6,
            n_embeddings=4,
            memory_size=10,
        )
        x = torch.randn(2, 64)
        output = selector(x, sampling=True)
        self.assertEqual(output.shape, (2, 4, 6))

    def test_gradient_flow_soft(self):
        """Test gradient flow with soft selection."""
        selector = SelectorEmbeddingEncoder(
            in_features=32,
            out_features=4,
            n_embeddings=3,
            memory_size=8,
        )
        x = torch.randn(2, 32, requires_grad=True)
        output = selector(x, sampling=False)
        output.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_gradient_flow_hard(self):
        """Test gradient flow with hard selection."""
        selector = SelectorEmbeddingEncoder(
            in_features=32,
            out_features=4,
            n_embeddings=3,
            memory_size=8,
        )
        x = torch.randn(2, 32, requires_grad=True)
        output = selector(x, sampling=True)
        output.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_different_temperatures(self):
        """Test with different temperature values."""
        for temp in [0.1, 0.5, 1.0, 2.0]:
            selector = SelectorEmbeddingEncoder(
                in_features=32,
                out_features=4,
                n_embeddings=3,
                memory_size=8,
                temperature=temp,
            )
            self.assertEqual(selector.temperature, temp)
            x = torch.randn(2, 32)
            output = selector(x, sampling=False)
            self.assertEqual(output.shape, (2, 3, 4))

    def test_memory_initialization(self):
        """Test memory bank has correct shape."""
        selector = SelectorEmbeddingEncoder(
            in_features=32,
            out_features=8,
            n_embeddings=5,
            memory_size=10,
        )
        self.assertEqual(selector.memory.weight.shape, (5, 80))  # n_embeddings × (memory_size × out_features)

    def test_selector_network(self):
        """Test selector is a Sequential module."""
        selector = SelectorEmbeddingEncoder(
            in_features=64,
            out_features=6,
            n_embeddings=4,
            memory_size=10,
        )
        self.assertIsInstance(selector.selector, nn.Sequential)

    def test_batch_processing(self):
        """Test different batch sizes."""
        selector = SelectorEmbeddingEncoder(
            in_features=32,
            out_features=4,
            n_embeddings=3,
            memory_size=5,
        )
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 32)
            output = selector(x, sampling=False)
            self.assertEqual(output.shape, (batch_size, 3, 4))


if __name__ == '__main__':
    unittest.main()
