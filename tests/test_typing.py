"""
Comprehensive tests for torch_concepts/typing.py

This test suite covers type definitions and aliases used throughout the package.
"""
import unittest
import torch
from torch_concepts.typing import BackboneType


class TestTyping(unittest.TestCase):
    """Test suite for typing.py module."""

    def test_backbone_type_none(self):
        """Test BackboneType with None value."""
        backbone: BackboneType = None
        self.assertIsNone(backbone)

    def test_backbone_type_callable(self):
        """Test BackboneType with callable."""
        def backbone_fn(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        backbone: BackboneType = backbone_fn
        test_input = torch.tensor([1.0, 2.0, 3.0])
        result = backbone(test_input)
        self.assertTrue(torch.equal(result, test_input * 2))

    def test_backbone_type_nn_module(self):
        """Test BackboneType with nn.Module."""
        backbone: BackboneType = torch.nn.Linear(10, 5)
        test_input = torch.randn(2, 10)
        result = backbone(test_input)
        self.assertEqual(result.shape, (2, 5))

    def test_backbone_type_lambda(self):
        """Test BackboneType with lambda function."""
        backbone: BackboneType = lambda x: x ** 2
        test_input = torch.tensor([2.0, 3.0, 4.0])
        result = backbone(test_input)
        expected = torch.tensor([4.0, 9.0, 16.0])
        self.assertTrue(torch.equal(result, expected))

    def test_backbone_type_sequential(self):
        """Test BackboneType with nn.Sequential."""
        backbone: BackboneType = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 15)
        )
        test_input = torch.randn(5, 10)
        result = backbone(test_input)
        self.assertEqual(result.shape, (5, 15))


if __name__ == '__main__':
    unittest.main()

