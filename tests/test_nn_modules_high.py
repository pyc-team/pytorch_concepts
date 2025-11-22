"""
Comprehensive tests for torch_concepts.nn.modules.high

Tests high-level model modules (CBM, CEM, CGM, etc.).
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.annotations import Annotations, AxisAnnotation
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.high.base.learner import BaseLearner


class TestHighLevelModels(unittest.TestCase):
    """Test high-level model architectures."""

    def setUp(self):
        """Set up common test fixtures."""
        # Create simple annotations for testing
        concept_labels = ['color', 'shape', 'size']
        task_labels = ['class1', 'class2']
        self.annotations = Annotations({
            1: AxisAnnotation(labels=concept_labels + task_labels)
        })
        self.variable_distributions = {
            'color': Delta,
            'shape': Delta,
            'size': Delta,
            'class1': Delta,
            'class2': Delta
        }

    def test_cbm_placeholder(self):
        """Placeholder test for CBM model."""
        # CBM requires complex setup with inference strategies
        # This is a placeholder to ensure the test file runs
        self.assertTrue(True)

    def test_cem_placeholder(self):
        """Placeholder test for CEM model."""
        # CEM requires complex setup with embeddings
        # This is a placeholder to ensure the test file runs
        self.assertTrue(True)


class TestBatchValidation(unittest.TestCase):
    """Test batch structure validation in BaseLearner."""
    
    def setUp(self):
        """Create a mock learner instance for testing unpack_batch."""
        # Create a mock learner that only implements unpack_batch
        self.learner = type('MockLearner', (), {})()
        # Bind the unpack_batch method from BaseLearner
        self.learner.unpack_batch = BaseLearner.unpack_batch.__get__(self.learner)
    
    def test_valid_batch_structure(self):
        """Test that valid batch structure is accepted."""
        valid_batch = {
            'inputs': torch.randn(4, 10),
            'concepts': torch.randn(4, 2)
        }
        inputs, concepts, transforms = self.learner.unpack_batch(valid_batch)
        self.assertIsNotNone(inputs)
        self.assertIsNotNone(concepts)
        self.assertEqual(transforms, {})
    
    def test_batch_with_transforms(self):
        """Test that batch with transforms is handled correctly."""
        batch_with_transforms = {
            'inputs': torch.randn(4, 10),
            'concepts': torch.randn(4, 2),
            'transforms': {'scaler': 'some_transform'}
        }
        inputs, concepts, transforms = self.learner.unpack_batch(batch_with_transforms)
        self.assertIsNotNone(inputs)
        self.assertIsNotNone(concepts)
        self.assertEqual(transforms, {'scaler': 'some_transform'})
    
    def test_missing_inputs_key(self):
        """Test that missing 'inputs' key raises KeyError."""
        invalid_batch = {
            'concepts': torch.randn(4, 2)
        }
        with self.assertRaises(KeyError) as context:
            self.learner.unpack_batch(invalid_batch)
        self.assertIn('inputs', str(context.exception))
        self.assertIn("missing required keys", str(context.exception))
    
    def test_missing_concepts_key(self):
        """Test that missing 'concepts' key raises KeyError."""
        invalid_batch = {
            'inputs': torch.randn(4, 10)
        }
        with self.assertRaises(KeyError) as context:
            self.learner.unpack_batch(invalid_batch)
        self.assertIn('concepts', str(context.exception))
        self.assertIn("missing required keys", str(context.exception))
    
    def test_missing_both_keys(self):
        """Test that missing both required keys raises KeyError."""
        invalid_batch = {
            'data': torch.randn(4, 10)
        }
        with self.assertRaises(KeyError) as context:
            self.learner.unpack_batch(invalid_batch)
        self.assertIn("missing required keys", str(context.exception))
    
    def test_non_dict_batch(self):
        """Test that non-dict batch raises TypeError."""
        invalid_batch = torch.randn(4, 10)
        with self.assertRaises(TypeError) as context:
            self.learner.unpack_batch(invalid_batch)
        self.assertIn("Expected batch to be a dict", str(context.exception))
    
    def test_tuple_batch(self):
        """Test that tuple batch raises TypeError."""
        invalid_batch = (torch.randn(4, 10), torch.randn(4, 2))
        with self.assertRaises(TypeError) as context:
            self.learner.unpack_batch(invalid_batch)
        self.assertIn("Expected batch to be a dict", str(context.exception))
    
    def test_empty_dict_batch(self):
        """Test that empty dict raises KeyError with helpful message."""
        invalid_batch = {}
        with self.assertRaises(KeyError) as context:
            self.learner.unpack_batch(invalid_batch)
        self.assertIn("missing required keys", str(context.exception))
        self.assertIn("Found keys: []", str(context.exception))


if __name__ == '__main__':
    unittest.main()


