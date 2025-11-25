"""
Comprehensive tests for torch_concepts.nn.modules.mid

Tests mid-level modules (base, constructors, inference, models).
"""
import unittest
import torch.nn as nn
from torch_concepts.annotations import Annotations, AxisAnnotation
from torch_concepts.nn.modules.mid.base.model import BaseConstructor


class TestBaseConstructor(unittest.TestCase):
    """Test BaseConstructor."""

    def setUp(self):
        """Set up test annotations and layers."""
        concept_labels = ('color', 'shape', 'size')
        self.annotations = Annotations({
            1: AxisAnnotation(labels=concept_labels)
        })
        self.encoder = nn.Linear(784, 3)
        self.predictor = nn.Linear(3, 10)

    def test_initialization(self):
        """Test base constructor initialization."""
        constructor = BaseConstructor(
            input_size=784,
            annotations=self.annotations,
            encoder=self.encoder,
            predictor=self.predictor
        )
        self.assertEqual(constructor.input_size, 784)
        self.assertIsNotNone(constructor.annotations)
        self.assertEqual(len(constructor.labels), 3)

    def test_name_to_id_mapping(self):
        """Test name to ID mapping."""
        constructor = BaseConstructor(
            input_size=784,
            annotations=self.annotations,
            encoder=self.encoder,
            predictor=self.predictor
        )
        self.assertIn('color', constructor.name2id)
        self.assertIn('shape', constructor.name2id)
        self.assertIn('size', constructor.name2id)
        self.assertEqual(constructor.name2id['color'], 0)


if __name__ == '__main__':
    unittest.main()
