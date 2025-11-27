"""
Comprehensive tests for Concept Bottleneck Model (CBM).

Tests cover:
- Model initialization with various configurations
- Forward pass and output shapes
- Training modes (manual PyTorch and Lightning)
- Backbone integration
- Distribution handling
- Filter methods
"""
import pytest
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel, ConceptBottleneckModel_Joint
from torch_concepts.annotations import AxisAnnotation, Annotations
from torch_concepts.nn.modules.utils import GroupConfig


class DummyBackbone(nn.Module):
    """Simple backbone for testing."""
    def __init__(self, out_features=8):
        super().__init__()
        self.out_features = out_features
    
    def forward(self, x):
        return torch.ones(x.shape[0], self.out_features)


class TestCBMInitialization(unittest.TestCase):
    """Test CBM initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'size', 'task1'],
                cardinalities=[3, 2, 1, 1],
                metadata={
                    'color': {'type': 'discrete', 'distribution': Categorical},
                    'shape': {'type': 'discrete', 'distribution': Categorical},
                    'size': {'type': 'binary', 'distribution': Bernoulli},
                    'task1': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
    
    def test_init_with_distributions_in_annotations(self):
        """Test initialization when distributions are in annotations."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']
        )
        
        self.assertIsInstance(model.model, nn.Module)
        self.assertTrue(hasattr(model, 'inference'))
        self.assertEqual(model.concept_names, ['color', 'shape', 'size', 'task1'])
    
    def test_init_with_variable_distributions(self):
        """Test initialization with variable_distributions parameter."""
        ann_no_dist = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })
        
        variable_distributions = {
            'c1': Bernoulli,
            'c2': Bernoulli,
            'task': Bernoulli
        }
        
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=ann_no_dist,
            variable_distributions=variable_distributions,
            task_names=['task']
        )
        
        self.assertEqual(model.concept_names, ['c1', 'c2', 'task'])
    
    def test_init_with_backbone(self):
        """Test initialization with custom backbone."""
        backbone = DummyBackbone()
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            backbone=backbone,
            task_names=['task1']
        )
        
        self.assertIsNotNone(model.backbone)
    
    def test_init_with_latent_encoder(self):
        """Test initialization with latent encoder config."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 2}
        )
        
        self.assertEqual(model.latent_size, 16)


class TestCBMForward(unittest.TestCase):
    """Test CBM forward pass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'size', 'task1'],
                cardinalities=[3, 2, 1, 1],
                metadata={
                    'color': {'type': 'discrete', 'distribution': Categorical},
                    'shape': {'type': 'discrete', 'distribution': Categorical},
                    'size': {'type': 'binary', 'distribution': Bernoulli},
                    'task1': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        self.model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']
        )
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        x = torch.randn(2, 8)
        query = ['color', 'shape', 'size']
        out = self.model(x, query=query)
        
        # Output shape: batch_size x sum(cardinalities for queried variables)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3 + 2 + 1)  # color + shape + size
    
    def test_forward_all_concepts(self):
        """Test forward with all concepts."""
        x = torch.randn(4, 8)
        query = ['color', 'shape', 'size', 'task1']
        out = self.model(x, query=query)
        
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[1], 3 + 2 + 1 + 1)
    
    def test_forward_single_concept(self):
        """Test forward with single concept."""
        x = torch.randn(2, 8)
        query = ['color']
        out = self.model(x, query=query)
        
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3)
    
    def test_forward_with_backbone(self):
        """Test forward pass with backbone."""
        backbone = DummyBackbone(out_features=8)
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            backbone=backbone,
            task_names=['task1']
        )
        
        x = torch.randn(2, 100)  # Raw input size (before backbone)
        query = ['color', 'shape']
        out = model(x, query=query)
        
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3 + 2)


class TestCBMFilterMethods(unittest.TestCase):
    """Test CBM filter methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        self.model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
    
    def test_filter_output_for_loss(self):
        """Test filter_output_for_loss returns correct format."""
        x = torch.randn(2, 8)
        query = ['c1', 'c2', 'task']
        out = self.model(x, query=query)
        target = torch.randint(0, 2, out.shape).float()
        
        filtered = self.model.filter_output_for_loss(out, target)
        
        self.assertIsInstance(filtered, dict)
        self.assertIn('input', filtered)
        self.assertIn('target', filtered)
        self.assertTrue(torch.allclose(filtered['input'], out))
        self.assertTrue(torch.allclose(filtered['target'], target))
    
    def test_filter_output_for_metrics(self):
        """Test filter_output_for_metrics returns correct format."""
        x = torch.randn(2, 8)
        query = ['c1', 'c2', 'task']
        out = self.model(x, query=query)
        target = torch.randint(0, 2, out.shape).float()
        
        filtered = self.model.filter_output_for_metrics(out, target)
        
        self.assertIsInstance(filtered, dict)
        self.assertIn('preds', filtered)
        self.assertIn('target', filtered)
        self.assertTrue(torch.allclose(filtered['preds'], out))
        self.assertTrue(torch.allclose(filtered['target'], target))


class TestCBMTraining(unittest.TestCase):
    """Test CBM training scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
    
    def test_manual_training_mode(self):
        """Test manual PyTorch training (no loss in model)."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        # No loss configured (loss is None)
        self.assertIsNone(model.loss)
        
        # Can train manually
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()
        
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4, 3)).float()
        
        model.train()
        out = model(x, query=['c1', 'c2', 'task'])
        loss = loss_fn(out, y)
        
        self.assertTrue(loss.requires_grad)
    
    def test_gradients_flow(self):
        """Test that gradients flow through the model."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        x = torch.randn(4, 8, requires_grad=True)
        out = model(x, query=['c1', 'c2', 'task'])
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)


class TestCBMEdgeCases(unittest.TestCase):
    """Test CBM edge cases and error handling."""
    
    def test_empty_query(self):
        """Test behavior with empty query."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=ann,
            task_names=['c2']
        )
        
        x = torch.randn(2, 8)
        # Empty or None query should handle gracefully
        # Behavior depends on implementation
    
    def test_repr(self):
        """Test string representation."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1'],
                cardinalities=[1],
                metadata={'c1': {'type': 'binary', 'distribution': Bernoulli}}
            )
        })
        
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=ann,
            task_names=['c1']
        )
        
        repr_str = repr(model)
        self.assertIsInstance(repr_str, str)
        self.assertIn('ConceptBottleneckModel', repr_str)


if __name__ == '__main__':
    unittest.main()
