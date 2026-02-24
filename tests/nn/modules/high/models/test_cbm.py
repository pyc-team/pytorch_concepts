"""
Comprehensive tests for Concept Bottleneck Model (CBM).

Tests cover:
- Model initialization with various configurations
- Forward pass and output shapes
- Training modes (joint, independent, sequential)
- Backbone integration
- Distribution handling
- Filter methods
- Factory function behavior
"""
import pytest
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel
from torch_concepts.annotations import AxisAnnotation, Annotations


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
        out = self.model(query=query, x=x)
        
        # Output shape: batch_size x sum(cardinalities for queried variables)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3 + 2 + 1)  # color + shape + size
    
    def test_forward_all_concepts(self):
        """Test forward with all concepts."""
        x = torch.randn(4, 8)
        query = ['color', 'shape', 'size', 'task1']
        out = self.model(query=query, x=x)
        
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[1], 3 + 2 + 1 + 1)
    
    def test_forward_single_concept(self):
        """Test forward with single concept."""
        x = torch.randn(2, 8)
        query = ['color']
        out = self.model(query=query, x=x)
        
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
        out = model(query=query, x=x)
        
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
        out = self.model(query=query, x=x)
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
        out = self.model(query=query, x=x)
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
        """Test manual PyTorch training (no training mode)."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        # No training mode = pure PyTorch module
        self.assertIsNone(model._training_mode)
        
        # Can train manually
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()
        
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4, 3)).float()
        
        model.train()
        out = model(query=['c1', 'c2', 'task'], x=x)
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
        out = model(query=['c1', 'c2', 'task'], x=x)
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


# =============================================================================
# Tests for Factory Function and Training Modes
# =============================================================================

class TestCBMFactory(unittest.TestCase):
    """Test ConceptBottleneckModel factory function."""
    
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
    
    def test_factory_joint_mode(self):
        """Test factory creates joint training model."""
        model = ConceptBottleneckModel(
            training='joint',
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        self.assertEqual(model._training_mode, 'joint')
        self.assertIn('Joint', model.__class__.__name__)
    
    def test_factory_independent_mode(self):
        """Test factory creates independent training model."""
        model = ConceptBottleneckModel(
            training='independent',
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        self.assertEqual(model._training_mode, 'independent')
        self.assertIn('Independent', model.__class__.__name__)
    
    def test_factory_sequential_mode(self):
        """Test factory creates sequential training model."""
        model = ConceptBottleneckModel(
            training='sequential',
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        self.assertEqual(model._training_mode, 'sequential')
        self.assertIn('Sequential', model.__class__.__name__)
    
    def test_factory_default_is_pytorch(self):
        """Test default is pure PyTorch module (no training mode)."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        self.assertIsNone(model._training_mode)
    
    def test_factory_invalid_mode_raises(self):
        """Test factory raises error for invalid mode."""
        with self.assertRaises(ValueError) as context:
            ConceptBottleneckModel(
                training='invalid_mode',
                input_size=8,
                annotations=self.ann,
                task_names=['task']
            )
        
        self.assertIn('invalid_mode', str(context.exception))


class TestCBMUnifiedForward(unittest.TestCase):
    """Test unified forward pass works across all modes."""
    
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
        self.x = torch.randn(4, 8)
    
    def test_forward_with_x_only(self):
        """Test forward with x tensor only."""
        model = ConceptBottleneckModel(
            training='joint',
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        out = model(x=self.x, query=['c1', 'c2', 'task'])
        self.assertEqual(out.shape, (4, 3))
    
    def test_forward_with_evidence(self):
        """Test forward with combined x and evidence dict."""
        model = ConceptBottleneckModel(
            training='joint',
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        # Test that evidence dict can be provided along with x
        # Evidence is merged with input latent
        evidence = {'extra_key': torch.randn(4, 1)}  # Additional evidence
        out = model(x=self.x, query=['c1', 'c2', 'task'], evidence=evidence)
        
        self.assertEqual(out.shape, (4, 3))
    
    def test_forward_same_output_all_modes(self):
        """Test all training modes produce same forward output shape."""
        for mode in ['joint', 'independent', 'sequential']:
            model = ConceptBottleneckModel(
                training=mode,
                input_size=8,
                annotations=self.ann,
                task_names=['task']
            )
            
            out = model(x=self.x, query=['c1', 'c2', 'task'])
            self.assertEqual(out.shape, (4, 3), f"Failed for mode: {mode}")


class TestCBMGraphLevels(unittest.TestCase):
    """Test graph level computation."""
    
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
    
    def test_graph_levels_computed(self):
        """Test graph_levels attribute is populated."""
        model = ConceptBottleneckModel(
            training='joint',
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        self.assertTrue(hasattr(model, 'graph_levels'))
        self.assertTrue(hasattr(model, 'roots'))
        self.assertIsInstance(model.graph_levels, list)
        self.assertIsInstance(model.roots, list)
    
    def test_roots_are_non_task_concepts(self):
        """Test roots are encoder-level concepts."""
        model = ConceptBottleneckModel(
            training='joint',
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        # Roots should be concepts that depend on input
        # In bipartite model: c1, c2 are roots
        self.assertIn('c1', model.roots)
        self.assertIn('c2', model.roots)
        self.assertNotIn('task', model.roots)


class TestTrainingModes(unittest.TestCase):
    """Test different training modes."""
    
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
        self.kwargs = {
            'input_size': 8,
            'annotations': self.ann,
            'task_names': ['task']
        }
    
    def test_joint_mode_works(self):
        """Test ConceptBottleneckModel with joint training."""
        model = ConceptBottleneckModel(training='joint', **self.kwargs)
        
        self.assertEqual(model._training_mode, 'joint')
        x = torch.randn(2, 8)
        out = model(x=x, query=['c1', 'c2', 'task'])
        self.assertEqual(out.shape, (2, 3))
    
    def test_independent_mode_works(self):
        """Test ConceptBottleneckModel with independent training."""
        model = ConceptBottleneckModel(training='independent', **self.kwargs)
        
        self.assertEqual(model._training_mode, 'independent')
        x = torch.randn(2, 8)
        out = model(x=x, query=['c1', 'c2', 'task'])
        self.assertEqual(out.shape, (2, 3))
    
    def test_sequential_mode_works(self):
        """Test ConceptBottleneckModel with sequential training."""
        model = ConceptBottleneckModel(training='sequential', **self.kwargs)
        
        self.assertEqual(model._training_mode, 'sequential')
        x = torch.randn(2, 8)
        out = model(x=x, query=['c1', 'c2', 'task'])
        self.assertEqual(out.shape, (2, 3))


class TestSequentialTrainingPhases(unittest.TestCase):
    """Test sequential training phase management."""
    
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
            training='sequential',
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
    
    def test_default_phase_is_encoder(self):
        """Test default training phase is encoder."""
        self.assertEqual(self.model.training_phase, 'encoder')
    
    def test_set_training_phase_predictor(self):
        """Test switching to predictor phase."""
        self.model.set_training_phase('predictor')
        self.assertEqual(self.model.training_phase, 'predictor')
    
    def test_set_training_phase_encoder(self):
        """Test switching back to encoder phase."""
        self.model.set_training_phase('predictor')
        self.model.set_training_phase('encoder')
        self.assertEqual(self.model.training_phase, 'encoder')
    
    def test_invalid_phase_raises(self):
        """Test invalid phase raises ValueError."""
        with self.assertRaises(ValueError):
            self.model.set_training_phase('invalid')


class TestLearnerIntegration(unittest.TestCase):
    """Test learner training step integration."""
    
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
        self.batch = {
            'inputs': {'x': torch.randn(4, 8)},
            'concepts': {'c': torch.randint(0, 2, (4, 3)).float()}
        }
    
    def _make_model(self, mode, with_loss=True):
        """Helper to create model with optional loss."""
        loss = nn.BCEWithLogitsLoss() if with_loss else None
        return ConceptBottleneckModel(
            training=mode,
            input_size=8,
            annotations=self.ann,
            task_names=['task'],
            loss=loss,
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.01}
        )
    
    def test_joint_training_step(self):
        """Test joint learner training step."""
        model = self._make_model('joint')
        model.train()
        
        loss = model.training_step(self.batch)
        
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
    
    def test_independent_training_step(self):
        """Test independent learner training step."""
        model = self._make_model('independent')
        model.train()
        
        loss = model.training_step(self.batch)
        
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
    
    def test_sequential_training_step_encoder(self):
        """Test sequential learner encoder phase training step."""
        model = self._make_model('sequential')
        model.set_training_phase('encoder')
        model.train()
        
        loss = model.training_step(self.batch)
        
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
    
    def test_sequential_training_step_predictor(self):
        """Test sequential learner predictor phase training step."""
        model = self._make_model('sequential')
        model.set_training_phase('predictor')
        model.train()
        
        loss = model.training_step(self.batch)
        
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
    
    def test_configure_optimizers_joint(self):
        """Test optimizer configuration for joint mode."""
        model = self._make_model('joint')
        
        config = model.configure_optimizers()
        
        self.assertIn('optimizer', config)
        self.assertIsInstance(config['optimizer'], torch.optim.Adam)
    
    def test_configure_optimizers_sequential(self):
        """Test optimizer configuration for sequential mode."""
        model = self._make_model('sequential')
        
        # Encoder phase
        model.set_training_phase('encoder')
        config = model.configure_optimizers()
        self.assertIn('optimizer', config)
        
        # Predictor phase
        model.set_training_phase('predictor')
        config = model.configure_optimizers()
        self.assertIn('optimizer', config)


if __name__ == '__main__':
    unittest.main()
