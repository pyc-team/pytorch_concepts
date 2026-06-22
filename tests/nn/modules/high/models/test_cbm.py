"""
Comprehensive tests for Concept Bottleneck Model (CBM).

Tests cover:
- Model initialization with various configurations
- Forward pass and output shapes
- Training modes (joint, independent)
- Backbone integration
- Distribution handling
- Target preparation (prepare_target)
- Factory function behavior
"""
import pytest
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, OneHotCategorical, RelaxedBernoulli, RelaxedOneHotCategorical
from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.nn import MLP
from torch_concepts.annotations import AxisAnnotation, Annotations


def _logits(out, names):
    """Concatenate per-variable logits for the queried ``names`` -> (B, sum cardinalities)."""
    import torch
    return torch.cat([out.params[n]['logits'] for n in names], dim=1)


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
                    'color': {'type': 'discrete'},
                    'shape': {'type': 'discrete'},
                    'size': {'type': 'discrete'},
                    'task1': {'type': 'discrete'}
                }
            )
        })
    
    def test_init_defaults(self):
        """Test initialization with default distributions on the model."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']
        )

        self.assertIsInstance(model.pgm, nn.Module)
        self.assertTrue(hasattr(model, 'inference'))
        self.assertEqual(model.concept_names, ['color', 'shape', 'size', 'task1'])
        # Distributions live on the model, not on the annotation
        self.assertEqual(model.variable_distributions['categorical'], OneHotCategorical)
        self.assertEqual(model.variable_distributions['binary'], Bernoulli)

    def test_init_with_variable_distributions_param(self):
        """Test initialization passing per-type variable_distributions override."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            variable_distributions={'binary': RelaxedBernoulli},
        )

        self.assertEqual(model.variable_distributions['binary'], RelaxedBernoulli)
        self.assertEqual(model.variable_distributions['categorical'], OneHotCategorical)
    
    def test_init_with_backbone(self):
        """Test initialization with custom backbone (raw input -> latent)."""
        backbone = DummyBackbone(out_features=8)
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            backbone=backbone,
            latent_size=8,
            task_names=['task1']
        )

        self.assertIs(model.backbone, backbone)

    def test_init_with_mlp_backbone(self):
        """Test initialization with an MLP backbone resolving latent_size."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            backbone=MLP(input_size=8, hidden_size=16, n_layers=2),
            latent_size=16,
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
                    'color': {'type': 'discrete'},
                    'shape': {'type': 'discrete'},
                    'size': {'type': 'discrete'},
                    'task1': {'type': 'discrete'}
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
        out = self.model(query=query, input=x)

        # Output shape: batch_size x sum(cardinalities for queried variables)
        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 3 + 2 + 1)  # color + shape + size

    def test_forward_all_concepts(self):
        """Test forward with all concepts."""
        x = torch.randn(4, 8)
        query = ['color', 'shape', 'size', 'task1']
        out = self.model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 4)
        self.assertEqual(logits.shape[1], 3 + 2 + 1 + 1)

    def test_forward_single_concept(self):
        """Test forward with single concept."""
        x = torch.randn(2, 8)
        query = ['color']
        out = self.model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 3)

    def test_forward_with_backbone(self):
        """Test forward pass with backbone (raw input -> latent inside the PGM)."""
        backbone = DummyBackbone(out_features=8)
        model = ConceptBottleneckModel(
            input_size=100,  # raw input dim (the PGM 'input' node)
            annotations=self.ann,
            backbone=backbone,
            latent_size=8,   # backbone output dim (the PGM 'latent' node)
            task_names=['task1']
        )

        x = torch.randn(2, 100)  # Raw input size (before backbone)
        query = ['color', 'shape']
        out = model(query=query, input=x)

        logits = _logits(out, query)
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], 3 + 2)


class TestCBMPrepareTarget(unittest.TestCase):
    """Test CBM prepare_target."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
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
        
        self.model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
    
    def test_prepare_target(self):
        """Test prepare_target returns target unchanged for CBM."""
        target = torch.randint(0, 2, (2, 3)).float()
        
        prepared = self.model.prepare_target(target)
        self.assertTrue(torch.allclose(prepared, target))


class TestCBMTraining(unittest.TestCase):
    """Test CBM training scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
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
    
    def test_manual_training_mode(self):
        """Test manual PyTorch training (no lightning mode)."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        # No lightning mode = pure PyTorch module
        self.assertFalse(isinstance(model, BaseLearner))

        # Can train manually
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()

        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4, 3)).float()

        model.train()
        query = ['c1', 'c2', 'task']
        out = model(query=query, input=x)
        loss = loss_fn(_logits(out, query), y)

        self.assertTrue(loss.requires_grad)

    def test_gradients_flow(self):
        """Test that gradients flow through the model."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        x = torch.randn(4, 8, requires_grad=True)
        query = ['c1', 'c2', 'task']
        out = model(query=query, input=x)
        loss = _logits(out, query).sum()
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
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'}
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
                labels=['c1', 'task'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'task': {'type': 'discrete'},
                }
            )
        })

        model = ConceptBottleneckModel(
            input_size=8,
            annotations=ann,
            task_names=['task']
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
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })

    def test_factory_joint_mode(self):
        """Test factory creates Lightning model with lightning=True."""
        model = ConceptBottleneckModel(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        self.assertIsInstance(model, BaseLearner)

    def test_factory_independent_mode(self):
        """IndependentInference is a DeterministicInference subclass — now allowed."""
        from torch_concepts.nn import IndependentInference
        # Should succeed (no ValueError) because IndependentInference is a subclass
        model = ConceptBottleneckModel(
            lightning=True,
            train_inference=IndependentInference,
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        self.assertIsInstance(model, BaseLearner)

    def test_factory_default_is_pytorch(self):
        """Test default is pure PyTorch module (no lightning mode)."""
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        self.assertFalse(isinstance(model, BaseLearner))


class TestCBMUnifiedForward(unittest.TestCase):
    """Test unified forward pass works across all modes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
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
        self.x = torch.randn(4, 8)
    
    def test_forward_with_x_only(self):
        """Test forward with x tensor only via lightning=True."""
        model = ConceptBottleneckModel(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        out = model(query=['c1', 'c2', 'task'], input=self.x)
        self.assertEqual(_logits(out, ['c1', 'c2', 'task']).shape, (4, 3))

    def test_forward_with_evidence(self):
        """Test forward with evidence dict works without error."""
        model = ConceptBottleneckModel(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        out = model(query=['c1', 'c2', 'task'], input=self.x)
        self.assertEqual(_logits(out, ['c1', 'c2', 'task']).shape, (4, 3))

    def test_forward_same_output_all_modes(self):
        """Test Lightning and pure PyTorch modes produce same forward output shape."""
        for lightning_mode in [True, False]:
            model = ConceptBottleneckModel(
                lightning=lightning_mode,
                input_size=8,
                annotations=self.ann,
                task_names=['task']
            )

            out = model(query=['c1', 'c2', 'task'], input=self.x)
            self.assertEqual(
                _logits(out, ['c1', 'c2', 'task']).shape, (4, 3),
                f"Failed for lightning_mode: {lightning_mode}"
            )


class TestTrainingModes(unittest.TestCase):
    """Test different training modes."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
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
        self.kwargs = {
            'input_size': 8,
            'annotations': self.ann,
            'task_names': ['task']
        }

    def test_joint_mode_works(self):
        """Test ConceptBottleneckModel with Lightning training."""
        model = ConceptBottleneckModel(lightning=True, **self.kwargs)

        self.assertIsInstance(model, BaseLearner)
        x = torch.randn(2, 8)
        out = model(query=['c1', 'c2', 'task'], input=x)
        self.assertEqual(_logits(out, ['c1', 'c2', 'task']).shape, (2, 3))

    def test_independent_mode_works(self):
        """IndependentInference is a DeterministicInference subclass — now allowed."""
        from torch_concepts.nn import IndependentInference
        model = ConceptBottleneckModel(lightning=True, train_inference=IndependentInference, **self.kwargs)
        self.assertIsInstance(model, BaseLearner)


class TestLearnerIntegration(unittest.TestCase):
    """Test learner training step integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
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
        self.batch = {
            'inputs': {'x': torch.randn(4, 8)},
            'concepts': {'c': torch.randint(0, 2, (4, 3)).float()}
        }

    def _make_model(self, lightning=True, with_loss=True, train_inference=None):
        """Helper to create model with optional loss."""
        loss = nn.BCEWithLogitsLoss() if with_loss else None
        kwargs = {
            'lightning': lightning,
            'input_size': 8,
            'annotations': self.ann,
            'task_names': ['task'],
            'loss': loss,
            'optim_class': torch.optim.Adam,
            'optim_kwargs': {'lr': 0.01}
        }
        if train_inference is not None:
            kwargs['train_inference'] = train_inference
        return ConceptBottleneckModel(**kwargs)

    def test_joint_training_step(self):
        """Test Lightning learner training step."""
        model = self._make_model(lightning=True)
        model.train()

        loss = model.training_step(self.batch)

        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)

    def test_independent_training_step(self):
        """IndependentInference is a DeterministicInference subclass — now allowed."""
        from torch_concepts.nn import IndependentInference
        model = self._make_model(lightning=True, train_inference=IndependentInference)
        self.assertIsInstance(model, BaseLearner)

    def test_configure_optimizers_joint(self):
        """Test optimizer configuration for Lightning mode."""
        model = self._make_model(lightning=True)

        config = model.configure_optimizers()

        self.assertIn('optimizer', config)
        self.assertIsInstance(config['optimizer'], torch.optim.Adam)


if __name__ == '__main__':
    unittest.main()
