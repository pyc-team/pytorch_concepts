"""
Comprehensive tests for BlackBox models in torch_concepts.nn.modules.high.models.blackbox.

Tests cover:
- Model initialization with various configurations
- Forward pass and output shapes
- Training modes (manual PyTorch and Lightning)
- Backbone integration
- Distribution handling
- Filter methods for loss and metrics
- Edge cases and error handling
- Device consistency
- Gradient flow
- Lightning integration (BaseLearner, training_step, configure_optimizers)
- BlackBox and BlackBoxTaskOnly models
"""
import pytest
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from torch_concepts.nn.modules.high.models.blackbox import BlackBox, BlackBoxTaskOnly
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.annotations import AxisAnnotation, Annotations


# =============================================================================
# Test Fixtures and Helper Classes
# =============================================================================

class DummyBackbone(nn.Module):
    """Simple backbone for testing."""
    def __init__(self, out_features=8):
        super().__init__()
        self.out_features = out_features
    
    def forward(self, x):
        return torch.ones(x.shape[0], self.out_features)


class DummyLatentEncoder(nn.Module):
    """Simple latent encoder for testing."""
    def __init__(self, input_size, hidden_size=4):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        return self.linear(x)


def make_annotations(labels, cardinalities, distributions=None):
    """Helper to create annotations (defaults will fill in distributions)."""
    metadata = {}
    for label, card in zip(labels, cardinalities):
        metadata[label] = {'type': 'discrete'}
    return Annotations({
        1: AxisAnnotation(
            labels=labels,
            cardinalities=cardinalities,
            metadata=metadata
        )
    })


# =============================================================================
# BlackBox Model Tests
# =============================================================================

class TestBlackBoxInitialization(unittest.TestCase):
    """Test BlackBox initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = make_annotations(
            ['c1', 'c2', 'task'],
            [1, 2, 1]
        )
    
    def test_basic_init(self):
        """Test basic initialization."""
        model = BlackBox(input_size=8, annotations=self.ann)
        
        self.assertIsInstance(model, nn.Module)
        self.assertTrue(hasattr(model, 'linear'))
        self.assertTrue(hasattr(model, 'latent_encoder'))
    
    def test_init_with_backbone(self):
        """Test initialization with custom backbone."""
        backbone = DummyBackbone(out_features=16)
        model = BlackBox(
            input_size=8,
            annotations=self.ann,
            backbone=backbone
        )
        
        self.assertIsInstance(model.backbone, DummyBackbone)
        self.assertEqual(model.backbone.out_features, 16)
    
    def test_init_with_latent_encoder(self):
        """Test initialization with custom latent encoder."""
        model = BlackBox(
            input_size=8,
            annotations=self.ann,
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
        
        self.assertIsInstance(model.latent_encoder, DummyLatentEncoder)
        self.assertEqual(model.latent_encoder.linear.in_features, 8)
        self.assertEqual(model.latent_encoder.linear.out_features, 4)
    
    def test_output_size_calculation(self):
        """Test that output size equals sum of cardinalities."""
        model = BlackBox(input_size=8, annotations=self.ann)
        
        # Sum of cardinalities: 1 + 2 + 1 = 4
        expected_output_size = sum(self.ann[1].cardinalities)
        self.assertEqual(model.linear.out_features, expected_output_size)
    
    def test_init_with_defaults(self):
        """Test initialization with default distributions."""
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
        
        model = BlackBox(
            input_size=8,
            annotations=ann,
        )
        
        self.assertIsInstance(model, nn.Module)
    
    def test_latent_size_with_no_encoder(self):
        """Test latent_size equals input_size when no encoder is used."""
        model = BlackBox(input_size=16, annotations=self.ann)
        self.assertEqual(model.latent_size, 16)
    
    def test_latent_size_with_encoder_kwargs(self):
        """Test latent_size equals hidden_size from encoder kwargs."""
        model = BlackBox(
            input_size=8,
            annotations=self.ann,
            latent_encoder_kwargs={'hidden_size': 32}
        )
        self.assertEqual(model.latent_size, 32)
    
    def test_linear_layer_input_matches_latent_size(self):
        """Test that the linear layer input features match latent_size."""
        model = BlackBox(
            input_size=8,
            annotations=self.ann,
            latent_encoder_kwargs={'hidden_size': 16}
        )
        self.assertEqual(model.linear.in_features, 16)
    
    def test_concept_names_stored(self):
        """Test that concept names are stored correctly."""
        model = BlackBox(input_size=8, annotations=self.ann)
        self.assertEqual(model.concept_names, ['c1', 'c2', 'task'])
    
    def test_no_inference_engine(self):
        """Test that BlackBox does not set up inference engines."""
        model = BlackBox(input_size=8, annotations=self.ann)
        
        # BlackBox doesn't create model/inference, so accessing the
        # inference property should raise AttributeError (caught by hasattr)
        self.assertFalse(hasattr(model, 'eval_inference'))
        self.assertFalse(hasattr(model, 'train_inference'))


class TestBlackBoxForward(unittest.TestCase):
    """Test BlackBox forward pass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = make_annotations(
            ['c1', 'c2', 'task'],
            [1, 3, 2]
        )
    
    def _make_model(self, **kwargs):
        defaults = dict(
            input_size=8,
            annotations=self.ann,
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
        defaults.update(kwargs)
        return BlackBox(**defaults)
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = self._make_model()
        x = torch.randn(2, 8)
        out = model(x)
        
        # Output size is sum of cardinalities: 1 + 3 + 2 = 6
        expected_output_size = sum(self.ann[1].cardinalities)
        self.assertEqual(out.shape, (2, expected_output_size))
    
    def test_forward_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = self._make_model()
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 8)
            out = model(x)
            self.assertEqual(out.shape[0], batch_size)
    
    def test_forward_query_filters_output(self):
        """Test that query selects only the queried concept columns."""
        model = self._make_model()
        
        x = torch.randn(2, 8)
        # Annotations: c1(1), c2(3), task(2) — total 6
        out_all = model(x)
        self.assertEqual(out_all.shape, (2, 6))
        
        # Query single binary concept
        out_c1 = model(x, query=['c1'])
        self.assertEqual(out_c1.shape, (2, 1))
        self.assertTrue(torch.allclose(out_c1, out_all[:, 0:1]))
        
        # Query single categorical concept
        out_c2 = model(x, query=['c2'])
        self.assertEqual(out_c2.shape, (2, 3))
        self.assertTrue(torch.allclose(out_c2, out_all[:, 1:4]))
        
        # Query subset
        out_subset = model(x, query=['c1', 'task'])
        self.assertEqual(out_subset.shape, (2, 3))  # 1 + 2
        self.assertTrue(torch.allclose(out_subset[:, 0:1], out_all[:, 0:1]))
        self.assertTrue(torch.allclose(out_subset[:, 1:3], out_all[:, 4:6]))
    
    def test_forward_query_all_same_as_none(self):
        """Test that querying all concepts returns same as query=None."""
        model = self._make_model()
        model.eval()
        
        x = torch.randn(2, 8)
        out_none = model(x)
        out_all = model(x, query=['c1', 'c2', 'task'])
        
        self.assertEqual(out_none.shape, out_all.shape)
        self.assertTrue(torch.allclose(out_none, out_all))
    
    def test_forward_query_single_concept(self):
        """Test querying a single concept with large cardinality."""
        model = self._make_model()
        
        x = torch.randn(4, 8)
        out = model(x, query=['task'])
        self.assertEqual(out.shape, (4, 2))
    
    def test_forward_with_evidence_ignored(self):
        """Test that evidence parameter is accepted but doesn't affect output."""
        model = self._make_model()
        
        x = torch.randn(2, 8)
        out1 = model(x)
        out2 = model(x, evidence=torch.randn(2, 4))
        
        self.assertEqual(out1.shape, out2.shape)
    
    def test_forward_deterministic(self):
        """Test that forward pass is deterministic with same input."""
        model = self._make_model()
        model.eval()
        
        x = torch.randn(2, 8)
        out1 = model(x)
        out2 = model(x)
        
        self.assertTrue(torch.allclose(out1, out2))
    
    def test_forward_no_backbone(self):
        """Test forward pass with no backbone (identity)."""
        model = BlackBox(input_size=8, annotations=self.ann)
        x = torch.randn(2, 8)
        out = model(x)
        
        self.assertEqual(out.shape, (2, 6))
    
    def test_forward_no_latent_encoder(self):
        """Test forward pass with identity latent encoder."""
        model = BlackBox(input_size=8, annotations=self.ann)
        x = torch.randn(3, 8)
        out = model(x)
        
        self.assertEqual(out.shape[0], 3)
        self.assertEqual(out.shape[1], sum(self.ann[1].cardinalities))
    
    def test_forward_extra_kwargs_ignored(self):
        """Test that extra kwargs in forward are silently ignored."""
        model = self._make_model()
        x = torch.randn(2, 8)
        # These kwargs should be captured by **kwargs and ignored
        out = model(x, ground_truth=torch.ones(2, 6), concept_names=['a'])
        self.assertEqual(out.shape, (2, 6))


class TestBlackBoxFilterMethods(unittest.TestCase):
    """Test BlackBox filter methods inherited from BaseModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = make_annotations(['c1', 'task'], [1, 1])
        self.model = BlackBox(
            input_size=8,
            annotations=self.ann,
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
    
    def test_filter_output_for_loss(self):
        """Test filter_output_for_loss returns correct format."""
        x = torch.randn(2, 8)
        out = self.model(x)
        target = torch.randint(0, 2, out.shape)
        
        loss_dict = self.model.filter_output_for_loss(out, target)
        
        self.assertIn('input', loss_dict)
        self.assertIn('target', loss_dict)
        self.assertTrue(torch.allclose(loss_dict['input'], out))
        self.assertTrue(torch.allclose(loss_dict['target'], target))
    
    def test_filter_output_for_metrics(self):
        """Test filter_output_for_metrics returns correct format."""
        x = torch.randn(2, 8)
        out = self.model(x)
        target = torch.randint(0, 2, out.shape)
        
        metric_dict = self.model.filter_output_for_metrics(out, target)
        
        self.assertIn('preds', metric_dict)
        self.assertIn('target', metric_dict)
        self.assertTrue(torch.allclose(metric_dict['preds'], out))
        self.assertTrue(torch.allclose(metric_dict['target'], target))
    
    def test_filter_methods_inherited_from_base(self):
        """Test that filter methods are inherited from BaseModel (not overridden)."""
        # After cleanup, BlackBox should not have its own filter methods
        self.assertNotIn(
            'filter_output_for_loss', BlackBox.__dict__,
            "BlackBox should not override filter_output_for_loss"
        )
        self.assertNotIn(
            'filter_output_for_metrics', BlackBox.__dict__,
            "BlackBox should not override filter_output_for_metrics"
        )


class TestBlackBoxRepr(unittest.TestCase):
    """Test BlackBox string representation."""
    
    def test_repr_with_backbone(self):
        """Test __repr__ returns informative string with backbone."""
        ann = make_annotations(['output'], [1])
        model = BlackBox(
            input_size=8,
            annotations=ann,
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
        
        repr_str = repr(model)
        self.assertIsInstance(repr_str, str)
        self.assertIn('BlackBox', repr_str)
        self.assertIn('DummyBackbone', repr_str)
    
    def test_repr_without_backbone(self):
        """Test __repr__ when no backbone is used."""
        ann = make_annotations(['output'], [1])
        model = BlackBox(input_size=8, annotations=ann)
        
        repr_str = repr(model)
        self.assertIn('BlackBox', repr_str)
        self.assertIn('None', repr_str)  # backbone=None


# =============================================================================
# BlackBox Lightning Integration Tests
# =============================================================================

class TestBlackBoxLightning(unittest.TestCase):
    """Test BlackBox with Lightning training mode."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = make_annotations(['c1', 'task'], [1, 1])
    
    def test_lightning_creates_base_learner(self):
        """Test that lightning=True creates a BaseLearner instance."""
        model = BlackBox(
            lightning=True,
            input_size=8,
            annotations=self.ann,
        )
        
        self.assertIsInstance(model, BaseLearner)
    
    def test_default_is_pytorch(self):
        """Test that default is pure PyTorch module (no Lightning)."""
        model = BlackBox(input_size=8, annotations=self.ann)
        self.assertFalse(isinstance(model, BaseLearner))
    
    def test_lightning_forward_works(self):
        """Test forward pass works with Lightning mode."""
        model = BlackBox(
            lightning=True,
            input_size=8,
            annotations=self.ann,
        )
        
        x = torch.randn(2, 8)
        out = model(x)
        
        self.assertEqual(out.shape, (2, 2))
    
    def test_lightning_training_step(self):
        """Test Lightning training_step works for BlackBox."""
        model = BlackBox(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            loss=nn.BCEWithLogitsLoss(),
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.01}
        )
        model.train()
        
        batch = {
            'inputs': {'x': torch.randn(4, 8)},
            'concepts': {'c': torch.randint(0, 2, (4, 2)).float()}
        }
        
        loss = model.training_step(batch)
        
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
    
    def test_lightning_validation_step(self):
        """Test Lightning validation_step works for BlackBox."""
        model = BlackBox(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            loss=nn.BCEWithLogitsLoss(),
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.01}
        )
        model.eval()
        
        batch = {
            'inputs': {'x': torch.randn(4, 8)},
            'concepts': {'c': torch.randint(0, 2, (4, 2)).float()}
        }
        
        loss = model.validation_step(batch)
        
        self.assertIsNotNone(loss)
    
    def test_lightning_configure_optimizers(self):
        """Test optimizer configuration for Lightning mode."""
        model = BlackBox(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            loss=nn.BCEWithLogitsLoss(),
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.01}
        )
        
        config = model.configure_optimizers()
        
        self.assertIn('optimizer', config)
        self.assertIsInstance(config['optimizer'], torch.optim.Adam)
    
    def test_lightning_no_optimizer_returns_none(self):
        """Test configure_optimizers returns None when no optimizer set."""
        model = BlackBox(
            lightning=True,
            input_size=8,
            annotations=self.ann,
        )
        
        config = model.configure_optimizers()
        self.assertIsNone(config)
    
    def test_lightning_get_inference_kwargs_returns_empty(self):
        """Test that _get_inference_kwargs returns {} for BlackBox."""
        model = BlackBox(
            lightning=True,
            input_size=8,
            annotations=self.ann,
        )
        
        batch = {
            'inputs': {'x': torch.randn(2, 8)},
            'concepts': {'c': torch.randint(0, 2, (2, 2)).float()}
        }
        
        kwargs = model._get_inference_kwargs(batch)
        self.assertEqual(kwargs, {})


# =============================================================================
# BlackBoxTaskOnly Model Tests
# =============================================================================

class TestBlackBoxTaskOnlyInitialization(unittest.TestCase):
    """Test BlackBoxTaskOnly initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = make_annotations(
            ['c1', 'c2', 'task1', 'task2'],
            [1, 2, 1, 3]
        )
    
    def test_basic_init_single_task(self):
        """Test basic initialization with single task."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task1'
        )
        
        self.assertIsInstance(model, nn.Module)
        self.assertTrue(hasattr(model, 'linear'))
        # Only task1 cardinality (1)
        self.assertEqual(model.linear.out_features, 1)
    
    def test_basic_init_multiple_tasks(self):
        """Test basic initialization with multiple tasks."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names=['task1', 'task2']
        )
        
        # task1 (1) + task2 (3) = 4
        self.assertEqual(model.linear.out_features, 4)
    
    def test_init_with_backbone(self):
        """Test initialization with custom backbone."""
        backbone = DummyBackbone(out_features=16)
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task1',
            backbone=backbone
        )
        
        self.assertIsInstance(model.backbone, DummyBackbone)
    
    def test_init_with_latent_encoder(self):
        """Test initialization with custom latent encoder."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task1',
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
        
        self.assertIsInstance(model.latent_encoder, DummyLatentEncoder)
    
    def test_task_concept_idx_calculation(self):
        """Test that task concept-level indices are correctly calculated."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']  # task1 is at concept index 2
        )
        
        # c1=0, c2=1, task1=2, task2=3
        self.assertEqual(model.task_concept_idx, [2])
    
    def test_task_annotations_created(self):
        """Test that task_annotations sub-annotation is correctly created."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task1'
        )
        
        self.assertEqual(model.task_annotations.labels, ['task1'])
        self.assertEqual(model.task_annotations.cardinalities, [1])
    
    def test_task_annotations_multiple_tasks(self):
        """Test task_annotations with multiple tasks."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names=['task1', 'task2']
        )
        
        self.assertEqual(model.task_annotations.labels, ['task1', 'task2'])
        self.assertEqual(model.task_annotations.cardinalities, [1, 3])
    
    def test_init_with_defaults(self):
        """Test initialization with default distributions."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'task'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'task': {'type': 'discrete'}
                }
            )
        })
        
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task',
        )
        
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.linear.out_features, 1)
    
    def test_no_inference_engine(self):
        """Test that BlackBoxTaskOnly does not set up inference engines."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task1'
        )
        
        self.assertFalse(hasattr(model, 'eval_inference'))
        self.assertFalse(hasattr(model, 'train_inference'))


class TestBlackBoxTaskOnlyForward(unittest.TestCase):
    """Test BlackBoxTaskOnly forward pass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = make_annotations(
            ['c1', 'c2', 'task1', 'task2'],
            [1, 2, 1, 3]
        )
    
    def _make_model(self, task_names='task1', **kwargs):
        defaults = dict(
            input_size=8,
            annotations=self.ann,
            task_names=task_names,
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
        defaults.update(kwargs)
        return BlackBoxTaskOnly(**defaults)
    
    def test_forward_shape_single_task(self):
        """Test forward pass output shape with single task."""
        model = self._make_model(task_names='task1')
        x = torch.randn(2, 8)
        out = model(x)
        
        # Only task1 output (cardinality 1)
        self.assertEqual(out.shape, (2, 1))
    
    def test_forward_shape_multiple_tasks(self):
        """Test forward pass output shape with multiple tasks."""
        model = self._make_model(task_names=['task1', 'task2'])
        x = torch.randn(2, 8)
        out = model(x)
        
        # task1(1) + task2(3) = 4
        self.assertEqual(out.shape, (2, 4))
    
    def test_forward_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = self._make_model()
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 8)
            out = model(x)
            self.assertEqual(out.shape[0], batch_size)
    
    def test_forward_deterministic(self):
        """Test that forward pass is deterministic."""
        model = self._make_model()
        model.eval()
        
        x = torch.randn(2, 8)
        out1 = model(x)
        out2 = model(x)
        
        self.assertTrue(torch.allclose(out1, out2))
    
    def test_forward_extra_kwargs_ignored(self):
        """Test that extra kwargs in forward are silently ignored."""
        model = self._make_model()
        x = torch.randn(2, 8)
        out = model(x, ground_truth=torch.ones(2, 1), concept_names=['a'])
        self.assertEqual(out.shape, (2, 1))


class TestBlackBoxTaskOnlyFilterMethods(unittest.TestCase):
    """Test BlackBoxTaskOnly filter methods with padding logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = make_annotations(
            ['c1', 'c2', 'task1'],
            [1, 2, 1]
        )
        self.model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task1',
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
    
    def test_filter_output_for_loss_slices_target(self):
        """Test filter_output_for_loss slices target to task columns."""
        x = torch.randn(2, 8)
        out = self.model(x)  # Shape: (2, 1) for task1 only
        # Full concept-level target: c1, c2, task1
        target = torch.tensor([[0., 1., 1.],
                               [1., 0., 0.]])
        
        loss_dict = self.model.filter_output_for_loss(out, target)
        
        self.assertIn('input', loss_dict)
        self.assertIn('target', loss_dict)
        
        # Input should be the raw model output (task-only)
        self.assertEqual(loss_dict['input'].shape, (2, 1))
        self.assertTrue(torch.allclose(loss_dict['input'], out))
        
        # Target should be sliced to task1 column (concept index 2)
        self.assertEqual(loss_dict['target'].shape, (2, 1))
        self.assertTrue(torch.allclose(loss_dict['target'], target[:, 2:3]))
    
    def test_filter_output_for_metrics_slices_target(self):
        """Test filter_output_for_metrics slices target to task columns."""
        x = torch.randn(2, 8)
        out = self.model(x)
        target = torch.tensor([[0., 1., 1.],
                               [1., 0., 0.]])
        
        metric_dict = self.model.filter_output_for_metrics(out, target)
        
        self.assertIn('preds', metric_dict)
        self.assertIn('target', metric_dict)
        
        # Preds should be the raw model output (task-only)
        self.assertEqual(metric_dict['preds'].shape, (2, 1))
        # Target should be sliced to task1 column
        self.assertEqual(metric_dict['target'].shape, (2, 1))
    
    def test_filter_loss_preserves_gradients(self):
        """Test that filter operation preserves gradient flow."""
        x = torch.randn(2, 8)
        out = self.model(x)
        
        loss_dict = self.model.filter_output_for_loss(out, torch.zeros(2, 3))
        loss = loss_dict['input'].sum()
        loss.backward()
        
        # Check gradients exist on model parameters
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_filter_loss_multiple_tasks_slicing(self):
        """Test target slicing with multiple tasks."""
        ann = make_annotations(
            ['c1', 'task1', 'task2'],
            [2, 1, 3]
        )
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names=['task1', 'task2'],
        )
        
        x = torch.randn(3, 8)
        out = model(x)  # Shape: (3, 4) for task1(1) + task2(3)
        self.assertEqual(out.shape, (3, 4))
        
        # Full concept-level target: c1, task1, task2
        target = torch.tensor([[0., 1., 2.],
                               [1., 0., 1.],
                               [0., 1., 0.]])
        
        loss_dict = model.filter_output_for_loss(out, target)
        
        # Input is the raw task-only output
        self.assertEqual(loss_dict['input'].shape, (3, 4))
        self.assertTrue(torch.allclose(loss_dict['input'], out))
        
        # Target sliced to task1 (idx 1) and task2 (idx 2)
        self.assertEqual(loss_dict['target'].shape, (3, 2))
        self.assertTrue(torch.allclose(loss_dict['target'], target[:, [1, 2]]))
    
    def test_filter_metrics_uses_same_slicing(self):
        """Test that metrics filter uses same target-slicing logic as loss filter."""
        x = torch.randn(4, 8)
        out = self.model(x)
        target = torch.zeros(4, 3)  # concept-level: c1, c2, task1
        
        loss_dict = self.model.filter_output_for_loss(out, target)
        metric_dict = self.model.filter_output_for_metrics(out, target)
        
        # Both should produce same results (just different keys)
        self.assertTrue(torch.allclose(loss_dict['input'], metric_dict['preds']))
        self.assertTrue(torch.allclose(loss_dict['target'], metric_dict['target']))
    
    def test_filter_overrides_base_model(self):
        """Test that BlackBoxTaskOnly overrides BaseModel filter methods."""
        self.assertIn('filter_output_for_loss', BlackBoxTaskOnly.__dict__)
        self.assertIn('filter_output_for_metrics', BlackBoxTaskOnly.__dict__)


class TestBlackBoxTaskOnlyMultipleTasks(unittest.TestCase):
    """Test BlackBoxTaskOnly with multiple tasks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = make_annotations(
            ['c1', 'task1', 'c2', 'task2'],
            [1, 2, 1, 3]
        )
    
    def test_init_with_non_contiguous_tasks(self):
        """Test initialization when tasks are not contiguous in annotations."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']  # Only first task
        )
        
        self.assertIsInstance(model, nn.Module)
        # task1 cardinality is 2
        self.assertEqual(model.linear.out_features, 2)
    
    def test_task_at_beginning(self):
        """Test when task is at the beginning of annotations."""
        ann = make_annotations(
            ['task', 'c1', 'c2'],
            [1, 2, 3]
        )
        
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task'
        )
        
        # Task is at concept index 0
        self.assertEqual(model.task_concept_idx, [0])
    
    def test_task_at_end(self):
        """Test when task is at the end of annotations."""
        ann = make_annotations(
            ['c1', 'c2', 'task'],
            [2, 3, 1]
        )
        
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task'
        )
        
        # Task is at concept index 2
        self.assertEqual(model.task_concept_idx, [2])


# =============================================================================
# BlackBoxTaskOnly Lightning Integration Tests
# =============================================================================

class TestBlackBoxTaskOnlyLightning(unittest.TestCase):
    """Test BlackBoxTaskOnly with Lightning training mode."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = make_annotations(
            ['c1', 'c2', 'task'],
            [1, 2, 1]
        )
    
    def test_lightning_creates_base_learner(self):
        """Test that lightning=True creates a BaseLearner instance."""
        model = BlackBoxTaskOnly(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names='task',
        )
        
        self.assertIsInstance(model, BaseLearner)
    
    def test_default_is_pytorch(self):
        """Test that default is pure PyTorch module."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task',
        )
        self.assertFalse(isinstance(model, BaseLearner))
    
    def test_lightning_training_step(self):
        """Test Lightning training_step works for BlackBoxTaskOnly."""
        model = BlackBoxTaskOnly(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names='task',
            loss=nn.BCEWithLogitsLoss(),
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.01}
        )
        model.train()
        
        # Concept-level target: one column per concept (c1, c2, task)
        batch = {
            'inputs': {'x': torch.randn(4, 8)},
            'concepts': {'c': torch.randint(0, 2, (4, 3)).float()}
        }
        
        loss = model.training_step(batch)
        
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
    
    def test_lightning_configure_optimizers(self):
        """Test optimizer configuration for Lightning mode."""
        model = BlackBoxTaskOnly(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names='task',
            optim_class=torch.optim.AdamW,
            optim_kwargs={'lr': 0.001}
        )
        
        config = model.configure_optimizers()
        
        self.assertIn('optimizer', config)
        self.assertIsInstance(config['optimizer'], torch.optim.AdamW)
    
    def test_lightning_get_inference_kwargs_returns_empty(self):
        """Test that _get_inference_kwargs returns {} for BlackBoxTaskOnly."""
        model = BlackBoxTaskOnly(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names='task',
        )
        
        batch = {
            'inputs': {'x': torch.randn(2, 8)},
            'concepts': {'c': torch.randint(0, 2, (2, 4)).float()}
        }
        
        kwargs = model._get_inference_kwargs(batch)
        self.assertEqual(kwargs, {})


class TestBlackBoxTaskOnlyRepr(unittest.TestCase):
    """Test BlackBoxTaskOnly string representation."""
    
    def test_repr(self):
        """Test __repr__ returns informative string."""
        ann = make_annotations(['c1', 'task'], [1, 1])
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task'
        )
        
        repr_str = repr(model)
        self.assertIsInstance(repr_str, str)
        self.assertIn('BlackBoxTaskOnly', repr_str)


# =============================================================================
# Device Consistency Tests
# =============================================================================

class TestBlackBoxDeviceConsistency(unittest.TestCase):
    """Test device consistency for BlackBox models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = make_annotations(['c1', 'task'], [1, 1])
    
    def test_blackbox_device_consistency(self):
        """Test that BlackBox maintains device consistency."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        model = BlackBox(input_size=8, annotations=self.ann)
        device = torch.device('cuda')
        model = model.to(device)
        
        x = torch.randn(2, 8, device=device)
        out = model(x)
        
        self.assertEqual(out.device.type, device.type)
    
    def test_blackbox_task_only_device_consistency(self):
        """Test that BlackBoxTaskOnly maintains device consistency."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task'
        )
        device = torch.device('cuda')
        model = model.to(device)
        
        x = torch.randn(2, 8, device=device)
        out = model(x)
        
        self.assertEqual(out.device.type, device.type)
    
    def test_blackbox_task_only_filter_preserves_device(self):
        """Test that filter methods preserve tensor device."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task'
        )
        device = torch.device('cuda')
        model = model.to(device)
        
        x = torch.randn(2, 8, device=device)
        out = model(x)
        target = torch.zeros(2, 2, device=device)
        
        loss_dict = model.filter_output_for_loss(out, target)
        
        self.assertEqual(loss_dict['input'].device.type, device.type)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestBlackBoxEdgeCases(unittest.TestCase):
    """Test edge cases for BlackBox models."""
    
    def test_single_concept(self):
        """Test with single concept."""
        ann = make_annotations(['output'], [1])
        model = BlackBox(input_size=8, annotations=ann)
        x = torch.randn(2, 8)
        out = model(x)
        
        self.assertEqual(out.shape, (2, 1))
    
    def test_large_cardinalities(self):
        """Test with large cardinalities."""
        ann = make_annotations(['c1', 'c2'], [100, 50])
        model = BlackBox(input_size=8, annotations=ann)
        x = torch.randn(2, 8)
        out = model(x)
        
        self.assertEqual(out.shape, (2, 150))
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        ann = make_annotations(['output'], [1])
        model = BlackBox(input_size=8, annotations=ann)
        x = torch.randn(1, 8)
        out = model(x)
        
        self.assertEqual(out.shape, (1, 1))
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        ann = make_annotations(['output'], [1])
        model = BlackBox(input_size=8, annotations=ann)
        x = torch.randn(2, 8, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))
    
    def test_many_concepts(self):
        """Test with many concepts."""
        labels = [f'c{i}' for i in range(20)]
        cardinalities = [1] * 20
        ann = make_annotations(labels, cardinalities)
        
        model = BlackBox(input_size=8, annotations=ann)
        x = torch.randn(2, 8)
        out = model(x)
        
        self.assertEqual(out.shape, (2, 20))
    
    def test_large_input_size(self):
        """Test with large input size."""
        ann = make_annotations(['c1'], [1])
        model = BlackBox(input_size=512, annotations=ann)
        x = torch.randn(2, 512)
        out = model(x)
        
        self.assertEqual(out.shape, (2, 1))


class TestBlackBoxTaskOnlyEdgeCases(unittest.TestCase):
    """Test edge cases for BlackBoxTaskOnly."""
    
    def test_gradient_flow(self):
        """Test that gradients flow through BlackBoxTaskOnly."""
        ann = make_annotations(['c1', 'task'], [1, 1])
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task'
        )
        
        x = torch.randn(2, 8, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))
    
    def test_gradient_flow_through_filter(self):
        """Test that gradients flow through filter output."""
        ann = make_annotations(['c1', 'task'], [1, 1])
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task'
        )
        
        x = torch.randn(2, 8, requires_grad=True)
        out = model(x)
        loss_dict = model.filter_output_for_loss(out, torch.zeros(2, 2))
        loss = loss_dict['input'].sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
    
    def test_large_task_cardinality(self):
        """Test with large task cardinality."""
        ann = make_annotations(['c1', 'task'], [1, 50])
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task'
        )
        
        x = torch.randn(2, 8)
        out = model(x)
        
        self.assertEqual(out.shape, (2, 50))
        
        # Full target has 2 concept-level columns (c1, task)
        target = torch.zeros(2, 2)
        loss_dict = model.filter_output_for_loss(out, target)
        # Input stays task-only
        self.assertEqual(loss_dict['input'].shape, (2, 50))
        # Target sliced to task column only
        self.assertEqual(loss_dict['target'].shape, (2, 1))
    
    def test_single_task_is_only_concept(self):
        """Test when the only concept is also the task."""
        ann = make_annotations(['task'], [1])
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task'
        )
        
        self.assertEqual(model.task_concept_idx, [0])
        
        x = torch.randn(2, 8)
        out = model(x)
        self.assertEqual(out.shape, (2, 1))
        
        # When task is the only concept, filter is identity on both sides
        target = torch.zeros(2, 1)
        loss_dict = model.filter_output_for_loss(out, target)
        self.assertTrue(torch.allclose(loss_dict['input'], out))
        self.assertTrue(torch.allclose(loss_dict['target'], target))
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        ann = make_annotations(['c1', 'task'], [1, 1])
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task'
        )
        
        x = torch.randn(1, 8)
        out = model(x)
        self.assertEqual(out.shape, (1, 1))
        
        # Full target has 2 concept-level columns
        target = torch.zeros(1, 2)
        loss_dict = model.filter_output_for_loss(out, target)
        self.assertEqual(loss_dict['input'].shape, (1, 1))
        self.assertEqual(loss_dict['target'].shape, (1, 1))


# =============================================================================
# Training Integration Tests
# =============================================================================

class TestBlackBoxTraining(unittest.TestCase):
    """Test BlackBox models in manual training scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = make_annotations(['c1', 'task'], [1, 1])
    
    def test_training_step(self):
        """Test a basic training step."""
        model = BlackBox(input_size=8, annotations=self.ann)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        x = torch.randn(4, 8)
        target = torch.zeros(4, 2)
        
        # Forward pass
        out = model(x)
        loss_dict = model.filter_output_for_loss(out, target)
        loss = nn.functional.binary_cross_entropy_with_logits(
            loss_dict['input'], 
            loss_dict['target']
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        self.assertTrue(loss.item() > 0)
    
    def test_eval_mode(self):
        """Test model in evaluation mode."""
        model = BlackBox(input_size=8, annotations=self.ann)
        model.eval()
        
        x = torch.randn(2, 8)
        
        with torch.no_grad():
            out = model(x)
        
        self.assertFalse(out.requires_grad)
    
    def test_train_eval_toggle(self):
        """Test toggling between train and eval modes."""
        model = BlackBox(input_size=8, annotations=self.ann)
        
        model.train()
        self.assertTrue(model.training)
        
        model.eval()
        self.assertFalse(model.training)
        
        model.train()
        self.assertTrue(model.training)
    
    def test_task_only_training_step(self):
        """Test a training step with BlackBoxTaskOnly."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task'
        )
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        x = torch.randn(4, 8)
        # Concept-level target: c1, task
        target = torch.zeros(4, 2)
        
        out = model(x)
        loss_dict = model.filter_output_for_loss(out, target)
        loss = nn.functional.binary_cross_entropy_with_logits(
            loss_dict['input'], 
            loss_dict['target']
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.assertTrue(loss.item() > 0)
    
    def test_parameter_update(self):
        """Test that parameters actually change after optimization step."""
        ann = make_annotations(['c1'], [1])
        model = BlackBox(input_size=4, annotations=ann)
        model.train()
        
        # Store initial parameters
        initial_weight = model.linear.weight.data.clone()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.randn(2, 4)
        target = torch.ones(2, 1)
        
        out = model(x)
        loss = nn.functional.mse_loss(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Parameters should have changed
        self.assertFalse(torch.allclose(model.linear.weight.data, initial_weight))


# =============================================================================
# Backbone Integration Tests
# =============================================================================

class TestBlackBoxBackboneIntegration(unittest.TestCase):
    """Test BlackBox models with various backbone configurations."""
    
    def test_blackbox_with_backbone_transforms_input(self):
        """Test that backbone transforms input before linear layer."""
        ann = make_annotations(['c1'], [1])
        backbone = DummyBackbone(out_features=8)
        
        model = BlackBox(
            input_size=8,
            annotations=ann,
            backbone=backbone
        )
        
        # Input size doesn't matter since DummyBackbone produces fixed size
        x = torch.randn(2, 100)
        out = model(x)
        
        self.assertEqual(out.shape, (2, 1))
    
    def test_task_only_with_backbone(self):
        """Test BlackBoxTaskOnly with backbone."""
        ann = make_annotations(['c1', 'task'], [1, 1])
        backbone = DummyBackbone(out_features=8)
        
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task',
            backbone=backbone
        )
        
        x = torch.randn(2, 50)
        out = model(x)
        
        self.assertEqual(out.shape, (2, 1))
    
    def test_no_backbone_uses_identity(self):
        """Test that no backbone means identity transform."""
        ann = make_annotations(['c1'], [1])
        model = BlackBox(input_size=8, annotations=ann)
        
        self.assertIsNone(model.backbone)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
