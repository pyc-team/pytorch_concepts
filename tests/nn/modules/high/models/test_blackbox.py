"""
Comprehensive tests for BlackBox models in torch_concepts.nn.modules.high.models.blackbox.

Tests cover:
- Model initialization with various configurations
- Forward pass and output shapes
- Training modes (manual PyTorch and Lightning)
- Backbone integration
- Distribution handling
- Target preparation (prepare_target)
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
from torch_concepts.nn.modules.loss import ConceptLoss
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torch_concepts.nn import MLP
from torch_concepts.annotations import AxisAnnotation, Annotations


# =============================================================================
# Test Fixtures and Helper Classes
# =============================================================================

def _logits(out, names):
    """Concatenate the per-concept logits for ``names`` along the feature axis."""
    import torch
    return torch.cat([out.params[n]['logits'] for n in names], dim=1)

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
        self.assertTrue(hasattr(model, 'backbone'))

    def test_init_with_backbone(self):
        """Test initialization with custom backbone."""
        backbone = DummyBackbone(out_features=16)
        model = BlackBox(
            input_size=8,
            annotations=self.ann,
            backbone=backbone,
            latent_size=16,
        )

        self.assertIsInstance(model.backbone, DummyBackbone)
        self.assertEqual(model.backbone.out_features, 16)

    def test_init_with_backbone_encoder(self):
        """Test initialization with a custom backbone mapping input to latent."""
        model = BlackBox(
            input_size=8,
            annotations=self.ann,
            backbone=DummyLatentEncoder(8, hidden_size=4),
            latent_size=4,
        )

        self.assertIsInstance(model.backbone, DummyLatentEncoder)
        self.assertEqual(model.backbone.linear.in_features, 8)
        self.assertEqual(model.backbone.linear.out_features, 4)
    
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
    
    def test_latent_size_with_no_backbone(self):
        """Test latent_size equals input_size when no backbone is used."""
        model = BlackBox(input_size=16, annotations=self.ann)
        self.assertEqual(model.latent_size, 16)

    def test_latent_size_with_backbone(self):
        """Test latent_size equals the backbone's output size."""
        model = BlackBox(
            input_size=8,
            annotations=self.ann,
            backbone=MLP(input_size=8, hidden_size=32),
            latent_size=32,
        )
        self.assertEqual(model.latent_size, 32)

    def test_linear_layer_input_matches_latent_size(self):
        """Test that the linear layer input features match latent_size."""
        model = BlackBox(
            input_size=8,
            annotations=self.ann,
            backbone=MLP(input_size=8, hidden_size=16),
            latent_size=16,
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
            backbone=DummyLatentEncoder(8, hidden_size=4),
            latent_size=4,
        )
        defaults.update(kwargs)
        return BlackBox(**defaults)
    
    ALL = ['c1', 'c2', 'task']

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = self._make_model()
        x = torch.randn(2, 8)
        out = model(x)

        # Output size is sum of cardinalities: 1 + 3 + 2 = 6
        expected_output_size = sum(self.ann[1].cardinalities)
        self.assertEqual(_logits(out, self.ALL).shape, (2, expected_output_size))

    def test_forward_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = self._make_model()

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 8)
            out = model(x)
            self.assertEqual(_logits(out, self.ALL).shape[0], batch_size)

    def test_forward_query_filters_output(self):
        """Test that query selects only the queried concept columns."""
        model = self._make_model()

        x = torch.randn(2, 8)
        # Annotations: c1(1), c2(3), task(2) — total 6
        out_all = model(x)
        all_logits = _logits(out_all, self.ALL)
        self.assertEqual(all_logits.shape, (2, 6))

        # Query single binary concept
        out_c1 = model(x, query=['c1'])
        c1_logits = _logits(out_c1, ['c1'])
        self.assertEqual(c1_logits.shape, (2, 1))
        self.assertTrue(torch.allclose(c1_logits, all_logits[:, 0:1]))

        # Query single categorical concept
        out_c2 = model(x, query=['c2'])
        c2_logits = _logits(out_c2, ['c2'])
        self.assertEqual(c2_logits.shape, (2, 3))
        self.assertTrue(torch.allclose(c2_logits, all_logits[:, 1:4]))

        # Query subset
        out_subset = model(x, query=['c1', 'task'])
        subset_logits = _logits(out_subset, ['c1', 'task'])
        self.assertEqual(subset_logits.shape, (2, 3))  # 1 + 2
        self.assertTrue(torch.allclose(subset_logits[:, 0:1], all_logits[:, 0:1]))
        self.assertTrue(torch.allclose(subset_logits[:, 1:3], all_logits[:, 4:6]))

    def test_forward_query_all_same_as_none(self):
        """Test that querying all concepts returns same as query=None."""
        model = self._make_model()
        model.eval()

        x = torch.randn(2, 8)
        out_none = model(x)
        out_all = model(x, query=['c1', 'c2', 'task'])

        none_logits = _logits(out_none, self.ALL)
        all_logits = _logits(out_all, self.ALL)
        self.assertEqual(none_logits.shape, all_logits.shape)
        self.assertTrue(torch.allclose(none_logits, all_logits))

    def test_forward_query_single_concept(self):
        """Test querying a single concept with large cardinality."""
        model = self._make_model()

        x = torch.randn(4, 8)
        out = model(x, query=['task'])
        self.assertEqual(_logits(out, ['task']).shape, (4, 2))

    def test_forward_with_evidence_ignored(self):
        """Test that evidence parameter is accepted but doesn't affect output."""
        model = self._make_model()

        x = torch.randn(2, 8)
        out1 = model(x)
        out2 = model(x, evidence=torch.randn(2, 4))

        self.assertEqual(
            _logits(out1, self.ALL).shape, _logits(out2, self.ALL).shape
        )

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic with same input."""
        model = self._make_model()
        model.eval()

        x = torch.randn(2, 8)
        out1 = model(x)
        out2 = model(x)

        self.assertTrue(
            torch.allclose(_logits(out1, self.ALL), _logits(out2, self.ALL))
        )

    def test_forward_no_backbone(self):
        """Test forward pass with no backbone (identity)."""
        model = BlackBox(input_size=8, annotations=self.ann)
        x = torch.randn(2, 8)
        out = model(x)

        self.assertEqual(_logits(out, self.ALL).shape, (2, 6))

    def test_forward_no_backbone_shape(self):
        """Test forward pass with identity backbone preserves shapes."""
        model = BlackBox(input_size=8, annotations=self.ann)
        x = torch.randn(3, 8)
        out = model(x)

        logits = _logits(out, self.ALL)
        self.assertEqual(logits.shape[0], 3)
        self.assertEqual(logits.shape[1], sum(self.ann[1].cardinalities))

    def test_forward_extra_kwargs_ignored(self):
        """Test that extra kwargs in forward are silently ignored."""
        model = self._make_model()
        x = torch.randn(2, 8)
        # These kwargs should be captured by **kwargs and ignored
        out = model(x, ground_truth=torch.ones(2, 6), concept_names=['a'])
        self.assertEqual(_logits(out, self.ALL).shape, (2, 6))


class TestBlackBoxPrepareTarget(unittest.TestCase):
    """Test BlackBox prepare_target inherited from BaseModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = make_annotations(['c1', 'task'], [1, 1])
        self.model = BlackBox(
            input_size=8,
            annotations=self.ann,
            backbone=DummyLatentEncoder(8, hidden_size=4),
            latent_size=4,
        )

    def test_prepare_target(self):
        """Test prepare_target returns target unchanged for BlackBox."""
        x = torch.randn(2, 8)
        out = self.model(x)
        target = torch.randint(0, 2, _logits(out, ['c1', 'task']).shape)

        prepared = self.model.prepare_target(target)
        self.assertTrue(torch.allclose(prepared, target))

    def test_prepare_target_inherited_from_base(self):
        """Test that prepare_target is inherited from BaseModel (not overridden)."""
        # After cleanup, BlackBox should not have its own prepare_target method
        self.assertNotIn(
            'prepare_target', BlackBox.__dict__,
            "BlackBox should not override prepare_target"
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
            latent_size=8,
        )

        repr_str = repr(model)
        self.assertIsInstance(repr_str, str)
        self.assertIn('BlackBox', repr_str)
        self.assertIn('backbone=DummyBackbone', repr_str)
        self.assertNotIn('latent_encoder=', repr_str)

    def test_repr_without_backbone(self):
        """Test __repr__ when no backbone is used (defaults to Identity)."""
        ann = make_annotations(['output'], [1])
        model = BlackBox(input_size=8, annotations=ann)

        repr_str = repr(model)
        self.assertIn('BlackBox', repr_str)
        self.assertIn('backbone=Identity', repr_str)
        self.assertNotIn('latent_encoder=', repr_str)


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
        
        self.assertEqual(out.logits.shape, (2, 2))
    
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
    
    def test_lightning_no_inference_engine(self):
        """Test that BlackBox with lightning=True does not have inference engines."""
        model = BlackBox(
            lightning=True,
            input_size=8,
            annotations=self.ann,
        )

        self.assertFalse(hasattr(model, 'eval_inference'))
        self.assertFalse(hasattr(model, 'train_inference'))


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
            backbone=backbone,
            latent_size=16,
        )

        self.assertIsInstance(model.backbone, DummyBackbone)

    def test_init_with_backbone_encoder(self):
        """Test initialization with a custom backbone mapping input to latent."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task1',
            backbone=DummyLatentEncoder(8, hidden_size=4),
            latent_size=4,
        )

        self.assertIsInstance(model.backbone, DummyLatentEncoder)
    
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
            backbone=DummyLatentEncoder(8, hidden_size=4),
            latent_size=4,
        )
        defaults.update(kwargs)
        return BlackBoxTaskOnly(**defaults)
    
    def test_forward_shape_single_task(self):
        """Test forward pass output shape with single task."""
        model = self._make_model(task_names='task1')
        x = torch.randn(2, 8)
        out = model(x)

        # Only task1 output (cardinality 1)
        self.assertEqual(_logits(out, ['task1']).shape, (2, 1))

    def test_forward_shape_multiple_tasks(self):
        """Test forward pass output shape with multiple tasks."""
        model = self._make_model(task_names=['task1', 'task2'])
        x = torch.randn(2, 8)
        out = model(x)

        # task1(1) + task2(3) = 4
        self.assertEqual(_logits(out, ['task1', 'task2']).shape, (2, 4))

    def test_forward_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = self._make_model()

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 8)
            out = model(x)
            self.assertEqual(_logits(out, ['task1']).shape[0], batch_size)

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic."""
        model = self._make_model()
        model.eval()

        x = torch.randn(2, 8)
        out1 = model(x)
        out2 = model(x)

        self.assertTrue(
            torch.allclose(_logits(out1, ['task1']), _logits(out2, ['task1']))
        )

    def test_forward_extra_kwargs_ignored(self):
        """Test that extra kwargs in forward are silently ignored."""
        model = self._make_model()
        x = torch.randn(2, 8)
        out = model(x, ground_truth=torch.ones(2, 1), concept_names=['a'])
        self.assertEqual(_logits(out, ['task1']).shape, (2, 1))


class TestBlackBoxTaskOnlyPrepareTarget(unittest.TestCase):
    """Test BlackBoxTaskOnly prepare_target with padding logic."""
    
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
            backbone=DummyLatentEncoder(8, hidden_size=4),
            latent_size=4,
        )

    def test_prepare_target_slices_target(self):
        """Test prepare_target slices target to task columns."""
        # Full concept-level target: c1, c2, task1
        target = torch.tensor([[0., 1., 1.],
                               [1., 0., 0.]])
        
        prepared = self.model.prepare_target(target)
        
        # Target should be sliced to task1 column (concept index 2)
        self.assertEqual(prepared.shape, (2, 1))
        self.assertTrue(torch.allclose(prepared, target[:, 2:3]))
    
    def test_forward_preserves_gradients(self):
        """Test that forward pass preserves gradient flow."""
        x = torch.randn(2, 8)
        out = self.model(x)

        loss = _logits(out, ['task1']).sum()
        loss.backward()
        
        # Check gradients exist on model parameters
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_prepare_target_multiple_tasks_slicing(self):
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
        
        # Full concept-level target: c1, task1, task2
        target = torch.tensor([[0., 1., 2.],
                               [1., 0., 1.],
                               [0., 1., 0.]])
        
        prepared = model.prepare_target(target)
        
        # Target sliced to task1 (idx 1) and task2 (idx 2)
        self.assertEqual(prepared.shape, (3, 2))
        self.assertTrue(torch.allclose(prepared, target[:, [1, 2]]))
    
    def test_prepare_target_overrides_base_model(self):
        """Test that BlackBoxTaskOnly overrides BaseModel prepare_target."""
        self.assertIn('prepare_target', BlackBoxTaskOnly.__dict__)


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
    
    def test_lightning_no_inference_engine(self):
        """Test that BlackBoxTaskOnly with lightning=True does not have inference engines."""
        model = BlackBoxTaskOnly(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names='task',
        )

        self.assertFalse(hasattr(model, 'eval_inference'))
        self.assertFalse(hasattr(model, 'train_inference'))


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


class TestBlackBoxTaskOnlyConceptLossRebuild(unittest.TestCase):
    """Test that BlackBoxTaskOnly rebuilds ConceptLoss with task-only annotations."""

    def setUp(self):
        self.ann = make_annotations(
            ['c1', 'c2', 'task'],
            [1, 2, 1]
        )

    def test_concept_loss_is_rebuilt_for_task_only(self):
        """Test that passing a ConceptLoss triggers the rebuild branch."""
        # All concepts are binary (cardinality 1), so only binary loss needed
        ann = make_annotations(['c1', 'c2', 'task'], [1, 1, 1])
        concept_loss = ConceptLoss(
            annotations=ann,
            binary=nn.BCEWithLogitsLoss(),
        )
        model = BlackBoxTaskOnly(
            lightning=True,
            input_size=8,
            annotations=ann,
            task_names='task',
            loss=concept_loss,
        )
        # The rebuilt loss should use task-only annotations
        self.assertIsInstance(model.loss, ConceptLoss)

    def test_concept_loss_rebuild_with_categorical(self):
        """Test ConceptLoss rebuild when task is categorical."""
        ann = make_annotations(['c1', 'task'], [1, 3])
        concept_loss = ConceptLoss(
            annotations=ann,
            binary=nn.BCEWithLogitsLoss(),
            categorical=nn.CrossEntropyLoss(),
        )
        model = BlackBoxTaskOnly(
            lightning=True,
            input_size=8,
            annotations=ann,
            task_names='task',
            loss=concept_loss,
        )
        self.assertIsInstance(model.loss, ConceptLoss)


class TestBlackBoxTaskOnlySetupMetrics(unittest.TestCase):
    """Test that setup_metrics rebuilds ConceptMetrics with task-only annotations."""

    def setUp(self):
        self.ann = make_annotations(
            ['c1', 'task'],
            [1, 1]
        )

    def test_setup_metrics_rebuilds_with_task_annotations(self):
        """Test setup_metrics reconstructs metrics for task-only outputs."""
        from torchmetrics.classification import BinaryAccuracy
        metrics = ConceptMetrics(
            annotations=self.ann,
            binary={'accuracy': BinaryAccuracy()},
            summary=False,
            per_concept=True,
        )
        model = BlackBoxTaskOnly(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names='task',
            metrics=metrics,
        )
        # After setup_metrics, model should have train/val/test metrics
        self.assertTrue(hasattr(model, 'train_metrics'))
        self.assertTrue(hasattr(model, 'val_metrics'))
        self.assertTrue(hasattr(model, 'test_metrics'))

    def test_setup_metrics_called_during_init(self):
        """Test that setup_metrics is invoked during __init__ when metrics are provided."""
        from torchmetrics.classification import BinaryAccuracy
        metrics = ConceptMetrics(
            annotations=self.ann,
            binary={'accuracy': BinaryAccuracy()},
            summary=True,
            per_concept=False,
        )
        model = BlackBoxTaskOnly(
            lightning=True,
            input_size=8,
            annotations=self.ann,
            task_names='task',
            metrics=metrics,
        )
        self.assertIsNotNone(model.train_metrics)


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

        self.assertEqual(_logits(out, ['c1', 'task']).device.type, device.type)

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

        self.assertEqual(_logits(out, ['task']).device.type, device.type)

    def test_blackbox_task_only_prepare_target_preserves_device(self):
        """Test that prepare_target preserves tensor device."""
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
        
        prepared = model.prepare_target(target)
        
        self.assertEqual(prepared.device.type, device.type)


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

        self.assertEqual(_logits(out, ['output']).shape, (2, 1))

    def test_large_cardinalities(self):
        """Test with large cardinalities."""
        ann = make_annotations(['c1', 'c2'], [100, 50])
        model = BlackBox(input_size=8, annotations=ann)
        x = torch.randn(2, 8)
        out = model(x)

        self.assertEqual(_logits(out, ['c1', 'c2']).shape, (2, 150))

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        ann = make_annotations(['output'], [1])
        model = BlackBox(input_size=8, annotations=ann)
        x = torch.randn(1, 8)
        out = model(x)

        self.assertEqual(_logits(out, ['output']).shape, (1, 1))

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        ann = make_annotations(['output'], [1])
        model = BlackBox(input_size=8, annotations=ann)
        x = torch.randn(2, 8, requires_grad=True)
        out = model(x)
        loss = _logits(out, ['output']).sum()
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

        self.assertEqual(_logits(out, labels).shape, (2, 20))

    def test_large_input_size(self):
        """Test with large input size."""
        ann = make_annotations(['c1'], [1])
        model = BlackBox(input_size=512, annotations=ann)
        x = torch.randn(2, 512)
        out = model(x)

        self.assertEqual(_logits(out, ['c1']).shape, (2, 1))


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
        loss = _logits(out, ['task']).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))

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

        self.assertEqual(_logits(out, ['task']).shape, (2, 50))

        # Full target has 2 concept-level columns (c1, task)
        target = torch.zeros(2, 2)
        prepared = model.prepare_target(target)
        # Target sliced to task column only
        self.assertEqual(prepared.shape, (2, 1))

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
        self.assertEqual(_logits(out, ['task']).shape, (2, 1))

        # When task is the only concept, prepare_target is identity
        target = torch.zeros(2, 1)
        prepared = model.prepare_target(target)
        self.assertTrue(torch.allclose(prepared, target))

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
        self.assertEqual(_logits(out, ['task']).shape, (1, 1))
        
        # Full target has 2 concept-level columns
        target = torch.zeros(1, 2)
        prepared = model.prepare_target(target)
        self.assertEqual(prepared.shape, (1, 1))


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
        prepared = model.prepare_target(target)
        loss = nn.functional.binary_cross_entropy_with_logits(
            _logits(out, ['c1', 'task']),
            prepared
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

        self.assertFalse(_logits(out, ['c1', 'task']).requires_grad)
    
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
        prepared = model.prepare_target(target)
        loss = nn.functional.binary_cross_entropy_with_logits(
            _logits(out, ['task']),
            prepared
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
        loss = nn.functional.mse_loss(_logits(out, ['c1']), target)
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
            backbone=backbone,
            latent_size=8,
        )

        # Input size doesn't matter since DummyBackbone produces fixed size
        x = torch.randn(2, 100)
        out = model(x)

        self.assertEqual(_logits(out, ['c1']).shape, (2, 1))

    def test_task_only_with_backbone(self):
        """Test BlackBoxTaskOnly with backbone."""
        ann = make_annotations(['c1', 'task'], [1, 1])
        backbone = DummyBackbone(out_features=8)

        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task',
            backbone=backbone,
            latent_size=8,
        )

        x = torch.randn(2, 50)
        out = model(x)

        self.assertEqual(_logits(out, ['task']).shape, (2, 1))

    def test_no_backbone_uses_identity(self):
        """Test that no backbone means identity transform."""
        ann = make_annotations(['c1'], [1])
        model = BlackBox(input_size=8, annotations=ann)

        self.assertIsInstance(model.backbone, nn.Identity)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
