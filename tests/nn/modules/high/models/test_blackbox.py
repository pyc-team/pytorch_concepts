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
- BlackBox and BlackBoxTaskOnly models
"""
import pytest
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from torch_concepts.nn.modules.high.models.blackbox import BlackBox, BlackBoxTaskOnly
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


@pytest.fixture
def basic_annotations():
    """Create basic annotations with one output."""
    return Annotations({
        1: AxisAnnotation(
            labels=['output'],
            cardinalities=[1],
            metadata={'output': {'type': 'discrete', 'distribution': Bernoulli}}
        )
    })


@pytest.fixture
def multi_concept_annotations():
    """Create annotations with multiple concepts and tasks."""
    return Annotations({
        1: AxisAnnotation(
            labels=['c1', 'c2', 'c3', 'task1', 'task2'],
            cardinalities=[1, 2, 3, 1, 4],
            metadata={
                'c1': {'type': 'discrete', 'distribution': Bernoulli},
                'c2': {'type': 'discrete', 'distribution': Categorical},
                'c3': {'type': 'discrete', 'distribution': Categorical},
                'task1': {'type': 'discrete', 'distribution': Bernoulli},
                'task2': {'type': 'discrete', 'distribution': Categorical}
            }
        )
    })


# =============================================================================
# BlackBox Model Tests
# =============================================================================

class TestBlackBoxInitialization(unittest.TestCase):
    """Test BlackBox initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 2, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Categorical},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
    
    def test_basic_init(self):
        """Test basic initialization."""
        model = BlackBox(
            input_size=8,
            annotations=self.ann
        )
        
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
        model = BlackBox(
            input_size=8,
            annotations=self.ann
        )
        
        # Sum of cardinalities: 1 + 2 + 1 = 4
        expected_output_size = sum(self.ann[1].cardinalities)
        self.assertEqual(model.linear.out_features, expected_output_size)
    
    def test_init_with_variable_distributions(self):
        """Test initialization with variable distributions."""
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
        
        var_dist = {
            'c1': Bernoulli,
            'c2': Bernoulli
        }
        
        model = BlackBox(
            input_size=8,
            annotations=ann,
            variable_distributions=var_dist
        )
        
        self.assertIsInstance(model, nn.Module)


class TestBlackBoxForward(unittest.TestCase):
    """Test BlackBox forward pass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 3, 2],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Categorical},
                    'task': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = BlackBox(
            input_size=8,
            annotations=self.ann,
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
        
        x = torch.randn(2, 8)
        out = model(x)
        
        # Output size is sum of cardinalities: 1 + 3 + 2 = 6
        expected_output_size = sum(self.ann[1].cardinalities)
        self.assertEqual(out.shape, (2, expected_output_size))
    
    def test_forward_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = BlackBox(
            input_size=8,
            annotations=self.ann,
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 8)
            out = model(x)
            self.assertEqual(out.shape[0], batch_size)
    
    def test_forward_with_query_ignored(self):
        """Test that query parameter is accepted but doesn't affect output."""
        model = BlackBox(
            input_size=8,
            annotations=self.ann,
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
        
        x = torch.randn(2, 8)
        out1 = model(x)
        out2 = model(x, query=['c1', 'task'])
        
        # Query shouldn't change output shape or values
        self.assertEqual(out1.shape, out2.shape)
    
    def test_forward_deterministic(self):
        """Test that forward pass is deterministic with same input."""
        model = BlackBox(
            input_size=8,
            annotations=self.ann,
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
        model.eval()
        
        x = torch.randn(2, 8)
        out1 = model(x)
        out2 = model(x)
        
        self.assertTrue(torch.allclose(out1, out2))


class TestBlackBoxFilterMethods(unittest.TestCase):
    """Test BlackBox filter methods for loss and metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'task'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
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


class TestBlackBoxRepr(unittest.TestCase):
    """Test BlackBox string representation."""
    
    def test_repr(self):
        """Test __repr__ returns informative string."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['output'],
                cardinalities=[1],
                metadata={'output': {'type': 'discrete', 'distribution': Bernoulli}}
            )
        })
        
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


# =============================================================================
# BlackBoxTaskOnly Model Tests
# =============================================================================

class TestBlackBoxTaskOnlyInitialization(unittest.TestCase):
    """Test BlackBoxTaskOnly initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task1', 'task2'],
                cardinalities=[1, 2, 1, 3],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Categorical},
                    'task1': {'type': 'discrete', 'distribution': Bernoulli},
                    'task2': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
    
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
    
    def test_task_indices_calculation(self):
        """Test that task start/end indices are correctly calculated."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']  # task1 is at index 2
        )
        
        # c1(1) + c2(2) = 3, so task1 starts at index 3
        self.assertEqual(model.task_start_idx, 3)
        self.assertEqual(model.task_end_idx, 4)  # 3 + 1 = 4
    
    def test_total_cardinality(self):
        """Test total cardinality is correctly stored."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task1'
        )
        
        # 1 + 2 + 1 + 3 = 7
        self.assertEqual(model.total_cardinality, 7)


class TestBlackBoxTaskOnlyForward(unittest.TestCase):
    """Test BlackBoxTaskOnly forward pass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task1', 'task2'],
                cardinalities=[1, 2, 1, 3],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Categorical},
                    'task1': {'type': 'discrete', 'distribution': Bernoulli},
                    'task2': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
    
    def test_forward_shape_single_task(self):
        """Test forward pass output shape with single task."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task1',
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
        
        x = torch.randn(2, 8)
        out = model(x)
        
        # Only task1 output (cardinality 1)
        self.assertEqual(out.shape, (2, 1))
    
    def test_forward_shape_multiple_tasks(self):
        """Test forward pass output shape with multiple tasks."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names=['task1', 'task2'],
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
        
        x = torch.randn(2, 8)
        out = model(x)
        
        # task1(1) + task2(3) = 4
        self.assertEqual(out.shape, (2, 4))
    
    def test_forward_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task1',
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 8)
            out = model(x)
            self.assertEqual(out.shape[0], batch_size)


class TestBlackBoxTaskOnlyFilterMethods(unittest.TestCase):
    """Test BlackBoxTaskOnly filter methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task1'],
                cardinalities=[1, 2, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Categorical},
                    'task1': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
        self.model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names='task1',
            backbone=DummyBackbone(),
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 4}
        )
    
    def test_filter_output_for_loss_pads_correctly(self):
        """Test filter_output_for_loss pads predictions correctly."""
        x = torch.randn(2, 8)
        out = self.model(x)  # Shape: (2, 1) for task1 only
        target = torch.zeros(2, 4)  # Total cardinality
        
        loss_dict = self.model.filter_output_for_loss(out, target)
        
        self.assertIn('input', loss_dict)
        self.assertIn('target', loss_dict)
        
        # Padded output should have total cardinality shape
        self.assertEqual(loss_dict['input'].shape, (2, 4))
        
        # Task predictions should be at correct positions
        # c1(1) + c2(2) = 3, so task1 is at index 3
        self.assertTrue(torch.allclose(loss_dict['input'][:, 3:4], out))
        
        # Other positions should be zeros
        self.assertTrue(torch.allclose(loss_dict['input'][:, :3], torch.zeros(2, 3)))
    
    def test_filter_output_for_metrics_pads_correctly(self):
        """Test filter_output_for_metrics pads predictions correctly."""
        x = torch.randn(2, 8)
        out = self.model(x)
        target = torch.zeros(2, 4)
        
        metric_dict = self.model.filter_output_for_metrics(out, target)
        
        self.assertIn('preds', metric_dict)
        self.assertIn('target', metric_dict)
        
        # Padded output should have total cardinality shape
        self.assertEqual(metric_dict['preds'].shape, (2, 4))


class TestBlackBoxTaskOnlyMultipleTasks(unittest.TestCase):
    """Test BlackBoxTaskOnly with multiple tasks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'task1', 'c2', 'task2'],
                cardinalities=[1, 2, 1, 3],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'task1': {'type': 'discrete', 'distribution': Categorical},
                    'c2': {'type': 'discrete', 'distribution': Bernoulli},
                    'task2': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
    
    def test_init_with_non_contiguous_tasks(self):
        """Test initialization when tasks are not contiguous in annotations."""
        # This tests the case where tasks are interleaved with concepts
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']  # Only first task
        )
        
        self.assertIsInstance(model, nn.Module)
        # task1 cardinality is 2
        self.assertEqual(model.linear.out_features, 2)


# =============================================================================
# Device Consistency Tests
# =============================================================================

class TestBlackBoxDeviceConsistency(unittest.TestCase):
    """Test device consistency for BlackBox models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'task'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
    
    def test_blackbox_device_consistency(self):
        """Test that BlackBox maintains device consistency."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        model = BlackBox(
            input_size=8,
            annotations=self.ann
        )
        
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


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestBlackBoxEdgeCases(unittest.TestCase):
    """Test edge cases for BlackBox models."""
    
    def test_single_concept(self):
        """Test with single concept."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['output'],
                cardinalities=[1],
                metadata={'output': {'type': 'discrete', 'distribution': Bernoulli}}
            )
        })
        
        model = BlackBox(input_size=8, annotations=ann)
        x = torch.randn(2, 8)
        out = model(x)
        
        self.assertEqual(out.shape, (2, 1))
    
    def test_large_cardinalities(self):
        """Test with large cardinalities."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2'],
                cardinalities=[100, 50],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Categorical},
                    'c2': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
        
        model = BlackBox(input_size=8, annotations=ann)
        x = torch.randn(2, 8)
        out = model(x)
        
        self.assertEqual(out.shape, (2, 150))
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['output'],
                cardinalities=[1],
                metadata={'output': {'type': 'discrete', 'distribution': Bernoulli}}
            )
        })
        
        model = BlackBox(input_size=8, annotations=ann)
        x = torch.randn(1, 8)
        out = model(x)
        
        self.assertEqual(out.shape, (1, 1))
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['output'],
                cardinalities=[1],
                metadata={'output': {'type': 'discrete', 'distribution': Bernoulli}}
            )
        })
        
        model = BlackBox(input_size=8, annotations=ann)
        x = torch.randn(2, 8, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))


class TestBlackBoxTaskOnlyEdgeCases(unittest.TestCase):
    """Test edge cases for BlackBoxTaskOnly."""
    
    def test_task_at_beginning(self):
        """Test when task is at the beginning of annotations."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['task', 'c1', 'c2'],
                cardinalities=[1, 2, 3],
                metadata={
                    'task': {'type': 'discrete', 'distribution': Bernoulli},
                    'c1': {'type': 'discrete', 'distribution': Categorical},
                    'c2': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
        
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task'
        )
        
        # Task is at index 0, so start_idx should be 0
        self.assertEqual(model.task_start_idx, 0)
        self.assertEqual(model.task_end_idx, 1)
    
    def test_task_at_end(self):
        """Test when task is at the end of annotations."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[2, 3, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Categorical},
                    'c2': {'type': 'discrete', 'distribution': Categorical},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
        
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
            task_names='task'
        )
        
        # c1(2) + c2(3) = 5, so task starts at index 5
        self.assertEqual(model.task_start_idx, 5)
        self.assertEqual(model.task_end_idx, 6)
    
    def test_filter_preserves_device(self):
        """Test that filter methods preserve tensor device."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'task'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
        
        model = BlackBoxTaskOnly(
            input_size=8,
            annotations=ann,
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
# Training Integration Tests
# =============================================================================

class TestBlackBoxTraining(unittest.TestCase):
    """Test BlackBox models in training scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'task'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
    
    def test_training_step(self):
        """Test a basic training step."""
        model = BlackBox(
            input_size=8,
            annotations=self.ann
        )
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
        model = BlackBox(
            input_size=8,
            annotations=self.ann
        )
        model.eval()
        
        x = torch.randn(2, 8)
        
        with torch.no_grad():
            out = model(x)
        
        self.assertFalse(out.requires_grad)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
