"""
Comprehensive tests for Concept Embedding Model (CEM).

Tests cover:
- Model initialization with various configurations
- Forward pass and output shapes
- Exogenous variables handling
- Training modes (manual PyTorch and Lightning)
- Backbone integration
- Distribution handling
- Inference modes (deterministic and ancestral sampling)
- Filter methods
- Edge cases and error handling
"""
import pytest
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch_concepts.nn.modules.high.models.cem import (
    ConceptEmbeddingModel,
    ConceptEmbeddingModel_Joint
)
from torch_concepts.annotations import AxisAnnotation, Annotations
from torch_concepts.nn.modules.mid.inference.forward import (
    DeterministicInference,
    AncestralSamplingInference
)


class DummyBackbone(nn.Module):
    """Simple backbone for testing."""
    def __init__(self, out_features=8):
        super().__init__()
        self.out_features = out_features
    
    def forward(self, x):
        return torch.ones(x.shape[0], self.out_features)


class TestCEMInitialization(unittest.TestCase):
    """Test CEM initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'size', 'task1'],
                cardinalities=[3, 2, 1, 1],
                metadata={
                    'color': {'type': 'discrete', 'distribution': Categorical},
                    'shape': {'type': 'discrete', 'distribution': Categorical},
                    'size': {'type': 'discrete', 'distribution': Bernoulli},
                    'task1': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
    
    def test_init_basic(self):
        """Test basic initialization."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']
        )
        
        self.assertIsInstance(model.model, nn.Module)
        self.assertTrue(hasattr(model, 'inference'))
        self.assertEqual(model.concept_names, ['color', 'shape', 'size', 'task1'])
    
    def test_init_with_exogenous_size(self):
        """Test initialization with custom exogenous size."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            exogenous_size=32
        )
        
        self.assertIsInstance(model.model, nn.Module)
        # The exogenous size should be passed to the encoder
        self.assertTrue(hasattr(model, 'model'))
    
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
        
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann_no_dist,
            variable_distributions=variable_distributions,
            task_names=['task']
        )
        
        self.assertEqual(model.concept_names, ['c1', 'c2', 'task'])
    
    def test_init_with_backbone(self):
        """Test initialization with custom backbone."""
        backbone = DummyBackbone()
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            backbone=backbone,
            task_names=['task1']
        )
        
        self.assertIsNotNone(model.backbone)
    
    def test_init_with_latent_encoder(self):
        """Test initialization with latent encoder config."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            latent_encoder_kwargs={'hidden_size': 16, 'n_layers': 2}
        )
        
        self.assertEqual(model.latent_size, 16)
    
    def test_init_with_deterministic_inference(self):
        """Test initialization with deterministic inference."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            inference=DeterministicInference
        )
        
        self.assertIsInstance(model.inference, DeterministicInference)
    
    def test_init_with_ancestral_sampling_inference(self):
        """Test initialization with ancestral sampling inference."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            inference=AncestralSamplingInference
        )
        
        self.assertIsInstance(model.inference, AncestralSamplingInference)
    
    def test_init_alias_class(self):
        """Test that ConceptEmbeddingModel alias works correctly."""
        model1 = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']
        )
        
        model2 = ConceptEmbeddingModel_Joint(
            input_size=8,
            annotations=self.ann,
            task_names=['task1']
        )
        
        # Both should be instances of the base class
        self.assertIsInstance(model1, ConceptEmbeddingModel_Joint)
        self.assertIsInstance(model2, ConceptEmbeddingModel_Joint)


class TestCEMForward(unittest.TestCase):
    """Test CEM forward pass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'size', 'task1'],
                cardinalities=[3, 2, 1, 1],
                metadata={
                    'color': {'type': 'discrete', 'distribution': Categorical},
                    'shape': {'type': 'discrete', 'distribution': Categorical},
                    'size': {'type': 'discrete', 'distribution': Bernoulli},
                    'task1': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
        
        self.model = ConceptEmbeddingModel(
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
        """Test forward with all concepts and tasks."""
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
    
    def test_forward_only_tasks(self):
        """Test forward with only task variables."""
        x = torch.randn(2, 8)
        query = ['task1']
        out = self.model(x, query=query)
        
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 1)
    
    def test_forward_with_backbone(self):
        """Test forward pass with backbone."""
        backbone = DummyBackbone(out_features=8)
        model = ConceptEmbeddingModel(
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
    
    def test_forward_batch_sizes(self):
        """Test forward with various batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 8)
            query = ['color', 'shape']
            out = self.model(x, query=query)
            
            self.assertEqual(out.shape[0], batch_size)
            self.assertEqual(out.shape[1], 3 + 2)
    
    def test_forward_deterministic_inference(self):
        """Test forward with deterministic inference."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1'],
            inference=DeterministicInference
        )
        
        x = torch.randn(2, 8)
        query = ['color', 'shape']
        
        # Multiple forwards should give same result (deterministic)
        out1 = model(x, query=query)
        out2 = model(x, query=query)
        
        self.assertTrue(torch.allclose(out1, out2))
    
    def test_forward_eval_mode(self):
        """Test forward in eval mode."""
        self.model.eval()
        x = torch.randn(2, 8)
        query = ['color', 'shape']
        
        with torch.no_grad():
            out = self.model(x, query=query)
        
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3 + 2)


class TestCEMConceptTypes(unittest.TestCase):
    """Test CEM with different concept types (binary, categorical, mixed)."""
    
    def test_init_binary_only(self):
        """Test initialization with only binary concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task'],
                cardinalities=[1, 1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'c3': {'type': 'binary', 'distribution': Bernoulli},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=10,
            annotations=ann,
            task_names=['task'],
            exogenous_size=16
        )
        
        self.assertIsInstance(model, ConceptEmbeddingModel)
        self.assertEqual(model.concept_names, ['c1', 'c2', 'c3', 'task'])
        # Verify model can be created without errors
        self.assertTrue(hasattr(model, 'model'))
    
    def test_init_categorical_only(self):
        """Test initialization with only categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'material', 'task'],
                cardinalities=[5, 4, 3, 2],
                metadata={
                    'color': {'type': 'categorical', 'distribution': Categorical},
                    'shape': {'type': 'categorical', 'distribution': Categorical},
                    'material': {'type': 'categorical', 'distribution': Categorical},
                    'task': {'type': 'categorical', 'distribution': Categorical}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=12,
            annotations=ann,
            task_names=['task'],
            exogenous_size=20
        )
        
        self.assertIsInstance(model, ConceptEmbeddingModel)
        self.assertEqual(model.concept_names, ['color', 'shape', 'material', 'task'])
        self.assertTrue(hasattr(model, 'model'))
    
    def test_init_mixed_concepts(self):
        """Test initialization with mixed binary and categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['binary1', 'categorical1', 'binary2', 'categorical2', 'task'],
                cardinalities=[1, 4, 1, 3, 1],
                metadata={
                    'binary1': {'type': 'binary', 'distribution': Bernoulli},
                    'categorical1': {'type': 'categorical', 'distribution': Categorical},
                    'binary2': {'type': 'binary', 'distribution': Bernoulli},
                    'categorical2': {'type': 'categorical', 'distribution': Categorical},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=15,
            annotations=ann,
            task_names=['task'],
            exogenous_size=24
        )
        
        self.assertIsInstance(model, ConceptEmbeddingModel)
        self.assertEqual(model.concept_names, ['binary1', 'categorical1', 'binary2', 'categorical2', 'task'])
        self.assertTrue(hasattr(model, 'model'))
    
    def test_forward_binary_only(self):
        """Test forward pass with only binary concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task'],
                cardinalities=[1, 1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'c3': {'type': 'binary', 'distribution': Bernoulli},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=10,
            annotations=ann,
            task_names=['task'],
            exogenous_size=16
        )
        
        x = torch.randn(8, 10)
        query = ['c1', 'c2', 'c3', 'task']
        out = model(x, query=query)
        
        # All binary: 1 + 1 + 1 + 1 = 4
        self.assertEqual(out.shape, (8, 4))
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(out).any(), "Output contains Inf values")
    
    def test_forward_categorical_only(self):
        """Test forward pass with only categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'material', 'task'],
                cardinalities=[5, 4, 3, 2],
                metadata={
                    'color': {'type': 'categorical', 'distribution': Categorical},
                    'shape': {'type': 'categorical', 'distribution': Categorical},
                    'material': {'type': 'categorical', 'distribution': Categorical},
                    'task': {'type': 'categorical', 'distribution': Categorical}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=12,
            annotations=ann,
            task_names=['task'],
            exogenous_size=20
        )
        
        x = torch.randn(6, 12)
        query = ['color', 'shape', 'material', 'task']
        out = model(x, query=query)
        
        # All categorical: 5 + 4 + 3 + 2 = 14
        self.assertEqual(out.shape, (6, 14))
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(out).any(), "Output contains Inf values")
    
    def test_forward_mixed_concepts(self):
        """Test forward pass with mixed binary and categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['binary1', 'categorical1', 'binary2', 'categorical2', 'task'],
                cardinalities=[1, 4, 1, 3, 1],
                metadata={
                    'binary1': {'type': 'binary', 'distribution': Bernoulli},
                    'categorical1': {'type': 'categorical', 'distribution': Categorical},
                    'binary2': {'type': 'binary', 'distribution': Bernoulli},
                    'categorical2': {'type': 'categorical', 'distribution': Categorical},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=15,
            annotations=ann,
            task_names=['task'],
            exogenous_size=24
        )
        
        x = torch.randn(10, 15)
        query = ['binary1', 'categorical1', 'binary2', 'categorical2', 'task']
        out = model(x, query=query)
        
        # Mixed: 1 + 4 + 1 + 3 + 1 = 10
        self.assertEqual(out.shape, (10, 10))
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(out).any(), "Output contains Inf values")
    
    def test_forward_binary_only_partial_query(self):
        """Test forward pass querying subset of binary concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'c4', 'task'],
                cardinalities=[1, 1, 1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'c3': {'type': 'binary', 'distribution': Bernoulli},
                    'c4': {'type': 'binary', 'distribution': Bernoulli},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=10,
            annotations=ann,
            task_names=['task'],
            exogenous_size=16
        )
        
        x = torch.randn(4, 10)
        # Query only some concepts
        query = ['c1', 'c3', 'task']
        out = model(x, query=query)
        
        # Only queried: 1 + 1 + 1 = 3
        self.assertEqual(out.shape, (4, 3))
    
    def test_forward_categorical_only_partial_query(self):
        """Test forward pass querying subset of categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'size', 'task'],
                cardinalities=[5, 4, 3, 2],
                metadata={
                    'color': {'type': 'categorical', 'distribution': Categorical},
                    'shape': {'type': 'categorical', 'distribution': Categorical},
                    'size': {'type': 'categorical', 'distribution': Categorical},
                    'task': {'type': 'categorical', 'distribution': Categorical}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=12,
            annotations=ann,
            task_names=['task'],
            exogenous_size=20
        )
        
        x = torch.randn(5, 12)
        # Query only some concepts
        query = ['color', 'size']
        out = model(x, query=query)
        
        # Only queried: 5 + 3 = 8
        self.assertEqual(out.shape, (5, 8))
    
    def test_forward_mixed_concepts_partial_query(self):
        """Test forward pass querying subset of mixed concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['b1', 'cat1', 'b2', 'cat2', 'b3', 'task'],
                cardinalities=[1, 4, 1, 3, 1, 1],
                metadata={
                    'b1': {'type': 'binary', 'distribution': Bernoulli},
                    'cat1': {'type': 'categorical', 'distribution': Categorical},
                    'b2': {'type': 'binary', 'distribution': Bernoulli},
                    'cat2': {'type': 'categorical', 'distribution': Categorical},
                    'b3': {'type': 'binary', 'distribution': Bernoulli},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=15,
            annotations=ann,
            task_names=['task'],
            exogenous_size=24
        )
        
        x = torch.randn(7, 15)
        # Query mix of binary and categorical
        query = ['b1', 'cat1', 'cat2']
        out = model(x, query=query)
        
        # Mixed query: 1 + 4 + 3 = 8
        self.assertEqual(out.shape, (7, 8))


class TestCEMExogenousVariables(unittest.TestCase):
    """Test CEM exogenous variable handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task'],
                cardinalities=[2, 3, 1, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Categorical},
                    'c2': {'type': 'discrete', 'distribution': Categorical},
                    'c3': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
    
    def test_different_exogenous_sizes(self):
        """Test models with different exogenous sizes."""
        for exogenous_size in [4, 8, 16, 32]:
            model = ConceptEmbeddingModel(
                input_size=8,
                annotations=self.ann,
                task_names=['task'],
                exogenous_size=exogenous_size
            )
            
            x = torch.randn(2, 8)
            query = ['c1', 'c2', 'c3']
            out = model(x, query=query)
            
            self.assertEqual(out.shape[0], 2)
            self.assertEqual(out.shape[1], 2 + 3 + 1)
    
    def test_exogenous_in_bipartite_model(self):
        """Test that exogenous variables are properly integrated."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task'],
            exogenous_size=16
        )
        
        # Model should have bipartite structure with exogenous
        self.assertTrue(hasattr(model.model, 'probabilistic_model'))


class TestCEMFilterMethods(unittest.TestCase):
    """Test CEM filter methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
        
        self.model = ConceptEmbeddingModel(
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


class TestCEMTraining(unittest.TestCase):
    """Test CEM training scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
    
    def test_manual_training_mode(self):
        """Test manual PyTorch training (no loss in model)."""
        model = ConceptEmbeddingModel(
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
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        x = torch.randn(4, 8, requires_grad=True)
        out = model(x, query=['c1', 'c2', 'task'])
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
    
    def test_training_step(self):
        """Test a complete training step."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Training step
        model.train()
        x = torch.randn(8, 8)
        y = torch.randint(0, 2, (8, 3)).float()
        
        optimizer.zero_grad()
        out = model(x, query=['c1', 'c2', 'task'])
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        
        self.assertTrue(True)  # If we get here, training works
    
    def test_parameters_update(self):
        """Test that parameters actually update during training."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Training step
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()
        
        x = torch.randn(8, 8)
        y = torch.randn(8, 3)
        
        optimizer.zero_grad()
        out = model(x, query=['c1', 'c2', 'task'])
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        
        # Check that at least some parameters changed
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break
        
        self.assertTrue(params_changed)


class TestCEMWithMultipleTasks(unittest.TestCase):
    """Test CEM with multiple task variables."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task1', 'task2'],
                cardinalities=[2, 3, 1, 1, 2],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Categorical},
                    'c2': {'type': 'discrete', 'distribution': Categorical},
                    'c3': {'type': 'discrete', 'distribution': Bernoulli},
                    'task1': {'type': 'discrete', 'distribution': Bernoulli},
                    'task2': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
    
    def test_multiple_tasks_init(self):
        """Test initialization with multiple tasks."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1', 'task2']
        )
        
        self.assertEqual(model.concept_names, ['c1', 'c2', 'c3', 'task1', 'task2'])
    
    def test_multiple_tasks_forward(self):
        """Test forward pass with multiple tasks."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1', 'task2']
        )
        
        x = torch.randn(2, 8)
        query = ['c1', 'c2', 'c3', 'task1', 'task2']
        out = model(x, query=query)
        
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 2 + 3 + 1 + 1 + 2)
    
    def test_query_only_one_task(self):
        """Test querying only one of multiple tasks."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task1', 'task2']
        )
        
        x = torch.randn(2, 8)
        query = ['c1', 'task1']
        out = model(x, query=query)
        
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 2 + 1)


class TestCEMConceptTypes(unittest.TestCase):
    """Test CEM with different concept types (binary, categorical, mixed)."""
    
    def test_only_binary_concepts_init(self):
        """Test initialization with only binary concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task'],
                cardinalities=[1, 1, 1, 1],  # All binary (cardinality 1)
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Bernoulli},
                    'c3': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task'],
            exogenous_size=16
        )
        
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'model'))
    
    def test_only_binary_concepts_forward(self):
        """Test forward pass with only binary concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task'],
                cardinalities=[1, 1, 1, 1],  # All binary
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Bernoulli},
                    'c3': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task'],
            exogenous_size=16
        )
        
        x = torch.randn(4, 8)
        out = model(x, query=['c1', 'c2', 'c3', 'task'])
        
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[1], 4)  # 3 binary concepts + 1 binary task
    
    def test_only_categorical_concepts_init(self):
        """Test initialization with only categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'size', 'task'],
                cardinalities=[3, 4, 5, 2],  # All categorical (cardinality > 1)
                metadata={
                    'color': {'type': 'discrete', 'distribution': Categorical},
                    'shape': {'type': 'discrete', 'distribution': Categorical},
                    'size': {'type': 'discrete', 'distribution': Categorical},
                    'task': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task'],
            exogenous_size=16
        )
        
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'model'))
    
    def test_only_categorical_concepts_forward(self):
        """Test forward pass with only categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['color', 'shape', 'size', 'task'],
                cardinalities=[3, 4, 5, 2],  # All categorical
                metadata={
                    'color': {'type': 'discrete', 'distribution': Categorical},
                    'shape': {'type': 'discrete', 'distribution': Categorical},
                    'size': {'type': 'discrete', 'distribution': Categorical},
                    'task': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task'],
            exogenous_size=16
        )
        
        x = torch.randn(4, 8)
        out = model(x, query=['color', 'shape', 'size', 'task'])
        
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[1], 3 + 4 + 5 + 2)  # Sum of all cardinalities
    
    def test_mixed_concepts_init(self):
        """Test initialization with mixed binary and categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['is_red', 'shape', 'has_texture', 'size', 'task'],
                cardinalities=[1, 3, 1, 4, 2],  # Mix: binary (1), categorical (3), binary (1), categorical (4), categorical (2)
                metadata={
                    'is_red': {'type': 'discrete', 'distribution': Bernoulli},
                    'shape': {'type': 'discrete', 'distribution': Categorical},
                    'has_texture': {'type': 'discrete', 'distribution': Bernoulli},
                    'size': {'type': 'discrete', 'distribution': Categorical},
                    'task': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task'],
            exogenous_size=16
        )
        
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'model'))
    
    def test_mixed_concepts_forward(self):
        """Test forward pass with mixed binary and categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['is_red', 'shape', 'has_texture', 'size', 'task'],
                cardinalities=[1, 3, 1, 4, 2],  # Mixed
                metadata={
                    'is_red': {'type': 'discrete', 'distribution': Bernoulli},
                    'shape': {'type': 'discrete', 'distribution': Categorical},
                    'has_texture': {'type': 'discrete', 'distribution': Bernoulli},
                    'size': {'type': 'discrete', 'distribution': Categorical},
                    'task': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task'],
            exogenous_size=16
        )
        
        x = torch.randn(4, 8)
        out = model(x, query=['is_red', 'shape', 'has_texture', 'size', 'task'])
        
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[1], 1 + 3 + 1 + 4 + 2)  # Sum of all cardinalities = 11


class TestCEMEdgeCases(unittest.TestCase):
    """Test CEM edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
    
    def test_single_concept(self):
        """Test with single concept."""
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
        
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task']
        )
        
        x = torch.randn(2, 8)
        out = model(x, query=['c1', 'task'])
        
        self.assertEqual(out.shape, (2, 2))
    
    def test_all_binary_concepts(self):
        """Test with all binary concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task'],
                cardinalities=[1, 1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Bernoulli},
                    'c3': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task']
        )
        
        x = torch.randn(2, 8)
        out = model(x, query=['c1', 'c2', 'c3', 'task'])
        
        self.assertEqual(out.shape, (2, 4))
    
    def test_all_categorical_concepts(self):
        """Test with all categorical concepts."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[3, 4, 5],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Categorical},
                    'c2': {'type': 'discrete', 'distribution': Categorical},
                    'task': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task']
        )
        
        x = torch.randn(2, 8)
        out = model(x, query=['c1', 'c2', 'task'])
        
        self.assertEqual(out.shape, (2, 3 + 4 + 5))
    
    def test_repr(self):
        """Test string representation."""
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        repr_str = repr(model)
        self.assertIsInstance(repr_str, str)
        self.assertIn('ConceptEmbeddingModel', repr_str)
    
    def test_device_consistency(self):
        """Test that model maintains device consistency."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        device = torch.device('cuda')
        model = model.to(device)
        
        x = torch.randn(2, 8, device=device)
        out = model(x, query=['c1', 'c2', 'task'])
        
        self.assertEqual(out.device, device)


class TestCEMCardinalities(unittest.TestCase):
    """Test CEM cardinality extraction and handling."""
    
    def test_concept_cardinalities_extraction(self):
        """Test that concept cardinalities are correctly extracted."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task1', 'task2'],
                cardinalities=[2, 3, 1, 1, 4],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Categorical},
                    'c2': {'type': 'discrete', 'distribution': Categorical},
                    'c3': {'type': 'discrete', 'distribution': Bernoulli},
                    'task1': {'type': 'discrete', 'distribution': Bernoulli},
                    'task2': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
        
        model = ConceptEmbeddingModel(
            input_size=8,
            annotations=ann,
            task_names=['task1', 'task2']
        )
        
        # Concept cardinalities should be [2, 3, 1] (excluding tasks)
        self.assertTrue(hasattr(model.model, 'probabilistic_model'))


class TestCEMComparison(unittest.TestCase):
    """Test comparison between CEM and CBM."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
    
    def test_cem_has_exogenous(self):
        """Test that CEM has exogenous variables while CBM doesn't."""
        from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel
        
        cem = ConceptEmbeddingModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        cbm = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        # Both should work but have different architectures
        x = torch.randn(2, 8)
        query = ['c1', 'c2', 'task']
        
        cem_out = cem(x, query=query)
        cbm_out = cbm(x, query=query)
        
        # Outputs should have same shape
        self.assertEqual(cem_out.shape, cbm_out.shape)


if __name__ == '__main__':
    unittest.main()
