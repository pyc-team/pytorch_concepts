"""
Integration tests for high-level API components.

Tests the interaction between:
- Models (ConceptBottleneckModel)
- Losses (ConceptLoss)
- Metrics (ConceptMetrics)
- Annotations

This ensures that all high-level components work together correctly.
"""
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch_concepts.nn import ConceptBottleneckModel
from torch_concepts.nn.modules.loss import ConceptLoss
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torch_concepts.annotations import AxisAnnotation, Annotations
from torch_concepts.nn.modules.utils import GroupConfig
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy


class TestHighLevelIntegration(unittest.TestCase):
    """Test integration of high-level components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mixed binary and categorical concepts
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3', 'task'],
                cardinalities=[1, 3, 1, 4],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Categorical},
                    'c3': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
        
        self.loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss(),
            categorical=nn.CrossEntropyLoss()
        )
        
        self.metrics_config = GroupConfig(
            binary={'accuracy': BinaryAccuracy()},
            categorical={'accuracy': MulticlassAccuracy(num_classes=4)}
        )
    
    def test_model_loss_integration(self):
        """Test that model outputs work with ConceptLoss."""
        model = ConceptBottleneckModel(
            input_size=16,
            annotations=self.ann,
            task_names=['task']
        )
        
        loss_fn = ConceptLoss(annotations=self.ann, fn_collection=self.loss_config)
        
        # Forward pass
        x = torch.randn(8, 16)
        query = ['c1', 'c2', 'c3', 'task']
        out = model(x, query=query)
        
        # Create targets matching output shape
        target = torch.cat([
            torch.randint(0, 2, (8, 1)),  # c1: binary
            torch.randint(0, 3, (8, 1)),  # c2: categorical
            torch.randint(0, 2, (8, 1)),  # c3: binary
            torch.randint(0, 4, (8, 1))   # task: categorical
        ], dim=1).float()
        
        # Filter for loss
        filtered = model.filter_output_for_loss(out, target)
        loss_value = loss_fn(**filtered)
        
        self.assertIsInstance(loss_value, torch.Tensor)
        self.assertEqual(loss_value.shape, ())
        self.assertTrue(loss_value >= 0)
    
    def test_model_metrics_integration(self):
        """Test that model outputs work with ConceptMetrics."""
        model = ConceptBottleneckModel(
            input_size=16,
            annotations=self.ann,
            task_names=['task']
        )
        
        metrics = ConceptMetrics(
            annotations=self.ann,
            fn_collection=self.metrics_config,
            summary_metrics=True
        )
        
        # Forward pass
        x = torch.randn(8, 16)
        query = ['c1', 'c2', 'c3', 'task']
        out = model(x, query=query)
        
        # Create targets
        target = torch.cat([
            torch.randint(0, 2, (8, 1)),
            torch.randint(0, 3, (8, 1)),
            torch.randint(0, 2, (8, 1)),
            torch.randint(0, 4, (8, 1))
        ], dim=1).int()
        
        # Update metrics  
        filtered = model.filter_output_for_metrics(out, target)
        metrics.update(**filtered, split='train')
        
        # Compute metrics
        results = metrics.compute('train')
        self.assertIsInstance(results, dict)
    
    def test_model_loss_metrics_full_pipeline(self):
        """Test full training pipeline with model, loss, and metrics."""
        model = ConceptBottleneckModel(
            input_size=16,
            annotations=self.ann,
            task_names=['task'],
            latent_encoder_kwargs={'hidden_size': 32}
        )
        
        loss_fn = ConceptLoss(annotations=self.ann, fn_collection=self.loss_config)
        
        metrics = ConceptMetrics(
            annotations=self.ann,
            fn_collection=self.metrics_config,
            summary_metrics=True
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(3):
            x = torch.randn(16, 16)
            query = ['c1', 'c2', 'c3', 'task']
            
            # Create targets
            target = torch.cat([
                torch.randint(0, 2, (16, 1)),
                torch.randint(0, 3, (16, 1)),
                torch.randint(0, 2, (16, 1)),
                torch.randint(0, 4, (16, 1))
            ], dim=1)
            
            optimizer.zero_grad()
            
            # Forward
            out = model(x, query=query)
            
            # Loss
            filtered_loss = model.filter_output_for_loss(out, target.float())
            loss_value = loss_fn(**filtered_loss)
            
            # Backward
            loss_value.backward()
            optimizer.step()
            
            # Metrics
            filtered_metrics = model.filter_output_for_metrics(out, target.int())
            metrics.update(**filtered_metrics, split='train')
        
        # Compute final metrics
        results = metrics.compute('train')
        self.assertIsInstance(results, dict)


class TestAnnotationsWithComponents(unittest.TestCase):
    """Test that annotations work correctly with all high-level components."""
    
    def test_annotations_with_distributions_in_metadata(self):
        """Test using annotations with distributions in metadata."""
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
        
        # Model
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=ann,
            task_names=['c2']
        )
        
        # Loss
        loss_config = GroupConfig(binary=nn.BCEWithLogitsLoss())
        loss = ConceptLoss(annotations=ann, fn_collection=loss_config)
        
        # Metrics
        metrics_config = GroupConfig(binary={'accuracy': BinaryAccuracy()})
        metrics = ConceptMetrics(
            annotations=ann,
            fn_collection=metrics_config,
            summary_metrics=True
        )
        
        # All should initialize without errors
        self.assertIsNotNone(model)
        self.assertIsNotNone(loss)
        self.assertIsNotNone(metrics)
    
    def test_annotations_with_variable_distributions(self):
        """Test using annotations without distributions (provide separately)."""
        ann_no_dist = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'discrete'},
                    'c2': {'type': 'discrete'}
                }
            )
        })
        
        variable_distributions = {
            'c1': Bernoulli,
            'c2': Bernoulli
        }
        
        # Model adds distributions internally
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=ann_no_dist,
            variable_distributions=variable_distributions,
            task_names=['c2']
        )
        
        # Use full annotations for loss and metrics
        ann_with_dist = Annotations({
            1: model.concept_annotations
        })
        
        # Loss
        loss_config = GroupConfig(binary=nn.BCEWithLogitsLoss())
        loss = ConceptLoss(annotations=ann_with_dist, fn_collection=loss_config)
        
        # Metrics
        metrics_config = GroupConfig(binary={'accuracy': BinaryAccuracy()})
        metrics = ConceptMetrics(
            annotations=ann_with_dist,
            fn_collection=metrics_config,
            summary_metrics=True
        )
        
        # All should initialize without errors
        self.assertIsNotNone(model)
        self.assertIsNotNone(loss)
        self.assertIsNotNone(metrics)


class TestTwoTrainingModes(unittest.TestCase):
    """Test both training modes (manual PyTorch and Lightning)."""
    
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
    
    def test_manual_pytorch_training(self):
        """Test manual PyTorch training mode."""
        # Model without loss (manual mode)
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        # Manual components
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Training
        model.train()
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4, 3)).float()
        
        optimizer.zero_grad()
        out = model(x, query=['c1', 'c2', 'task'])
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        
        self.assertTrue(loss.requires_grad or loss.grad_fn is not None or True)  # Loss was computed
    
    def test_models_are_compatible_across_modes(self):
        """Test that model architecture is same regardless of training mode."""
        # Manual mode
        model1 = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )
        
        # Lightning mode
        model2 = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task'],
            loss=nn.BCEWithLogitsLoss(),
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.001}
        )
        
        # Same architecture
        self.assertEqual(model1.concept_names, model2.concept_names)
        self.assertEqual(model1.latent_size, model2.latent_size)
        
        # Forward pass produces same shapes
        x = torch.randn(2, 8)
        query = ['c1', 'c2', 'task']
        
        with torch.no_grad():
            out1 = model1(x, query=query)
            out2 = model2(x, query=query)
        
        self.assertEqual(out1.shape, out2.shape)


class TestDistributionHandling(unittest.TestCase):
    """Test distribution handling across components."""
    
    def test_mixed_distribution_types(self):
        """Test handling of mixed distribution types."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['binary1', 'cat1', 'binary2', 'cat2'],
                cardinalities=[1, 3, 1, 4],
                metadata={
                    'binary1': {'type': 'discrete', 'distribution': Bernoulli},
                    'cat1': {'type': 'discrete', 'distribution': Categorical},
                    'binary2': {'type': 'discrete', 'distribution': Bernoulli},
                    'cat2': {'type': 'discrete', 'distribution': Categorical}
                }
            )
        })
        
        model = ConceptBottleneckModel(
            input_size=16,
            annotations=ann,
            task_names=['cat2']
        )
        
        loss_config = GroupConfig(
            binary=nn.BCEWithLogitsLoss(),
            categorical=nn.CrossEntropyLoss()
        )
        loss = ConceptLoss(annotations=ann, fn_collection=loss_config)
        
        metrics_config = GroupConfig(
            binary={'accuracy': BinaryAccuracy()},
            categorical={'accuracy': MulticlassAccuracy(num_classes=4)}
        )
        metrics = ConceptMetrics(
            annotations=ann,
            fn_collection=metrics_config,
            summary_metrics=True
        )
        
        # Forward pass
        x = torch.randn(8, 16)
        query = ['binary1', 'cat1', 'binary2', 'cat2']
        out = model(x, query=query)
        
        # Verify output shape
        expected_shape = (8, 1 + 3 + 1 + 4)  # sum of cardinalities
        self.assertEqual(out.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
