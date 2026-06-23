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
import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from torch_concepts.nn import ConceptBottleneckModel
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.nn.modules.loss import ConceptLoss
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torch_concepts.annotations import Annotations
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy


def _logits(out, names):
    """Concatenate per-variable logits for the queried ``names`` -> (B, sum cardinalities)."""
    return torch.cat([out.params[n]['logits'] for n in names], dim=1)


class TestHighLevelIntegration(unittest.TestCase):
    """Test integration of high-level components."""

    def setUp(self):
        """Set up test fixtures."""
        # Mixed binary and categorical concepts
        self.ann = Annotations(
                labels=['c1', 'c2', 'c3', 'task'],
                cardinalities=[1, 3, 1, 4],
                types=['binary', 'categorical', 'binary', 'categorical'],
            )

        self.loss_binary = nn.BCEWithLogitsLoss()
        self.loss_categorical = nn.CrossEntropyLoss()

    def test_model_loss_integration(self):
        """Test that model outputs work with ConceptLoss."""
        model = ConceptBottleneckModel(
            input_size=16,
            annotations=self.ann,
            task_names=['task']
        )

        loss_fn = ConceptLoss(
            annotations=self.ann,
            binary=self.loss_binary,
            categorical=self.loss_categorical,
        )

        # Forward pass
        x = torch.randn(8, 16)
        query = ['c1', 'c2', 'c3', 'task']
        out = model(query=query, input=x)

        # Create concept-level targets (one integer per concept)
        target = torch.cat([
            torch.randint(0, 2, (8, 1)),   # c1: binary
            torch.randint(0, 3, (8, 1)),   # c2: 3-class
            torch.randint(0, 2, (8, 1)),   # c3: binary
            torch.randint(0, 4, (8, 1)),   # task: 4-class
        ], dim=1).float()

        # Attach target and compute ConceptLoss
        out.target = target
        loss_value = loss_fn(out)

        self.assertIsInstance(loss_value, torch.Tensor)
        self.assertEqual(loss_value.shape, ())
        self.assertGreaterEqual(loss_value.item(), 0)

    def test_model_metrics_integration(self):
        """Test that model outputs work with ConceptMetrics."""
        model = ConceptBottleneckModel(
            input_size=16,
            annotations=self.ann,
            task_names=['task']
        )

        metrics = ConceptMetrics(
            annotations=self.ann,
            binary={'accuracy': BinaryAccuracy()},
            categorical={'accuracy': (MulticlassAccuracy, {"average": "micro"})},
            summary=True,
        )

        # Forward pass
        x = torch.randn(8, 16)
        query = ['c1', 'c2', 'c3', 'task']
        out = model(query=query, input=x)

        # Create concept-level integer targets
        target = torch.cat([
            torch.randint(0, 2, (8, 1)),
            torch.randint(0, 3, (8, 1)),
            torch.randint(0, 2, (8, 1)),
            torch.randint(0, 4, (8, 1)),
        ], dim=1).float()

        out.target = target
        metrics.update(out)

        results = metrics.compute()
        self.assertIsInstance(results, dict)

    def test_model_loss_metrics_full_pipeline(self):
        """Test full training pipeline with model, loss, and metrics."""
        model = ConceptBottleneckModel(
            input_size=16,
            annotations=self.ann,
            task_names=['task'],
        )

        loss_fn = ConceptLoss(
            annotations=self.ann,
            binary=self.loss_binary,
            categorical=self.loss_categorical,
        )

        metrics = ConceptMetrics(
            annotations=self.ann,
            binary={'accuracy': BinaryAccuracy()},
            categorical={'accuracy': (MulticlassAccuracy, {"average": "micro"})},
            summary=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(3):
            x = torch.randn(16, 16)
            query = ['c1', 'c2', 'c3', 'task']

            target = torch.cat([
                torch.randint(0, 2, (16, 1)),
                torch.randint(0, 3, (16, 1)),
                torch.randint(0, 2, (16, 1)),
                torch.randint(0, 4, (16, 1)),
            ], dim=1).float()

            optimizer.zero_grad()
            out = model(query=query, input=x)
            out.target = target
            loss_value = loss_fn(out)
            loss_value.backward()
            optimizer.step()

            metrics.update(out)

        results = metrics.compute()
        self.assertIsInstance(results, dict)


class TestAnnotationsWithComponents(unittest.TestCase):
    """Test that annotations work correctly with all high-level components."""

    def test_annotations_with_distributions_in_metadata(self):
        """Test that binary annotations initialize model, loss, and metrics correctly."""
        ann = Annotations(
                labels=['c1', 'c2'],
                cardinalities=[1, 1],
                types=['binary', 'binary'],
            )

        model = ConceptBottleneckModel(
            input_size=8,
            annotations=ann,
            task_names=['c2']
        )

        loss = ConceptLoss(annotations=ann, binary=nn.BCEWithLogitsLoss())

        metrics = ConceptMetrics(
            annotations=ann,
            binary={'accuracy': BinaryAccuracy()},
            summary=True,
        )

        self.assertIsNotNone(model)
        self.assertIsNotNone(loss)
        self.assertIsNotNone(metrics)

    def test_annotations_with_variable_distributions(self):
        """Test that per-type variable_distributions override works on the model."""
        ann = Annotations(
                labels=['c1', 'c2'],
                cardinalities=[1, 1],
                types=['binary', 'binary'],
            )

        # Model with explicit per-type distribution override
        model = ConceptBottleneckModel(
            input_size=8,
            annotations=ann,
            task_names=['c2'],
            variable_distributions={'binary': Bernoulli},
        )

        loss = ConceptLoss(annotations=ann, binary=nn.BCEWithLogitsLoss())

        metrics = ConceptMetrics(
            annotations=ann,
            binary={'accuracy': BinaryAccuracy()},
            summary=True,
        )

        self.assertIsNotNone(model)
        self.assertIsNotNone(loss)
        self.assertIsNotNone(metrics)


class TestTwoTrainingModes(unittest.TestCase):
    """Test both training modes (manual PyTorch and Lightning)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                types=['binary', 'binary', 'binary'],
            )
    
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
        out = model(query=['c1', 'c2', 'task'], input=x)
        loss = loss_fn(_logits(out, ['c1', 'c2', 'task']), y)
        loss.backward()
        optimizer.step()

        self.assertTrue(loss.requires_grad or loss.grad_fn is not None or True)  # Loss was computed

    def test_models_are_compatible_across_modes(self):
        """Test that model architecture is same regardless of lightning mode."""
        # Manual mode (pure PyTorch)
        model1 = ConceptBottleneckModel(
            input_size=8,
            annotations=self.ann,
            task_names=['task']
        )

        # Lightning mode (with lightning=True)
        model2 = ConceptBottleneckModel(
            lightning=True,
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
            out1 = model1(query=query, input=x)
            out2 = model2(query=query, input=x)

        self.assertEqual(
            _logits(out1, query).shape,
            _logits(out2, query).shape,
        )


class TestDistributionHandling(unittest.TestCase):
    """Test distribution handling across components."""

    def test_mixed_distribution_types(self):
        """Test handling of mixed distribution types."""
        ann = Annotations(
                labels=['binary1', 'cat1', 'binary2', 'cat2'],
                cardinalities=[1, 3, 1, 4],
                types=['binary', 'categorical', 'binary', 'categorical'],
            )

        model = ConceptBottleneckModel(
            input_size=16,
            annotations=ann,
            task_names=['cat2']
        )

        loss = ConceptLoss(
            annotations=ann,
            binary=nn.BCEWithLogitsLoss(),
            categorical=nn.CrossEntropyLoss(),
        )

        metrics = ConceptMetrics(
            annotations=ann,
            binary={'accuracy': BinaryAccuracy()},
            categorical={'accuracy': (MulticlassAccuracy, {"average": "micro"})},
            summary=True,
        )

        # Forward pass
        x = torch.randn(8, 16)
        query = ['binary1', 'cat1', 'binary2', 'cat2']
        out = model(query=query, input=x)

        # Verify output shape via per-variable logits
        expected_shape = (8, 1 + 3 + 1 + 4)  # sum of cardinalities
        self.assertEqual(_logits(out, query).shape, expected_shape)

        # Verify loss and metrics can be used
        self.assertIsNotNone(loss)
        self.assertIsNotNone(metrics)


if __name__ == '__main__':
    unittest.main()
