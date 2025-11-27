"""
Tests for torch_concepts.nn.modules.high.base.learner.BaseLearner

BaseLearner is now a lightweight training orchestrator that handles:
- Loss computation
- Metrics tracking (ConceptMetrics or dict of MetricCollections)  
- Optimizer and scheduler configuration

Note: Annotations and concept management are now handled by BaseModel,
not BaseLearner. These tests focus on the core orchestration functionality.
"""
import unittest
import torch
import torch.nn as nn
import torchmetrics
from torch.distributions import Bernoulli
from torch_concepts.annotations import Annotations, AxisAnnotation
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torch_concepts.nn.modules.utils import GroupConfig


class MockLearner(BaseLearner):
    """Mock implementation of BaseLearner for testing."""
    def __init__(self, n_concepts=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store n_concepts for testing (would normally come from model)
        self.n_concepts = n_concepts
        # Add a dummy parameter so optimizer has parameters
        self.dummy_param = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """Simple forward pass for testing."""
        return torch.randn(x.shape[0], self.n_concepts)


class TestBaseLearnerInitialization(unittest.TestCase):
    """Test BaseLearner initialization."""

    def test_basic_initialization(self):
        """Test initialization without parameters."""
        learner = MockLearner(n_concepts=3)
        self.assertEqual(learner.n_concepts, 3)
        self.assertIsNone(learner.loss)
        self.assertIsNone(learner.metrics)
        self.assertIsNone(learner.optim_class)

    def test_initialization_with_loss(self):
        """Test initialization with loss function."""
        loss_fn = nn.MSELoss()
        learner = MockLearner(n_concepts=2, loss=loss_fn)
        self.assertEqual(learner.loss, loss_fn)

    def test_initialization_with_optimizer(self):
        """Test initialization with optimizer configuration."""
        learner = MockLearner(
            n_concepts=3,
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}
        )
        self.assertEqual(learner.optim_class, torch.optim.Adam)
        self.assertEqual(learner.optim_kwargs, {'lr': 0.001, 'weight_decay': 0.0001})

    def test_initialization_with_scheduler(self):
        """Test initialization with scheduler configuration."""
        learner = MockLearner(
            n_concepts=2,
            optim_class=torch.optim.Adam,
            scheduler_class=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={'step_size': 10, 'gamma': 0.1}
        )
        self.assertEqual(learner.scheduler_class, torch.optim.lr_scheduler.StepLR)
        self.assertEqual(learner.scheduler_kwargs, {'step_size': 10, 'gamma': 0.1})

    def test_repr_with_optimizer_and_scheduler(self):
        """Test __repr__ method with optimizer and scheduler."""
        learner = MockLearner(
            n_concepts=3,
            optim_class=torch.optim.Adam,
            scheduler_class=torch.optim.lr_scheduler.StepLR
        )
        repr_str = repr(learner)
        self.assertIn("MockLearner", repr_str)
        self.assertIn("n_concepts=3", repr_str)
        self.assertIn("Adam", repr_str)
        self.assertIn("StepLR", repr_str)

    def test_repr_without_scheduler(self):
        """Test __repr__ method without scheduler."""
        learner = MockLearner(
            n_concepts=2,
            optim_class=torch.optim.SGD
        )
        repr_str = repr(learner)
        self.assertIn("scheduler=None", repr_str)


class TestBaseLearnerMetrics(unittest.TestCase):
    """Test metrics handling in BaseLearner."""

    def setUp(self):
        """Set up annotations for ConceptMetrics testing."""
        self.annotations = Annotations({
            1: AxisAnnotation(
                labels=('C1', 'C2'),
                metadata={
                    'C1': {'type': 'discrete', 'distribution': Bernoulli},
                    'C2': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })

    def test_metrics_none(self):
        """Test initialization with no metrics."""
        learner = MockLearner(metrics=None)
        self.assertIsNone(learner.metrics)
        self.assertIsNone(learner.train_metrics)
        self.assertIsNone(learner.val_metrics)
        self.assertIsNone(learner.test_metrics)

    def test_metrics_with_concept_metrics(self):
        """Test initialization with ConceptMetrics object."""
        metrics = ConceptMetrics(
            annotations=self.annotations,
            summary_metrics=True,
            fn_collection=GroupConfig(
                binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
            )
        )
        learner = MockLearner(metrics=metrics)
        
        # Verify metrics object is stored
        self.assertIs(learner.metrics, metrics)
        
        # Verify pointers to individual collections
        self.assertIs(learner.train_metrics, metrics.train_metrics)
        self.assertIs(learner.val_metrics, metrics.val_metrics)
        self.assertIs(learner.test_metrics, metrics.test_metrics)

    def test_metrics_with_dict(self):
        """Test initialization with dict of MetricCollections."""
        from torchmetrics import MetricCollection
        
        train_collection = MetricCollection({
            'accuracy': torchmetrics.classification.BinaryAccuracy()
        })
        val_collection = MetricCollection({
            'accuracy': torchmetrics.classification.BinaryAccuracy()
        })
        test_collection = MetricCollection({
            'accuracy': torchmetrics.classification.BinaryAccuracy()
        })
        
        metrics_dict = {
            'train_metrics': train_collection,
            'val_metrics': val_collection,
            'test_metrics': test_collection
        }
        
        learner = MockLearner(metrics=metrics_dict)
        
        # Verify dict is stored
        self.assertIs(learner.metrics, metrics_dict)
        
        # Verify pointers to individual collections
        self.assertIs(learner.train_metrics, train_collection)
        self.assertIs(learner.val_metrics, val_collection)
        self.assertIs(learner.test_metrics, test_collection)

    def test_metrics_dict_with_invalid_keys(self):
        """Test that dict with invalid keys raises assertion error."""
        from torchmetrics import MetricCollection
        
        invalid_dict = {
            'training': MetricCollection({'acc': torchmetrics.classification.BinaryAccuracy()}),
            'validation': MetricCollection({'acc': torchmetrics.classification.BinaryAccuracy()})
        }
        
        with self.assertRaises(AssertionError) as context:
            MockLearner(metrics=invalid_dict)
        self.assertIn("train_metrics", str(context.exception))
        self.assertIn("val_metrics", str(context.exception))
        self.assertIn("test_metrics", str(context.exception))

    def test_update_metrics_with_concept_metrics(self):
        """Test update_metrics method with ConceptMetrics."""
        metrics = ConceptMetrics(
            annotations=self.annotations,
            summary_metrics=True,
            fn_collection=GroupConfig(
                binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
            )
        )
        learner = MockLearner(metrics=metrics)
        
        # Create dummy predictions and targets (2 samples, 2 concepts)
        preds = torch.tensor([[0.8, 0.7], [0.2, 0.3]])
        targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        
        # Update metrics - should not raise error
        learner.update_metrics(preds, targets, step='train')

    def test_update_metrics_with_dict(self):
        """Test update_metrics method with dict of MetricCollections."""
        from torchmetrics import MetricCollection
        
        train_collection = MetricCollection({
            'accuracy': torchmetrics.classification.BinaryAccuracy()
        })
        
        metrics_dict = {
            'train_metrics': train_collection,
            'val_metrics': None,
            'test_metrics': None
        }
        
        learner = MockLearner(metrics=metrics_dict)
        
        # Create dummy predictions and targets
        preds = torch.tensor([0.8, 0.2])
        targets = torch.tensor([1, 0])
        
        # Update metrics - should not raise error
        learner.update_metrics(preds, targets, step='train')

    def test_update_metrics_with_none(self):
        """Test update_metrics when metrics is None."""
        learner = MockLearner(metrics=None)
        
        # Should not raise error even with None metrics
        preds = torch.tensor([0.8, 0.2])
        targets = torch.tensor([1, 0])
        learner.update_metrics(preds, targets, step='train')


class TestBaseLearnerUpdateAndLogMetrics(unittest.TestCase):
    """Test update_and_log_metrics method."""

    def setUp(self):
        """Set up annotations for testing."""
        self.annotations = Annotations({
            1: AxisAnnotation(
                labels=('C1', 'C2'),
                metadata={
                    'C1': {'type': 'discrete', 'distribution': Bernoulli},
                    'C2': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })

    def test_update_and_log_metrics(self):
        """Test update_and_log_metrics method."""
        metrics = ConceptMetrics(
            annotations=self.annotations,
            summary_metrics=True,
            fn_collection=GroupConfig(
                binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
            )
        )
        learner = MockLearner(metrics=metrics)
        
        # Create metrics args (2 samples, 2 concepts)
        metrics_args = {
            'preds': torch.tensor([[0.8, 0.7], [0.2, 0.3]]),
            'target': torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        }
        
        # Should not raise error
        learner.update_and_log_metrics(metrics_args, step='train', batch_size=2)


if __name__ == '__main__':
    unittest.main()
