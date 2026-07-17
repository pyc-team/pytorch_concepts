"""
Tests for torch_concepts.nn.modules.high.base.learner.BaseLearner

BaseLearner is now a lightweight training orchestrator that handles:
- Loss computation (single nn.Module)
- Metrics tracking (ConceptMetrics or dict of MetricCollections)  
- Optimizer and scheduler configuration
- shared_step / training_step / validation_step / test_step
- _get_inference_kwargs
- log_loss
- update_metrics error path

Note: Annotations and concept management are now handled by BaseModel,
not BaseLearner. These tests focus on the core orchestration functionality.
"""
import unittest
import pytest
import torch
import torch.nn as nn
import torchmetrics
from torch.distributions import Bernoulli
from torch_concepts.annotations import Annotations
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.nn.modules.loss import ConceptLoss
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torch_concepts.nn.modules.outputs import ModelOutput


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


class FullMockLearner(BaseLearner):
    """Mock that satisfies all shared_step requirements.

    Mimics the interface that BaseModel normally provides:
    concept_names, concept_annotations, prepare_target,
    and a full forward(x, query, evidence, **kw).
    """

    def __init__(self, annotations, n_concepts=2, **kwargs):
        super().__init__(**kwargs)
        self.n_concepts = n_concepts
        self.concept_annotations = annotations
        self.concept_names = self.concept_annotations.labels
        self.dummy_param = nn.Parameter(torch.randn(1))

    def build_query(self, ground_truth):
        if ground_truth is None:
            return {}
        return {
            name: ground_truth[:, i].float().unsqueeze(-1)
            for i, name in enumerate(self.concept_names)
        }

    def forward(self, x=None, query=None, evidence=None, **kwargs):
        if evidence is not None:
            batch_size = evidence['input'].shape[0]
        elif x is not None:
            batch_size = x.shape[0]
        else:
            batch_size = 2
        logits = torch.randn(batch_size, self.n_concepts, requires_grad=True)
        params = {}
        if query:
            for i, name in enumerate(query.keys()):
                params[name] = {'logits': logits[:, i:i+1]}
        return ModelOutput(logits=logits, params=params)

    def prepare_target(self, target):
        return target


class TestBaseLearnerInitialization(unittest.TestCase):
    """Test BaseLearner initialization."""

    def test_basic_initialization(self):
        """Test initialization without parameters."""
        learner = MockLearner(n_concepts=3)
        self.assertEqual(learner.n_concepts, 3)
        self.assertIsNone(learner.loss)
        self.assertIsNone(learner.train_metrics)
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
        self.annotations = Annotations(
                labels=('C1', 'C2'),
                metadata={
                    'C1': {'type': 'discrete', 'distribution': Bernoulli},
                    'C2': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )

    def test_metrics_none(self):
        """Test initialization with no metrics."""
        learner = MockLearner(metrics=None)
        self.assertIsNone(learner.train_metrics)
        self.assertIsNone(learner.val_metrics)
        self.assertIsNone(learner.test_metrics)

    def test_metrics_with_concept_metrics(self):
        """Test initialization with ConceptMetrics object."""
        metrics = ConceptMetrics(
            annotations=self.annotations,
            summary=True,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
        )
        learner = MockLearner(metrics=metrics)
        
        # Verify split metrics are created
        self.assertIsNotNone(learner.train_metrics)
        self.assertIsNotNone(learner.val_metrics)
        self.assertIsNotNone(learner.test_metrics)

    def test_metrics_with_dict_raises(self):
        """Test that passing a dict raises AssertionError."""
        with self.assertRaises(AssertionError):
            MockLearner(metrics={'train_metrics': None})

    def test_metrics_with_invalid_type_raises(self):
        """Test that passing an invalid type raises AssertionError."""
        with self.assertRaises(AssertionError):
            MockLearner(metrics="not_metrics")

    def test_update_metrics_with_concept_metrics(self):
        """Test update_metrics method with ConceptMetrics."""
        metrics = ConceptMetrics(
            annotations=self.annotations,
            summary=True,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
        )
        learner = MockLearner(metrics=metrics)
        
        # Create ModelOutput (2 samples, 2 concepts)
        out = ModelOutput(
            logits=torch.tensor([[0.8, 0.7], [0.2, 0.3]]),
            target=torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        )
        
        # Update metrics - should not raise error
        learner.update_metrics(out, step='train')

    def test_update_metrics_with_none(self):
        """Test update_metrics when metrics is None."""
        learner = MockLearner(metrics=None)
        
        # Should not raise error even with None metrics
        out = ModelOutput(
            logits=torch.tensor([0.8, 0.2]),
            target=torch.tensor([1, 0])
        )
        learner.update_metrics(out, step='train')


class TestBaseLearnerUpdateAndLogMetrics(unittest.TestCase):
    """Test update_and_log_metrics method."""

    def setUp(self):
        """Set up annotations for testing."""
        self.annotations = Annotations(
                labels=('C1', 'C2'),
                metadata={
                    'C1': {'type': 'discrete', 'distribution': Bernoulli},
                    'C2': {'type': 'discrete', 'distribution': Bernoulli}
                }
            )

    def test_update_and_log_metrics(self):
        """Test update_and_log_metrics method."""
        metrics = ConceptMetrics(
            annotations=self.annotations,
            summary=True,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
        )
        learner = MockLearner(metrics=metrics)
        
        # Create ModelOutput (2 samples, 2 concepts)
        out = ModelOutput(
            logits=torch.tensor([[0.8, 0.7], [0.2, 0.3]]),
            target=torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        )
        
        # Should not raise error
        learner.update_and_log_metrics(out, step='train', batch_size=2)


class TestBaseLearnerBatchHandling(unittest.TestCase):
    """Test batch handling methods (_check_batch, unpack_batch)."""

    def test_check_batch_valid(self):
        """Test _check_batch with valid batch."""
        learner = MockLearner(n_concepts=2)
        batch = {
            'inputs': {'x': torch.randn(4, 8)},
            'concepts': {'c': torch.randint(0, 2, (4, 2)).float()}
        }
        
        # Should not raise error
        learner._check_batch(batch)

    def test_check_batch_missing_inputs(self):
        """Test _check_batch with missing 'inputs' key."""
        learner = MockLearner(n_concepts=2)
        batch = {
            'concepts': {'c': torch.randint(0, 2, (4, 2)).float()}
        }
        
        with self.assertRaises(KeyError) as context:
            learner._check_batch(batch)
        self.assertIn("inputs", str(context.exception))

    def test_check_batch_missing_concepts(self):
        """Test _check_batch with missing 'concepts' key."""
        learner = MockLearner(n_concepts=2)
        batch = {
            'inputs': {'x': torch.randn(4, 8)}
        }
        
        with self.assertRaises(KeyError) as context:
            learner._check_batch(batch)
        self.assertIn("concepts", str(context.exception))

    def test_check_batch_not_dict(self):
        """Test _check_batch with non-dict batch."""
        learner = MockLearner(n_concepts=2)
        batch = [torch.randn(4, 8)]
        
        with self.assertRaises(TypeError) as context:
            learner._check_batch(batch)
        self.assertIn("dict", str(context.exception))

    def test_unpack_batch_returns_tuple(self):
        """Test unpack_batch returns (inputs, concepts, transforms)."""
        learner = MockLearner(n_concepts=2)
        x = torch.randn(4, 8)
        c = torch.randint(0, 2, (4, 2)).float()
        batch = {
            'inputs': {'x': x},
            'concepts': {'c': c}
        }
        
        inputs, concepts, transforms = learner.unpack_batch(batch)
        
        self.assertEqual(inputs, {'x': x})
        self.assertEqual(concepts, {'c': c})
        self.assertEqual(transforms, {})

    def test_unpack_batch_with_transforms(self):
        """Test unpack_batch extracts transforms when present."""
        learner = MockLearner(n_concepts=2)
        mock_transform = {'c': 'some_transform'}
        batch = {
            'inputs': {'x': torch.randn(4, 8)},
            'concepts': {'c': torch.randint(0, 2, (4, 2)).float()},
            'transforms': mock_transform
        }
        
        inputs, concepts, transforms = learner.unpack_batch(batch)
        
        self.assertEqual(transforms, mock_transform)


class TestBaseLearnerConfigureOptimizers(unittest.TestCase):
    """Test configure_optimizers method."""

    def test_configure_optimizers_none(self):
        """Test configure_optimizers returns None when no optimizer set."""
        learner = MockLearner(n_concepts=2)
        result = learner.configure_optimizers()
        self.assertIsNone(result)

    def test_configure_optimizers_with_optimizer_only(self):
        """Test configure_optimizers with optimizer only."""
        learner = MockLearner(
            n_concepts=2,
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.001}
        )
        
        result = learner.configure_optimizers()
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimizer', result)
        self.assertIsInstance(result['optimizer'], torch.optim.Adam)

    def test_configure_optimizers_with_scheduler(self):
        """Test configure_optimizers with optimizer and scheduler."""
        learner = MockLearner(
            n_concepts=2,
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.001},
            scheduler_class=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={'step_size': 10}
        )
        
        result = learner.configure_optimizers()
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimizer', result)
        self.assertIn('lr_scheduler', result)
        self.assertIsInstance(result['optimizer'], torch.optim.Adam)
        self.assertIsInstance(result['lr_scheduler'], torch.optim.lr_scheduler.StepLR)

    def test_configure_optimizers_with_monitor(self):
        """Test configure_optimizers extracts monitor from scheduler_kwargs."""
        learner = MockLearner(
            n_concepts=2,
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.001},
            scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_kwargs={'mode': 'min', 'monitor': 'val_loss'}
        )
        
        result = learner.configure_optimizers()
        
        self.assertIn('monitor', result)
        self.assertEqual(result['monitor'], 'val_loss')

    def test_configure_optimizers_without_kwargs(self):
        """Test configure_optimizers works without kwargs."""
        learner = MockLearner(
            n_concepts=2,
            optim_class=torch.optim.SGD
        )
        
        result = learner.configure_optimizers()
        
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result['optimizer'], torch.optim.SGD)


# ======================================================================
# update_metrics error path
# ======================================================================

class TestBaseLearnerUpdateMetricsError(unittest.TestCase):
    """Test update_metrics with invalid metrics is a no-op."""

    def test_update_metrics_invalid_type_is_noop(self):
        """When no split metrics are set, update_metrics is a no-op."""
        learner = MockLearner(n_concepts=2)
        out = ModelOutput(
            logits=torch.tensor([0.8, 0.2]),
            target=torch.tensor([1, 0])
        )
        # Should not raise — train_metrics is None so nothing happens
        learner.update_metrics(out, step='train')


# ======================================================================
# _get_inference_kwargs
# ======================================================================

class TestGetInferenceKwargs(unittest.TestCase):
    """Test shared_step's build_query/forward integration replaces the old _get_inference_kwargs."""

    def setUp(self):
        self.annotations = Annotations(
                labels=('C1', 'C2'),
                metadata={
                    'C1': {'type': 'discrete', 'distribution': Bernoulli},
                    'C2': {'type': 'discrete', 'distribution': Bernoulli},
                }
            )

    def test_no_inference_returns_empty(self):
        """FullMockLearner.build_query with None ground_truth returns empty dict."""
        learner = FullMockLearner(self.annotations, n_concepts=2)
        # build_query(None) returns an empty dict (no teacher-forcing)
        result = learner.build_query(None)
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {})

    def test_with_inference_returns_kwargs(self):
        """FullMockLearner.build_query with a real ground_truth returns per-concept tensors."""
        learner = FullMockLearner(self.annotations, n_concepts=2)
        c = torch.randint(0, 2, (4, 2)).float()
        result = learner.build_query(c)
        self.assertIsInstance(result, dict)
        # build_query maps each concept name to a (batch, 1) tensor
        for name in ('C1', 'C2'):
            self.assertIn(name, result)
            self.assertEqual(result[name].shape[0], 4)


# ======================================================================
# shared_step, training_step, validation_step, test_step, log_loss
# ======================================================================

class TestBaseLearnerSharedStep(unittest.TestCase):
    """Test shared_step and the per-split step methods."""

    def setUp(self):
        self.annotations = Annotations(
                labels=('C1', 'C2'),
                metadata={
                    'C1': {'type': 'discrete', 'distribution': Bernoulli},
                    'C2': {'type': 'discrete', 'distribution': Bernoulli},
                }
            )
        self.loss_fn = ConceptLoss(
            self.annotations,
            binary=nn.BCEWithLogitsLoss(),
        )
        self.batch = {
            'inputs': {'x': torch.randn(8, 3)},
            'concepts': {'c': torch.randint(0, 2, (8, 2)).float()},
        }

    # -- helpers to capture Lightning self.log / self.log_dict calls ----

    @staticmethod
    def _patch_logging(learner):
        """Monkey-patch self.log and self.log_dict so they don't need a Trainer."""
        learner._logged = {}
        def _fake_log(name, value, **kw):
            learner._logged[name] = value
        def _fake_log_dict(d, **kw):
            learner._logged.update(d)
        learner.log = _fake_log
        learner.log_dict = _fake_log_dict

    # -- shared_step ---------------------------------------------------

    def test_shared_step_computes_loss(self):
        """shared_step returns a scalar loss and logs it."""
        learner = FullMockLearner(
            self.annotations, n_concepts=2,
            loss=self.loss_fn,
        )
        self._patch_logging(learner)
        loss = learner.shared_step(self.batch, step='train')
        self.assertEqual(loss.shape, ())
        self.assertIn('train_loss', learner._logged)

    def test_shared_step_no_loss(self):
        """shared_step with loss=None returns None."""
        learner = FullMockLearner(
            self.annotations, n_concepts=2,
            loss=None,
        )
        self._patch_logging(learner)
        result = learner.shared_step(self.batch, step='train')
        self.assertIsNone(result)

    def test_shared_step_with_composite_loss(self):
        """shared_step works when loss uses per-type composition."""
        loss = ConceptLoss(
            self.annotations,
            binary=[nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss()],
            binary_weights=[1.0, 0.5],
        )
        learner = FullMockLearner(
            self.annotations, n_concepts=2,
            loss=loss,
        )
        self._patch_logging(learner)
        loss = learner.shared_step(self.batch, step='val')
        self.assertEqual(loss.shape, ())
        self.assertIn('val_loss', learner._logged)

    # -- training_step / validation_step / test_step -------------------

    def test_training_step(self):
        """training_step delegates to shared_step('train')."""
        learner = FullMockLearner(
            self.annotations, n_concepts=2, loss=self.loss_fn,
        )
        self._patch_logging(learner)
        loss = learner.training_step(self.batch)
        self.assertEqual(loss.shape, ())
        self.assertIn('train_loss', learner._logged)

    def test_validation_step(self):
        """validation_step delegates to shared_step('val')."""
        learner = FullMockLearner(
            self.annotations, n_concepts=2, loss=self.loss_fn,
        )
        self._patch_logging(learner)
        loss = learner.validation_step(self.batch)
        self.assertEqual(loss.shape, ())
        self.assertIn('val_loss', learner._logged)

    def test_test_step(self):
        """test_step delegates to shared_step('test')."""
        learner = FullMockLearner(
            self.annotations, n_concepts=2, loss=self.loss_fn,
        )
        self._patch_logging(learner)
        loss = learner.test_step(self.batch)
        self.assertEqual(loss.shape, ())
        self.assertIn('test_loss', learner._logged)

    # -- log_loss (line 171) -------------------------------------------

    def test_log_loss_called(self):
        """log_loss records '<step>_loss'."""
        learner = FullMockLearner(
            self.annotations, n_concepts=2, loss=self.loss_fn,
        )
        self._patch_logging(learner)
        fake_loss = torch.tensor(0.42, requires_grad=True)
        learner.log_loss('train', fake_loss, batch_size=8)
        self.assertIn('train_loss', learner._logged)
        self.assertAlmostEqual(learner._logged['train_loss'].item(), 0.42, places=5)

    def test_shared_step_standard_loss(self):
        """shared_step dispatches standard PyTorch loss as loss(logits, target)."""
        learner = FullMockLearner(
            self.annotations, n_concepts=2,
            loss=nn.MSELoss(),
        )
        self._patch_logging(learner)
        loss = learner.shared_step(self.batch, step='train')
        self.assertEqual(loss.shape, ())
        self.assertIn('train_loss', learner._logged)


class TestBaseLearnerMetricsEdgeCases(unittest.TestCase):
    """Cover ConceptMetrics with empty collection and log_metrics else branch."""

    def setUp(self):
        self.annotations = Annotations(
                labels=('C1', 'C2'),
                metadata={
                    'C1': {'type': 'discrete', 'distribution': Bernoulli},
                    'C2': {'type': 'discrete', 'distribution': Bernoulli},
                }
            )

    def test_concept_metrics_no_collection(self):
        """ConceptMetrics with empty collection sets split metrics to None."""
        metrics = ConceptMetrics(
            annotations=self.annotations,
            summary=True,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
        )
        # Simulate an empty collection (all sub-collections cleared)
        metrics.binary = torchmetrics.MetricCollection({})
        learner = MockLearner(n_concepts=2, metrics=metrics)
        self.assertIsNone(learner.train_metrics)
        self.assertIsNone(learner.val_metrics)
        self.assertIsNone(learner.test_metrics)

    def test_log_metrics_non_concept_metrics(self):
        """log_metrics with a plain MetricCollection uses the else branch."""
        learner = MockLearner(n_concepts=2)
        learner._logged = {}
        def _fake_log_dict(d, **kw):
            learner._logged.update(d)
        learner.log_dict = _fake_log_dict

        coll = torchmetrics.MetricCollection({
            'acc': torchmetrics.classification.BinaryAccuracy(),
        })
        # Should go through the else branch (not ConceptMetrics)
        learner.log_metrics(coll)


if __name__ == '__main__':
    unittest.main()
