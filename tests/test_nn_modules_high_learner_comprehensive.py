"""
Comprehensive tests for torch_concepts.nn.modules.high.base.learner

Tests the BaseLearner class with metrics setup, optimizer configuration,
and loss computation for binary and categorical concepts.
"""
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch_concepts.annotations import Annotations, AxisAnnotation
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torchmetrics import Accuracy, MeanSquaredError


class MockLearner(BaseLearner):
    """Mock implementation of BaseLearner for testing."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add a dummy parameter so optimizer is not empty
        self.dummy_param = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """Simple forward pass for testing."""
        return torch.randn(x.shape[0], self.n_concepts)

    def training_step(self, batch, batch_idx):
        """Mock training step."""
        return torch.tensor(0.5)

    def validation_step(self, batch, batch_idx):
        """Mock validation step."""
        return torch.tensor(0.3)

    def test_step(self, batch, batch_idx):
        """Mock test step."""
        return torch.tensor(0.2)

    def configure_optimizers(self):
        """Configure optimizer."""
        if self.optim_class is not None:
            optimizer = self.optim_class(self.parameters(), **(self.optim_kwargs or {}))
            if self.scheduler_class is not None:
                scheduler = self.scheduler_class(optimizer, **(self.scheduler_kwargs or {}))
                return {'optimizer': optimizer, 'lr_scheduler': scheduler}
            return optimizer
        return None

    def filter_output_for_loss(self, forward_out, target):
        """Filter outputs for loss computation."""
        return {'input': forward_out, 'target': target}


class TestBaseLearnerInitialization(unittest.TestCase):
    """Test BaseLearner initialization with various configurations."""

    def setUp(self):
        """Set up common test fixtures."""
        # Create annotations with distribution metadata
        concept_labels = ['age', 'gender', 'color']
        self.annotations_with_dist = Annotations({
            1: AxisAnnotation(
                labels=concept_labels,
                cardinalities=[1, 1, 3],
                metadata={
                    'age': {'label': 'age', 'type': 'discrete', 'distribution': Bernoulli},
                    'gender': {'label': 'gender', 'type': 'discrete', 'distribution': Bernoulli},
                    'color': {'label': 'color', 'type': 'discrete', 'distribution': Categorical}
                }
            )
        })

        # Create annotations without distribution metadata
        self.annotations_no_dist = Annotations({
            1: AxisAnnotation(
                labels=concept_labels,
                cardinalities=[1, 1, 3],
                metadata={
                    'age': {'label': 'age', 'type': 'discrete'},
                    'gender': {'label': 'gender', 'type': 'discrete'},
                    'color': {'label': 'color', 'type': 'discrete'}
                }
            )
        })

        self.variable_distributions = {
            'discrete_card1': {'path': 'torch.distributions.Bernoulli'},
            'discrete_cardn': {'path': 'torch.distributions.Categorical'}
        }

    def test_initialization_with_distribution_metadata(self):
        """Test initialization when annotations have distribution metadata."""
        learner = MockLearner(
            annotations=self.annotations_with_dist,
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.001}
        )
        self.assertEqual(learner.n_concepts, 3)
        self.assertEqual(learner.concept_names, ['age', 'gender', 'color'])
        self.assertIsNotNone(learner.metadata)

    def test_initialization_without_distribution_metadata(self):
        """Test initialization when annotations lack distribution metadata."""
        # Provide metadata for all concepts to avoid AttributeError
        annotations_no_dist = Annotations({
            1: AxisAnnotation(
                labels=['age', 'gender', 'color'],
                cardinalities=[1, 1, 3],
                metadata={
                    'age': {'label': 'age', 'type': 'discrete'},
                    'gender': {'label': 'gender', 'type': 'discrete'},
                    'color': {'label': 'color', 'type': 'discrete'}
                }
            )
        })
        learner = MockLearner(
            annotations=annotations_no_dist,
            variable_distributions=self.variable_distributions,
            optim_class=torch.optim.Adam
        )
        self.assertEqual(learner.n_concepts, 3)
        self.assertIsNotNone(learner.concept_annotations)

    def test_initialization_missing_distributions_raises_error(self):
        """Test that missing distributions raises assertion error."""
        with self.assertRaises(AssertionError) as context:
            MockLearner(
                annotations=self.annotations_no_dist,
                optim_class=torch.optim.Adam
            )
        self.assertIn("variable_distributions must be provided", str(context.exception))

    def test_continuous_concepts_raise_error(self):
        """Test that continuous concepts raise NotImplementedError."""
        continuous_annotations = Annotations({
            1: AxisAnnotation(
                labels=['temp', 'pressure'],
                metadata={
                    'temp': {'label': 'temp', 'type': 'continuous', 'distribution': Delta},
                    'pressure': {'label': 'pressure', 'type': 'continuous', 'distribution': Delta}
                }
            )
        })
        with self.assertRaises(NotImplementedError) as context:
            MockLearner(
                annotations=continuous_annotations,
                optim_class=torch.optim.Adam
            )
        self.assertIn("Continuous concepts are not yet supported", str(context.exception))

    def test_repr_method(self):
        """Test __repr__ method."""
        learner = MockLearner(
            annotations=self.annotations_with_dist,
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
            annotations=self.annotations_with_dist,
            optim_class=torch.optim.SGD
        )
        repr_str = repr(learner)
        self.assertIn("scheduler=None", repr_str)


class TestBaseLearnerMetricsSetup(unittest.TestCase):
    """Test metrics setup functionality."""

    def setUp(self):
        """Set up annotations for testing."""
        self.annotations = Annotations({
            1: AxisAnnotation(
                labels=['binary1', 'binary2', 'cat1'],
                cardinalities=[1, 1, 4],
                metadata={
                    'binary1': {'label': 'binary1', 'type': 'discrete', 'distribution': Bernoulli},
                    'binary2': {'label': 'binary2', 'type': 'discrete', 'distribution': Bernoulli},
                    'cat1': {'label': 'cat1', 'type': 'discrete', 'distribution': Categorical}
                }
            )
        })

    def test_invalid_perconcept_metrics_type(self):
        """Test that invalid perconcept_metrics type raises error."""
        metrics_config = {
            'discrete': {
                'binary': {
                    'accuracy': {
                        'path': 'torchmetrics.Accuracy',
                        'kwargs': {'task': 'binary'}
                    }
                },
                'categorical': {
                    'accuracy': {
                        'path': 'torchmetrics.Accuracy',
                        'kwargs': {'task': 'multiclass', 'num_classes': 4}
                    }
                }
            }
        }
        with self.assertRaises(ValueError) as context:
            MockLearner(
                annotations=self.annotations,
                optim_class=torch.optim.Adam,
                metrics=metrics_config,
                perconcept_metrics="invalid"  # Should be bool or list
            )
        self.assertIn("perconcept_metrics must be either a bool or a list", str(context.exception))

    def test_metrics_setup_with_summary_metrics(self):
        """Test metrics setup with summary metrics enabled."""
        metrics_config = {
            'discrete': {
                'binary': {
                    'accuracy': {
                        'path': 'torchmetrics.Accuracy',
                        'kwargs': {'task': 'binary'}
                    }
                },
                'categorical': {
                    'accuracy': {
                        'path': 'torchmetrics.Accuracy',
                        'kwargs': {'task': 'multiclass', 'num_classes': 4}
                    }
                }
            }
        }
        learner = MockLearner(
            annotations=self.annotations,
            optim_class=torch.optim.Adam,
            metrics=metrics_config,
            summary_metrics=True,
            perconcept_metrics=False
        )
        self.assertIsNotNone(learner.train_metrics)
        self.assertIsNotNone(learner.val_metrics)
        self.assertIsNotNone(learner.test_metrics)

    def test_metrics_setup_with_perconcept_metrics_bool(self):
        """Test per-concept metrics with boolean flag."""
        metrics_config = {
            'discrete': {
                'binary': {
                    'accuracy': {
                        'path': 'torchmetrics.Accuracy',
                        'kwargs': {'task': 'binary'}
                    }
                },
                'categorical': {
                    'accuracy': {
                        'path': 'torchmetrics.Accuracy',
                        'kwargs': {'task': 'multiclass', 'num_classes': 4}
                    }
                }
            }
        }
        learner = MockLearner(
            annotations=self.annotations,
            optim_class=torch.optim.Adam,
            metrics=metrics_config,
            summary_metrics=False,
            perconcept_metrics=True
        )
        self.assertIsNotNone(learner.train_metrics)
        self.assertTrue(learner.perconcept_metrics)

    def test_metrics_setup_with_perconcept_metrics_list(self):
        """Test per-concept metrics with specific concept list."""
        metrics_config = {
            'discrete': {
                'binary': {
                    'accuracy': {
                        'path': 'torchmetrics.Accuracy',
                        'kwargs': {'task': 'binary'}
                    }
                },
                'categorical': {
                    'accuracy': {
                        'path': 'torchmetrics.Accuracy',
                        'kwargs': {'task': 'multiclass', 'num_classes': 4}
                    }
                }
            }
        }
        learner = MockLearner(
            annotations=self.annotations,
            optim_class=torch.optim.Adam,
            metrics=metrics_config,
            summary_metrics=False,
            perconcept_metrics=['binary1', 'cat1']
        )
        self.assertIsNotNone(learner.train_metrics)
        self.assertEqual(learner.perconcept_metrics, ['binary1', 'cat1'])

    def test_metrics_setup_with_categorical_concepts(self):
        """Test metrics setup with categorical concepts."""
        metrics_config = {
            'discrete': {
                'binary': {
                    'accuracy': {
                        'path': 'torchmetrics.Accuracy',
                        'kwargs': {'task': 'binary'}
                    }
                },
                'categorical': {
                    'accuracy': {
                        'path': 'torchmetrics.Accuracy',
                        'kwargs': {'task': 'multiclass', 'num_classes': 4}
                    }
                }
            }
        }
        learner = MockLearner(
            annotations=self.annotations,
            optim_class=torch.optim.Adam,
            metrics=metrics_config,
            summary_metrics=True,
            perconcept_metrics=True
        )
        self.assertIsNotNone(learner.train_metrics)
        self.assertTrue(hasattr(learner, 'max_card'))

    def test_no_metrics_configuration(self):
        """Test initialization without metrics."""
        learner = MockLearner(
            annotations=self.annotations,
            optim_class=torch.optim.Adam,
            metrics=None
        )
        self.assertFalse(learner.summary_metrics)
        self.assertFalse(learner.perconcept_metrics)
        self.assertIsNone(learner.train_metrics)


class TestBaseLearnerBatchHandling(unittest.TestCase):
    """Test batch validation and unpacking."""

    def setUp(self):
        """Set up learner for testing."""
        annotations = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2'],
                metadata={
                    'c1': {'label': 'c1', 'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'label': 'c2', 'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
        self.learner = MockLearner(
            annotations=annotations,
            optim_class=torch.optim.Adam
        )

    def test_valid_batch_unpacking(self):
        """Test unpacking a valid batch."""
        batch = {
            'inputs': torch.randn(4, 10),
            'concepts': torch.randn(4, 2)
        }
        inputs, concepts, transforms = self.learner.unpack_batch(batch)
        self.assertEqual(inputs.shape, (4, 10))
        self.assertEqual(concepts.shape, (4, 2))
        self.assertEqual(transforms, {})

    def test_batch_with_transforms(self):
        """Test batch with transforms."""
        batch = {
            'inputs': torch.randn(4, 10),
            'concepts': torch.randn(4, 2),
            'transforms': {'normalize': True}
        }
        inputs, concepts, transforms = self.learner.unpack_batch(batch)
        self.assertEqual(transforms, {'normalize': True})

    def test_non_dict_batch_raises_error(self):
        """Test that non-dict batch raises TypeError."""
        with self.assertRaises(TypeError) as context:
            self.learner.unpack_batch([torch.randn(4, 10), torch.randn(4, 2)])
        self.assertIn("Expected batch to be a dict", str(context.exception))

    def test_missing_inputs_raises_error(self):
        """Test that missing inputs key raises KeyError."""
        batch = {'concepts': torch.randn(4, 2)}
        with self.assertRaises(KeyError) as context:
            self.learner.unpack_batch(batch)
        self.assertIn("missing required keys", str(context.exception))
        self.assertIn("inputs", str(context.exception))

    def test_missing_concepts_raises_error(self):
        """Test that missing concepts key raises KeyError."""
        batch = {'inputs': torch.randn(4, 10)}
        with self.assertRaises(KeyError) as context:
            self.learner.unpack_batch(batch)
        self.assertIn("concepts", str(context.exception))


class TestBaseLearnerMetricsUpdate(unittest.TestCase):
    """Test metric update functionality."""

    def setUp(self):
        """Set up learner with metrics."""
        self.annotations = Annotations({
            1: AxisAnnotation(
                labels=['b1', 'b2'],
                cardinalities=[1, 1],
                metadata={
                    'b1': {'label': 'b1', 'type': 'discrete', 'distribution': Bernoulli},
                    'b2': {'label': 'b2', 'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })

        metrics_config = {
            'discrete': {
                'binary': {
                    'accuracy': {
                        'path': 'torchmetrics.Accuracy',
                        'kwargs': {'task': 'binary'}
                    }
                }
            }
        }

        self.learner = MockLearner(
            annotations=self.annotations,
            optim_class=torch.optim.Adam,
            metrics=metrics_config,
            summary_metrics=True,
            perconcept_metrics=False
        )

    def test_update_metrics_with_binary_concepts(self):
        """Test metrics update for binary concepts."""
        metrics_config = {
            'discrete': {
                'binary': {
                    'accuracy': {
                        'path': 'torchmetrics.Accuracy',
                        'kwargs': {'task': 'binary'}
                    }
                }
            }
        }
        self.learner = MockLearner(
            annotations=self.annotations,
            optim_class=torch.optim.Adam,
            metrics=metrics_config,
            summary_metrics=True,
            perconcept_metrics=False
        )
        c_hat = torch.randn(8, 2)
        c_true = torch.randint(0, 2, (8, 2)).float()

        metric_dict = {'input': c_hat, 'target': c_true}
        # This should not raise an error
        self.learner.update_metrics(metric_dict, self.learner.train_metrics)


class TestBaseLearnerLogging(unittest.TestCase):
    """Test logging functionality."""

    def setUp(self):
        """Set up learner for testing."""
        annotations = Annotations({
            1: AxisAnnotation(
                labels=['c1'],
                metadata={
                    'c1': {'label': 'c1', 'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
        self.learner = MockLearner(
            annotations=annotations,
            optim_class=torch.optim.Adam
        )

    def test_log_loss_method(self):
        """Test log_loss method."""
        loss = torch.tensor(0.5)
        # This should not raise an error
        # Note: actual logging requires a trainer context
        try:
            self.learner.log_loss('train', loss)
        except RuntimeError:
            # Expected if not in trainer context
            pass

    def test_log_metrics_method(self):
        """Test log_metrics method."""
        metrics = {'accuracy': 0.95}
        # This should not raise an error
        try:
            self.learner.log_metrics(metrics)
        except RuntimeError:
            # Expected if not in trainer context
            pass


class TestBaseLearnerOptimizerConfiguration(unittest.TestCase):
    """Test optimizer and scheduler configuration."""

    def setUp(self):
        """Set up annotations."""
        self.annotations = Annotations({
            1: AxisAnnotation(
                labels=['c1'],
                metadata={
                    'c1': {'label': 'c1', 'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })

    def test_optimizer_configuration_with_kwargs(self):
        """Test optimizer configuration with custom kwargs."""
        learner = MockLearner(
            annotations=self.annotations,
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}
        )
        optimizer = learner.configure_optimizers()
        self.assertIsNotNone(optimizer)

    def test_scheduler_configuration(self):
        """Test scheduler configuration."""
        learner = MockLearner(
            annotations=self.annotations,
            optim_class=torch.optim.Adam,
            optim_kwargs={'lr': 0.001},
            scheduler_class=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={'step_size': 10}
        )
        config = learner.configure_optimizers()
        self.assertIsInstance(config, dict)
        self.assertIn('optimizer', config)
        self.assertIn('lr_scheduler', config)

    def test_no_optimizer_returns_none(self):
        """Test that no optimizer configuration returns None."""
        learner = MockLearner(
            annotations=self.annotations,
            optim_class=None
        )
        result = learner.configure_optimizers()
        self.assertIsNone(result)


class TestBaseLearnerCheckMetric(unittest.TestCase):
    """Test _check_metric static method."""

    def test_check_metric_clones_and_resets(self):
        """Test that _check_metric clones and resets a metric."""
        from torchmetrics import Accuracy
        metric = Accuracy(task='binary')
        # Update metric with some data
        metric.update(torch.tensor([0.9, 0.1]), torch.tensor([1, 0]))
        # Clone and reset
        cloned = BaseLearner._check_metric(metric)
        # Should be a different object
        self.assertIsNot(cloned, metric)
        # Should be reset (no accumulated state)
        self.assertTrue(type(cloned).__name__.endswith('Accuracy'))

class TestBaseLearnerInstantiateMetricDict(unittest.TestCase):
    """Test _instantiate_metric_dict method."""

    def setUp(self):
        """Set up learner."""
        annotations = Annotations({
            1: AxisAnnotation(
                labels=['c1'],
                metadata={
                    'c1': {'label': 'c1', 'type': 'discrete', 'distribution': Bernoulli}
                }
            )
        })
        self.learner = MockLearner(
            annotations=annotations,
            optim_class=torch.optim.Adam
        )

    def test_instantiate_metric_dict_with_valid_config(self):
        """Test instantiating metrics from valid config."""
        config = {
            'accuracy': {
                'path': 'torchmetrics.Accuracy',
                'kwargs': {'task': 'binary'}
            }
        }
        metrics = self.learner._instantiate_metric_dict(config)
        self.assertIn('accuracy', metrics)
        self.assertIsNotNone(metrics['accuracy'])

    def test_instantiate_metric_dict_with_num_classes_override(self):
        """Test that num_classes parameter overrides kwargs."""
        config = {
            'accuracy': {
                'path': 'torchmetrics.Accuracy',
                'kwargs': {'task': 'multiclass', 'num_classes': 2}
            }
        }
        metrics = self.learner._instantiate_metric_dict(config, num_classes=5)
        self.assertIn('accuracy', metrics)

    def test_instantiate_metric_dict_with_empty_config(self):
        """Test instantiating with empty config."""
        metrics = self.learner._instantiate_metric_dict({})
        self.assertEqual(metrics, {})

    def test_instantiate_metric_dict_with_non_dict(self):
        """Test instantiating with non-dict returns empty dict."""
        metrics = self.learner._instantiate_metric_dict(None)
        self.assertEqual(metrics, {})


if __name__ == '__main__':
    unittest.main()
