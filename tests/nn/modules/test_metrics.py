"""
Comprehensive tests for torch_concepts.nn.modules.metrics

Tests metrics module for concept-based models:
- Completeness score, intervention score, CACE score (functional metrics)
- ConceptMetrics: Unified metric tracking for different concept types
- Edge cases, error handling, and advanced scenarios
- Integration with PyTorch Lightning workflows
"""
import unittest
import torch
import torchmetrics
from sklearn.metrics import f1_score

from torch_concepts.nn.functional import completeness_score, intervention_score, cace_score
from torch_concepts.nn.modules.metrics import ConceptMetrics, Metric
from torch_concepts.annotations import AxisAnnotation, Annotations


class ANDModel(torch.nn.Module):
    """Helper model for testing intervention scores."""
    
    def __init__(self):
        super(ANDModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1, bias=True)

        # Manually set weights and bias to perform AND operation
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(torch.tensor([[1.0, 1.0]]))
            self.linear.bias = torch.nn.Parameter(torch.tensor([-1.5]))

    def forward(self, x):
        return self.linear(x)


class TestCompletenessScore(unittest.TestCase):
    """Test completeness score metric."""
    def test_completeness_score_accuracy(self):
        y_true = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
        y_pred_blackbox = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
        y_pred_whitebox = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])

        score = completeness_score(y_true, y_pred_blackbox, y_pred_whitebox, scorer=f1_score)
        self.assertAlmostEqual(score, 1.0, places=2, msg="Completeness score with f1_score should be 1.0")

    def test_completeness_score_f1(self):
        y_true = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 2])
        y_pred_blackbox = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 2])
        y_pred_whitebox = torch.tensor([0, 1, 2, 2, 1, 0, 2, 1, 1])

        score = completeness_score(y_true, y_pred_blackbox, y_pred_whitebox, scorer=f1_score)
        self.assertAlmostEqual(score, 0.3, places=1, msg="Completeness score with f1_score should be 0.0")

    def test_completeness_score_higher_than_1(self):
        y_true = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])
        y_pred_blackbox = torch.tensor([0, 1, 1, 1, 0, 2, 1, 2])
        y_pred_whitebox = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])

        score = completeness_score(y_true, y_pred_blackbox, y_pred_whitebox, scorer=f1_score)
        self.assertTrue(score > 1, msg="Completeness score should be higher than 1 when the whitebox model is better than the blackbox model")

class TestInterventionScore(unittest.TestCase):
    """Test intervention score metric."""

    def test_intervention_score_basic(self):
        y_predictor = ANDModel()
        c_true = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        c_pred = torch.FloatTensor([[.8, .2], [.8, .8], [.8, .2], [.8, .8]])
        y_true = torch.tensor([0, 0, 0, 1])
        intervention_groups = [[], [0], [1]]

        scores = intervention_score(y_predictor, c_pred, c_true, y_true, intervention_groups, auc=False)
        self.assertTrue(isinstance(scores, list))
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores[1], 1.0)

        auc_score = intervention_score(y_predictor, c_pred, c_true, y_true, intervention_groups, auc=True)
        self.assertTrue(isinstance(auc_score, float))
        self.assertEqual(round(auc_score*100)/100, 0.89)


class TestCaceScore(unittest.TestCase):
    """Test CACE (Concept Activation Causal Effect) score metric."""
    
    def test_cace_score_basic(self):
        y_pred_c0 = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
        y_pred_c1 = torch.tensor([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])
        expected_result = torch.tensor([0.15, 0.1, -0.25])
        result = cace_score(y_pred_c0, y_pred_c1)
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-6))

    def test_cace_score_zero_effect(self):
        y_pred_c0 = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
        y_pred_c1 = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
        expected_result = torch.tensor([0.0, 0.0, 0.0])
        result = cace_score(y_pred_c0, y_pred_c1)
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-6))

    def test_cace_score_negative_effect(self):
        y_pred_c0 = torch.tensor([[0.3, 0.4, 0.3], [0.4, 0.3, 0.3]])
        y_pred_c1 = torch.tensor([[0.1, 0.1, 0.8], [0.2, 0.1, 0.7]])
        expected_result = torch.tensor([-0.2, -0.25, 0.45])
        result = cace_score(y_pred_c0, y_pred_c1)
        self.assertTrue(torch.allclose(result, expected_result, atol=1e-6))

    def test_cace_score_different_shapes(self):
        y_pred_c0 = torch.tensor([[0.3, 0.4, 0.3], [0.4, 0.3, 0.3]])
        y_pred_c1 = torch.tensor([[0.1, 0.1, 0.8]])
        with self.assertRaises(RuntimeError):
            cace_score(y_pred_c0, y_pred_c1)


class TestConceptMetricsModule(unittest.TestCase):
    """Test metrics module structure and imports."""

    def test_module_imports(self):
        """Test that metrics module can be imported."""
        from torch_concepts.nn.modules import metrics
        self.assertIsNotNone(metrics)

    def test_module_has_metric_class(self):
        """Test that Metric base class is accessible."""
        self.assertIsNotNone(Metric)

    def test_placeholder(self):
        """Placeholder test for commented out code."""
        self.assertTrue(True)


class TestComputeCace(unittest.TestCase):
    """Test compute_cace utility function."""

    def setUp(self):
        """Build a minimal CBM and fake dataloader."""
        from torch.distributions import Bernoulli
        from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel

        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli},
                }
            )
        })
        self.model = ConceptBottleneckModel(
            input_size=4, annotations=ann, task_names=['task']
        )
        x = torch.randn(8, 4)
        self.dataloader = [{"inputs": {"x": x[:4]}, "concepts": {}},
                           {"inputs": {"x": x[4:]}, "concepts": {}}]

    def test_basic(self):
        """Returns a scalar tensor."""
        from torch_concepts.nn.modules.metrics import compute_cace
        result = compute_cace(
            model=self.model,
            dataloader=self.dataloader,
            source_concept='c1',
            target_concept='task',
        )
        self.assertEqual(result.dim(), 0)

    def test_custom_v_high_v_low(self):
        """Custom intervention values."""
        from torch_concepts.nn.modules.metrics import compute_cace
        result = compute_cace(
            model=self.model,
            dataloader=self.dataloader,
            source_concept='c1',
            target_concept='task',
            prob_high=0.8,
            prob_low=0.2,
        )
        self.assertEqual(result.dim(), 0)

    def test_empty_dataloader_raises(self):
        """Empty dataloader should raise ValueError."""
        from torch_concepts.nn.modules.metrics import compute_cace
        with self.assertRaises(ValueError):
            compute_cace(
                model=self.model,
                dataloader=[],
                source_concept='c1',
                target_concept='task',
            )

    def test_identical_interventions_give_zero(self):
        """do(C=v) vs do(C=v) should yield zero CaCE."""
        from torch_concepts.nn.modules.metrics import compute_cace
        result = compute_cace(
            model=self.model,
            dataloader=self.dataloader,
            source_concept='c1',
            target_concept='task',
            prob_high=0.5,
            prob_low=0.5,
        )
        self.assertTrue(torch.allclose(result, torch.tensor(0.0), atol=1e-6))

    def test_restores_training_mode(self):
        """Model training mode is restored after compute_cace."""
        from torch_concepts.nn.modules.metrics import compute_cace
        self.model.train()
        compute_cace(
            model=self.model,
            dataloader=self.dataloader,
            source_concept='c1',
            target_concept='task',
        )
        self.assertTrue(self.model.training)

    def test_keeps_eval_mode(self):
        """If model was in eval mode, stays in eval mode."""
        from torch_concepts.nn.modules.metrics import compute_cace
        self.model.eval()
        compute_cace(
            model=self.model,
            dataloader=self.dataloader,
            source_concept='c1',
            target_concept='task',
        )
        self.assertFalse(self.model.training)


class TestConceptMetrics(unittest.TestCase):
    """Test ConceptMetrics for unified metric tracking."""

    def setUp(self):
        """Set up test fixtures."""
        # Create annotations with mixed concept types (binary and categorical only)
        axis_mixed = AxisAnnotation(
            labels=('binary1', 'binary2', 'cat1', 'cat2'),
            cardinalities=[1, 1, 3, 4],
            metadata={
                'binary1': {'type': 'discrete'},
                'binary2': {'type': 'discrete'},
                'cat1': {'type': 'discrete'},
                'cat2': {'type': 'discrete'},
            }
        )
        self.annotations_mixed = Annotations({1: axis_mixed})
        
        # All binary
        axis_binary = AxisAnnotation(
            labels=('b1', 'b2', 'b3'),
            cardinalities=[1, 1, 1],
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'},
                'b3': {'type': 'discrete'},
            }
        )
        self.annotations_binary = Annotations({1: axis_binary})
        
        # All categorical
        axis_categorical = AxisAnnotation(
            labels=('cat1', 'cat2'),
            cardinalities=(3, 5),
            metadata={
                'cat1': {'type': 'discrete'},
                'cat2': {'type': 'discrete'},
            }
        )
        self.annotations_categorical = Annotations({1: axis_categorical})

    def test_binary_only_metrics(self):
        """Test ConceptMetrics with only binary concepts."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )
        
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        metrics.update(preds=endogenous, target=targets)
        result = metrics.compute()
        
        self.assertIn('train/SUMMARY-binary_accuracy', result)
        self.assertIsInstance(result['train/SUMMARY-binary_accuracy'], torch.Tensor)
        self.assertTrue(0 <= result['train/SUMMARY-binary_accuracy'] <= 1)

    def test_categorical_only_metrics(self):
        """Test ConceptMetrics with only categorical concepts."""
        metrics = ConceptMetrics(
            self.annotations_categorical,
            categorical={
                'accuracy': torchmetrics.classification.MulticlassAccuracy(
                    num_classes=5, average='micro'
                )
            },
            summary=True,
            prefix='val'
        )
        
        endogenous = torch.randn(16, 8)
        targets = torch.cat([
            torch.randint(0, 3, (16, 1)),
            torch.randint(0, 5, (16, 1))
        ], dim=1)
        
        metrics.update(preds=endogenous, target=targets)
        result = metrics.compute()
        
        self.assertIn('val/SUMMARY-categorical_accuracy', result)
        self.assertTrue(0 <= result['val/SUMMARY-categorical_accuracy'] <= 1)

    def test_mixed_concepts_metrics(self):
        """Test ConceptMetrics with mixed concept types."""
        metrics = ConceptMetrics(
            self.annotations_mixed,
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy(),
                'f1': torchmetrics.classification.BinaryF1Score()
            },
            categorical={
                'accuracy': torchmetrics.classification.MulticlassAccuracy(
                    num_classes=4, average='micro'
                )
            },
            summary=True,
            prefix='test'
        )
        
        endogenous = torch.randn(16, 9)
        targets = torch.cat([
            torch.randint(0, 2, (16, 2)).float(),
            torch.randint(0, 3, (16, 1)),
            torch.randint(0, 4, (16, 1)),
        ], dim=1)
        
        metrics.update(preds=endogenous, target=targets)
        result = metrics.compute()
        
        self.assertIn('test/SUMMARY-binary_accuracy', result)
        self.assertIn('test/SUMMARY-binary_f1', result)
        self.assertIn('test/SUMMARY-categorical_accuracy', result)

    def test_per_concept(self):
        """Test per-concept metric tracking."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=False,
            per_concept=['b1', 'b2'],
            prefix='train'
        )
        
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        metrics.update(preds=endogenous, target=targets)
        result = metrics.compute()
        
        self.assertIn('train/b1_accuracy', result)
        self.assertIn('train/b2_accuracy', result)
        self.assertNotIn('train/b3_accuracy', result)

    def test_summary_and_per_concept(self):
        """Test combining summary and per-concept metrics."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            per_concept=True,
            prefix='val'
        )
        
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        metrics.update(preds=endogenous, target=targets)
        result = metrics.compute()
        
        self.assertIn('val/SUMMARY-binary_accuracy', result)
        self.assertIn('val/b1_accuracy', result)
        self.assertIn('val/b2_accuracy', result)
        self.assertIn('val/b3_accuracy', result)

    def test_multiple_splits_via_clone(self):
        """Test independent tracking for train/val/test via clone."""
        base = ConceptMetrics(
            self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True
        )
        train_metrics = base.clone(prefix='train')
        val_metrics = base.clone(prefix='val')
        
        torch.manual_seed(42)
        train_endogenous = torch.randn(16, 3)
        train_targets = torch.randint(0, 2, (16, 3)).float()
        
        torch.manual_seed(43)
        val_endogenous = torch.randn(16, 3)
        val_targets = torch.randint(0, 2, (16, 3)).float()
        
        train_metrics.update(preds=train_endogenous, target=train_targets)
        val_metrics.update(preds=val_endogenous, target=val_targets)
        
        train_result = train_metrics.compute()
        val_result = val_metrics.compute()
        
        self.assertIn('train/SUMMARY-binary_accuracy', train_result)
        self.assertIn('val/SUMMARY-binary_accuracy', val_result)

    def test_reset_metrics(self):
        """Test metric reset functionality."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )
        
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        metrics.update(preds=endogenous, target=targets)
        result1 = metrics.compute()
        
        metrics.reset()
        endogenous2 = torch.randn(16, 3)
        targets2 = torch.randint(0, 2, (16, 3)).float()
        metrics.update(preds=endogenous2, target=targets2)
        result2 = metrics.compute()
        
        self.assertIsInstance(result1['train/SUMMARY-binary_accuracy'], torch.Tensor)
        self.assertIsInstance(result2['train/SUMMARY-binary_accuracy'], torch.Tensor)

    def test_missing_required_metrics(self):
        """Test that missing required metrics raises error."""
        with self.assertRaises(ValueError):
            ConceptMetrics(
                self.annotations_binary,
                categorical={
                    'accuracy': torchmetrics.classification.MulticlassAccuracy(
                        num_classes=3, average='micro'
                    )
                },
                summary=True
            )

    def test_unused_metrics_warning(self):
        """Test that unused metrics produce warnings."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConceptMetrics(
                self.annotations_binary,
                binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
                continuous={'mse': torchmetrics.regression.MeanSquaredError()},
                summary=True
            )
            self.assertTrue(any("continuous" in str(warning.message).lower() 
                              for warning in w))

    def test_metric_class_with_kwargs(self):
        """Test passing metric class with user kwargs as tuple."""
        metrics = ConceptMetrics(
            self.annotations_categorical,
            categorical={
                'accuracy': (
                    torchmetrics.classification.MulticlassAccuracy,
                    {'average': 'macro'}
                )
            },
            summary=True,
            prefix='train'
        )
        
        endogenous = torch.randn(16, 8)
        targets = torch.cat([
            torch.randint(0, 3, (16, 1)),
            torch.randint(0, 5, (16, 1))
        ], dim=1)
        
        metrics.update(preds=endogenous, target=targets)
        result = metrics.compute()
        
        self.assertIn('train/SUMMARY-categorical_accuracy', result)
        self.assertTrue(0 <= result['train/SUMMARY-categorical_accuracy'] <= 1)

    def test_metric_class_without_kwargs(self):
        """Test passing just metric class (no instantiation)."""
        metrics = ConceptMetrics(
            self.annotations_categorical,
            categorical={'accuracy': torchmetrics.classification.MulticlassAccuracy},
            summary=True,
            prefix='val'
        )
        
        endogenous = torch.randn(16, 8)
        targets = torch.cat([
            torch.randint(0, 3, (16, 1)),
            torch.randint(0, 5, (16, 1))
        ], dim=1)
        
        metrics.update(preds=endogenous, target=targets)
        result = metrics.compute()
        
        self.assertIn('val/SUMMARY-categorical_accuracy', result)

    def test_mixed_metric_specs(self):
        """Test mixing instantiated, class+kwargs, and class-only metrics."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy(),
                'f1': (torchmetrics.classification.BinaryF1Score, {'threshold': 0.5}),
                'precision': torchmetrics.classification.BinaryPrecision
            },
            summary=True,
            prefix='test'
        )
        
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        metrics.update(preds=endogenous, target=targets)
        result = metrics.compute()
        
        self.assertIn('test/SUMMARY-binary_accuracy', result)
        self.assertIn('test/SUMMARY-binary_f1', result)
        self.assertIn('test/SUMMARY-binary_precision', result)

    def test_num_classes_in_kwargs_raises_error(self):
        """Test that providing num_classes in kwargs raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            ConceptMetrics(
                self.annotations_categorical,
                categorical={
                    'accuracy': (
                        torchmetrics.classification.MulticlassAccuracy,
                        {'num_classes': 10, 'average': 'macro'}
                    )
                },
                summary=True
            )
        
        self.assertIn('num_classes', str(cm.exception))
        self.assertIn('automatically', str(cm.exception).lower())


class TestConceptMetricsEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in ConceptMetrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        axis_binary = AxisAnnotation(
            labels=('b1', 'b2'),
            cardinalities=[1, 1],
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'}
            }
        )
        self.annotations_binary = Annotations({1: axis_binary})
    
    def test_empty_batch_update(self):
        """Test updating with empty batch."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )
        
        endogenous = torch.randn(0, 2)
        targets = torch.randint(0, 2, (0, 2)).float()
        
        metrics.update(preds=endogenous, target=targets)
        result = metrics.compute()
        
        self.assertIn('train/SUMMARY-binary_accuracy', result)
    
    def test_single_sample_batch(self):
        """Test with batch size of 1."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )
        
        endogenous = torch.randn(1, 2)
        targets = torch.randint(0, 2, (1, 2)).float()
        
        metrics.update(preds=endogenous, target=targets)
        result = metrics.compute()
        
        self.assertIn('train/SUMMARY-binary_accuracy', result)
        self.assertTrue(0 <= result['train/SUMMARY-binary_accuracy'] <= 1)
    
    def test_very_large_batch(self):
        """Test with large batch size."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )
        
        batch_size = 10000
        endogenous = torch.randn(batch_size, 2)
        targets = torch.randint(0, 2, (batch_size, 2)).float()
        
        metrics.update(preds=endogenous, target=targets)
        result = metrics.compute()
        
        self.assertIn('train/SUMMARY-binary_accuracy', result)
    
    def test_no_prefix(self):
        """Test creating metrics without prefix."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True
        )
        
        endogenous = torch.randn(16, 2)
        targets = torch.randint(0, 2, (16, 2)).float()
        
        metrics.update(preds=endogenous, target=targets)
        result = metrics.compute()
        
        self.assertIn('SUMMARY-binary_accuracy', result)
    
    def test_empty_collection(self):
        """Test creating metrics with empty config."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={},
            summary=True
        )
        
        self.assertEqual(len(metrics.collection), 0)
    
    def test_perconcept_invalid_name(self):
        """Test that invalid concept names in per_concept are handled."""
        with self.assertRaises(ValueError):
            ConceptMetrics(
                self.annotations_binary,
                binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
                summary=True,
                per_concept=['nonexistent_concept']
            )
    
    def test_perconcept_invalid_type(self):
        """Test that invalid type for per_concept raises error."""
        with self.assertRaises(ValueError):
            ConceptMetrics(
                self.annotations_binary,
                binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
                summary=True,
                per_concept="invalid_string"
            )


class TestConceptMetricsAccuracy(unittest.TestCase):
    """Test that metrics compute accurate values."""
    
    def setUp(self):
        """Set up test fixtures."""
        axis_binary = AxisAnnotation(
            labels=('b1', 'b2'),
            cardinalities=[1, 1],
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'}
            }
        )
        self.annotations_binary = Annotations({1: axis_binary})
    
    def test_perfect_accuracy(self):
        """Test that perfect predictions give 100% accuracy."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )
        
        torch.manual_seed(42)
        targets = torch.randint(0, 2, (32, 2)).float()
        predictions = targets.clone()
        
        metrics.update(preds=predictions, target=targets)
        result = metrics.compute()
        
        self.assertAlmostEqual(
            result['train/SUMMARY-binary_accuracy'].item(), 
            1.0, 
            places=5
        )
    
    def test_zero_accuracy(self):
        """Test that completely wrong predictions give 0% accuracy."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )
        
        torch.manual_seed(42)
        targets = torch.randint(0, 2, (32, 2)).float()
        predictions = 1 - targets
        
        metrics.update(preds=predictions, target=targets)
        result = metrics.compute()
        
        self.assertAlmostEqual(
            result['train/SUMMARY-binary_accuracy'].item(), 
            0.0, 
            places=5
        )
    
    def test_known_accuracy_value(self):
        """Test with known accuracy value."""
        metrics = ConceptMetrics(
            self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )
        
        targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        predictions = torch.tensor([[1.0, 1.0], [1.0, 0.0]])  # 3 out of 4
        
        metrics.update(preds=predictions, target=targets)
        result = metrics.compute()
        
        self.assertAlmostEqual(
            result['train/SUMMARY-binary_accuracy'].item(), 
            0.75, 
            places=5
        )


class TestConceptMetricsMultipleBatches(unittest.TestCase):
    """Test metrics with multiple batch updates."""
    
    def setUp(self):
        """Set up test fixtures."""
        axis_binary = AxisAnnotation(
            labels=('b1',),
            cardinalities=[1],
            metadata={'b1': {'type': 'discrete'}}
        )
        self.annotations = Annotations({1: axis_binary})
    
    def test_accumulation_across_batches(self):
        """Test that metrics correctly accumulate across batches."""
        metrics = ConceptMetrics(
            self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )
        
        targets1 = torch.tensor([[1.0], [1.0]])
        preds1 = torch.tensor([[1.0], [1.0]])
        
        targets2 = torch.tensor([[1.0], [1.0]])
        preds2 = torch.tensor([[0.0], [0.0]])
        
        metrics.update(preds=preds1, target=targets1)
        metrics.update(preds=preds2, target=targets2)
        
        result = metrics.compute()
        
        self.assertAlmostEqual(
            result['train/SUMMARY-binary_accuracy'].item(),
            0.5,
            places=5
        )
    
    def test_reset_clears_accumulation(self):
        """Test that reset clears accumulated state."""
        metrics = ConceptMetrics(
            self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )
        
        targets1 = torch.tensor([[1.0], [1.0]])
        preds1 = torch.tensor([[0.0], [0.0]])
        
        metrics.update(preds=preds1, target=targets1)
        result1 = metrics.compute()
        self.assertAlmostEqual(result1['train/SUMMARY-binary_accuracy'].item(), 0.0)
        
        metrics.reset()
        
        targets2 = torch.tensor([[1.0], [1.0]])
        preds2 = torch.tensor([[1.0], [1.0]])
        
        metrics.update(preds=preds2, target=targets2)
        result2 = metrics.compute()
        
        self.assertAlmostEqual(result2['train/SUMMARY-binary_accuracy'].item(), 1.0)


class TestConceptMetricsRepr(unittest.TestCase):
    """Test string representations and display methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        axis_binary = AxisAnnotation(
            labels=('b1', 'b2'),
            cardinalities=[1, 1],
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'}
            }
        )
        self.annotations = Annotations({1: axis_binary})
    
    def test_repr_with_metrics(self):
        """Test __repr__ method."""
        metrics = ConceptMetrics(
            self.annotations,
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy(),
                'f1': torchmetrics.classification.BinaryF1Score()
            },
            summary=True,
            per_concept=False
        )
        
        repr_str = repr(metrics)
        
        self.assertIn('ConceptMetrics', repr_str)
        self.assertIn('n_concepts=2', repr_str)
        self.assertIn('summary=True', repr_str)
        self.assertIn('BinaryAccuracy', repr_str)
        self.assertIn('BinaryF1Score', repr_str)
    
    def test_repr_with_mixed_metric_specs(self):
        """Test __repr__ with different metric specification methods."""
        metrics = ConceptMetrics(
            self.annotations,
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy(),
                'f1': (torchmetrics.classification.BinaryF1Score, {}),
                'precision': torchmetrics.classification.BinaryPrecision
            },
            summary=True
        )
        
        repr_str = repr(metrics)
        
        self.assertIn('BinaryAccuracy', repr_str)
        self.assertIn('BinaryF1Score', repr_str)
        self.assertIn('BinaryPrecision', repr_str)


class TestConceptMetricsClone(unittest.TestCase):
    """Test the clone() method for creating independent copies."""
    
    def setUp(self):
        """Set up test fixtures."""
        axis_binary = AxisAnnotation(
            labels=('b1',),
            cardinalities=[1],
            metadata={'b1': {'type': 'discrete'}}
        )
        self.annotations = Annotations({1: axis_binary})
    
    def test_clone_with_prefix(self):
        """Test cloning with a new prefix."""
        base = ConceptMetrics(
            self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True
        )
        cloned = base.clone(prefix='train')
        
        targets = torch.tensor([[1.0], [1.0]])
        preds = torch.tensor([[1.0], [1.0]])
        cloned.update(preds=preds, target=targets)
        result = cloned.compute()
        
        self.assertIn('train/SUMMARY-binary_accuracy', result)
    
    def test_clones_are_independent(self):
        """Test that cloned instances have independent state."""
        base = ConceptMetrics(
            self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True
        )
        train = base.clone(prefix='train')
        val = base.clone(prefix='val')
        
        # Update only train
        targets = torch.tensor([[1.0], [1.0]])
        preds = torch.tensor([[1.0], [1.0]])
        train.update(preds=preds, target=targets)
        
        train_result = train.compute()
        val_result = val.compute()
        
        self.assertAlmostEqual(train_result['train/SUMMARY-binary_accuracy'].item(), 1.0)
        # val was never updated, so its result should be the default (0.0 for accuracy)
        self.assertIn('val/SUMMARY-binary_accuracy', val_result)
    
    def test_collection_property(self):
        """Test the collection property returns non-empty sub-collections."""
        metrics = ConceptMetrics(
            self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True
        )
        
        coll = metrics.collection
        self.assertIn('binary', coll)
        self.assertTrue(len(coll) > 0)


class TestConceptMetricsIntegration(unittest.TestCase):
    """Integration tests simulating real training scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        axis_mixed = AxisAnnotation(
            labels=('binary1', 'binary2', 'cat1'),
            cardinalities=[1, 1, 3],
            metadata={
                'binary1': {'type': 'discrete'},
                'binary2': {'type': 'discrete'},
                'cat1': {'type': 'discrete'}
            }
        )
        self.annotations = Annotations({1: axis_mixed})
    
    def test_full_training_epoch_simulation(self):
        """Simulate a complete training epoch with multiple batches."""
        base = ConceptMetrics(
            self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            categorical={'accuracy': torchmetrics.classification.MulticlassAccuracy},
            summary=True,
            per_concept=True
        )
        train_metrics = base.clone(prefix='train')
        
        num_batches = 10
        batch_size = 32
        
        for _ in range(num_batches):
            predictions = torch.randn(batch_size, 5)
            targets = torch.cat([
                torch.randint(0, 2, (batch_size, 2)),
                torch.randint(0, 3, (batch_size, 1))
            ], dim=1)
            
            train_metrics.update(preds=predictions, target=targets)
        
        results = train_metrics.compute()
        
        self.assertIn('train/SUMMARY-binary_accuracy', results)
        self.assertIn('train/SUMMARY-categorical_accuracy', results)
        self.assertIn('train/binary1_accuracy', results)
        self.assertIn('train/binary2_accuracy', results)
        self.assertIn('train/cat1_accuracy', results)
        
        train_metrics.reset()
        
        results_after_reset = train_metrics.compute()
        self.assertIn('train/SUMMARY-binary_accuracy', results_after_reset)
    
    def test_train_val_test_workflow(self):
        """Simulate complete train/val/test workflow."""
        base = ConceptMetrics(
            self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            categorical={'accuracy': torchmetrics.classification.MulticlassAccuracy},
            summary=True
        )
        train_metrics = base.clone(prefix='train')
        val_metrics = base.clone(prefix='val')
        test_metrics = base.clone(prefix='test')
        
        batch_size = 16
        
        for _ in range(5):
            predictions = torch.randn(batch_size, 5)
            targets = torch.cat([
                torch.randint(0, 2, (batch_size, 2)),
                torch.randint(0, 3, (batch_size, 1))
            ], dim=1)
            train_metrics.update(preds=predictions, target=targets)
        
        for _ in range(2):
            predictions = torch.randn(batch_size, 5)
            targets = torch.cat([
                torch.randint(0, 2, (batch_size, 2)),
                torch.randint(0, 3, (batch_size, 1))
            ], dim=1)
            val_metrics.update(preds=predictions, target=targets)
        
        for _ in range(3):
            predictions = torch.randn(batch_size, 5)
            targets = torch.cat([
                torch.randint(0, 2, (batch_size, 2)),
                torch.randint(0, 3, (batch_size, 1))
            ], dim=1)
            test_metrics.update(preds=predictions, target=targets)
        
        train_results = train_metrics.compute()
        val_results = val_metrics.compute()
        test_results = test_metrics.compute()
        
        self.assertIn('train/SUMMARY-binary_accuracy', train_results)
        self.assertIn('val/SUMMARY-binary_accuracy', val_results)
        self.assertIn('test/SUMMARY-binary_accuracy', test_results)


if __name__ == '__main__':
    unittest.main()
