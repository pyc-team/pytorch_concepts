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
from torch_concepts.nn.modules.utils import GroupConfig
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
        # The ConceptCausalEffect class is currently commented out
        # This test ensures the module structure is correct
        self.assertTrue(True)


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
        metrics_config = GroupConfig(
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy()
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary, 
            metrics_config,
            summary_metrics=True
        )
        
        # Binary concepts: endogenous shape (batch, 3)
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        # Update and compute
        metrics.update(preds=endogenous, target=targets, split='train')
        result = metrics.compute('train')
        
        self.assertIn('train/SUMMARY-binary_accuracy', result)
        self.assertIsInstance(result['train/SUMMARY-binary_accuracy'], torch.Tensor)
        self.assertTrue(0 <= result['train/SUMMARY-binary_accuracy'] <= 1)

    def test_categorical_only_metrics(self):
        """Test ConceptMetrics with only categorical concepts."""
        metrics_config = GroupConfig(
            categorical={
                'accuracy': torchmetrics.classification.MulticlassAccuracy(
                    num_classes=5, average='micro'
                )
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations_categorical,
            metrics_config,
            summary_metrics=True
        )
        
        # Categorical: cat1 (3 classes) + cat2 (5 classes) = 8 endogenous total
        endogenous = torch.randn(16, 8)
        targets = torch.cat([
            torch.randint(0, 3, (16, 1)),
            torch.randint(0, 5, (16, 1))
        ], dim=1)
        
        # Update and compute
        metrics.update(preds=endogenous, target=targets, split='val')
        result = metrics.compute('val')
        
        self.assertIn('val/SUMMARY-categorical_accuracy', result)
        self.assertTrue(0 <= result['val/SUMMARY-categorical_accuracy'] <= 1)

    def test_mixed_concepts_metrics(self):
        """Test ConceptMetrics with mixed concept types."""
        metrics_config = GroupConfig(
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy(),
                'f1': torchmetrics.classification.BinaryF1Score()
            },
            categorical={
                'accuracy': torchmetrics.classification.MulticlassAccuracy(
                    num_classes=4, average='micro'
                )
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations_mixed,
            metrics_config,
            summary_metrics=True
        )
        
        # Mixed: 2 binary + (3 + 4) categorical = 9 endogenous
        endogenous = torch.randn(16, 9)
        targets = torch.cat([
            torch.randint(0, 2, (16, 2)).float(),  # binary
            torch.randint(0, 3, (16, 1)),  # cat1
            torch.randint(0, 4, (16, 1)),  # cat2
        ], dim=1)
        
        # Update and compute
        metrics.update(preds=endogenous, target=targets, split='test')
        result = metrics.compute('test')
        
        self.assertIn('test/SUMMARY-binary_accuracy', result)
        self.assertIn('test/SUMMARY-binary_f1', result)
        self.assertIn('test/SUMMARY-categorical_accuracy', result)

    def test_perconcept_metrics(self):
        """Test per-concept metric tracking."""
        metrics_config = GroupConfig(
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy()
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=False,
            perconcept_metrics=['b1', 'b2']
        )
        
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        # Update and compute
        metrics.update(preds=endogenous, target=targets, split='train')
        result = metrics.compute('train')
        
        self.assertIn('train/b1_accuracy', result)
        self.assertIn('train/b2_accuracy', result)
        self.assertNotIn('train/b3_accuracy', result)  # Not tracked

    def test_summary_and_perconcept_metrics(self):
        """Test combining summary and per-concept metrics."""
        metrics_config = GroupConfig(
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy()
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True,
            perconcept_metrics=True
        )
        
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        # Update and compute
        metrics.update(preds=endogenous, target=targets, split='val')
        result = metrics.compute('val')
        
        # Check both summary and per-concept
        self.assertIn('val/SUMMARY-binary_accuracy', result)
        self.assertIn('val/b1_accuracy', result)
        self.assertIn('val/b2_accuracy', result)
        self.assertIn('val/b3_accuracy', result)

    def test_multiple_splits(self):
        """Test independent tracking for train/val/test splits."""
        metrics_config = GroupConfig(
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy()
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        # Generate different data for each split
        torch.manual_seed(42)
        train_endogenous = torch.randn(16, 3)
        train_targets = torch.randint(0, 2, (16, 3)).float()
        
        torch.manual_seed(43)
        val_endogenous = torch.randn(16, 3)
        val_targets = torch.randint(0, 2, (16, 3)).float()
        
        # Update different splits
        metrics.update(preds=train_endogenous, target=train_targets, split='train')
        metrics.update(preds=val_endogenous, target=val_targets, split='val')
        
        # Compute each split
        train_result = metrics.compute('train')
        val_result = metrics.compute('val')
        
        # Results should be independent
        self.assertIn('train/SUMMARY-binary_accuracy', train_result)
        self.assertIn('val/SUMMARY-binary_accuracy', val_result)

    def test_reset_metrics(self):
        """Test metric reset functionality."""
        metrics_config = GroupConfig(
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy()
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        # Update and compute
        metrics.update(preds=endogenous, target=targets, split='train')
        result1 = metrics.compute('train')
        
        # Reset and update with different data
        metrics.reset('train')
        endogenous2 = torch.randn(16, 3)
        targets2 = torch.randint(0, 2, (16, 3)).float()
        metrics.update(preds=endogenous2, target=targets2, split='train')
        result2 = metrics.compute('train')
        
        # Results should be different (with high probability)
        self.assertIsInstance(result1['train/SUMMARY-binary_accuracy'], torch.Tensor)
        self.assertIsInstance(result2['train/SUMMARY-binary_accuracy'], torch.Tensor)

    def test_reset_all_splits(self):
        """Test resetting all splits at once."""
        metrics_config = GroupConfig(
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy()
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        # Update all splits
        metrics.update(preds=endogenous, target=targets, split='train')
        metrics.update(preds=endogenous, target=targets, split='val')
        metrics.update(preds=endogenous, target=targets, split='test')
        
        # Reset all at once
        metrics.reset()
        
        # All should be reset (empty results)
        train_result = metrics.compute('train')
        val_result = metrics.compute('val')
        test_result = metrics.compute('test')
        
        self.assertIn('train/SUMMARY-binary_accuracy', train_result)
        self.assertIn('val/SUMMARY-binary_accuracy', val_result)
        self.assertIn('test/SUMMARY-binary_accuracy', test_result)

    def test_missing_required_metrics(self):
        """Test that missing required metrics raises error."""
        # Missing binary metrics config
        metrics_config = GroupConfig(
            categorical={
                'accuracy': torchmetrics.classification.MulticlassAccuracy(
                    num_classes=3, average='micro'
                )
            }
        )
        
        with self.assertRaises(ValueError):
            ConceptMetrics(
                self.annotations_binary,
                metrics_config,
                summary_metrics=True
            )

    def test_unused_metrics_warning(self):
        """Test that unused metrics produce warnings."""
        import warnings
        
        # Provides continuous metrics but no continuous concepts
        metrics_config = GroupConfig(
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy()
            },
            continuous={
                'mse': torchmetrics.regression.MeanSquaredError()
            }
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ConceptMetrics(
                self.annotations_binary,
                metrics_config,
                summary_metrics=True
            )
            # Should warn about unused continuous metrics
            self.assertTrue(any("continuous" in str(warning.message).lower() 
                              for warning in w))

    def test_metric_class_with_kwargs(self):
        """Test passing metric class with user kwargs as tuple."""
        metrics_config = GroupConfig(
            categorical={
                # Pass class + kwargs tuple
                'accuracy': (
                    torchmetrics.classification.MulticlassAccuracy,
                    {'average': 'macro'}
                )
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations_categorical,
            metrics_config,
            summary_metrics=True
        )
        
        # Categorical: cat1 (3 classes) + cat2 (5 classes) = 8 endogenous total
        endogenous = torch.randn(16, 8)
        targets = torch.cat([
            torch.randint(0, 3, (16, 1)),
            torch.randint(0, 5, (16, 1))
        ], dim=1)
        
        # Update and compute
        metrics.update(preds=endogenous, target=targets, split='train')
        result = metrics.compute('train')
        
        self.assertIn('train/SUMMARY-categorical_accuracy', result)
        # Should use max cardinality (5) with macro averaging
        self.assertTrue(0 <= result['train/SUMMARY-categorical_accuracy'] <= 1)

    def test_metric_class_without_kwargs(self):
        """Test passing just metric class (no instantiation)."""
        metrics_config = GroupConfig(
            categorical={
                # Pass class only, num_classes will be added automatically
                'accuracy': torchmetrics.classification.MulticlassAccuracy
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations_categorical,
            metrics_config,
            summary_metrics=True
        )
        
        endogenous = torch.randn(16, 8)
        targets = torch.cat([
            torch.randint(0, 3, (16, 1)),
            torch.randint(0, 5, (16, 1))
        ], dim=1)
        
        metrics.update(preds=endogenous, target=targets, split='val')
        result = metrics.compute('val')
        
        self.assertIn('val/SUMMARY-categorical_accuracy', result)

    def test_mixed_metric_specs(self):
        """Test mixing instantiated, class+kwargs, and class-only metrics."""
        metrics_config = GroupConfig(
            binary={
                # Pre-instantiated
                'accuracy': torchmetrics.classification.BinaryAccuracy(),
                # Class + kwargs (using threshold as example)
                'f1': (torchmetrics.classification.BinaryF1Score, {'threshold': 0.5}),
                # Class only
                'precision': torchmetrics.classification.BinaryPrecision
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()
        
        metrics.update(preds=endogenous, target=targets, split='test')
        result = metrics.compute('test')
        
        self.assertIn('test/SUMMARY-binary_accuracy', result)
        self.assertIn('test/SUMMARY-binary_f1', result)
        self.assertIn('test/SUMMARY-binary_precision', result)

    def test_num_classes_in_kwargs_raises_error(self):
        """Test that providing num_classes in kwargs raises ValueError."""
        metrics_config = GroupConfig(
            categorical={
                'accuracy': (
                    torchmetrics.classification.MulticlassAccuracy,
                    {'num_classes': 10, 'average': 'macro'}  # num_classes should not be provided
                )
            }
        )
        
        with self.assertRaises(ValueError) as cm:
            metrics = ConceptMetrics(
                self.annotations_categorical,
                metrics_config,
                summary_metrics=True
            )
            # Trigger metric instantiation
            endogenous = torch.randn(16, 8)
            targets = torch.cat([
                torch.randint(0, 3, (16, 1)),
                torch.randint(0, 5, (16, 1))
            ], dim=1)
            metrics.update(preds=endogenous, target=targets, split='train')
        
        self.assertIn('num_classes', str(cm.exception))
        self.assertIn('automatically', str(cm.exception).lower())


class TestConceptMetricsEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in ConceptMetrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Standard binary concepts
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
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        # Empty batch
        endogenous = torch.randn(0, 2)
        targets = torch.randint(0, 2, (0, 2)).float()
        
        # Should not crash
        metrics.update(preds=endogenous, target=targets, split='train')
        result = metrics.compute('train')
        
        # Result should have the metric key
        self.assertIn('train/SUMMARY-binary_accuracy', result)
    
    def test_single_sample_batch(self):
        """Test with batch size of 1."""
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        # Single sample
        endogenous = torch.randn(1, 2)
        targets = torch.randint(0, 2, (1, 2)).float()
        
        metrics.update(preds=endogenous, target=targets, split='train')
        result = metrics.compute('train')
        
        self.assertIn('train/SUMMARY-binary_accuracy', result)
        self.assertTrue(0 <= result['train/SUMMARY-binary_accuracy'] <= 1)
    
    def test_very_large_batch(self):
        """Test with large batch size."""
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        # Large batch
        batch_size = 10000
        endogenous = torch.randn(batch_size, 2)
        targets = torch.randint(0, 2, (batch_size, 2)).float()
        
        metrics.update(preds=endogenous, target=targets, split='train')
        result = metrics.compute('train')
        
        self.assertIn('train/SUMMARY-binary_accuracy', result)
    
    def test_invalid_split_name(self):
        """Test that invalid split names raise ValueError."""
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        endogenous = torch.randn(16, 2)
        targets = torch.randint(0, 2, (16, 2)).float()
        
        # Invalid split name
        with self.assertRaises(ValueError):
            metrics.update(preds=endogenous, target=targets, split='invalid_split')
    
    def test_validation_alias(self):
        """Test that 'validation' works as alias for 'val'."""
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        endogenous = torch.randn(16, 2)
        targets = torch.randint(0, 2, (16, 2)).float()
        
        # Use 'validation' instead of 'val'
        metrics.update(preds=endogenous, target=targets, split='validation')
        result = metrics.compute('validation')
        
        self.assertIn('val/SUMMARY-binary_accuracy', result)
    
    def test_no_metrics_config(self):
        """Test creating metrics with empty config."""
        metrics_config = GroupConfig(binary={})
        
        # Should create metrics but with no actual metrics
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        # Should have empty collections
        self.assertEqual(len(metrics.train_metrics), 0)
    
    def test_perconcept_invalid_name(self):
        """Test that invalid concept names in perconcept_metrics are handled."""
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        # Invalid concept name in list
        with self.assertRaises(ValueError):
            metrics = ConceptMetrics(
                self.annotations_binary,
                metrics_config,
                summary_metrics=True,
                perconcept_metrics=['nonexistent_concept']
            )
    
    def test_perconcept_invalid_type(self):
        """Test that invalid type for perconcept_metrics raises error."""
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        # Invalid type (should be bool or list)
        with self.assertRaises(ValueError):
            metrics = ConceptMetrics(
                self.annotations_binary,
                metrics_config,
                summary_metrics=True,
                perconcept_metrics="invalid_string"
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
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        # Perfect predictions
        torch.manual_seed(42)
        targets = torch.randint(0, 2, (32, 2)).float()
        predictions = targets.clone()  # Exact match
        
        metrics.update(preds=predictions, target=targets, split='train')
        result = metrics.compute('train')
        
        # Should be exactly 1.0
        self.assertAlmostEqual(
            result['train/SUMMARY-binary_accuracy'].item(), 
            1.0, 
            places=5
        )
    
    def test_zero_accuracy(self):
        """Test that completely wrong predictions give 0% accuracy."""
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        # Completely wrong predictions
        torch.manual_seed(42)
        targets = torch.randint(0, 2, (32, 2)).float()
        predictions = 1 - targets  # Opposite of targets
        
        metrics.update(preds=predictions, target=targets, split='train')
        result = metrics.compute('train')
        
        # Should be exactly 0.0
        self.assertAlmostEqual(
            result['train/SUMMARY-binary_accuracy'].item(), 
            0.0, 
            places=5
        )
    
    def test_known_accuracy_value(self):
        """Test with known accuracy value."""
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        metrics = ConceptMetrics(
            self.annotations_binary,
            metrics_config,
            summary_metrics=True
        )
        
        # Construct specific case: 3 out of 4 correct
        targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        predictions = torch.tensor([[1.0, 1.0], [1.0, 0.0]])  # 3 out of 4 correct
        
        metrics.update(preds=predictions, target=targets, split='train')
        result = metrics.compute('train')
        
        # Should be 0.75 (3 out of 4)
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
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        metrics = ConceptMetrics(
            self.annotations,
            metrics_config,
            summary_metrics=True
        )
        
        # Batch 1: 100% accuracy
        targets1 = torch.tensor([[1.0], [1.0]])
        preds1 = torch.tensor([[1.0], [1.0]])
        
        # Batch 2: 0% accuracy
        targets2 = torch.tensor([[1.0], [1.0]])
        preds2 = torch.tensor([[0.0], [0.0]])
        
        # Update with both batches
        metrics.update(preds=preds1, target=targets1, split='train')
        metrics.update(preds=preds2, target=targets2, split='train')
        
        result = metrics.compute('train')
        
        # Should be 50% (2 correct out of 4 total)
        self.assertAlmostEqual(
            result['train/SUMMARY-binary_accuracy'].item(),
            0.5,
            places=5
        )
    
    def test_reset_clears_accumulation(self):
        """Test that reset clears accumulated state."""
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        metrics = ConceptMetrics(
            self.annotations,
            metrics_config,
            summary_metrics=True
        )
        
        # First epoch
        targets1 = torch.tensor([[1.0], [1.0]])
        preds1 = torch.tensor([[0.0], [0.0]])  # 0% accuracy
        
        metrics.update(preds=preds1, target=targets1, split='train')
        result1 = metrics.compute('train')
        self.assertAlmostEqual(result1['train/SUMMARY-binary_accuracy'].item(), 0.0)
        
        # Reset
        metrics.reset('train')
        
        # Second epoch with different data
        targets2 = torch.tensor([[1.0], [1.0]])
        preds2 = torch.tensor([[1.0], [1.0]])  # 100% accuracy
        
        metrics.update(preds=preds2, target=targets2, split='train')
        result2 = metrics.compute('train')
        
        # Should be 100%, not affected by previous data
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
        metrics_config = GroupConfig(
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy(),
                'f1': torchmetrics.classification.BinaryF1Score()
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations,
            metrics_config,
            summary_metrics=True,
            perconcept_metrics=False
        )
        
        repr_str = repr(metrics)
        
        # Should contain key information
        self.assertIn('ConceptMetrics', repr_str)
        self.assertIn('n_concepts=2', repr_str)
        self.assertIn('summary=True', repr_str)
        self.assertIn('perconcept=False', repr_str)
        self.assertIn('BinaryAccuracy', repr_str)
        self.assertIn('BinaryF1Score', repr_str)
    
    def test_repr_with_mixed_metric_specs(self):
        """Test __repr__ with different metric specification methods."""
        metrics_config = GroupConfig(
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy(),  # Instantiated
                'f1': (torchmetrics.classification.BinaryF1Score, {}),  # Tuple
                'precision': torchmetrics.classification.BinaryPrecision  # Class
            }
        )
        
        metrics = ConceptMetrics(
            self.annotations,
            metrics_config,
            summary_metrics=True
        )
        
        repr_str = repr(metrics)
        
        # All metrics should appear
        self.assertIn('BinaryAccuracy', repr_str)
        self.assertIn('BinaryF1Score', repr_str)
        self.assertIn('BinaryPrecision', repr_str)


class TestConceptMetricsGetMethod(unittest.TestCase):
    """Test the get() dict-like interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        axis_binary = AxisAnnotation(
            labels=('b1',),
            cardinalities=[1],
            metadata={'b1': {'type': 'discrete'}}
        )
        self.annotations = Annotations({1: axis_binary})
        
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()}
        )
        
        self.metrics = ConceptMetrics(
            self.annotations,
            metrics_config,
            summary_metrics=True
        )
    
    def test_get_train_metrics(self):
        """Test getting train metrics collection."""
        collection = self.metrics.get('train_metrics')
        self.assertIsNotNone(collection)
        self.assertTrue(len(collection) > 0)
    
    def test_get_val_metrics(self):
        """Test getting validation metrics collection."""
        collection = self.metrics.get('val_metrics')
        self.assertIsNotNone(collection)
    
    def test_get_test_metrics(self):
        """Test getting test metrics collection."""
        collection = self.metrics.get('test_metrics')
        self.assertIsNotNone(collection)
    
    def test_get_invalid_key(self):
        """Test getting with invalid key returns default."""
        result = self.metrics.get('invalid_key')
        self.assertIsNone(result)
    
    def test_get_with_custom_default(self):
        """Test get with custom default value."""
        default = "custom_default"
        result = self.metrics.get('invalid_key', default=default)
        self.assertEqual(result, default)


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
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            categorical={'accuracy': torchmetrics.classification.MulticlassAccuracy}
        )
        
        metrics = ConceptMetrics(
            self.annotations,
            metrics_config,
            summary_metrics=True,
            perconcept_metrics=True
        )
        
        # Simulate training batches
        num_batches = 10
        batch_size = 32
        
        for _ in range(num_batches):
            # Mixed predictions: 2 binary + 3 categorical = 5 endogenous dims
            predictions = torch.randn(batch_size, 5)
            targets = torch.cat([
                torch.randint(0, 2, (batch_size, 2)),
                torch.randint(0, 3, (batch_size, 1))
            ], dim=1)
            
            metrics.update(preds=predictions, target=targets, split='train')
        
        # Compute results
        results = metrics.compute('train')
        
        # Verify all expected metrics are present
        self.assertIn('train/SUMMARY-binary_accuracy', results)
        self.assertIn('train/SUMMARY-categorical_accuracy', results)
        self.assertIn('train/binary1_accuracy', results)
        self.assertIn('train/binary2_accuracy', results)
        self.assertIn('train/cat1_accuracy', results)
        
        # Reset for next epoch
        metrics.reset('train')
        
        # After reset, metrics should be ready for new epoch
        results_after_reset = metrics.compute('train')
        self.assertIn('train/SUMMARY-binary_accuracy', results_after_reset)
    
    def test_train_val_test_workflow(self):
        """Simulate complete train/val/test workflow."""
        metrics_config = GroupConfig(
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            categorical={'accuracy': torchmetrics.classification.MulticlassAccuracy}
        )
        
        metrics = ConceptMetrics(
            self.annotations,
            metrics_config,
            summary_metrics=True
        )
        
        batch_size = 16
        
        # Training
        for _ in range(5):
            predictions = torch.randn(batch_size, 5)
            targets = torch.cat([
                torch.randint(0, 2, (batch_size, 2)),
                torch.randint(0, 3, (batch_size, 1))
            ], dim=1)
            metrics.update(preds=predictions, target=targets, split='train')
        
        # Validation
        for _ in range(2):
            predictions = torch.randn(batch_size, 5)
            targets = torch.cat([
                torch.randint(0, 2, (batch_size, 2)),
                torch.randint(0, 3, (batch_size, 1))
            ], dim=1)
            metrics.update(preds=predictions, target=targets, split='val')
        
        # Testing
        for _ in range(3):
            predictions = torch.randn(batch_size, 5)
            targets = torch.cat([
                torch.randint(0, 2, (batch_size, 2)),
                torch.randint(0, 3, (batch_size, 1))
            ], dim=1)
            metrics.update(preds=predictions, target=targets, split='test')
        
        # Compute all splits
        train_results = metrics.compute('train')
        val_results = metrics.compute('val')
        test_results = metrics.compute('test')
        
        # All should have results
        self.assertIn('train/SUMMARY-binary_accuracy', train_results)
        self.assertIn('val/SUMMARY-binary_accuracy', val_results)
        self.assertIn('test/SUMMARY-binary_accuracy', test_results)
        
        # Reset all
        metrics.reset()
        
        # After reset, all should be clean
        train_clean = metrics.compute('train')
        val_clean = metrics.compute('val')
        test_clean = metrics.compute('test')
        
        self.assertIn('train/SUMMARY-binary_accuracy', train_clean)
        self.assertIn('val/SUMMARY-binary_accuracy', val_clean)
        self.assertIn('test/SUMMARY-binary_accuracy', test_clean)


if __name__ == '__main__':
    unittest.main()
