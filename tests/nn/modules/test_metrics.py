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
from torch_concepts.annotations import Annotations
from torch_concepts.tensor import AnnotatedTensor


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


@unittest.skip("compute_cace disabled — intervention module not yet ported")
class TestComputeCace(unittest.TestCase):
    """Test compute_cace utility function."""

    def setUp(self):
        """Build a minimal CBM and fake dataloader."""
        from torch.distributions import Bernoulli
        from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel

        ann = Annotations(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'discrete', 'distribution': Bernoulli},
                    'c2': {'type': 'discrete', 'distribution': Bernoulli},
                    'task': {'type': 'discrete', 'distribution': Bernoulli},
                }
            )
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
        axis_mixed = Annotations(
            labels=('binary1', 'binary2', 'cat1', 'cat2'),
            cardinalities=[1, 1, 3, 4],
            metadata={
                'binary1': {'type': 'discrete'},
                'binary2': {'type': 'discrete'},
                'cat1': {'type': 'discrete'},
                'cat2': {'type': 'discrete'},
            }
        )
        self.annotations_mixed = axis_mixed
        
        # All binary
        axis_binary = Annotations(
            labels=('b1', 'b2', 'b3'),
            cardinalities=[1, 1, 1],
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'},
                'b3': {'type': 'discrete'},
            }
        )
        self.annotations_binary = axis_binary
        
        # All categorical
        axis_categorical = Annotations(
            labels=('cat1', 'cat2'),
            cardinalities=(3, 5),
            metadata={
                'cat1': {'type': 'discrete'},
                'cat2': {'type': 'discrete'},
            }
        )
        self.annotations_categorical = axis_categorical

    def test_binary_only_metrics(self):
        """Test ConceptMetrics with only binary concepts."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )

        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()

        preds = AnnotatedTensor(endogenous, self.annotations_binary, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertIn('train/SUMMARY-binary_accuracy', result)
        self.assertIsInstance(result['train/SUMMARY-binary_accuracy'], torch.Tensor)
        self.assertTrue(0 <= result['train/SUMMARY-binary_accuracy'] <= 1)

    def test_categorical_only_metrics(self):
        """Test ConceptMetrics with only categorical concepts."""
        metrics = ConceptMetrics(
            annotations=self.annotations_categorical,
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

        preds = AnnotatedTensor(endogenous, self.annotations_categorical, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_categorical.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertIn('val/SUMMARY-categorical_accuracy', result)
        self.assertTrue(0 <= result['val/SUMMARY-categorical_accuracy'] <= 1)

    def test_mixed_concepts_metrics(self):
        """Test ConceptMetrics with mixed concept types."""
        metrics = ConceptMetrics(
            annotations=self.annotations_mixed,
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

        preds = AnnotatedTensor(endogenous, self.annotations_mixed, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_mixed.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertIn('test/SUMMARY-binary_accuracy', result)
        self.assertIn('test/SUMMARY-binary_f1', result)
        self.assertIn('test/SUMMARY-categorical_accuracy', result)

    def test_per_concept(self):
        """Test per-concept metric tracking."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=False,
            per_concept=['b1', 'b2'],
            prefix='train'
        )

        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()

        preds = AnnotatedTensor(endogenous, self.annotations_binary, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertIn('train/b1_accuracy', result)
        self.assertIn('train/b2_accuracy', result)
        self.assertNotIn('train/b3_accuracy', result)

    def test_summary_and_per_concept(self):
        """Test combining summary and per-concept metrics."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            per_concept=True,
            prefix='val'
        )

        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()

        preds = AnnotatedTensor(endogenous, self.annotations_binary, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertIn('val/SUMMARY-binary_accuracy', result)
        self.assertIn('val/b1_accuracy', result)
        self.assertIn('val/b2_accuracy', result)
        self.assertIn('val/b3_accuracy', result)

    def test_multiple_splits_via_clone(self):
        """Test independent tracking for train/val/test via clone."""
        base = ConceptMetrics(
            annotations=self.annotations_binary,
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

        train_preds = AnnotatedTensor(train_endogenous, self.annotations_binary, axis=-1)
        train_target = AnnotatedTensor(train_targets, self.annotations_binary.to_concept_space(), axis=-1)
        val_preds = AnnotatedTensor(val_endogenous, self.annotations_binary, axis=-1)
        val_target = AnnotatedTensor(val_targets, self.annotations_binary.to_concept_space(), axis=-1)
        train_metrics.update(preds=train_preds, target=train_target)
        val_metrics.update(preds=val_preds, target=val_target)
        
        train_result = train_metrics.compute()
        val_result = val_metrics.compute()
        
        self.assertIn('train/SUMMARY-binary_accuracy', train_result)
        self.assertIn('val/SUMMARY-binary_accuracy', val_result)

    def test_reset_metrics(self):
        """Test metric reset functionality."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )

        endogenous = torch.randn(16, 3)
        targets = torch.randint(0, 2, (16, 3)).float()

        preds = AnnotatedTensor(endogenous, self.annotations_binary, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result1 = metrics.compute()

        metrics.reset()
        endogenous2 = torch.randn(16, 3)
        targets2 = torch.randint(0, 2, (16, 3)).float()
        preds2 = AnnotatedTensor(endogenous2, self.annotations_binary, axis=-1)
        target2 = AnnotatedTensor(targets2, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds2, target=target2)
        result2 = metrics.compute()
        
        self.assertIsInstance(result1['train/SUMMARY-binary_accuracy'], torch.Tensor)
        self.assertIsInstance(result2['train/SUMMARY-binary_accuracy'], torch.Tensor)

    def test_metric_class_with_kwargs(self):
        """Test passing metric class with user kwargs as tuple."""
        metrics = ConceptMetrics(
            annotations=self.annotations_categorical,
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

        preds = AnnotatedTensor(endogenous, self.annotations_categorical, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_categorical.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertIn('train/SUMMARY-categorical_accuracy', result)
        self.assertTrue(0 <= result['train/SUMMARY-categorical_accuracy'] <= 1)

    def test_metric_class_without_kwargs(self):
        """Test passing just metric class (no instantiation)."""
        metrics = ConceptMetrics(
            annotations=self.annotations_categorical,
            categorical={'accuracy': torchmetrics.classification.MulticlassAccuracy},
            summary=True,
            prefix='val'
        )

        endogenous = torch.randn(16, 8)
        targets = torch.cat([
            torch.randint(0, 3, (16, 1)),
            torch.randint(0, 5, (16, 1))
        ], dim=1)

        preds = AnnotatedTensor(endogenous, self.annotations_categorical, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_categorical.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertIn('val/SUMMARY-categorical_accuracy', result)

    def test_mixed_metric_specs(self):
        """Test mixing instantiated, class+kwargs, and class-only metrics."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
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

        preds = AnnotatedTensor(endogenous, self.annotations_binary, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertIn('test/SUMMARY-binary_accuracy', result)
        self.assertIn('test/SUMMARY-binary_f1', result)
        self.assertIn('test/SUMMARY-binary_precision', result)

    def test_num_classes_in_kwargs_raises_error(self):
        """Test that providing num_classes in kwargs raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            ConceptMetrics(
                annotations=self.annotations_categorical,
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
        axis_binary = Annotations(
            labels=('b1', 'b2'),
            cardinalities=[1, 1],
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'}
            }
        )
        self.annotations_binary = axis_binary
    
    def test_empty_batch_update(self):
        """Test updating with empty batch."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )

        endogenous = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4, 2)).float()
        preds = AnnotatedTensor(endogenous, self.annotations_binary, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)

        empty_endogenous = torch.randn(0, 2)
        empty_targets = torch.randint(0, 2, (0, 2)).float()
        empty_preds = AnnotatedTensor(empty_endogenous, self.annotations_binary, axis=-1)
        empty_target = AnnotatedTensor(empty_targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=empty_preds, target=empty_target)

        result = metrics.compute()

        self.assertIn('train/SUMMARY-binary_accuracy', result)
    
    def test_single_sample_batch(self):
        """Test with batch size of 1."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )

        endogenous = torch.randn(1, 2)
        targets = torch.randint(0, 2, (1, 2)).float()

        preds = AnnotatedTensor(endogenous, self.annotations_binary, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertIn('train/SUMMARY-binary_accuracy', result)
        self.assertTrue(0 <= result['train/SUMMARY-binary_accuracy'] <= 1)
    
    def test_very_large_batch(self):
        """Test with large batch size."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )

        batch_size = 10000
        endogenous = torch.randn(batch_size, 2)
        targets = torch.randint(0, 2, (batch_size, 2)).float()

        preds = AnnotatedTensor(endogenous, self.annotations_binary, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertIn('train/SUMMARY-binary_accuracy', result)
    
    def test_no_prefix(self):
        """Test creating metrics without prefix."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True
        )

        endogenous = torch.randn(16, 2)
        targets = torch.randint(0, 2, (16, 2)).float()

        preds = AnnotatedTensor(endogenous, self.annotations_binary, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertIn('SUMMARY-binary_accuracy', result)

    def test_empty_collection(self):
        """Test creating metrics with empty config."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
            binary={},
            summary=True
        )

        self.assertEqual(len(metrics.collection), 0)

    def test_perconcept_invalid_name(self):
        """Test that invalid concept names in per_concept are handled."""
        with self.assertRaises(ValueError):
            ConceptMetrics(
                annotations=self.annotations_binary,
                binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
                summary=True,
                per_concept=['nonexistent_concept']
            )

    def test_perconcept_invalid_type(self):
        """Test that invalid type for per_concept raises error."""
        with self.assertRaises(ValueError):
            ConceptMetrics(
                annotations=self.annotations_binary,
                binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
                summary=True,
                per_concept="invalid_string"
            )


class TestConceptMetricsAccuracy(unittest.TestCase):
    """Test that metrics compute accurate values."""
    
    def setUp(self):
        """Set up test fixtures."""
        axis_binary = Annotations(
            labels=('b1', 'b2'),
            cardinalities=[1, 1],
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'}
            }
        )
        self.annotations_binary = axis_binary
    
    def test_perfect_accuracy(self):
        """Test that perfect predictions give 100% accuracy."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )

        torch.manual_seed(42)
        targets = torch.randint(0, 2, (32, 2)).float()
        predictions = targets.clone()

        preds = AnnotatedTensor(predictions, self.annotations_binary, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertAlmostEqual(
            result['train/SUMMARY-binary_accuracy'].item(),
            1.0,
            places=5
        )
    
    def test_zero_accuracy(self):
        """Test that completely wrong predictions give 0% accuracy."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )

        torch.manual_seed(42)
        targets = torch.randint(0, 2, (32, 2)).float()
        predictions = 1 - targets

        preds = AnnotatedTensor(predictions, self.annotations_binary, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
        result = metrics.compute()

        self.assertAlmostEqual(
            result['train/SUMMARY-binary_accuracy'].item(),
            0.0,
            places=5
        )
    
    def test_known_accuracy_value(self):
        """Test with known accuracy value."""
        metrics = ConceptMetrics(
            annotations=self.annotations_binary,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )

        targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        predictions = torch.tensor([[1.0, 1.0], [1.0, 0.0]])  # 3 out of 4

        preds = AnnotatedTensor(predictions, self.annotations_binary, axis=-1)
        target = AnnotatedTensor(targets, self.annotations_binary.to_concept_space(), axis=-1)
        metrics.update(preds=preds, target=target)
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
        axis_binary = Annotations(
            labels=('b1',),
            cardinalities=[1],
            metadata={'b1': {'type': 'discrete'}}
        )
        self.annotations = axis_binary
    
    def test_accumulation_across_batches(self):
        """Test that metrics correctly accumulate across batches."""
        metrics = ConceptMetrics(
            annotations=self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )

        targets1 = torch.tensor([[1.0], [1.0]])
        preds1 = torch.tensor([[1.0], [1.0]])

        targets2 = torch.tensor([[1.0], [1.0]])
        preds2 = torch.tensor([[0.0], [0.0]])

        ann_concept = self.annotations.to_concept_space()
        metrics.update(
            preds=AnnotatedTensor(preds1, self.annotations, axis=-1),
            target=AnnotatedTensor(targets1, ann_concept, axis=-1),
        )
        metrics.update(
            preds=AnnotatedTensor(preds2, self.annotations, axis=-1),
            target=AnnotatedTensor(targets2, ann_concept, axis=-1),
        )

        result = metrics.compute()

        self.assertAlmostEqual(
            result['train/SUMMARY-binary_accuracy'].item(),
            0.5,
            places=5
        )
    
    def test_reset_clears_accumulation(self):
        """Test that reset clears accumulated state."""
        metrics = ConceptMetrics(
            annotations=self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True,
            prefix='train'
        )

        ann_concept = self.annotations.to_concept_space()

        targets1 = torch.tensor([[1.0], [1.0]])
        preds1 = torch.tensor([[0.0], [0.0]])

        metrics.update(
            preds=AnnotatedTensor(preds1, self.annotations, axis=-1),
            target=AnnotatedTensor(targets1, ann_concept, axis=-1),
        )
        result1 = metrics.compute()
        self.assertAlmostEqual(result1['train/SUMMARY-binary_accuracy'].item(), 0.0)

        metrics.reset()

        targets2 = torch.tensor([[1.0], [1.0]])
        preds2 = torch.tensor([[1.0], [1.0]])

        metrics.update(
            preds=AnnotatedTensor(preds2, self.annotations, axis=-1),
            target=AnnotatedTensor(targets2, ann_concept, axis=-1),
        )
        result2 = metrics.compute()

        self.assertAlmostEqual(result2['train/SUMMARY-binary_accuracy'].item(), 1.0)


class TestConceptMetricsRepr(unittest.TestCase):
    """Test string representations and display methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        axis_binary = Annotations(
            labels=('b1', 'b2'),
            cardinalities=[1, 1],
            metadata={
                'b1': {'type': 'discrete'},
                'b2': {'type': 'discrete'}
            }
        )
        self.annotations = axis_binary
    
    def test_repr_with_metrics(self):
        """Test __repr__ method."""
        metrics = ConceptMetrics(
            annotations=self.annotations,
            binary={
                'accuracy': torchmetrics.classification.BinaryAccuracy(),
                'f1': torchmetrics.classification.BinaryF1Score()
            },
            summary=True,
            per_concept=False
        )

        repr_str = repr(metrics)

        self.assertIn('ConceptMetrics', repr_str)
        self.assertIn('summary=True', repr_str)
        self.assertIn('BinaryAccuracy', repr_str)
        self.assertIn('BinaryF1Score', repr_str)
    
    def test_repr_with_mixed_metric_specs(self):
        """Test __repr__ with different metric specification methods."""
        metrics = ConceptMetrics(
            annotations=self.annotations,
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
        axis_binary = Annotations(
            labels=('b1',),
            cardinalities=[1],
            metadata={'b1': {'type': 'discrete'}}
        )
        self.annotations = axis_binary
    
    def test_clone_with_prefix(self):
        """Test cloning with a new prefix."""
        base = ConceptMetrics(
            annotations=self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True
        )
        cloned = base.clone(prefix='train')

        targets = torch.tensor([[1.0], [1.0]])
        preds = torch.tensor([[1.0], [1.0]])
        cloned.update(
            preds=AnnotatedTensor(preds, self.annotations, axis=-1),
            target=AnnotatedTensor(targets, self.annotations.to_concept_space(), axis=-1),
        )
        result = cloned.compute()

        self.assertIn('train/SUMMARY-binary_accuracy', result)

    def test_clones_are_independent(self):
        """Test that cloned instances have independent state."""
        base = ConceptMetrics(
            annotations=self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True
        )
        train = base.clone(prefix='train')
        val = base.clone(prefix='val')

        ann_concept = self.annotations.to_concept_space()

        # Update train and val with different data
        targets = torch.tensor([[1.0], [1.0]])
        preds = torch.tensor([[1.0], [1.0]])
        train.update(
            preds=AnnotatedTensor(preds, self.annotations, axis=-1),
            target=AnnotatedTensor(targets, ann_concept, axis=-1),
        )

        val_targets = torch.tensor([[1.0], [1.0]])
        val_preds = torch.tensor([[0.0], [0.0]])
        val.update(
            preds=AnnotatedTensor(val_preds, self.annotations, axis=-1),
            target=AnnotatedTensor(val_targets, ann_concept, axis=-1),
        )

        train_result = train.compute()
        val_result = val.compute()

        self.assertAlmostEqual(train_result['train/SUMMARY-binary_accuracy'].item(), 1.0)
        # val was updated with different data, so its result differs from train
        self.assertAlmostEqual(val_result['val/SUMMARY-binary_accuracy'].item(), 0.0)

    def test_collection_property(self):
        """Test the collection property returns non-empty sub-collections."""
        metrics = ConceptMetrics(
            annotations=self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            summary=True
        )
        preds = AnnotatedTensor(torch.tensor([[1.0], [1.0]]), self.annotations, axis=-1)
        target = AnnotatedTensor(
            torch.tensor([[1.0], [1.0]]), self.annotations.to_concept_space(), axis=-1
        )
        metrics.update(preds=preds, target=target)

        coll = metrics.collection
        self.assertIn('binary', coll)
        self.assertTrue(len(coll) > 0)


class TestConceptMetricsIntegration(unittest.TestCase):
    """Integration tests simulating real training scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        axis_mixed = Annotations(
            labels=('binary1', 'binary2', 'cat1'),
            cardinalities=[1, 1, 3],
            metadata={
                'binary1': {'type': 'discrete'},
                'binary2': {'type': 'discrete'},
                'cat1': {'type': 'discrete'}
            }
        )
        self.annotations = axis_mixed
    
    def test_full_training_epoch_simulation(self):
        """Simulate a complete training epoch with multiple batches."""
        base = ConceptMetrics(
            annotations=self.annotations,
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

            preds = AnnotatedTensor(predictions, self.annotations, axis=-1)
            target = AnnotatedTensor(targets, self.annotations.to_concept_space(), axis=-1)
            train_metrics.update(preds=preds, target=target)
        
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
            annotations=self.annotations,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            categorical={'accuracy': torchmetrics.classification.MulticlassAccuracy},
            summary=True
        )
        train_metrics = base.clone(prefix='train')
        val_metrics = base.clone(prefix='val')
        test_metrics = base.clone(prefix='test')

        batch_size = 16
        ann_concept = self.annotations.to_concept_space()

        for _ in range(5):
            predictions = torch.randn(batch_size, 5)
            targets = torch.cat([
                torch.randint(0, 2, (batch_size, 2)),
                torch.randint(0, 3, (batch_size, 1))
            ], dim=1)
            train_metrics.update(
                preds=AnnotatedTensor(predictions, self.annotations, axis=-1),
                target=AnnotatedTensor(targets, ann_concept, axis=-1),
            )

        for _ in range(2):
            predictions = torch.randn(batch_size, 5)
            targets = torch.cat([
                torch.randint(0, 2, (batch_size, 2)),
                torch.randint(0, 3, (batch_size, 1))
            ], dim=1)
            val_metrics.update(
                preds=AnnotatedTensor(predictions, self.annotations, axis=-1),
                target=AnnotatedTensor(targets, ann_concept, axis=-1),
            )

        for _ in range(3):
            predictions = torch.randn(batch_size, 5)
            targets = torch.cat([
                torch.randint(0, 2, (batch_size, 2)),
                torch.randint(0, 3, (batch_size, 1))
            ], dim=1)
            test_metrics.update(
                preds=AnnotatedTensor(predictions, self.annotations, axis=-1),
                target=AnnotatedTensor(targets, ann_concept, axis=-1),
            )
        
        train_results = train_metrics.compute()
        val_results = val_metrics.compute()
        test_results = test_metrics.compute()
        
        self.assertIn('train/SUMMARY-binary_accuracy', train_results)
        self.assertIn('val/SUMMARY-binary_accuracy', val_results)
        self.assertIn('test/SUMMARY-binary_accuracy', test_results)


if __name__ == '__main__':
    unittest.main()


class TestConceptMetricsEdgeCases(unittest.TestCase):
    """Test edge cases and missing coverage in ConceptMetrics."""

    def setUp(self):
        import torchmetrics.classification as tc
        self.BinaryAccuracy = tc.BinaryAccuracy
        self.MulticlassAccuracy = tc.MulticlassAccuracy

        # binary-only annotations
        self.ann_binary = Annotations(
                labels=['b1', 'b2'],
                cardinalities=[1, 1],
                types=['binary', 'binary'],
            )
        # categorical-only annotations
        self.ann_cat = Annotations(
                labels=['c1'],
                cardinalities=[3],
                types=['categorical'],
            )
        # mixed
        self.ann_mixed = Annotations(
                labels=['b1', 'c1'],
                cardinalities=[1, 3],
                types=['binary', 'categorical'],
            )
        # dicts of metrics in the format ConceptMetrics expects
        self.bin_metrics = {'accuracy': self.BinaryAccuracy()}
        self.cat_metrics = {'accuracy': self.MulticlassAccuracy}

    def test_clone_metric_helper(self):
        from torch_concepts.nn.modules.metrics import clone_metric
        m = self.BinaryAccuracy()
        cloned = clone_metric(m)
        assert cloned is not m

    def test_repr_non_empty(self):
        m = ConceptMetrics(
            annotations=self.ann_binary,
            binary=self.bin_metrics,
        )
        r = repr(m)
        assert "ConceptMetrics" in r

    def test_collection_property_binary(self):
        m = ConceptMetrics(annotations=self.ann_binary, binary=self.bin_metrics)
        preds = AnnotatedTensor(torch.randn(4, 2), self.ann_binary, axis=-1)
        targets = AnnotatedTensor(torch.randint(0, 2, (4, 2)), self.ann_binary.to_concept_space(), axis=-1)
        m.update(preds, targets)
        coll = m.collection
        assert 'binary' in coll

    def test_collection_property_categorical(self):
        m = ConceptMetrics(annotations=self.ann_cat, categorical=self.cat_metrics)
        preds = AnnotatedTensor(torch.randn(4, 3), self.ann_cat, axis=-1)
        targets = AnnotatedTensor(torch.randint(0, 3, (4, 1)), self.ann_cat.to_concept_space(), axis=-1)
        m.update(preds, targets)
        coll = m.collection
        assert 'categorical' in coll

    def test_clone_with_prefix(self):
        m = ConceptMetrics(annotations=self.ann_binary, binary=self.bin_metrics)
        cloned = m.clone(prefix='val')
        preds = AnnotatedTensor(torch.randn(4, 2), self.ann_binary, axis=-1)
        targets = AnnotatedTensor(torch.randint(0, 2, (4, 2)), self.ann_binary.to_concept_space(), axis=-1)
        cloned.update(preds, targets)
        assert 'val/SUMMARY-binary_' in cloned.binary.prefix

    def test_per_concept_tracking(self):
        m = ConceptMetrics(
            annotations=self.ann_binary,
            binary=self.bin_metrics,
            per_concept=True,
        )
        preds = AnnotatedTensor(torch.randn(4, 2), self.ann_binary, axis=-1)
        targets = AnnotatedTensor(torch.randint(0, 2, (4, 2)), self.ann_binary.to_concept_space(), axis=-1)
        m.update(preds, targets)
        results = m.compute()
        assert len(results) > 0

    def test_per_concept_tracking_subset(self):
        m = ConceptMetrics(
            annotations=self.ann_binary,
            binary=self.bin_metrics,
            per_concept=['b1'],
        )
        preds = AnnotatedTensor(torch.randn(4, 2), self.ann_binary, axis=-1)
        targets = AnnotatedTensor(torch.randint(0, 2, (4, 2)), self.ann_binary.to_concept_space(), axis=-1)
        m.update(preds, targets)
        results = m.compute()
        # b1 per-concept collection present
        assert any('b1' in k for k in results)

    def test_update_skips_empty_batch(self):
        m = ConceptMetrics(annotations=self.ann_binary, binary=self.bin_metrics)
        # Empty batch — should return without error
        preds = AnnotatedTensor(torch.zeros(0, 2), self.ann_binary, axis=-1)
        targets = AnnotatedTensor(torch.zeros(0, 2), self.ann_binary.to_concept_space(), axis=-1)
        m.update(preds, targets)

    def test_instantiate_metric_from_tuple(self):
        m = ConceptMetrics(
            annotations=self.ann_cat,
            categorical={'acc': (self.MulticlassAccuracy, {})},
        )
        preds = AnnotatedTensor(torch.randn(4, 3), self.ann_cat, axis=-1)
        targets = AnnotatedTensor(torch.randint(0, 3, (4, 1)), self.ann_cat.to_concept_space(), axis=-1)
        m.update(preds, targets)
        results = m.compute()
        assert len(results) > 0

    def test_instantiate_metric_raises_on_num_classes_conflict(self):
        with self.assertRaises(ValueError):
            ConceptMetrics(
                annotations=self.ann_cat,
                categorical={'acc': (self.MulticlassAccuracy, {'num_classes': 3})},
            )

    def test_reset_clears_state(self):
        m = ConceptMetrics(annotations=self.ann_binary, binary=self.bin_metrics)
        preds = AnnotatedTensor(torch.randn(4, 2), self.ann_binary, axis=-1)
        targets = AnnotatedTensor(torch.randint(0, 2, (4, 2)), self.ann_binary.to_concept_space(), axis=-1)
        m.update(preds, targets)
        m.reset()
        # After reset computing should still work (fresh state)
        m.update(preds, targets)
        results = m.compute()
        assert len(results) > 0

    def test_update_with_model_output(self):
        from torch_concepts.nn.modules.outputs import ModelOutput
        m = ConceptMetrics(annotations=self.ann_binary, binary=self.bin_metrics)
        preds = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4, 2))
        mo = ModelOutput()
        mo.logits = AnnotatedTensor(preds, self.ann_binary, axis=-1)
        mo.target = AnnotatedTensor(targets, self.ann_binary.to_concept_space(), axis=-1)
        m.update(mo)
        results = m.compute()
        assert len(results) > 0

    def test_continuous_supported(self):
        """A continuous concept builds and updates its regression metric on ``loc``."""
        from torchmetrics.regression import MeanSquaredError
        from torch_concepts.tensor import AnnotatedTensor
        from torch_concepts.nn.modules.outputs import ModelOutput

        ann_cont = Annotations(labels=['cont1'], cardinalities=[1], types=['continuous'])
        m = ConceptMetrics(annotations=ann_cont, continuous={'mse': MeanSquaredError()},
                           per_concept=True)
        out = ModelOutput()
        out.loc = AnnotatedTensor(torch.randn(4, 1), ann_cont, axis=-1)
        out.target = AnnotatedTensor(torch.randn(4, 1), ann_cont.to_concept_space(), axis=-1)
        m.update(out)
        self.assertIn('cont1_mse', m.compute())

    def test_per_concept_categorical(self):
        m = ConceptMetrics(
            annotations=self.ann_cat,
            categorical=self.cat_metrics,
            per_concept=True,
        )
        preds = AnnotatedTensor(torch.randn(4, 3), self.ann_cat, axis=-1)
        targets = AnnotatedTensor(torch.randint(0, 3, (4, 1)), self.ann_cat.to_concept_space(), axis=-1)
        m.update(preds, targets)
        results = m.compute()
        assert len(results) > 0


class TestConceptMetricsMissingLines(unittest.TestCase):
    """Tests targeting specific missing lines in ConceptMetrics."""

    def _make_binary_ann(self):
        return Annotations(labels=['b1', 'b2'], cardinalities=[1, 1], types=['binary', 'binary'])

    def _make_cat_ann(self):
        return Annotations(labels=['c1'], cardinalities=[3], types=['categorical'])

    def test_per_concept_invalid_name_raises(self):
        """per_concept list with unknown name raises ValueError."""
        ann = self._make_binary_ann()
        with self.assertRaises(ValueError):
            ConceptMetrics(
                annotations=ann,
                binary={'acc': torchmetrics.classification.BinaryAccuracy()},
                per_concept=['nonexistent'],
            )

    def test_per_concept_invalid_type_raises(self):
        """per_concept with a non-bool/non-list raises ValueError."""
        ann = self._make_binary_ann()
        with self.assertRaises(ValueError):
            ConceptMetrics(
                annotations=ann,
                binary={'acc': torchmetrics.classification.BinaryAccuracy()},
                per_concept=42,
            )

    def test_clone_with_prefix_updates_per_concept(self):
        """clone(prefix=...) updates per-concept collection prefixes."""
        ann = self._make_binary_ann()
        m = ConceptMetrics(
            annotations=ann,
            binary={'acc': torchmetrics.classification.BinaryAccuracy()},
            per_concept=True,
        )
        preds = AnnotatedTensor(torch.randn(4, 2), ann, axis=-1)
        targets = AnnotatedTensor(torch.randint(0, 2, (4, 2)), ann.to_concept_space(), axis=-1)
        m.update(preds, targets)

        cloned = m.clone(prefix='test')
        # Check b1 collection has updated prefix
        assert cloned._per_concept  # non-empty
        for coll in cloned._per_concept.values():
            assert 'test/' in coll.prefix

    def test_clone_with_none_prefix_no_change(self):
        """clone(prefix=None) keeps existing prefix."""
        m = ConceptMetrics(
            annotations=self._make_binary_ann(),
            binary={'acc': torchmetrics.classification.BinaryAccuracy()},
        )
        original_prefix = m.binary.prefix
        cloned = m.clone(prefix=None)
        assert cloned.binary.prefix == original_prefix

    def test_categorical_summary_update(self):
        """Summary update for categorical concepts."""
        ann = self._make_cat_ann()
        m = ConceptMetrics(
            annotations=ann,
            categorical={'acc': torchmetrics.classification.MulticlassAccuracy},
        )
        preds = AnnotatedTensor(torch.randn(4, 3), ann, axis=-1)
        targets = AnnotatedTensor(torch.randint(0, 3, (4, 1)), ann.to_concept_space(), axis=-1)
        m.update(preds, targets)
        results = m.compute()
        assert len(results) > 0


class TestConceptMetricsContinuousPaths(unittest.TestCase):
    """Continuous-concept code paths.

    Continuous concepts are supported: a continuous concept reports ``loc`` (its
    mean), so metrics score it on that quantity rather than on logits. The update
    tests build a real continuous :class:`ModelOutput`; the collection/clone/compute
    tests reuse a binary metric with an injected continuous collection for brevity.
    """

    def _make_binary_metrics(self):
        """Return a ready-to-use binary-only ConceptMetrics instance."""
        ann = Annotations(
                labels=['b1'],
                cardinalities=[1],
                types=['binary'],
            )
        return ConceptMetrics(
            annotations=ann,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            prefix='t',
        )

    def _inject_continuous_collection(self, m):
        """Replace m.continuous with a populated MetricCollection."""
        from torchmetrics import MetricCollection
        from torchmetrics.regression import MeanSquaredError
        m.continuous = MetricCollection(
            {'mse': MeanSquaredError()},
            prefix='t/SUMMARY-continuous_',
        )
        return m

    # ------------------------------------------------------------------
    # collection property — line 168
    # ------------------------------------------------------------------
    def test_collection_includes_continuous_when_non_empty(self):
        """collection property adds 'continuous' key when non-empty (line 168)."""
        m = self._make_binary_metrics()
        self._inject_continuous_collection(m)
        coll = m.collection
        self.assertIn('continuous', coll)

    def test_collection_includes_per_concept_when_non_empty(self):
        """collection property adds per-concept keys when non-empty (lines 170-171)."""
        ann = Annotations(labels=['b1'], cardinalities=[1], types=['binary'])
        m = ConceptMetrics(
            annotations=ann,
            binary={'accuracy': torchmetrics.classification.BinaryAccuracy()},
            per_concept=True,
        )
        preds = AnnotatedTensor(torch.randn(4, 1), ann, axis=-1)
        targets = AnnotatedTensor(torch.randint(0, 2, (4, 1)), ann.to_concept_space(), axis=-1)
        m.update(preds, targets)
        coll = m.collection
        # _per_concept has 'b1' with a BinaryAccuracy inside — len > 0
        self.assertIn('b1', coll)

    # ------------------------------------------------------------------
    # clone() — line 189
    # ------------------------------------------------------------------
    def test_clone_updates_continuous_prefix(self):
        """clone(prefix=...) updates continuous collection prefix (line 189)."""
        m = self._make_binary_metrics()
        ann = Annotations(labels=['b1'], cardinalities=[1], types=['binary'])
        preds = AnnotatedTensor(torch.randn(4, 1), ann, axis=-1)
        targets = AnnotatedTensor(torch.randint(0, 2, (4, 1)), ann.to_concept_space(), axis=-1)
        m.update(preds, targets)
        self._inject_continuous_collection(m)
        cloned = m.clone(prefix='val')
        # The prefix of continuous should contain 'val/'
        self.assertIn('val/', cloned.continuous.prefix)

    # ------------------------------------------------------------------
    # update() continuous summary path — line 331
    # ------------------------------------------------------------------
    def test_update_summary_continuous(self):
        """Summary metric for continuous concepts is computed on ``loc``."""
        from torchmetrics.regression import MeanSquaredError
        from torch_concepts.tensor import AnnotatedTensor
        from torch_concepts.nn.modules.outputs import ModelOutput

        ann = Annotations(labels=['x', 'y'], cardinalities=[1, 1],
                          types=['continuous', 'continuous'])
        m = ConceptMetrics(
            annotations=ann,
            continuous={'mse': MeanSquaredError()},
            summary=True, per_concept=False,
        )
        out = ModelOutput()
        out.loc = AnnotatedTensor(torch.randn(4, 2), ann, axis=-1)
        out.target = AnnotatedTensor(torch.randn(4, 2), ann.to_concept_space(), axis=-1)
        m.update(out)
        result = m.compute()
        self.assertTrue(any('continuous' in k for k in result))

    # ------------------------------------------------------------------
    # update() per-concept continuous path — lines 343-344
    # ------------------------------------------------------------------
    def test_update_per_concept_continuous(self):
        """Per-concept metric for a continuous concept reads its ``loc`` and updates."""
        from torchmetrics.regression import MeanSquaredError
        from torch_concepts.tensor import AnnotatedTensor
        from torch_concepts.nn.modules.outputs import ModelOutput

        ann = Annotations(labels=['x'], cardinalities=[1], types=['continuous'])
        m = ConceptMetrics(
            annotations=ann,
            continuous={'mse': MeanSquaredError()},
            per_concept=True, summary=False,
        )
        out = ModelOutput()
        out.loc = AnnotatedTensor(torch.randn(4, 1), ann, axis=-1)
        out.target = AnnotatedTensor(torch.randn(4, 1), ann.to_concept_space(), axis=-1)
        m.update(out)
        result = m.compute()
        self.assertIn('x_mse', result)

    # ------------------------------------------------------------------
    # compute() continuous branch — line 354
    # ------------------------------------------------------------------
    def test_compute_includes_continuous(self):
        """compute() includes continuous results when collection is non-empty (line 354)."""
        m = self._make_binary_metrics()
        self._inject_continuous_collection(m)
        # Provide some data to the continuous collection so compute returns a value
        cont_val = torch.randn(4, 1)
        m.continuous.update(cont_val, cont_val)  # MSE of identical tensors = 0
        results = m.compute()
        self.assertTrue(any('continuous' in k.lower() or 'mse' in k.lower()
                            for k in results))
