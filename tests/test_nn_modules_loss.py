"""
Comprehensive tests for torch_concepts.nn.modules.loss

Tests weighted loss functions for concept-based learning:
- WeightedBCEWithLogitsLoss
- WeightedCrossEntropyLoss
- WeightedMSELoss
"""
import unittest
import torch
from torch_concepts.nn.modules.loss import (
    WeightedBCEWithLogitsLoss,
    WeightedCrossEntropyLoss,
    WeightedMSELoss,
)


class TestWeightedBCEWithLogitsLoss(unittest.TestCase):
    """Test weighted BCE with logits loss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = WeightedBCEWithLogitsLoss()

        concept_logits = torch.randn(32, 10)
        task_logits = torch.randn(32, 5)
        concept_targets = torch.randint(0, 2, (32, 10)).float()
        task_targets = torch.randint(0, 2, (32, 5)).float()

        loss = loss_fn(concept_logits, task_logits, concept_targets, task_targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertTrue(loss >= 0)

    def test_weighted_loss(self):
        """Test with concept loss weight."""
        loss_fn = WeightedBCEWithLogitsLoss(concept_loss_weight=0.8)

        concept_logits = torch.randn(16, 8)
        task_logits = torch.randn(16, 3)
        concept_targets = torch.randint(0, 2, (16, 8)).float()
        task_targets = torch.randint(0, 2, (16, 3)).float()

        loss = loss_fn(concept_logits, task_logits, concept_targets, task_targets)

        self.assertTrue(loss >= 0)

    def test_weight_extremes(self):
        """Test with extreme weight values."""
        # All weight on concepts
        loss_fn_concepts = WeightedBCEWithLogitsLoss(concept_loss_weight=1.0)
        # All weight on tasks
        loss_fn_tasks = WeightedBCEWithLogitsLoss(concept_loss_weight=0.0)

        concept_logits = torch.randn(10, 5)
        task_logits = torch.randn(10, 3)
        concept_targets = torch.randint(0, 2, (10, 5)).float()
        task_targets = torch.randint(0, 2, (10, 3)).float()

        loss_concepts = loss_fn_concepts(concept_logits, task_logits, concept_targets, task_targets)
        loss_tasks = loss_fn_tasks(concept_logits, task_logits, concept_targets, task_targets)

        # Both should be valid
        self.assertTrue(loss_concepts >= 0)
        self.assertTrue(loss_tasks >= 0)

    def test_no_weight_unweighted_sum(self):
        """Test that None weight gives unweighted sum."""
        loss_fn = WeightedBCEWithLogitsLoss(concept_loss_weight=None)

        concept_logits = torch.randn(8, 4)
        task_logits = torch.randn(8, 2)
        concept_targets = torch.randint(0, 2, (8, 4)).float()
        task_targets = torch.randint(0, 2, (8, 2)).float()

        loss = loss_fn(concept_logits, task_logits, concept_targets, task_targets)
        self.assertTrue(loss >= 0)

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        loss_fn = WeightedBCEWithLogitsLoss(concept_loss_weight=0.5)

        concept_logits = torch.randn(4, 3, requires_grad=True)
        task_logits = torch.randn(4, 2, requires_grad=True)
        concept_targets = torch.randint(0, 2, (4, 3)).float()
        task_targets = torch.randint(0, 2, (4, 2)).float()

        loss = loss_fn(concept_logits, task_logits, concept_targets, task_targets)
        loss.backward()

        self.assertIsNotNone(concept_logits.grad)
        self.assertIsNotNone(task_logits.grad)


class TestWeightedCrossEntropyLoss(unittest.TestCase):
    """Test weighted cross-entropy loss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = WeightedCrossEntropyLoss()

        # CrossEntropyLoss expects (batch, n_classes) for logits and (batch,) for targets
        concept_logits = torch.randn(32, 10)  # 32 samples, 10 classes
        task_logits = torch.randn(32, 3)      # 32 samples, 3 classes
        concept_targets = torch.randint(0, 10, (32,))
        task_targets = torch.randint(0, 3, (32,))

        loss = loss_fn(concept_logits, concept_targets, task_logits, task_targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    def test_weighted_loss(self):
        """Test with concept loss weight."""
        loss_fn = WeightedCrossEntropyLoss(concept_loss_weight=0.6)

        concept_logits = torch.randn(16, 5)
        task_logits = torch.randn(16, 4)
        concept_targets = torch.randint(0, 5, (16,))
        task_targets = torch.randint(0, 4, (16,))

        loss = loss_fn(concept_logits, concept_targets, task_logits, task_targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss >= 0)

    def test_multiclass_classification(self):
        """Test with multi-class classification."""
        loss_fn = WeightedCrossEntropyLoss(concept_loss_weight=0.7)

        # Many classes
        concept_logits = torch.randn(8, 20)
        task_logits = torch.randn(8, 15)
        concept_targets = torch.randint(0, 20, (8,))
        task_targets = torch.randint(0, 15, (8,))

        loss = loss_fn(concept_logits, concept_targets, task_logits, task_targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss >= 0)

    def test_gradient_flow(self):
        """Test gradient flow."""
        loss_fn = WeightedCrossEntropyLoss(concept_loss_weight=0.5)

        concept_logits = torch.randn(4, 5, requires_grad=True)
        task_logits = torch.randn(4, 4, requires_grad=True)
        concept_targets = torch.randint(0, 5, (4,))
        task_targets = torch.randint(0, 4, (4,))

        loss = loss_fn(concept_logits, concept_targets, task_logits, task_targets)
        loss.backward()

        self.assertIsNotNone(concept_logits.grad)
        self.assertIsNotNone(task_logits.grad)


class TestWeightedMSELoss(unittest.TestCase):
    """Test weighted MSE loss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = WeightedMSELoss()

        concept_preds = torch.randn(32, 10)
        task_preds = torch.randn(32, 5)
        concept_targets = torch.randn(32, 10)
        task_targets = torch.randn(32, 5)

        loss = loss_fn(concept_preds, concept_targets, task_preds, task_targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

    def test_weighted_loss(self):
        """Test with concept loss weight."""
        loss_fn = WeightedMSELoss(concept_loss_weight=0.75)

        concept_preds = torch.randn(16, 8)
        task_preds = torch.randn(16, 3)
        concept_targets = torch.randn(16, 8)
        task_targets = torch.randn(16, 3)

        loss = loss_fn(concept_preds, concept_targets, task_preds, task_targets)
        self.assertTrue(loss >= 0)

    def test_regression_task(self):
        """Test with continuous regression values."""
        loss_fn = WeightedMSELoss(concept_loss_weight=0.5)

        concept_preds = torch.randn(10, 5) * 100  # Large values
        task_preds = torch.randn(10, 2) * 100
        concept_targets = torch.randn(10, 5) * 100
        task_targets = torch.randn(10, 2) * 100

        loss = loss_fn(concept_preds, concept_targets, task_preds, task_targets)
        self.assertTrue(loss >= 0)

    def test_perfect_predictions(self):
        """Test with perfect predictions (zero loss)."""
        loss_fn = WeightedMSELoss(concept_loss_weight=0.5)

        concept_preds = torch.randn(5, 3)
        task_preds = torch.randn(5, 2)

        # Targets same as predictions
        loss = loss_fn(concept_preds, concept_preds, task_preds, task_preds)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_gradient_flow(self):
        """Test gradient flow."""
        loss_fn = WeightedMSELoss(concept_loss_weight=0.5)

        concept_preds = torch.randn(4, 3, requires_grad=True)
        task_preds = torch.randn(4, 2, requires_grad=True)
        concept_targets = torch.randn(4, 3)
        task_targets = torch.randn(4, 2)

        loss = loss_fn(concept_preds, concept_targets, task_preds, task_targets)
        loss.backward()

        self.assertIsNotNone(concept_preds.grad)
        self.assertIsNotNone(task_preds.grad)

    def test_reduction_modes(self):
        """Test different reduction modes."""
        for reduction in ['mean', 'sum']:
            loss_fn = WeightedMSELoss(concept_loss_weight=0.5, reduction=reduction)

            concept_preds = torch.randn(8, 4)
            task_preds = torch.randn(8, 2)
            concept_targets = torch.randn(8, 4)
            task_targets = torch.randn(8, 2)

            loss = loss_fn(concept_preds, concept_targets, task_preds, task_targets)
            self.assertTrue(loss >= 0)


class TestLossComparison(unittest.TestCase):
    """Test comparisons between different loss weighting strategies."""

    def test_weight_effect(self):
        """Test that weight actually affects loss distribution."""
        torch.manual_seed(42)

        # Create data where concept loss is much higher
        concept_logits = torch.randn(10, 5) * 5  # High variance
        task_logits = torch.randn(10, 2)
        concept_targets = torch.randint(0, 2, (10, 5)).float()
        task_targets = torch.randint(0, 2, (10, 2)).float()

        loss_fn_high_concept = WeightedBCEWithLogitsLoss(concept_loss_weight=0.9)
        loss_fn_high_task = WeightedBCEWithLogitsLoss(concept_loss_weight=0.1)

        loss_high_concept = loss_fn_high_concept(concept_logits, task_logits, concept_targets, task_targets)
        loss_high_task = loss_fn_high_task(concept_logits, task_logits, concept_targets, task_targets)

        # Losses should be different
        self.assertNotAlmostEqual(loss_high_concept.item(), loss_high_task.item(), places=2)


if __name__ == '__main__':
    unittest.main()
