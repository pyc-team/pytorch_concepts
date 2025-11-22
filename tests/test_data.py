import unittest
import torch
from torch import nn

from torch_concepts.data.datasets import ToyDataset, CompletenessDataset
from torch_concepts.data.datasets.toy import _xor, _trigonometry, _dot, _checkmark, _complete
from torch_concepts.data.backbone import compute_backbone_embs


class TestToyDataset(unittest.TestCase):

    def setUp(self):
        self.size = 100
        self.random_state = 42
        self.xor_data = ToyDataset('xor', size=self.size, random_state=self.random_state)
        self.trigonometry_data = ToyDataset('trigonometry', size=self.size, random_state=self.random_state)
        self.dot_data = ToyDataset('dot', size=self.size, random_state=self.random_state)
        self.checkmark_data = ToyDataset('checkmark', size=self.size, random_state=self.random_state)
        self.complete = CompletenessDataset(n_samples=self.size, n_concepts=7, n_hidden_concepts=0,
                                            n_tasks=3, random_state=self.random_state)
        self.incomplete = CompletenessDataset(n_samples=self.size, n_concepts=7, n_hidden_concepts=4,
                                                n_tasks=3, random_state=self.random_state)

    def test_length(self):
        self.assertEqual(len(self.xor_data), self.size)
        self.assertEqual(len(self.trigonometry_data), self.size)
        self.assertEqual(len(self.dot_data), self.size)
        self.assertEqual(len(self.checkmark_data), self.size)

    def test_label_names(self):
        self.assertEqual(self.xor_data.concept_attr_names, ['C1', 'C2'])
        self.assertEqual(self.xor_data.task_attr_names, ['xor'])
        self.assertEqual(self.trigonometry_data.concept_attr_names, ['C1', 'C2', 'C3'])
        self.assertEqual(self.trigonometry_data.task_attr_names, ['sumGreaterThan1'])
        self.assertEqual(self.dot_data.concept_attr_names, ['dotV1V2GreaterThan0', 'dotV3V4GreaterThan0'])
        self.assertEqual(self.dot_data.task_attr_names, ['dotV1V3GreaterThan0'])
        self.assertEqual(self.checkmark_data.concept_attr_names, ['A', 'B', 'C'])
        self.assertEqual(self.checkmark_data.task_attr_names, ['D'])
        self.assertEqual(self.complete.concept_attr_names, ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
        self.assertEqual(self.complete.task_attr_names, ['y0', 'y1', 'y2'])
        self.assertEqual(self.incomplete.concept_attr_names, ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
        self.assertEqual(self.incomplete.task_attr_names, ['y0', 'y1', 'y2'])

    def test_xor_item(self):
        x, c, y, dag, concept_names, target_names = _xor(self.size, self.random_state)
        for i in range(self.size):
            data, concept_label, target_label = self.xor_data[i]
            self.assertTrue(torch.equal(data, x[i]))
            self.assertTrue(torch.equal(concept_label, c[i]))
            self.assertTrue(torch.equal(target_label, y[i]))

    def test_trigonometric_item(self):
        x, c, y, dag, concept_names, target_names = _trigonometry(self.size, self.random_state)
        for i in range(self.size):
            data, concept_label, target_label = self.trigonometry_data[i]
            self.assertTrue(torch.equal(data, x[i]))
            self.assertTrue(torch.equal(concept_label, c[i]))
            self.assertTrue(torch.equal(target_label, y[i]))

    def test_dot_item(self):
        x, c, y, dag, concept_names, target_names = _dot(self.size, self.random_state)
        for i in range(self.size):
            data, concept_label, target_label = self.dot_data[i]
            self.assertTrue(torch.equal(data, x[i]))
            self.assertTrue(torch.equal(concept_label, c[i]))
            self.assertTrue(torch.equal(target_label, y[i]))

    def test_checkmark_item(self):
        x, c, y, dag, concept_names, target_names = _checkmark(self.size, self.random_state)
        for i in range(self.size):
            data, concept_label, target_label = self.checkmark_data[i]
            self.assertTrue(torch.equal(data, x[i]))
            self.assertTrue(torch.equal(concept_label, c[i]))
            self.assertTrue(torch.equal(target_label, y[i]))


class TestBackboneTrainingStatePreservation(unittest.TestCase):
    """Test that compute_backbone_embs preserves the training state of the model."""

    def setUp(self):
        # Create a simple backbone model
        self.backbone = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU()
        )
        # Create a simple dataset
        X = torch.randn(20, 10)
        self.dataset = [{'x': X[i]} for i in range(len(X))]

    def test_preserves_training_mode(self):
        """Test that a model in training mode is restored to training mode."""
        self.backbone.train()
        self.assertTrue(self.backbone.training, "Model should start in training mode")
        
        _ = compute_backbone_embs(
            self.dataset, 
            self.backbone, 
            batch_size=4, 
            verbose=False
        )
        
        self.assertTrue(
            self.backbone.training, 
            "Model should be restored to training mode after compute_backbone_embs"
        )

    def test_preserves_eval_mode(self):
        """Test that a model in eval mode remains in eval mode."""
        self.backbone.eval()
        self.assertFalse(self.backbone.training, "Model should start in eval mode")
        
        _ = compute_backbone_embs(
            self.dataset, 
            self.backbone, 
            batch_size=4, 
            verbose=False
        )
        
        self.assertFalse(
            self.backbone.training, 
            "Model should remain in eval mode after compute_backbone_embs"
        )

    def test_embeddings_computed_correctly(self):
        """Test that embeddings are computed with correct shape."""
        embs = compute_backbone_embs(
            self.dataset, 
            self.backbone, 
            batch_size=4, 
            verbose=False
        )
        
        self.assertEqual(embs.shape[0], len(self.dataset), "Should have one embedding per sample")
        self.assertEqual(embs.shape[1], 5, "Embedding dimension should match backbone output")


if __name__ == '__main__':
    unittest.main()
