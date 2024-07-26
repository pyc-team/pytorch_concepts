import unittest
import torch

from torch_concepts.data import ToyDataset, CompletenessDataset
from torch_concepts.data.toy import _xor, _trigonometry, _dot, _checkmark, _complete


class TestToyDataset(unittest.TestCase):

    def setUp(self):
        self.size = 100
        self.random_state = 42
        self.xor_data = ToyDataset('xor', size=self.size, random_state=self.random_state)
        self.trigonometry_data = ToyDataset('trigonometry', size=self.size, random_state=self.random_state)
        self.dot_data = ToyDataset('dot', size=self.size, random_state=self.random_state)
        self.checkmark_data = ToyDataset('checkmark', size=self.size, random_state=self.random_state)
        self.complete = CompletenessDataset(n_samples=self.size, n_features=100, n_concepts=7, n_hidden_concepts=0,
                                            n_tasks=3, emb_size=8, random_state=self.random_state)
        self.incomplete = CompletenessDataset(n_samples=self.size, n_features=100, n_concepts=7, n_hidden_concepts=4,
                                                n_tasks=3, emb_size=8, random_state=self.random_state)

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
        self.assertEqual(self.complete.concept_attr_names, ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'])
        self.assertEqual(self.complete.task_attr_names, ['y0', 'y1', 'y2'])
        self.assertEqual(self.incomplete.concept_attr_names, ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'R0', 'R1', 'R2', 'R3'])
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

    def test_completeness_item(self):
        x, c, y, dag, concept_names, target_names = _complete(self.size, 100, 7, 0, 3, 8, self.random_state)
        for i in range(self.size):
            data, concept_label, target_label = self.complete[i]
            self.assertTrue(torch.equal(data, x[i]))
            self.assertTrue(torch.equal(concept_label, c[i]))
            self.assertTrue(torch.equal(target_label, y[i]))

        x, c, y, dag, concept_names, target_names = _complete(self.size, 100, 7, 4, 3, 8, self.random_state)
        for i in range(self.size):
            data, concept_label, target_label = self.incomplete[i]
            self.assertTrue(torch.equal(data, x[i]))
            self.assertTrue(torch.equal(concept_label, c[i]))
            self.assertTrue(torch.equal(target_label, y[i]))


if __name__ == '__main__':
    unittest.main()
