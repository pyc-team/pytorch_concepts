import unittest
import torch

from torch_concepts.base import AnnotatedTensor
from torch_concepts.nn import Annotate, LinearConceptLayer


class TestAnnotate(unittest.TestCase):
    def setUp(self):
        self.annotations = [["concept1", "concept2"], ["concept3", "concept4"]]
        self.annotated_axis = [1, 2]
        self.annotate_layer = Annotate(self.annotations, self.annotated_axis)
        self.input_tensor = torch.randn(5, 2, 2)

    def test_forward(self):
        annotated_tensor = self.annotate_layer(self.input_tensor)
        self.assertIsInstance(annotated_tensor, AnnotatedTensor)
        self.assertTrue(torch.equal(annotated_tensor.to_standard_tensor(), self.input_tensor))
        self.assertEqual(annotated_tensor.annotations, [None, *self.annotations])

class TestLinearConceptLayer(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.annotations = [["concept1", "concept2"], 4, ["concept3", "concept4", "concept5"]]
        self.layer = LinearConceptLayer(self.in_features, self.annotations)
        self.input_tensor = torch.randn(5, self.in_features)

    def test_shape(self):
        expected_shape = [2, 4, 3]
        self.assertEqual(self.layer.shape(), expected_shape)

    def test_forward(self):
        output = self.layer(self.input_tensor)
        self.assertIsInstance(output, AnnotatedTensor)
        self.assertEqual(output.shape, (5, *self.layer.shape()))
        self.assertEqual(output.annotations, [None, ["concept1", "concept2"], None, ["concept3", "concept4", "concept5"]])


if __name__ == '__main__':
    unittest.main()
