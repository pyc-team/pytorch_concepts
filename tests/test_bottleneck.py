import unittest
import torch
import torch.nn.functional as F
from torch_concepts.nn.bottleneck import LinearConceptBottleneck, LinearConceptResidualBottleneck, ConceptEmbeddingBottleneck
from torch_concepts.concepts.base import AnnotatedTensor

class TestLinearConceptBottleneck(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.annotations = ["concept1", "concept2"]
        self.activation = F.sigmoid
        self.bottleneck = LinearConceptBottleneck(self.in_features, self.annotations, self.activation)
        self.input_tensor = torch.randn(5, self.in_features)

    def test_predict(self):
        output = self.bottleneck.predict(self.input_tensor)
        self.assertEqual(output.shape, (5, len(self.annotations)))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))

    def test_transform(self):
        c_int, intermediate = self.bottleneck.transform(self.input_tensor)
        self.assertIsInstance(c_int, AnnotatedTensor)
        self.assertIn('c_pred', intermediate)
        self.assertIn('c_int', intermediate)

    def test_annotations(self):
        # throw error if annotations is not a list
        with self.assertRaises(AssertionError):
            LinearConceptBottleneck(self.in_features, [self.annotations, 3], self.activation)

class TestLinearConceptResidualBottleneck(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.annotations = ["concept1", "concept2"]
        self.residual_size = 5
        self.activation = F.sigmoid
        self.bottleneck = LinearConceptResidualBottleneck(self.in_features, self.annotations, self.residual_size, self.activation)
        self.input_tensor = torch.randn(5, self.in_features)

    def test_predict(self):
        output = self.bottleneck.predict(self.input_tensor)
        self.assertEqual(output.shape, (5, len(self.annotations)))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))

    def test_transform(self):
        c_new, intermediate = self.bottleneck.transform(self.input_tensor)
        self.assertIsInstance(c_new, AnnotatedTensor)
        self.assertIn('c_pred', intermediate)
        self.assertIn('c_int', intermediate)
        self.assertEqual(c_new.shape[-1], len(self.annotations) + self.residual_size)

class TestConceptEmbeddingBottleneck(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.annotations = ["concept1", "concept2"]
        self.concept_embedding_size = 7
        self.activation = F.sigmoid
        self.bottleneck = ConceptEmbeddingBottleneck(self.in_features, self.annotations,
                                                     self.concept_embedding_size, self.activation)
        self.input_tensor = torch.randn(5, self.in_features)

    def test_predict(self):
        output = self.bottleneck.predict(self.input_tensor)
        self.assertEqual(output.shape, (5, 2))

    def test_transform(self):
        c_mix, intermediate = self.bottleneck.transform(self.input_tensor)
        self.assertIsInstance(c_mix, AnnotatedTensor)
        self.assertEqual(c_mix.shape[-1], 7)
        self.assertIn('c_pred', intermediate)
        self.assertIn('c_int', intermediate)

if __name__ == "__main__":
    unittest.main()
