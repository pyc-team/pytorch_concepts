import unittest
import torch

from torch_concepts.base import ConceptTensor, ConceptDistribution
from torch_concepts.nn import ConceptLayer, ProbabilisticConceptLayer, ConceptMemory


class TestConceptClasses(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.concept_dimensions = {
            1: ["concept_a", "concept_b"],
            2: 3  # This should create 3 default concept names
        }
        self.input_tensor = torch.randn(5, self.in_features)
        self.encoder = ConceptLayer(self.in_features, self.concept_dimensions)
        self.prob_encoder = ProbabilisticConceptLayer(self.in_features, self.concept_dimensions)
        self.memory_size = 10
        self.emb_size = 5
        self.memory = ConceptMemory(self.memory_size, self.emb_size, self.concept_dimensions)

    def test_concept_encoder(self):
        output = self.encoder.forward(self.input_tensor)
        self.assertIsInstance(output, ConceptTensor)
        expected_shape = (5, 2, 3)  # Batch size 5, 2 concepts, 3 concepts in the second dimension
        self.assertEqual(output.shape, expected_shape)
        expected_concept_names = {
            1: ["concept_a", "concept_b"],
            2: ["concept_2_0", "concept_2_1", "concept_2_2"],
            0: ['concept_0_0', 'concept_0_1', 'concept_0_2', 'concept_0_3', 'concept_0_4']
        }
        self.assertEqual(output.concept_names, expected_concept_names)

    def test_generative_concept_encoder(self):
        output_dist = self.prob_encoder.forward(self.input_tensor)
        self.assertIsInstance(output_dist, ConceptDistribution)
        expected_shape = (5, 2, 3)  # Batch size 5, 2 concepts, 3 concepts in the second dimension
        self.assertEqual(output_dist.base_dist.mean.shape, expected_shape)
        self.assertEqual(output_dist.base_dist.variance.shape, expected_shape)
        expected_concept_names = {
            1: ["concept_a", "concept_b"],
            2: ["concept_2_0", "concept_2_1", "concept_2_2"]
        }
        self.assertEqual(self.prob_encoder.concept_names, expected_concept_names)
        output = output_dist.rsample()
        self.assertIsInstance(output, ConceptTensor)
        self.assertEqual(output.shape, expected_shape)

    def test_concept_memory(self):
        output = self.memory.forward()
        self.assertIsInstance(output, ConceptTensor)
        expected_shape = (10, 2, 3)  # Memory size 10, 2 concepts, 3 concepts in the second dimension
        self.assertEqual(output.shape, expected_shape)
        expected_concept_names = {
            1: ["concept_a", "concept_b"],
            2: ["concept_2_0", "concept_2_1", "concept_2_2"]
        }
        self.assertEqual(self.memory.concept_names, expected_concept_names)


if __name__ == '__main__':
    unittest.main()
