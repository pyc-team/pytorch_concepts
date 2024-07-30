import unittest
import torch

from torch_concepts.base import ConceptTensor
from torch_concepts.nn import ConceptEncoder, ConceptScorer, ProbabilisticConceptEncoder, LogicMemory


class TestConceptClasses(unittest.TestCase):

    def setUp(self):
        self.in_features = 10
        self.n_concepts = 5
        self.emb_size = 4
        self.batch_size = 3
        self.concept_names = ["A", "B", "C", "D", "E"]

    def test_concept_encoder(self):
        encoder = ConceptEncoder(self.in_features, self.n_concepts, self.emb_size)
        x = torch.randn(self.batch_size, self.in_features)
        result = encoder(x)

        # Test output shape
        self.assertEqual(result.shape, (self.batch_size, self.n_concepts, self.emb_size))

        # Test emb_size=1 case
        encoder = ConceptEncoder(self.in_features, self.n_concepts, emb_size=1)
        result = encoder(x)
        self.assertEqual(result.shape, (self.batch_size, self.n_concepts))

    def test_generative_concept_encoder(self):
        encoder = ProbabilisticConceptEncoder(self.in_features, self.n_concepts, self.emb_size, concept_names=self.concept_names)
        x = torch.randn(self.batch_size, self.in_features)
        qz_x = encoder(x)
        result = qz_x.rsample()

        # Test output type
        self.assertIsInstance(result, ConceptTensor)

        # Test output shape
        self.assertEqual(result.shape, (self.batch_size, self.n_concepts, self.emb_size))

        # Test concept names
        self.assertEqual(result.concept_names, self.concept_names)

        # Test sampling properties
        self.assertIsNotNone(qz_x)
        self.assertIsNotNone(encoder.p_z)

        # Test emb_size=1 case
        encoder = ProbabilisticConceptEncoder(self.in_features, self.n_concepts, emb_size=1, concept_names=self.concept_names)
        qz_x = encoder(x)
        result = qz_x.rsample()
        self.assertEqual(result.shape, (self.batch_size, self.n_concepts))
        self.assertEqual(result.concept_names, self.concept_names)

    def test_concept_scorer(self):
        scorer = ConceptScorer(self.emb_size)
        x = ConceptTensor.concept(torch.randn(self.batch_size, self.n_concepts, self.emb_size), concept_names=self.concept_names)
        result = scorer(x)

        # Test output shape
        self.assertEqual(result.shape, (self.batch_size, self.n_concepts))

    def test_logic_memory(self):
        n_rules, rule_emb_size, n_tasks = 2, 6, 7
        memory = LogicMemory(self.n_concepts, n_tasks, memory_size=n_rules, rule_emb_size=rule_emb_size, concept_names=self.concept_names)

        decoded_rules = memory._decode_rules()

        # Test output shape
        self.assertEqual(decoded_rules.shape, (n_tasks, n_rules, self.n_concepts, 3))

        x = ConceptTensor.concept(torch.randn(self.batch_size, self.n_concepts), concept_names=self.concept_names)
        y_per_rule, c_reconstruction = memory(x)

        # Test output shape
        self.assertEqual(y_per_rule.shape, (self.batch_size, n_tasks, n_rules))
        self.assertEqual(c_reconstruction.shape, (self.batch_size, n_tasks, n_rules, self.n_concepts))

        # Test sampling case
        y_per_rule, c_reconstruction = memory(x, idxs=torch.randint(0, n_rules, (self.batch_size, n_tasks,)).long())
        self.assertEqual(y_per_rule.shape, (self.batch_size, n_tasks, 1))
        self.assertEqual(c_reconstruction.shape, (self.batch_size, n_tasks, 1, self.n_concepts))

    def test_concept_tensor_creation(self):
        x = torch.randn(self.batch_size, self.n_concepts)
        c = ConceptTensor.concept(x, concept_names=self.concept_names)

        # Test concept names
        self.assertEqual(c.concept_names, self.concept_names)

        # Test default concept names
        c_default = ConceptTensor.concept(x)
        self.assertEqual(c_default.concept_names, [f"concept_{i}" for i in range(self.n_concepts)])

        # Test mismatch in concept names length
        with self.assertRaises(ValueError):
            ConceptTensor.concept(x, concept_names=["A", "B"])

    def test_assign_concept_names(self):
        x = torch.randn(self.batch_size, self.n_concepts)
        c = ConceptTensor.concept(x)

        # Assign new concept names
        new_concept_names = ["V", "W", "X", "Y", "Z"]
        c.assign_concept_names(new_concept_names)

        # Test new concept names
        self.assertEqual(c.concept_names, new_concept_names)

        # Test mismatch in concept names length
        with self.assertRaises(ValueError):
            c.assign_concept_names(["A", "B"])

    def test_extract_by_concept_names(self):
        x = torch.randn(self.batch_size, self.n_concepts)
        c = ConceptTensor.concept(x, concept_names=self.concept_names)

        # Extract by concept names
        extracted_c = c.extract_by_concept_names(["A", "C", "E"])

        # Test shape of extracted tensor
        self.assertEqual(extracted_c.shape, (self.batch_size, 3))

        # Test concept names of extracted tensor
        self.assertEqual(extracted_c.concept_names, ["A", "C", "E"])


if __name__ == '__main__':
    unittest.main()
