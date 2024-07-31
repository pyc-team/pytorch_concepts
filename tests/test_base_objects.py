import unittest
import torch

from torch_concepts.base import ConceptTensor


class TestConceptTensor(unittest.TestCase):
    def setUp(self):
        self.data = torch.randn(5, 4)
        self.concept_names = {
            1: ["concept_a", "concept_b", "concept_c", "concept_d"]
        }
        self.tensor = ConceptTensor(self.data, self.concept_names)

    def test_creation(self):
        self.assertEqual(self.tensor.shape, self.data.shape)
        self.assertEqual(self.tensor.concept_names, self.concept_names)

    def test_default_concept_names(self):
        data = torch.randn(3, 4, 5)
        tensor = ConceptTensor(data)
        expected_names = {
            0: [f"concept_0_{i}" for i in range(3)],
            1: [f"concept_1_{i}" for i in range(4)],
            2: [f"concept_2_{i}" for i in range(5)]
        }
        self.assertEqual(tensor.concept_names, expected_names)

    def test_assign_concept_names(self):
        new_concept_names = {
            1: ["new_a", "new_b", "new_c", "new_d"]
        }
        self.tensor.assign_concept_names(new_concept_names)
        self.assertEqual(self.tensor.concept_names, new_concept_names)

    def test_extract_by_concept_names(self):
        target_concepts = {1: ["concept_a", "concept_c"]}
        extracted_tensor = self.tensor.extract_by_concept_names(target_concepts)
        self.assertEqual(extracted_tensor.shape, (5, 2))
        self.assertEqual(extracted_tensor.concept_names[1], ["concept_a", "concept_c"])

    def test_extract_by_indices(self):
        target_concepts = {1: [0, 2]}
        extracted_tensor = self.tensor.extract_by_concept_names(target_concepts)
        self.assertEqual(extracted_tensor.shape, (5, 2))
        self.assertEqual(extracted_tensor.concept_names[1], ["concept_a", "concept_c"])

    def test_invalid_concept_names(self):
        with self.assertRaises(ValueError):
            ConceptTensor(self.data, {1: ["concept_a", "concept_b"]})

    def test_invalid_dimension(self):
        with self.assertRaises(ValueError):
            ConceptTensor(self.data, {2: ["concept_a", "concept_b", "concept_c", "concept_d"]})

    def test_new_empty(self):
        empty_tensor = self.tensor.new_empty(5, 4)
        self.assertEqual(empty_tensor.shape, (5, 4))
        self.assertEqual(empty_tensor.concept_names, self.concept_names)

    def test_to_standard_tensor(self):
        standard_tensor = self.tensor.to_standard_tensor()
        self.assertTrue(isinstance(standard_tensor, torch.Tensor))
        self.assertFalse(isinstance(standard_tensor, ConceptTensor))

    def test_transpose(self):
        transposed_tensor = self.tensor.transpose(0, 1)
        self.assertEqual(transposed_tensor.shape, (4, 5))
        self.assertEqual(transposed_tensor.concept_names[1], ["concept_0_0", "concept_0_1", "concept_0_2", "concept_0_3", "concept_0_4"])

    def test_permute(self):
        data = torch.randn(2, 3, 4)
        concept_names = {
            0: ["batch_0", "batch_1"],
            1: ["channel_0", "channel_1", "channel_2"],
            2: ["width_0", "width_1", "width_2", "width_3"]
        }
        tensor = ConceptTensor(data, concept_names)
        permuted_tensor = tensor.permute(2, 0, 1)
        self.assertEqual(permuted_tensor.shape, (4, 2, 3))
        self.assertEqual(permuted_tensor.concept_names[0], ["width_0", "width_1", "width_2", "width_3"])
        self.assertEqual(permuted_tensor.concept_names[1], ["batch_0", "batch_1"])
        self.assertEqual(permuted_tensor.concept_names[2], ["channel_0", "channel_1", "channel_2"])

    def test_squeeze(self):
        data = torch.randn(1, 5, 4)
        tensor = ConceptTensor(data, {1: ["concept_a", "concept_b", "concept_c", "concept_d", "concept_e"]})
        squeezed_tensor = tensor.squeeze(0)
        self.assertEqual(squeezed_tensor.shape, (5, 4))
        self.assertEqual(squeezed_tensor.concept_names, {0: ["concept_a", "concept_b", "concept_c", "concept_d", "concept_e"], 1: ["concept_2_0", "concept_2_1", "concept_2_2", "concept_2_3"]})
        squeezed_tensor = tensor.squeeze()
        self.assertEqual(squeezed_tensor.shape, (5, 4))

    def test_unsqueeze(self):
        unsqueezed_tensor = self.tensor.unsqueeze(0)
        self.assertEqual(unsqueezed_tensor.shape, (1, 5, 4))
        self.assertEqual(unsqueezed_tensor.concept_names[1], ["concept_0_0", "concept_0_1", "concept_0_2", "concept_0_3", "concept_0_4"])
        unsqueezed_tensor = self.tensor.unsqueeze(-1)
        self.assertEqual(unsqueezed_tensor.shape, (5, 4, 1))

    def test_update_concept_names(self):
        new_concept_names = {
            1: ["new_a", "new_b", "new_c", "new_d"]
        }
        self.tensor.update_concept_names(new_concept_names)
        self.assertEqual(self.tensor.concept_names[1], ["new_a", "new_b", "new_c", "new_d"])

    def test_view(self):
        view_tensor = self.tensor.view(10, 2)
        self.assertEqual(view_tensor.shape, (10, 2))
        self.assertEqual(view_tensor.concept_names[1], ["concept_1_0", "concept_1_1"])

    def test_ravel(self):
        raveled_tensor = self.tensor.ravel()
        self.assertEqual(raveled_tensor.shape, (20,))


if __name__ == '__main__':
    unittest.main()
