import unittest
import torch

from torch_concepts.data.base.dataset import ConceptDataset
from torch_concepts.annotations import Annotations, AxisAnnotation



class TestConceptSubset(unittest.TestCase):
    """Test concept_names_subset functionality in ConceptDataset."""

    def setUp(self):
        """Create a simple dataset with multiple concepts."""
        self.n_samples = 50
        self.X = torch.randn(self.n_samples, 10)
        self.C = torch.randint(0, 2, (self.n_samples, 5))
        self.all_concept_names = ['concept_0', 'concept_1', 'concept_2', 'concept_3', 'concept_4']
        self.annotations = Annotations({
            1: AxisAnnotation(
                labels=self.all_concept_names,
                cardinalities=(1, 1, 1, 1, 1),
                metadata={name: {'type': 'discrete'} for name in self.all_concept_names}
            )
        })

    def test_subset_selection(self):
        """Test that concept subset is correctly selected."""
        subset = ['concept_1', 'concept_3']
        dataset = ConceptDataset(
            self.X,
            self.C,
            annotations=self.annotations,
            concept_names_subset=subset
        )

        self.assertEqual(list(dataset.concept_names), subset)
        self.assertEqual(dataset.n_concepts, 2)
        self.assertEqual(dataset.concepts.shape[1], 2)

    def test_subset_preserves_order(self):
        """Test that concept subset preserves the order specified."""
        subset = ['concept_3', 'concept_0', 'concept_2']
        dataset = ConceptDataset(
            self.X,
            self.C,
            annotations=self.annotations,
            concept_names_subset=subset
        )

        self.assertEqual(list(dataset.concept_names), subset)

    def test_subset_missing_concepts_error(self):
        """Test that missing concepts raise clear error."""
        subset = ['concept_1', 'nonexistent_concept', 'another_missing']

        with self.assertRaises(AssertionError) as context:
            ConceptDataset(
                self.X,
                self.C,
                annotations=self.annotations,
                concept_names_subset=subset
            )

        error_msg = str(context.exception)
        self.assertIn('nonexistent_concept', error_msg)
        self.assertIn('another_missing', error_msg)
        self.assertIn('Concepts not found', error_msg)

    def test_subset_single_concept(self):
        """Test selecting a single concept."""
        subset = ['concept_2']
        dataset = ConceptDataset(
            self.X,
            self.C,
            annotations=self.annotations,
            concept_names_subset=subset
        )

        self.assertEqual(dataset.n_concepts, 1)
        self.assertEqual(dataset.concepts.shape[1], 1)

    def test_subset_metadata_preserved(self):
        """Test that metadata is correctly preserved for subset."""
        subset = ['concept_1', 'concept_3']
        dataset = ConceptDataset(
            self.X,
            self.C,
            annotations=self.annotations,
            concept_names_subset=subset
        )

        metadata = dataset.annotations[1].metadata
        self.assertEqual(set(metadata.keys()), set(subset))
        for name in subset:
            self.assertEqual(metadata[name]['type'], 'discrete')

    def test_subset_none_uses_all_concepts(self):
        """Test that None subset uses all concepts."""
        dataset = ConceptDataset(
            self.X,
            self.C,
            annotations=self.annotations,
            concept_names_subset=None
        )

        self.assertEqual(list(dataset.concept_names), self.all_concept_names)
        self.assertEqual(dataset.n_concepts, 5)


if __name__ == '__main__':
    unittest.main()
