import unittest
import torch
from torch import nn

from torch_concepts.data.backbone import compute_backbone_embs
from torch_concepts.data.base.dataset import ConceptDataset
from torch_concepts.annotations import Annotations, AxisAnnotation


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


class TestEnsureList(unittest.TestCase):
    """Test suite for ensure_list utility function."""

    def test_list_remains_list(self):
        """Test that a list remains unchanged."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])
        
    def test_tuple_converts_to_list(self):
        """Test that a tuple is converted to list."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list((1, 2, 3))
        self.assertEqual(result, [1, 2, 3])
        self.assertIsInstance(result, list)
        
    def test_single_value_wraps_in_list(self):
        """Test that a single value is wrapped in a list."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list(5)
        self.assertEqual(result, [5])
        
        result = ensure_list(3.14)
        self.assertEqual(result, [3.14])
        
    def test_string_wraps_in_list(self):
        """Test that a string is wrapped (not converted to list of chars)."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list('hello')
        self.assertEqual(result, ['hello'])
        self.assertEqual(len(result), 1)
        
    def test_set_converts_to_list(self):
        """Test that a set is converted to list."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list({1, 2, 3})
        self.assertEqual(set(result), {1, 2, 3})
        self.assertIsInstance(result, list)
        
    def test_range_converts_to_list(self):
        """Test that a range is converted to list."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list(range(5))
        self.assertEqual(result, [0, 1, 2, 3, 4])
        
    def test_generator_converts_to_list(self):
        """Test that a generator is consumed and converted to list."""
        from torch_concepts.data.utils import ensure_list
        
        gen = (x * 2 for x in range(3))
        result = ensure_list(gen)
        self.assertEqual(result, [0, 2, 4])
        
    def test_numpy_array_converts_to_list(self):
        """Test that a numpy array is converted to list."""
        from torch_concepts.data.utils import ensure_list
        import numpy as np
        
        arr = np.array([1, 2, 3])
        result = ensure_list(arr)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result, list)
        
    def test_torch_tensor_converts_to_list(self):
        """Test that a torch tensor is converted to list."""
        from torch_concepts.data.utils import ensure_list
        
        tensor = torch.tensor([1, 2, 3])
        result = ensure_list(tensor)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result, list)
        
    def test_none_wraps_in_list(self):
        """Test that None is wrapped in a list."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list(None)
        self.assertEqual(result, [None])
        
    def test_nested_list_preserved(self):
        """Test that nested lists are preserved."""
        from torch_concepts.data.utils import ensure_list
        
        nested = [[1, 2], [3, 4]]
        result = ensure_list(nested)
        self.assertEqual(result, [[1, 2], [3, 4]])
        
    def test_dict_raises_error(self):
        """Test that a dict raises TypeError with helpful message."""
        from torch_concepts.data.utils import ensure_list
        
        with self.assertRaises(TypeError) as context:
            ensure_list({'a': 1, 'b': 2})
        
        self.assertIn('Cannot convert dict to list', str(context.exception))
        self.assertIn('keys', str(context.exception))
        self.assertIn('values', str(context.exception))
        
    def test_empty_list_remains_empty(self):
        """Test that an empty list remains empty."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list([])
        self.assertEqual(result, [])
        
    def test_empty_tuple_converts_to_empty_list(self):
        """Test that an empty tuple converts to empty list."""
        from torch_concepts.data.utils import ensure_list
        
        result = ensure_list(())
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
