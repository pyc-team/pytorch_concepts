import unittest
import torch
import pandas as pd

from torch_concepts.data.base.dataset import ConceptDataset
from torch_concepts.annotations import Annotations



class TestConceptSubset(unittest.TestCase):
    """Test concept_names_subset functionality in ConceptDataset."""

    def setUp(self):
        """Create a simple dataset with multiple concepts."""
        self.n_samples = 50
        self.X = torch.randn(self.n_samples, 10)
        self.C = torch.randint(0, 2, (self.n_samples, 5))
        self.all_concept_names = ['concept_0', 'concept_1', 'concept_2', 'concept_3', 'concept_4']
        self.annotations = Annotations(
                labels=self.all_concept_names,
                cardinalities=(1, 1, 1, 1, 1),
            )

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


class TestConceptSubsetWithGraph(unittest.TestCase):
    """Test concept_names_subset also subsets the graph."""

    def setUp(self):
        """Create a dataset with concepts and a graph."""
        self.n_samples = 50
        self.X = torch.randn(self.n_samples, 10)
        self.C = torch.randint(0, 2, (self.n_samples, 5))
        self.all_concept_names = ['c0', 'c1', 'c2', 'c3', 'c4']
        self.annotations = Annotations(
                labels=self.all_concept_names,
                cardinalities=(1, 1, 1, 1, 1),
            )
        # Graph: c0 -> c1 -> c2 -> c3 -> c4
        self.graph = pd.DataFrame(0, index=self.all_concept_names, columns=self.all_concept_names)
        self.graph.loc['c0', 'c1'] = 1
        self.graph.loc['c1', 'c2'] = 1
        self.graph.loc['c2', 'c3'] = 1
        self.graph.loc['c3', 'c4'] = 1

    def test_graph_subsetted_with_concepts(self):
        """Test that the graph is subsetted to match the concept subset."""
        subset = ['c1', 'c2', 'c3']
        dataset = ConceptDataset(
            self.X, self.C,
            annotations=self.annotations,
            graph=self.graph,
            concept_names_subset=subset
        )

        self.assertIsNotNone(dataset.graph)
        self.assertEqual(list(dataset.graph.node_names), subset)
        self.assertEqual(dataset.graph.data.shape, (3, 3))
        # c1 -> c2 edge should be preserved
        self.assertEqual(dataset.graph.data[0, 1].item(), 1)
        # c2 -> c3 edge should be preserved
        self.assertEqual(dataset.graph.data[1, 2].item(), 1)
        # no other edges
        self.assertEqual(dataset.graph.data.sum().item(), 2)

    def test_graph_subsetted_removes_disconnected(self):
        """Test that edges to excluded concepts are removed."""
        subset = ['c0', 'c3']
        dataset = ConceptDataset(
            self.X, self.C,
            annotations=self.annotations,
            graph=self.graph,
            concept_names_subset=subset
        )

        self.assertEqual(list(dataset.graph.node_names), subset)
        self.assertEqual(dataset.graph.data.shape, (2, 2))
        # No direct edge between c0 and c3 in original
        self.assertEqual(dataset.graph.data.sum().item(), 0)

    def test_graph_none_without_subset(self):
        """Test that graph works normally without concept subset."""
        dataset = ConceptDataset(
            self.X, self.C,
            annotations=self.annotations,
            graph=self.graph,
            concept_names_subset=None
        )

        self.assertEqual(list(dataset.graph.node_names), self.all_concept_names)
        self.assertEqual(dataset.graph.data.shape, (5, 5))
        self.assertEqual(dataset.graph.data.sum().item(), 4)

    def test_graph_single_concept_subset(self):
        """Test graph with a single concept subset."""
        subset = ['c2']
        dataset = ConceptDataset(
            self.X, self.C,
            annotations=self.annotations,
            graph=self.graph,
            concept_names_subset=subset
        )

        self.assertEqual(list(dataset.graph.node_names), subset)
        self.assertEqual(dataset.graph.data.shape, (1, 1))
        self.assertEqual(dataset.graph.data.sum().item(), 0)

    def test_graph_subsetted_node_names(self):
        """Test that graph node_names match the concept subset."""
        subset = ['c1', 'c3']
        dataset = ConceptDataset(
            self.X, self.C,
            annotations=self.annotations,
            graph=self.graph,
            concept_names_subset=subset
        )

        self.assertEqual(list(dataset.graph.node_names), subset)
        self.assertEqual(dataset.graph.data.shape, (2, 2))


class TestReorderByType(unittest.TestCase):
    """Test that mixed-type concepts are grouped contiguously by type."""

    def setUp(self):
        self.n_samples = 20
        self.X = torch.randn(self.n_samples, 10)
        # Interleaved types: categorical(card 2), binary, binary, categorical(card 3).
        self.labels = ['cat_a', 'bin_a', 'bin_b', 'cat_b']
        self.annotations = Annotations(labels=self.labels, cardinalities=(2, 1, 1, 3))
        self.C = torch.stack([
            torch.randint(0, 2, (self.n_samples,)),
            torch.randint(0, 2, (self.n_samples,)),
            torch.randint(0, 2, (self.n_samples,)),
            torch.randint(0, 3, (self.n_samples,)),
        ], dim=1)

    def test_default_groups_by_type(self):
        dataset = ConceptDataset(self.X, self.C, annotations=self.annotations)

        self.assertEqual(list(dataset.concept_names), ['bin_a', 'bin_b', 'cat_a', 'cat_b'])
        # Values follow their label, not their original column position.
        bin_a_col = self.labels.index('bin_a')
        torch.testing.assert_close(
            dataset.concepts.tensor[:, dataset.concept_names.index('bin_a')],
            self.C[:, bin_a_col].to(dataset.concepts.tensor.dtype),
        )

    def test_disabled_keeps_original_order(self):
        dataset = ConceptDataset(self.X, self.C, annotations=self.annotations, reorder_by_type=False)

        self.assertEqual(list(dataset.concept_names), self.labels)


class TestReorderByCardinality(unittest.TestCase):
    """Within categorical concepts, group further by ascending cardinality."""

    def test_categoricals_sorted_by_cardinality(self):
        n_samples = 20
        X = torch.randn(n_samples, 10)
        # cat_5 and cat_3 out of order, plus a binary concept thrown in.
        labels = ['cat_5', 'bin_a', 'cat_3', 'cat_3b']
        annotations = Annotations(labels=labels, cardinalities=(5, 1, 3, 3))
        C = torch.stack([
            torch.randint(0, 5, (n_samples,)),
            torch.randint(0, 2, (n_samples,)),
            torch.randint(0, 3, (n_samples,)),
            torch.randint(0, 3, (n_samples,)),
        ], dim=1)

        dataset = ConceptDataset(X, C, annotations=annotations)

        # binary first, then categoricals ascending by cardinality;
        # cat_3 before cat_3b since it was first in the original order (stable tie-break).
        self.assertEqual(list(dataset.concept_names), ['bin_a', 'cat_3', 'cat_3b', 'cat_5'])


if __name__ == '__main__':
    unittest.main()
