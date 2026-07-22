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
                metadata={name: {'type': 'discrete'} for name in self.all_concept_names}
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

    def test_subset_metadata_preserved(self):
        """Test that metadata is correctly preserved for subset."""
        subset = ['concept_1', 'concept_3']
        dataset = ConceptDataset(
            self.X,
            self.C,
            annotations=self.annotations,
            concept_names_subset=subset
        )

        metadata = dataset.annotations.metadata
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
                metadata={name: {'type': 'discrete'} for name in self.all_concept_names}
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


class TestGeneratedGroundTruthSelection(unittest.TestCase):
    """Test selecting a generated concept source as ground truth."""

    def setUp(self):
        self.X = torch.randn(4, 3)
        self.C = torch.zeros(4, 2)
        self.annotations = Annotations(labels=["native_0", "native_1"])
        self.generated_annotations = {
            "first": Annotations(labels=["first_0"]),
            "second": Annotations(labels=["second_0", "second_1"]),
        }
        self.generated_values = {
            "first": torch.ones(4, 1),
            "second": torch.full((4, 2), 2.0),
        }

    def test_selects_named_generated_source_when_used_as_ground_truth(self):
        dataset = ConceptDataset(
            self.X,
            self.C,
            annotations=self.annotations,
        )

        dataset.set_generated_concepts(
            self.generated_annotations,
            self.generated_values,
            use_as_gt=True,
            generated_gt_name="second",
        )

        self.assertTrue(torch.equal(dataset.ground_truth, self.generated_values["second"]))
        self.assertEqual(dataset.concept_names, ["second_0", "second_1"])
        sample = dataset[0]
        self.assertNotIn("ground_truth", sample["concepts"])
        self.assertIs(sample["concepts"]["c"], sample["concepts"]["generated"]["second"])

    def test_missing_generated_ground_truth_source_raises(self):
        dataset = ConceptDataset(
            self.X,
            self.C,
            annotations=self.annotations,
        )

        with self.assertRaisesRegex(ValueError, "generated_gt_name='missing'"):
            dataset.set_generated_concepts(
                self.generated_annotations,
                self.generated_values,
                use_as_gt=True,
                generated_gt_name="missing",
            )

    def test_generate_concepts_runs_pipeline_with_kwargs(self):
        calls = []

        class Pipeline:
            def __call__(self, dataset, **kwargs):
                calls.append((dataset, kwargs))
                return self_values, self_annotations

        self_values = self.generated_values
        self_annotations = self.generated_annotations

        dataset = ConceptDataset(
            self.X,
            self.C,
            annotations=self.annotations,
        )
        dataset.generate_concepts(
            Pipeline(),
            num_concepts=2,
            use_as_gt=True,
            generated_gt_name="second",
        )

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], dataset)
        self.assertEqual(calls[0][1], {"class_names": None, "num_concepts": 2})
        self.assertTrue(torch.equal(dataset.ground_truth, self.generated_values["second"]))

    def test_generate_concepts_can_name_self_and_annotate_extra_datasets(self):
        calls = []
        val_dataset = ConceptDataset(
            torch.randn(2, 3),
            torch.zeros(2, 2),
            annotations=self.annotations,
        )

        class Pipeline:
            def __call__(self, dataset, **kwargs):
                calls.append((dataset, kwargs))
                return (
                    {
                        "train": torch.ones(len(dataset), 1),
                        "val": torch.zeros(len(val_dataset), 1),
                    },
                    {
                        "train": Annotations(labels=["generated"]),
                        "val": Annotations(labels=["generated"]),
                    },
                )

        dataset = ConceptDataset(
            self.X,
            self.C,
            annotations=self.annotations,
        )
        dataset.generate_concepts(
            Pipeline(),
            self_annotation_name="train",
            datasets_to_annotate={"val": val_dataset},
        )

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], dataset)
        self.assertEqual(calls[0][1]["class_names"], None)
        self.assertEqual(
            calls[0][1]["annotation_datasets"],
            {"train": dataset, "val": val_dataset},
        )
        self.assertEqual(set(dataset.generated_annotations), {"train", "val"})

    def test_native_c_reuses_native_sample_object(self):
        dataset = ConceptDataset(self.X, self.C, annotations=self.annotations)

        sample = dataset[0]

        self.assertIs(sample["concepts"]["c"], sample["concepts"]["native"])
        self.assertEqual(
            set(sample["concepts"]),
            {"c", "native", "generated"},
        )


if __name__ == '__main__':
    unittest.main()
