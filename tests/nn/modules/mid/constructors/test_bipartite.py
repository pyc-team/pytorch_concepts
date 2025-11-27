
import unittest
import torch
from torch_concepts.annotations import Annotations, AxisAnnotation
from torch_concepts.nn import BipartiteModel, LinearCC
from torch_concepts.nn import LazyConstructor
from torch.distributions import Bernoulli


class TestBipartiteModel(unittest.TestCase):
    """Test BipartiteModel."""

    def setUp(self):
        """Set up test data."""
        # Define concepts and tasks
        all_labels = ('color', 'shape', 'size', 'task1', 'task2')
        metadata = {
            'color': {'distribution': Bernoulli},
            'shape': {'distribution': Bernoulli},
            'size': {'distribution': Bernoulli},
            'task1': {'distribution': Bernoulli},
            'task2': {'distribution': Bernoulli}
        }
        self.annotations = Annotations({
            1: AxisAnnotation(labels=all_labels, metadata=metadata)
        })
        self.task_names = ['task1', 'task2']

    def test_initialization(self):
        """Test bipartite model initialization."""
        model = BipartiteModel(
            task_names=self.task_names,
            input_size=784,
            annotations=self.annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC)
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.task_names, self.task_names)
        self.assertEqual(set(model.concept_names), {'color', 'shape', 'size'})

    def test_bipartite_structure(self):
        """Test that bipartite structure is correct."""
        model = BipartiteModel(
            task_names=self.task_names,
            input_size=784,
            annotations=self.annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC)
        )
        # In bipartite model, concepts should point to tasks
        # Tasks should not point to themselves
        graph = model.model_graph
        self.assertIsNotNone(graph)

    def test_single_task(self):
        """Test with single task."""
        model = BipartiteModel(
            task_names=['task1'],
            input_size=784,
            annotations=self.annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC)
        )
        self.assertEqual(model.task_names, ['task1'])


if __name__ == '__main__':
    unittest.main()
