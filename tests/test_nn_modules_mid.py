"""
Comprehensive tests for torch_concepts.nn.modules.mid

Tests mid-level modules (base, constructors, inference, models).
"""
import unittest
import torch
import torch.nn as nn
from torch_concepts.annotations import Annotations, AxisAnnotation
from torch_concepts.nn.modules.mid.base.model import BaseConstructor
from torch_concepts.nn.modules.mid.constructors.concept_graph import ConceptGraph


class TestBaseConstructor(unittest.TestCase):
    """Test BaseConstructor."""

    def setUp(self):
        """Set up test annotations and layers."""
        concept_labels = ('color', 'shape', 'size')
        self.annotations = Annotations({
            1: AxisAnnotation(labels=concept_labels)
        })
        self.encoder = nn.Linear(784, 3)
        self.predictor = nn.Linear(3, 10)

    def test_initialization(self):
        """Test base constructor initialization."""
        constructor = BaseConstructor(
            input_size=784,
            annotations=self.annotations,
            encoder=self.encoder,
            predictor=self.predictor
        )
        self.assertEqual(constructor.input_size, 784)
        self.assertIsNotNone(constructor.annotations)
        self.assertEqual(len(constructor.labels), 3)

    def test_name_to_id_mapping(self):
        """Test name to ID mapping."""
        constructor = BaseConstructor(
            input_size=784,
            annotations=self.annotations,
            encoder=self.encoder,
            predictor=self.predictor
        )
        self.assertIn('color', constructor.name2id)
        self.assertIn('shape', constructor.name2id)
        self.assertIn('size', constructor.name2id)
        self.assertEqual(constructor.name2id['color'], 0)


class TestConceptGraph(unittest.TestCase):
    """Test ConceptGraph."""

    def test_initialization(self):
        """Test concept graph initialization."""
        adj = torch.tensor([[0., 1., 1.],
                           [0., 0., 1.],
                           [0., 0., 0.]])
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        self.assertEqual(graph.n_nodes, 3)
        self.assertEqual(len(graph.node_names), 3)

    def test_get_root_nodes(self):
        """Test getting root nodes."""
        adj = torch.tensor([[0., 1., 1.],
                           [0., 0., 1.],
                           [0., 0., 0.]])
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        roots = graph.get_root_nodes()
        self.assertIn('A', roots)
        self.assertEqual(len(roots), 1)

    def test_get_leaf_nodes(self):
        """Test getting leaf nodes."""
        adj = torch.tensor([[0., 1., 1.],
                           [0., 0., 1.],
                           [0., 0., 0.]])
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        leaves = graph.get_leaf_nodes()
        self.assertIn('C', leaves)

    def test_has_edge(self):
        """Test edge existence checking."""
        adj = torch.tensor([[0., 1., 0.],
                           [0., 0., 1.],
                           [0., 0., 0.]])
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        self.assertTrue(graph.has_edge('A', 'B'))
        self.assertTrue(graph.has_edge('B', 'C'))
        self.assertFalse(graph.has_edge('A', 'C'))
        self.assertFalse(graph.has_edge('B', 'A'))

    def test_get_successors(self):
        """Test getting successor nodes."""
        adj = torch.tensor([[0., 1., 1.],
                           [0., 0., 1.],
                           [0., 0., 0.]])
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        successors_a = graph.get_successors('A')
        self.assertIn('B', successors_a)
        self.assertIn('C', successors_a)

    def test_get_predecessors(self):
        """Test getting predecessor nodes."""
        adj = torch.tensor([[0., 1., 1.],
                           [0., 0., 1.],
                           [0., 0., 0.]])
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        predecessors_c = graph.get_predecessors('C')
        self.assertIn('A', predecessors_c)
        self.assertIn('B', predecessors_c)

    def test_is_dag(self):
        """Test DAG checking."""
        # Acyclic graph
        adj_dag = torch.tensor([[0., 1., 0.],
                                [0., 0., 1.],
                                [0., 0., 0.]])
        graph_dag = ConceptGraph(adj_dag, node_names=['A', 'B', 'C'])
        self.assertTrue(graph_dag.is_dag())

    def test_topological_sort(self):
        """Test topological sorting."""
        adj = torch.tensor([[0., 1., 1.],
                           [0., 0., 1.],
                           [0., 0., 0.]])
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        topo_order = graph.topological_sort()

        # A should come before B and C
        # B should come before C
        idx_a = topo_order.index('A')
        idx_b = topo_order.index('B')
        idx_c = topo_order.index('C')
        self.assertLess(idx_a, idx_b)
        self.assertLess(idx_a, idx_c)
        self.assertLess(idx_b, idx_c)

    def test_to_networkx(self):
        """Test conversion to NetworkX."""
        adj = torch.tensor([[0., 1., 0.],
                           [0., 0., 1.],
                           [0., 0., 0.]])
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        nx_graph = graph.to_networkx()

        self.assertEqual(nx_graph.number_of_nodes(), 3)
        self.assertTrue(nx_graph.has_edge('A', 'B'))

    def test_to_pandas(self):
        """Test conversion to pandas DataFrame."""
        adj = torch.tensor([[0., 1., 0.],
                           [0., 0., 1.],
                           [0., 0., 0.]])
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        df = graph.to_pandas()

        self.assertIsNotNone(df)
        # Should have at least 2 edges (A->B and B->C)
        self.assertGreaterEqual(len(df), 2)

    def test_from_sparse(self):
        """Test creation from sparse format."""
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
        edge_weight = torch.tensor([1.0, 1.0, 1.0])
        graph = ConceptGraph.from_sparse(
            edge_index, edge_weight, n_nodes=3,
            node_names=['X', 'Y', 'Z']
        )
        self.assertEqual(graph.n_nodes, 3)
        self.assertTrue(graph.has_edge('X', 'Y'))
        self.assertTrue(graph.has_edge('X', 'Z'))

    def test_empty_graph(self):
        """Test empty graph."""
        adj = torch.zeros(3, 3)
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])
        self.assertEqual(graph.n_nodes, 3)
        self.assertFalse(graph.has_edge('A', 'B'))


if __name__ == '__main__':
    unittest.main()
