"""Tests for ConceptGraph class."""
import unittest
import torch
from torch_concepts import ConceptGraph


class TestConceptGraph(unittest.TestCase):
    """Test suite for ConceptGraph functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple DAG: A -> B -> C
        #                       A -> C
        self.adj_matrix = torch.tensor([
            [0., 1., 1.],
            [0., 0., 1.],
            [0., 0., 0.]
        ])
        self.node_names = ['A', 'B', 'C']
        self.graph = ConceptGraph(self.adj_matrix, node_names=self.node_names)

    def test_initialization(self):
        """Test graph initialization."""
        self.assertEqual(self.graph.n_nodes, 3)
        self.assertEqual(self.graph.node_names, ['A', 'B', 'C'])
        self.assertTrue(torch.equal(self.graph.data, self.adj_matrix))

    def test_initialization_default_names(self):
        """Test graph initialization with default node names."""
        graph = ConceptGraph(self.adj_matrix)
        self.assertEqual(graph.node_names, ['node_0', 'node_1', 'node_2'])

    def test_initialization_validation(self):
        """Test graph initialization validation."""
        # Test non-2D tensor
        with self.assertRaises(ValueError):
            ConceptGraph(torch.randn(3))
        
        # Test non-square matrix
        with self.assertRaises(ValueError):
            ConceptGraph(torch.randn(3, 4))
        
        # Test mismatched node names
        with self.assertRaises(ValueError):
            ConceptGraph(self.adj_matrix, node_names=['A', 'B'])

    def test_indexing(self):
        """Test graph indexing."""
        # Test integer indexing
        self.assertEqual(self.graph[0, 1].item(), 1.0)
        self.assertEqual(self.graph[0, 2].item(), 1.0)
        self.assertEqual(self.graph[1, 2].item(), 1.0)
        
        # Test string indexing
        self.assertEqual(self.graph['A', 'B'].item(), 1.0)
        self.assertEqual(self.graph['A', 'C'].item(), 1.0)
        self.assertEqual(self.graph['B', 'C'].item(), 1.0)

    def test_get_edge_weight(self):
        """Test getting edge weights."""
        self.assertEqual(self.graph.get_edge_weight('A', 'B'), 1.0)
        self.assertEqual(self.graph.get_edge_weight('A', 'C'), 1.0)
        self.assertEqual(self.graph.get_edge_weight('B', 'A'), 0.0)

    def test_has_edge(self):
        """Test edge existence checking."""
        self.assertTrue(self.graph.has_edge('A', 'B'))
        self.assertTrue(self.graph.has_edge('A', 'C'))
        self.assertFalse(self.graph.has_edge('B', 'A'))
        self.assertFalse(self.graph.has_edge('C', 'A'))

    def test_to_pandas(self):
        """Test conversion to pandas DataFrame."""
        df = self.graph.to_pandas()
        self.assertEqual(list(df.index), ['A', 'B', 'C'])
        self.assertEqual(list(df.columns), ['A', 'B', 'C'])
        self.assertEqual(df.loc['A', 'B'], 1.0)
        self.assertEqual(df.loc['B', 'A'], 0.0)

    def test_to_networkx(self):
        """Test conversion to NetworkX graph."""
        G = self.graph.to_networkx()
        self.assertEqual(set(G.nodes()), {'A', 'B', 'C'})
        self.assertTrue(G.has_edge('A', 'B'))
        self.assertTrue(G.has_edge('A', 'C'))
        self.assertTrue(G.has_edge('B', 'C'))
        self.assertFalse(G.has_edge('B', 'A'))

    def test_dense_to_sparse(self):
        """Test conversion to sparse format."""
        edge_index, edge_weight = self.graph.dense_to_sparse()
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_index.shape[1], 3)  # 3 edges
        self.assertEqual(edge_weight.shape[0], 3)

    def test_get_root_nodes(self):
        """Test finding root nodes."""
        roots = self.graph.get_root_nodes()
        self.assertEqual(roots, ['A'])

    def test_get_leaf_nodes(self):
        """Test finding leaf nodes."""
        leaves = self.graph.get_leaf_nodes()
        self.assertEqual(leaves, ['C'])

    def test_topological_sort(self):
        """Test topological sorting."""
        # Create DAG: A -> B -> C
        adj = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32)
        graph = ConceptGraph(adj, node_names=['A', 'B', 'C'])

        topo_order = graph.topological_sort()
        # Verify A comes before B, B comes before C
        self.assertEqual(topo_order.index('A') < topo_order.index('B'), True)
        self.assertEqual(topo_order.index('B') < topo_order.index('C'), True)
    
    def test_from_sparse(self):
        """Test creating graph from sparse format directly."""
        # Create graph from sparse format
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
        edge_weight = torch.tensor([1.0, 2.0, 3.0])
        graph = ConceptGraph.from_sparse(
            edge_index, edge_weight, n_nodes=3, node_names=['A', 'B', 'C']
        )
        
        # Verify structure
        self.assertEqual(graph.n_nodes, 3)
        self.assertEqual(graph.node_names, ['A', 'B', 'C'])
        
        # Verify edges
        self.assertAlmostEqual(graph.get_edge_weight('A', 'B'), 1.0)
        self.assertAlmostEqual(graph.get_edge_weight('A', 'C'), 2.0)
        self.assertAlmostEqual(graph.get_edge_weight('B', 'C'), 3.0)
        self.assertAlmostEqual(graph.get_edge_weight('B', 'A'), 0.0)
        
        # Verify dense reconstruction matches
        expected_dense = torch.tensor([
            [0, 1, 2],
            [0, 0, 3],
            [0, 0, 0]
        ], dtype=torch.float32)
        self.assertTrue(torch.allclose(graph.data, expected_dense))

    def test_get_predecessors(self):
        """Test getting predecessors."""
        # C has predecessors A and B
        preds_c = set(self.graph.get_predecessors('C'))
        self.assertEqual(preds_c, {'A', 'B'})
        
        # B has predecessor A
        preds_b = self.graph.get_predecessors('B')
        self.assertEqual(preds_b, ['A'])
        
        # A has no predecessors
        preds_a = self.graph.get_predecessors('A')
        self.assertEqual(preds_a, [])

    def test_get_successors(self):
        """Test getting successors."""
        # A has successors B and C
        succs_a = set(self.graph.get_successors('A'))
        self.assertEqual(succs_a, {'B', 'C'})
        
        # B has successor C
        succs_b = self.graph.get_successors('B')
        self.assertEqual(succs_b, ['C'])
        
        # C has no successors
        succs_c = self.graph.get_successors('C')
        self.assertEqual(succs_c, [])

    def test_get_ancestors(self):
        """Test getting ancestors."""
        # C has ancestors A and B
        ancestors_c = self.graph.get_ancestors('C')
        self.assertEqual(ancestors_c, {'A', 'B'})
        
        # B has ancestor A
        ancestors_b = self.graph.get_ancestors('B')
        self.assertEqual(ancestors_b, {'A'})
        
        # A has no ancestors
        ancestors_a = self.graph.get_ancestors('A')
        self.assertEqual(ancestors_a, set())

    def test_get_descendants(self):
        """Test getting descendants."""
        # A has descendants B and C
        descendants_a = self.graph.get_descendants('A')
        self.assertEqual(descendants_a, {'B', 'C'})
        
        # B has descendant C
        descendants_b = self.graph.get_descendants('B')
        self.assertEqual(descendants_b, {'C'})
        
        # C has no descendants
        descendants_c = self.graph.get_descendants('C')
        self.assertEqual(descendants_c, set())

    def test_is_dag(self):
        """Test DAG checking."""
        self.assertTrue(self.graph.is_dag())
        self.assertTrue(self.graph.is_directed_acyclic())
        
        # Create a graph with a cycle
        cycle_adj = torch.tensor([
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.]
        ])
        cycle_graph = ConceptGraph(cycle_adj, node_names=['A', 'B', 'C'])
        self.assertFalse(cycle_graph.is_dag())


if __name__ == '__main__':
    unittest.main()
