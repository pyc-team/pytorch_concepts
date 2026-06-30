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


class TestConceptGraphCacheInvalidation(unittest.TestCase):
    """Test that NX graph cache is invalidated on edge mutation."""

    def test_edge_weight_mutation_invalidates_cache(self):
        """Modifying edge_weight should invalidate the cached NX graph."""
        adj = torch.tensor([[0., 1.], [0., 0.]])
        g = ConceptGraph(adj, node_names=['A', 'B'])

        # Trigger cache creation
        _ = g._nx_graph
        self.assertIsNotNone(g._nx_graph_cache)

        # Mutate edge weights
        g.edge_weight = torch.tensor([5.0])
        self.assertIsNone(g._nx_graph_cache)

        # New cache should reflect the mutation
        nx_g = g._nx_graph
        self.assertAlmostEqual(nx_g['A']['B']['weight'], 5.0)

    def test_edge_index_mutation_invalidates_cache(self):
        """Modifying edge_index should invalidate the cached NX graph."""
        adj = torch.tensor([[0., 1.], [0., 0.]])
        g = ConceptGraph(adj, node_names=['A', 'B'])

        _ = g._nx_graph
        self.assertIsNotNone(g._nx_graph_cache)

        # Reverse direction: B -> A
        g.edge_index = torch.tensor([[1], [0]])
        self.assertIsNone(g._nx_graph_cache)

        nx_g = g._nx_graph
        self.assertTrue(nx_g.has_edge('B', 'A'))
        self.assertFalse(nx_g.has_edge('A', 'B'))


if __name__ == '__main__':
    unittest.main()


class TestConceptGraphModuleFunctions(unittest.TestCase):
    """Test module-level functions in concept_graph.py."""

    def setUp(self):
        self.adj = torch.tensor([[0., 1., 1.], [0., 0., 1.], [0., 0., 0.]])
        self.names = ['A', 'B', 'C']
        self.graph = ConceptGraph(self.adj, node_names=self.names)

    def test_dense_to_sparse_empty_graph(self):
        """Empty graph returns empty tensors."""
        from torch_concepts.concept_graph import dense_to_sparse
        zero = torch.zeros(3, 3)
        g = ConceptGraph(zero, node_names=['X', 'Y', 'Z'])
        ei, ew = dense_to_sparse(g)
        self.assertEqual(ei.shape[1], 0)
        self.assertEqual(ew.shape[0], 0)

    def test_dense_to_sparse_tensor_input(self):
        from torch_concepts.concept_graph import dense_to_sparse
        ei, ew = dense_to_sparse(self.adj)
        self.assertEqual(ei.shape[0], 2)
        self.assertEqual(ew.shape[0], 3)

    def test_to_networkx_graph_function(self):
        from torch_concepts.concept_graph import to_networkx_graph
        G = to_networkx_graph(self.graph)
        self.assertTrue(G.has_edge('A', 'B'))

    def test_to_networkx_graph_with_threshold(self):
        from torch_concepts.concept_graph import to_networkx_graph
        adj = torch.tensor([[0., 0.1, 1.], [0., 0., 1.], [0., 0., 0.]])
        G = to_networkx_graph(adj, node_names=self.names, threshold=0.5)
        self.assertFalse(G.has_edge('A', 'B'))  # 0.1 <= 0.5 threshold
        self.assertTrue(G.has_edge('A', 'C'))   # 1.0 > 0.5

    def test_to_networkx_graph_tensor_default_names(self):
        from torch_concepts.concept_graph import to_networkx_graph
        adj = torch.tensor([[0., 1.], [0., 0.]])
        G = to_networkx_graph(adj)  # no names → integer indices
        self.assertIn(0, G.nodes())
        self.assertIn(1, G.nodes())

    def test_get_root_nodes_function(self):
        from torch_concepts.concept_graph import get_root_nodes
        roots = get_root_nodes(self.graph)
        self.assertEqual(roots, ['A'])

    def test_get_root_nodes_from_tensor(self):
        from torch_concepts.concept_graph import get_root_nodes
        roots = get_root_nodes(self.adj, node_names=self.names)
        self.assertIn('A', roots)

    def test_get_root_nodes_from_networkx(self):
        import networkx as nx
        from torch_concepts.concept_graph import get_root_nodes
        G = nx.DiGraph()
        G.add_edges_from([('X', 'Y'), ('X', 'Z')])
        roots = get_root_nodes(G)
        self.assertEqual(roots, ['X'])

    def test_get_leaf_nodes_function(self):
        from torch_concepts.concept_graph import get_leaf_nodes
        leaves = get_leaf_nodes(self.graph)
        self.assertEqual(leaves, ['C'])

    def test_get_leaf_nodes_from_tensor(self):
        from torch_concepts.concept_graph import get_leaf_nodes
        leaves = get_leaf_nodes(self.adj, node_names=self.names)
        self.assertIn('C', leaves)

    def test_get_leaf_nodes_from_networkx(self):
        import networkx as nx
        from torch_concepts.concept_graph import get_leaf_nodes
        G = nx.DiGraph()
        G.add_edges_from([('X', 'Y'), ('X', 'Z')])
        leaves = get_leaf_nodes(G)
        self.assertIn('Y', leaves)

    def test_topological_sort_function(self):
        from torch_concepts.concept_graph import topological_sort
        order = topological_sort(self.graph)
        self.assertLess(order.index('A'), order.index('B'))

    def test_topological_sort_from_tensor(self):
        from torch_concepts.concept_graph import topological_sort
        order = topological_sort(self.adj, node_names=self.names)
        self.assertLess(order.index('A'), order.index('C'))

    def test_topological_sort_from_networkx(self):
        import networkx as nx
        from torch_concepts.concept_graph import topological_sort
        G = nx.DiGraph()
        G.add_edges_from([('X', 'Y')])
        order = topological_sort(G)
        self.assertLess(order.index('X'), order.index('Y'))

    def test_get_predecessors_function_from_networkx(self):
        import networkx as nx
        from torch_concepts.concept_graph import get_predecessors
        G = nx.DiGraph()
        G.add_edges_from([('A', 'C'), ('B', 'C')])
        preds = set(get_predecessors(G, 'C'))
        self.assertEqual(preds, {'A', 'B'})

    def test_get_predecessors_from_tensor(self):
        from torch_concepts.concept_graph import get_predecessors
        preds = set(get_predecessors(self.adj, 'C', node_names=self.names))
        self.assertIn('A', preds)

    def test_get_successors_function_from_networkx(self):
        import networkx as nx
        from torch_concepts.concept_graph import get_successors
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('A', 'C')])
        succs = set(get_successors(G, 'A'))
        self.assertEqual(succs, {'B', 'C'})

    def test_get_successors_from_tensor(self):
        from torch_concepts.concept_graph import get_successors
        succs = set(get_successors(self.adj, 'A', node_names=self.names))
        self.assertIn('B', succs)

    def test_get_predecessors_int_node_with_names(self):
        import networkx as nx
        from torch_concepts.concept_graph import get_predecessors
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B')])
        # int node with node_names provided
        preds = get_predecessors(G, 1, node_names=['A', 'B'])
        self.assertEqual(preds, ['A'])

    def test_get_successors_int_node_with_names(self):
        import networkx as nx
        from torch_concepts.concept_graph import get_successors
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B')])
        succs = get_successors(G, 0, node_names=['A', 'B'])
        self.assertEqual(succs, ['B'])

    def test_get_levels_function(self):
        levels = self.graph.get_levels()
        # A is at level 0 (root), B and C after
        self.assertIn('A', levels[0])

    def test_from_sparse_node_name_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ConceptGraph.from_sparse(
                torch.tensor([[0], [1]]),
                torch.tensor([1.0]),
                n_nodes=3,
                node_names=['X', 'Y'],  # only 2 names for 3 nodes
            )

    def test_node_to_index_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            self.graph._node_to_index(3.14)

    def test_node_to_index_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            self.graph._node_to_index(99)

    def test_node_to_index_unknown_string_raises(self):
        with self.assertRaises(ValueError):
            self.graph._node_to_index('ZZ')

    def test_getitem_dense_fallback(self):
        # row-wise index returns the full row from dense matrix
        row = self.graph[0]
        self.assertEqual(row.shape[0], 3)

    def test_get_levels_isolated_nodes(self):
        """Isolated (no-edge) node defaults to depth 0."""
        adj = torch.zeros(2, 2)
        g = ConceptGraph(adj, node_names=['X', 'Y'])
        levels = g.get_levels()
        all_nodes = [n for level in levels for n in level]
        self.assertIn('X', all_nodes)
        self.assertIn('Y', all_nodes)
