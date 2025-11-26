"""
Comprehensive tests for torch_concepts.nn.modules.mid.constructors

Tests for BipartiteModel and GraphModel constructors.
"""
import unittest
import torch
import pandas as pd
from torch_concepts.annotations import Annotations, AxisAnnotation
from torch_concepts import ConceptGraph
from torch_concepts.nn import BipartiteModel, LinearCC
from torch_concepts.nn import GraphModel
from torch_concepts.nn import LazyConstructor
from torch.distributions import Bernoulli


class TestGraphModel(unittest.TestCase):
    """Test GraphModel."""

    def setUp(self):
        """Set up test data."""
        # Create a simple DAG: A -> C, B -> C, C -> D
        self.concept_names = ['A', 'B', 'C', 'D']
        graph_df = pd.DataFrame(0, index=self.concept_names, columns=self.concept_names)
        graph_df.loc['A', 'C'] = 1
        graph_df.loc['B', 'C'] = 1
        graph_df.loc['C', 'D'] = 1

        self.graph = ConceptGraph(
            torch.FloatTensor(graph_df.values),
            node_names=self.concept_names
        )

        # Create annotations
        metadata = {name: {'distribution': Bernoulli} for name in self.concept_names}
        self.annotations = Annotations({
            1: AxisAnnotation(labels=tuple(self.concept_names), metadata=metadata)
        })

    def test_initialization(self):
        """Test graph model initialization."""
        model = GraphModel(
            model_graph=self.graph,
            input_size=784,
            annotations=self.annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC)
        )
        self.assertIsNotNone(model)
        self.assertTrue(self.graph.is_dag())

    def test_root_and_internal_nodes(self):
        """Test identification of root and internal nodes."""
        model = GraphModel(
            model_graph=self.graph,
            input_size=784,
            annotations=self.annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC)
        )
        # A and B have no parents (root nodes)
        # C and D have parents (internal nodes)
        root_nodes = model.root_nodes
        internal_nodes = model.internal_nodes

        self.assertTrue('A' in root_nodes)
        self.assertTrue('B' in root_nodes)
        self.assertTrue('C' in internal_nodes or 'D' in internal_nodes)

    def test_topological_order(self):
        """Test topological ordering of graph."""
        model = GraphModel(
            model_graph=self.graph,
            input_size=784,
            annotations=self.annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC)
        )
        order = model.graph_order
        # Check that parents come before children
        a_idx = order.index('A')
        c_idx = order.index('C')
        d_idx = order.index('D')

        self.assertLess(a_idx, c_idx)
        self.assertLess(c_idx, d_idx)

    def test_simple_chain(self):
        """Test with simple chain graph: A -> B -> C."""
        chain_names = ['A', 'B', 'C']
        graph_df = pd.DataFrame(0, index=chain_names, columns=chain_names)
        graph_df.loc['A', 'B'] = 1
        graph_df.loc['B', 'C'] = 1

        graph = ConceptGraph(
            torch.FloatTensor(graph_df.values),
            node_names=chain_names
        )

        metadata = {name: {'distribution': Bernoulli} for name in chain_names}
        annotations = Annotations({
            1: AxisAnnotation(labels=tuple(chain_names), metadata=metadata)
        })

        model = GraphModel(
            model_graph=graph,
            input_size=784,
            annotations=annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC)
        )
        self.assertEqual(len(model.root_nodes), 1)
        self.assertIn('A', model.root_nodes)

    def test_disconnected_components(self):
        """Test with disconnected graph components."""
        names = ['A', 'B', 'C', 'D']
        graph_df = pd.DataFrame(0, index=names, columns=names)
        # A -> B (component 1)
        # C -> D (component 2)
        graph_df.loc['A', 'B'] = 1
        graph_df.loc['C', 'D'] = 1

        graph = ConceptGraph(
            torch.FloatTensor(graph_df.values),
            node_names=names
        )

        metadata = {name: {'distribution': Bernoulli} for name in names}
        annotations = Annotations({
            1: AxisAnnotation(labels=tuple(names), metadata=metadata)
        })

        model = GraphModel(
            model_graph=graph,
            input_size=784,
            annotations=annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC)
        )
        # Should have 2 root nodes (A and C)
        self.assertEqual(len(model.root_nodes), 2)
        self.assertIn('A', model.root_nodes)
        self.assertIn('C', model.root_nodes)

    def test_star_topology(self):
        """Test star topology: A -> B, A -> C, A -> D."""
        names = ['A', 'B', 'C', 'D']
        graph_df = pd.DataFrame(0, index=names, columns=names)
        graph_df.loc['A', 'B'] = 1
        graph_df.loc['A', 'C'] = 1
        graph_df.loc['A', 'D'] = 1

        graph = ConceptGraph(
            torch.FloatTensor(graph_df.values),
            node_names=names
        )

        metadata = {name: {'distribution': Bernoulli} for name in names}
        annotations = Annotations({
            1: AxisAnnotation(labels=tuple(names), metadata=metadata)
        })

        model = GraphModel(
            model_graph=graph,
            input_size=784,
            annotations=annotations,
            encoder=LazyConstructor(torch.nn.Linear),
            predictor=LazyConstructor(LinearCC)
        )
        # A is the only root
        self.assertEqual(len(model.root_nodes), 1)
        self.assertIn('A', model.root_nodes)
        # B, C, D are all internal
        self.assertEqual(len(model.internal_nodes), 3)


if __name__ == '__main__':
    unittest.main()
