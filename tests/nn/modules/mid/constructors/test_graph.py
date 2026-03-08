"""
Comprehensive tests for torch_concepts.nn.modules.mid.constructors

Tests for BipartiteModel and GraphModel constructors.
"""
import unittest
import torch
import pandas as pd
from torch_concepts.annotations import Annotations, AxisAnnotation
from torch_concepts import ConceptGraph
from torch_concepts.nn import BipartiteModel, LinearConceptToConcept
from torch_concepts.nn import GraphModel
from torch_concepts.nn import LazyConstructor
from torch_concepts.nn import (
    DeterministicInference,
    LinearLatentToExogenous,
    LinearExogenousToConcept,
    HyperlinearConceptExogenousToConcept,
)
from torch.distributions import Bernoulli, Categorical


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
            predictor=LazyConstructor(LinearConceptToConcept)
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
            predictor=LazyConstructor(LinearConceptToConcept)
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
            predictor=LazyConstructor(LinearConceptToConcept)
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
            predictor=LazyConstructor(LinearConceptToConcept)
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
            predictor=LazyConstructor(LinearConceptToConcept)
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
            predictor=LazyConstructor(LinearConceptToConcept)
        )
        # A is the only root
        self.assertEqual(len(model.root_nodes), 1)
        self.assertIn('A', model.root_nodes)
        # B, C, D are all internal
        self.assertEqual(len(model.internal_nodes), 3)

    def test_substring_concept_names_with_self_exogenous(self):
        """Test that concept names sharing substrings don't cause incorrect exogenous matching.

        Regression test for a bug where 'OtherCar' would incorrectly match
        exogenous variables for 'OtherCarCost' because startswith('exog_OtherCar')
        also matches 'exog_OtherCarCost_state_*'. The fix uses the more specific
        pattern startswith('exog_{name}_state_').
        """
        # Graph: OtherCar -> OtherCarCost (substring relationship)
        names = ['OtherCar', 'OtherCarCost']
        graph_df = pd.DataFrame(0, index=names, columns=names)
        graph_df.loc['OtherCar', 'OtherCarCost'] = 1

        graph = ConceptGraph(
            torch.FloatTensor(graph_df.values),
            node_names=names
        )

        # OtherCar: binary (cardinality=1), OtherCarCost: categorical with 4 classes
        metadata = {
            'OtherCar': {'distribution': Bernoulli},
            'OtherCarCost': {'distribution': Categorical},
        }
        annotations = Annotations({
            1: AxisAnnotation(
                labels=tuple(names),
                cardinalities=[1, 4],
                metadata=metadata,
            )
        })

        latent_dim = 16
        exog_size = 8

        # Build model with both source and internal exogenous variables
        model = GraphModel(
            model_graph=graph,
            input_size=latent_dim,
            annotations=annotations,
            source_exogenous=LazyConstructor(LinearLatentToExogenous, out_exogenous=exog_size),
            internal_exogenous=LazyConstructor(LinearLatentToExogenous, out_exogenous=exog_size),
            encoder=LazyConstructor(LinearExogenousToConcept),
            predictor=LazyConstructor(HyperlinearConceptExogenousToConcept, hidden_size=4),
        )

        # Verify exogenous variables are correctly assigned
        pm = model.probabilistic_model

        # The internal exogenous for OtherCarCost should have exactly 4 vars
        # (one per state), not 5 (which would happen if OtherCar's exog leaked in)
        othercar_exog = [v for v in pm.variables
                         if v.concept.startswith('exog_OtherCar_state_')]
        othercarcost_exog = [v for v in pm.variables
                             if v.concept.startswith('exog_OtherCarCost_state_')]
        self.assertEqual(len(othercar_exog), 1,
                         "OtherCar (binary) should have 1 source exogenous variable")
        self.assertEqual(len(othercarcost_exog), 4,
                         "OtherCarCost (4-class categorical) should have 4 internal exogenous variables")

        # Verify the OtherCarCost predictor variable only has correct parents
        othercarcost_var = pm.concept_to_variable['OtherCarCost']
        parent_names = [p.concept for p in othercarcost_var.parents]
        # Should NOT contain any 'exog_OtherCar_state_*' parents
        leaked_parents = [p for p in parent_names if p.startswith('exog_OtherCar_state_')]
        self.assertEqual(len(leaked_parents), 0,
                         f"OtherCarCost should not have OtherCar's exogenous as parents, "
                         f"but found: {leaked_parents}")

        # Run a forward pass to verify no shape mismatch
        inference = DeterministicInference(pm)
        x = torch.randn(4, latent_dim)
        result = inference.query(['OtherCar', 'OtherCarCost'], evidence={'input': x})
        # OtherCar: 1 output, OtherCarCost: 4 outputs => total 5
        self.assertEqual(result.shape, (4, 5),
                         f"Expected output shape (4, 5), got {result.shape}")

    def test_substring_concept_names_with_source_exogenous_only(self):
        """Test substring safety when using use_source_exogenous=True (no internal exogenous).

        Verifies that source exogenous variables for parent concepts are correctly
        matched when concept names share substrings.
        """
        # Graph: OtherCar -> OtherCarCost
        names = ['OtherCar', 'OtherCarCost']
        graph_df = pd.DataFrame(0, index=names, columns=names)
        graph_df.loc['OtherCar', 'OtherCarCost'] = 1

        graph = ConceptGraph(
            torch.FloatTensor(graph_df.values),
            node_names=names
        )

        metadata = {
            'OtherCar': {'distribution': Bernoulli},
            'OtherCarCost': {'distribution': Categorical},
        }
        annotations = Annotations({
            1: AxisAnnotation(
                labels=tuple(names),
                cardinalities=[1, 4],
                metadata=metadata,
            )
        })

        latent_dim = 16
        exog_size = 8

        model = GraphModel(
            model_graph=graph,
            input_size=latent_dim,
            annotations=annotations,
            source_exogenous=LazyConstructor(LinearLatentToExogenous, out_exogenous=exog_size),
            encoder=LazyConstructor(LinearExogenousToConcept),
            predictor=LazyConstructor(HyperlinearConceptExogenousToConcept, hidden_size=4),
            use_source_exogenous=True,
        )

        pm = model.probabilistic_model

        # Verify exogenous parents of OtherCarCost only reference OtherCar's exog vars
        othercarcost_var = pm.concept_to_variable['OtherCarCost']
        parent_names = [p.concept for p in othercarcost_var.parents]
        # Should only contain OtherCar (concept parent) and exog_OtherCar_state_* (source exog)
        # but NOT exog_OtherCarCost_state_* (those don't exist in this mode)
        for p_name in parent_names:
            if p_name.startswith('exog_'):
                self.assertTrue(
                    p_name.startswith('exog_OtherCar_state_'),
                    f"OtherCarCost has unexpected exogenous parent: {p_name}"
                )


if __name__ == '__main__':
    unittest.main()
