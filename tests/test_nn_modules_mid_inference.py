"""
Comprehensive tests for torch_concepts.nn.modules.mid.inference

Tests for ForwardInference engine.
"""
import unittest
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, RelaxedBernoulli, RelaxedOneHotCategorical
from torch_concepts.data.datasets import ToyDataset

from torch_concepts import InputVariable, EndogenousVariable, Annotations, AxisAnnotation, ConceptGraph
from torch_concepts.nn import AncestralSamplingInference, WANDAGraphLearner, GraphModel, LazyConstructor, LinearZU, \
    LinearUC, HyperLinearCUC
from torch_concepts.nn.modules.mid.models.variable import Variable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.inference.forward import ForwardInference
from torch_concepts.distributions import Delta


class SimpleForwardInference(ForwardInference):
    """Concrete implementation for testing."""

    def get_results(self, results: torch.Tensor, parent_variable: Variable):
        """Simple pass-through implementation."""
        return results


class TestForwardInference(unittest.TestCase):
    """Test ForwardInference class."""

    def test_initialization_simple_model(self):
        """Test initialization with simple model."""
        # Create simple model: latent -> A
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_a = EndogenousVariable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)
        self.assertIsNotNone(inference.sorted_variables)
        self.assertIsNotNone(inference.levels)
        self.assertEqual(len(inference.sorted_variables), 2)

    def test_topological_sort(self):
        """Test topological sorting of variables."""
        # Create chain: latent -> A -> B
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_a = EndogenousVariable('A', parents=[input_var], distribution=Bernoulli, size=1)
        var_b = EndogenousVariable('B', parents=[var_a], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_b = ParametricCPD('B', parametrization=nn.Linear(1, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a, var_b],
            parametric_cpds=[latent_factor, cpd_a, cpd_b]
        )

        inference = SimpleForwardInference(pgm)

        # Check topological order
        sorted_names = [v.concepts[0] for v in inference.sorted_variables]
        self.assertEqual(sorted_names, ['input', 'A', 'B'])

    def test_levels_computation(self):
        """Test level-based grouping for parallel computation."""
        # Create diamond structure
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_a = EndogenousVariable('A', parents=[input_var], distribution=Bernoulli, size=1)
        var_b = EndogenousVariable('B', parents=[input_var], distribution=Bernoulli, size=1)
        var_c = EndogenousVariable('C', parents=[var_a, var_b], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_b = ParametricCPD('B', parametrization=nn.Linear(10, 1))
        cpd_c = ParametricCPD('C', parametrization=nn.Linear(2, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a, var_b, var_c],
            parametric_cpds=[latent_factor, cpd_a, cpd_b, cpd_c]
        )

        inference = SimpleForwardInference(pgm)

        # Check levels
        self.assertEqual(len(inference.levels), 3)
        # Level 0: latent
        self.assertEqual(len(inference.levels[0]), 1)
        # Level 1: A and B (can be computed in parallel)
        self.assertEqual(len(inference.levels[1]), 2)
        # Level 2: C
        self.assertEqual(len(inference.levels[2]), 1)

    def test_predict_simple_chain(self):
        """Test predict method with simple chain."""
        input_var = InputVariable('input', parents=[], distribution=Delta, size=10)
        var_a = EndogenousVariable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)

        # Run prediction
        external_inputs = {'input': torch.randn(4, 10)}
        results = inference.predict(external_inputs)

        self.assertIn('input', results)
        self.assertIn('A', results)
        self.assertEqual(results['A'].shape[0], 4)

    def test_predict_with_debug_mode(self):
        """Test predict with debug mode (sequential execution)."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}
        results = inference.predict(external_inputs, debug=True)

        self.assertIn('input', results)
        self.assertIn('A', results)

    def test_predict_diamond_structure(self):
        """Test predict with diamond structure (parallel computation)."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)
        var_b = Variable('B', parents=[input_var], distribution=Bernoulli, size=1)
        var_c = Variable('C', parents=[var_a, var_b], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_b = ParametricCPD('B', parametrization=nn.Linear(10, 1))
        cpd_c = ParametricCPD('C', parametrization=nn.Linear(2, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a, var_b, var_c],
            parametric_cpds=[latent_factor, cpd_a, cpd_b, cpd_c]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}
        results = inference.predict(external_inputs)

        self.assertEqual(len(results), 4)
        self.assertIn('C', results)

    def test_compute_single_variable_root(self):
        """Test _compute_single_variable for root variable."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())

        pgm = ProbabilisticModel(
            variables=[input_var],
            parametric_cpds=[latent_factor]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}
        results = {}

        concept_name, output = inference._compute_single_variable(
            input_var, external_inputs, results
        )

        self.assertEqual(concept_name, 'input')
        self.assertEqual(output.shape[0], 4)

    def test_compute_single_variable_child(self):
        """Test _compute_single_variable for child variable."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}
        results = {'input': torch.randn(4, 10)}

        concept_name, output = inference._compute_single_variable(
            var_a, external_inputs, results
        )

        self.assertEqual(concept_name, 'A')
        self.assertIsNotNone(output)

    def test_missing_external_input(self):
        """Test error when root variable missing from external_inputs."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())

        pgm = ProbabilisticModel(
            variables=[input_var],
            parametric_cpds=[latent_factor]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {}  # Missing 'input'
        results = {}

        with self.assertRaises(ValueError):
            inference._compute_single_variable(input_var, external_inputs, results)

    def test_missing_parent_result(self):
        """Test error when parent hasn't been computed yet."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}
        results = {}  # Missing 'input' in results

        with self.assertRaises(RuntimeError):
            inference._compute_single_variable(var_a, external_inputs, results)

    def test_get_parent_kwargs(self):
        """Test get_parent_kwargs method."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)

        parent_input = [torch.randn(4, 10)]
        parent_endogenous = []

        kwargs = inference.get_parent_kwargs(cpd_a, parent_input, parent_endogenous)
        self.assertIsInstance(kwargs, dict)

    def test_concept_map(self):
        """Test concept_map creation."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor, cpd_a]
        )

        inference = SimpleForwardInference(pgm)

        self.assertIn('input', inference.concept_map)
        self.assertIn('A', inference.concept_map)
        self.assertEqual(inference.concept_map['input'], input_var)

    def test_categorical_parent(self):
        """Test with categorical parent variable."""
        var_a = Variable('A', parents=[], distribution=Categorical, size=3)
        var_b = Variable('B', parents=[var_a], distribution=Bernoulli, size=1)

        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 3))
        cpd_b = ParametricCPD('B', parametrization=nn.Linear(3, 1))

        pgm = ProbabilisticModel(
            variables=[var_a, var_b],
            parametric_cpds=[cpd_a, cpd_b]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'A': torch.randn(4, 10)}
        results = inference.predict(external_inputs)

        self.assertIn('B', results)

    def test_multiple_children_same_parent(self):
        """Test multiple children depending on same parent."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)
        var_b = Variable('B', parents=[input_var], distribution=Bernoulli, size=1)
        var_c = Variable('C', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        cpd_a = ParametricCPD('A', parametrization=nn.Linear(10, 1))
        cpd_b = ParametricCPD('B', parametrization=nn.Linear(10, 1))
        cpd_c = ParametricCPD('C', parametrization=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[input_var, var_a, var_b, var_c],
            parametric_cpds=[latent_factor, cpd_a, cpd_b, cpd_c]
        )

        inference = SimpleForwardInference(pgm)

        # All three children should be in the same level
        self.assertEqual(len(inference.levels[1]), 3)

    def test_missing_factor(self):
        """Test error when factor is missing for a variable."""
        input_var = Variable('input', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[input_var], distribution=Bernoulli, size=1)

        latent_factor = ParametricCPD('input', parametrization=nn.Identity())
        # Missing cpd_a

        pgm = ProbabilisticModel(
            variables=[input_var, var_a],
            parametric_cpds=[latent_factor]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'input': torch.randn(4, 10)}

        with self.assertRaises(RuntimeError):
            inference.predict(external_inputs)

    def test_unroll_pgm(self):
        latent_dims = 20
        n_epochs = 1000
        n_samples = 1000
        concept_reg = 0.5

        dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
        x_train = dataset.input_data
        concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
        task_idx = list(dataset.graph.edge_index[1].unique().numpy())
        c_train = dataset.concepts[:, concept_idx]
        y_train = dataset.concepts[:, task_idx]

        c_train = torch.cat([c_train, y_train], dim=1)
        y_train = deepcopy(c_train)
        cy_train = torch.cat([c_train, y_train], dim=1)
        c_train_one_hot = torch.cat(
            [cy_train[:, :2], torch.nn.functional.one_hot(cy_train[:, 2].long(), num_classes=2).float()], dim=1)
        cy_train_one_hot = torch.cat([c_train_one_hot, c_train_one_hot], dim=1)

        concept_names = ['c1', 'c2', 'xor']
        task_names = ['c1_copy', 'c2_copy', 'xor_copy']
        cardinalities = [1, 1, 2, 1, 1, 2]
        metadata = {
            'c1': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 1'},
            'c2': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 2'},
            'xor': {'distribution': RelaxedOneHotCategorical, 'type': 'categorical', 'description': 'XOR Task'},
            'c1_copy': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 1 Copy'},
            'c2_copy': {'distribution': RelaxedBernoulli, 'type': 'binary', 'description': 'Concept 2 Copy'},
            'xor_copy': {'distribution': RelaxedOneHotCategorical, 'type': 'categorical',
                         'description': 'XOR Task Copy'},
        }
        annotations = Annotations(
            {1: AxisAnnotation(concept_names + task_names, cardinalities=cardinalities, metadata=metadata)})

        model_graph = ConceptGraph(torch.tensor([[0, 0, 0, 0, 1, 1],
                                                 [0, 0, 0, 1, 0, 1],
                                                 [0, 0, 0, 1, 1, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0]]), list(annotations.get_axis_annotation(1).labels))

        # ProbabilisticModel Initialization
        encoder = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], latent_dims), torch.nn.LeakyReLU())
        concept_model = GraphModel(model_graph=model_graph,
                                   input_size=latent_dims,
                                   annotations=annotations,
                                   source_exogenous=LazyConstructor(LinearZU, exogenous_size=11),
                                   internal_exogenous=LazyConstructor(LinearZU, exogenous_size=7),
                                   encoder=LazyConstructor(LinearUC),
                                   predictor=LazyConstructor(HyperLinearCUC, embedding_size=20))

        # graph learning init
        graph_learner = WANDAGraphLearner(concept_names, task_names)

        inference_engine = AncestralSamplingInference(concept_model.probabilistic_model, graph_learner, temperature=0.1)
        query_concepts = ["c1", "c2", "xor", "c1_copy", "c2_copy", "xor_copy"]

        emb = encoder(x_train)
        cy_pred_before_unrolling = inference_engine.query(query_concepts, evidence={'input': emb}, debug=True)

        concept_model_new = inference_engine.unrolled_probabilistic_model()

        # identify available query concepts in the unrolled model
        query_concepts = [c for c in query_concepts if c in inference_engine.available_query_vars]
        concept_idx = {v: i for i, v in enumerate(concept_names)}
        reverse_c2t_mapping = dict(zip(task_names, concept_names))
        query_concepts = sorted(query_concepts, key=lambda x: concept_idx[x] if x in concept_idx else concept_idx[reverse_c2t_mapping[x]])

        inference_engine = AncestralSamplingInference(concept_model_new, temperature=0.1)
        cy_pred_after_unrolling = inference_engine.query(query_concepts, evidence={'input': emb}, debug=True)

        self.assertTrue(cy_pred_after_unrolling.shape == c_train_one_hot.shape)


if __name__ == '__main__':
    unittest.main()

