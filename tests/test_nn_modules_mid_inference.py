"""
Comprehensive tests for torch_concepts.nn.modules.mid.inference

Tests for ForwardInference engine.
"""
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch_concepts.nn.modules.mid.models.variable import Variable
from torch_concepts.nn.modules.mid.models.factor import Factor
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
        # Create simple model: embedding -> A
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        factor_a = Factor('A', module_class=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a],
            factors=[embedding_factor, factor_a]
        )

        inference = SimpleForwardInference(pgm)
        self.assertIsNotNone(inference.sorted_variables)
        self.assertIsNotNone(inference.levels)
        self.assertEqual(len(inference.sorted_variables), 2)

    def test_topological_sort(self):
        """Test topological sorting of variables."""
        # Create chain: embedding -> A -> B
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)
        var_b = Variable('B', parents=[var_a], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        factor_a = Factor('A', module_class=nn.Linear(10, 1))
        factor_b = Factor('B', module_class=nn.Linear(1, 1))

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a, var_b],
            factors=[embedding_factor, factor_a, factor_b]
        )

        inference = SimpleForwardInference(pgm)

        # Check topological order
        sorted_names = [v.concepts[0] for v in inference.sorted_variables]
        self.assertEqual(sorted_names, ['embedding', 'A', 'B'])

    def test_levels_computation(self):
        """Test level-based grouping for parallel computation."""
        # Create diamond structure
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)
        var_b = Variable('B', parents=[embedding_var], distribution=Bernoulli, size=1)
        var_c = Variable('C', parents=[var_a, var_b], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        factor_a = Factor('A', module_class=nn.Linear(10, 1))
        factor_b = Factor('B', module_class=nn.Linear(10, 1))
        factor_c = Factor('C', module_class=nn.Linear(2, 1))

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a, var_b, var_c],
            factors=[embedding_factor, factor_a, factor_b, factor_c]
        )

        inference = SimpleForwardInference(pgm)

        # Check levels
        self.assertEqual(len(inference.levels), 3)
        # Level 0: embedding
        self.assertEqual(len(inference.levels[0]), 1)
        # Level 1: A and B (can be computed in parallel)
        self.assertEqual(len(inference.levels[1]), 2)
        # Level 2: C
        self.assertEqual(len(inference.levels[2]), 1)

    def test_predict_simple_chain(self):
        """Test predict method with simple chain."""
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        factor_a = Factor('A', module_class=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a],
            factors=[embedding_factor, factor_a]
        )

        inference = SimpleForwardInference(pgm)

        # Run prediction
        external_inputs = {'embedding': torch.randn(4, 10)}
        results = inference.predict(external_inputs)

        self.assertIn('embedding', results)
        self.assertIn('A', results)
        self.assertEqual(results['A'].shape[0], 4)

    def test_predict_with_debug_mode(self):
        """Test predict with debug mode (sequential execution)."""
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        factor_a = Factor('A', module_class=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a],
            factors=[embedding_factor, factor_a]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'embedding': torch.randn(4, 10)}
        results = inference.predict(external_inputs, debug=True)

        self.assertIn('embedding', results)
        self.assertIn('A', results)

    def test_predict_diamond_structure(self):
        """Test predict with diamond structure (parallel computation)."""
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)
        var_b = Variable('B', parents=[embedding_var], distribution=Bernoulli, size=1)
        var_c = Variable('C', parents=[var_a, var_b], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        factor_a = Factor('A', module_class=nn.Linear(10, 1))
        factor_b = Factor('B', module_class=nn.Linear(10, 1))
        factor_c = Factor('C', module_class=nn.Linear(2, 1))

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a, var_b, var_c],
            factors=[embedding_factor, factor_a, factor_b, factor_c]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'embedding': torch.randn(4, 10)}
        results = inference.predict(external_inputs)

        self.assertEqual(len(results), 4)
        self.assertIn('C', results)

    def test_compute_single_variable_root(self):
        """Test _compute_single_variable for root variable."""
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)

        embedding_factor = Factor('embedding', module_class=nn.Identity())

        pgm = ProbabilisticModel(
            variables=[embedding_var],
            factors=[embedding_factor]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'embedding': torch.randn(4, 10)}
        results = {}

        concept_name, output = inference._compute_single_variable(
            embedding_var, external_inputs, results
        )

        self.assertEqual(concept_name, 'embedding')
        self.assertEqual(output.shape[0], 4)

    def test_compute_single_variable_child(self):
        """Test _compute_single_variable for child variable."""
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        factor_a = Factor('A', module_class=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a],
            factors=[embedding_factor, factor_a]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'embedding': torch.randn(4, 10)}
        results = {'embedding': torch.randn(4, 10)}

        concept_name, output = inference._compute_single_variable(
            var_a, external_inputs, results
        )

        self.assertEqual(concept_name, 'A')
        self.assertIsNotNone(output)

    def test_missing_external_input(self):
        """Test error when root variable missing from external_inputs."""
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)

        embedding_factor = Factor('embedding', module_class=nn.Identity())

        pgm = ProbabilisticModel(
            variables=[embedding_var],
            factors=[embedding_factor]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {}  # Missing 'embedding'
        results = {}

        with self.assertRaises(ValueError):
            inference._compute_single_variable(embedding_var, external_inputs, results)

    def test_missing_parent_result(self):
        """Test error when parent hasn't been computed yet."""
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        factor_a = Factor('A', module_class=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a],
            factors=[embedding_factor, factor_a]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'embedding': torch.randn(4, 10)}
        results = {}  # Missing 'embedding' in results

        with self.assertRaises(RuntimeError):
            inference._compute_single_variable(var_a, external_inputs, results)

    def test_get_parent_kwargs(self):
        """Test get_parent_kwargs method."""
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        factor_a = Factor('A', module_class=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a],
            factors=[embedding_factor, factor_a]
        )

        inference = SimpleForwardInference(pgm)

        parent_latent = [torch.randn(4, 10)]
        parent_logits = []

        kwargs = inference.get_parent_kwargs(factor_a, parent_latent, parent_logits)
        self.assertIsInstance(kwargs, dict)

    def test_concept_map(self):
        """Test concept_map creation."""
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        factor_a = Factor('A', module_class=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a],
            factors=[embedding_factor, factor_a]
        )

        inference = SimpleForwardInference(pgm)

        self.assertIn('embedding', inference.concept_map)
        self.assertIn('A', inference.concept_map)
        self.assertEqual(inference.concept_map['embedding'], embedding_var)

    def test_categorical_parent(self):
        """Test with categorical parent variable."""
        var_a = Variable('A', parents=[], distribution=Categorical, size=3)
        var_b = Variable('B', parents=[var_a], distribution=Bernoulli, size=1)

        factor_a = Factor('A', module_class=nn.Linear(10, 3))
        factor_b = Factor('B', module_class=nn.Linear(3, 1))

        pgm = ProbabilisticModel(
            variables=[var_a, var_b],
            factors=[factor_a, factor_b]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'A': torch.randn(4, 10)}
        results = inference.predict(external_inputs)

        self.assertIn('B', results)

    def test_multiple_children_same_parent(self):
        """Test multiple children depending on same parent."""
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)
        var_b = Variable('B', parents=[embedding_var], distribution=Bernoulli, size=1)
        var_c = Variable('C', parents=[embedding_var], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        factor_a = Factor('A', module_class=nn.Linear(10, 1))
        factor_b = Factor('B', module_class=nn.Linear(10, 1))
        factor_c = Factor('C', module_class=nn.Linear(10, 1))

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a, var_b, var_c],
            factors=[embedding_factor, factor_a, factor_b, factor_c]
        )

        inference = SimpleForwardInference(pgm)

        # All three children should be in the same level
        self.assertEqual(len(inference.levels[1]), 3)

    def test_missing_factor(self):
        """Test error when factor is missing for a variable."""
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        # Missing factor_a

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a],
            factors=[embedding_factor]
        )

        inference = SimpleForwardInference(pgm)

        external_inputs = {'embedding': torch.randn(4, 10)}

        with self.assertRaises(RuntimeError):
            inference.predict(external_inputs)

    def test_complex_multi_level_hierarchy(self):
        """Test complex multi-level hierarchy."""
        # Level 0: embedding
        embedding_var = Variable('embedding', parents=[], distribution=Delta, size=10)

        # Level 1: A, B
        var_a = Variable('A', parents=[embedding_var], distribution=Bernoulli, size=1)
        var_b = Variable('B', parents=[embedding_var], distribution=Categorical, size=3)

        # Level 2: C (depends on A and B)
        var_c = Variable('C', parents=[var_a, var_b], distribution=Bernoulli, size=1)

        # Level 3: D (depends on C)
        var_d = Variable('D', parents=[var_c], distribution=Bernoulli, size=1)

        embedding_factor = Factor('embedding', module_class=nn.Identity())
        factor_a = Factor('A', module_class=nn.Linear(10, 1))
        factor_b = Factor('B', module_class=nn.Linear(10, 3))
        factor_c = Factor('C', module_class=nn.Linear(4, 1))  # 1 + 3 inputs
        factor_d = Factor('D', module_class=nn.Linear(1, 1))

        pgm = ProbabilisticModel(
            variables=[embedding_var, var_a, var_b, var_c, var_d],
            factors=[embedding_factor, factor_a, factor_b, factor_c, factor_d]
        )

        inference = SimpleForwardInference(pgm)

        self.assertEqual(len(inference.levels), 4)

        external_inputs = {'embedding': torch.randn(4, 10)}
        results = inference.predict(external_inputs)

        self.assertEqual(len(results), 5)


if __name__ == '__main__':
    unittest.main()

