"""
Comprehensive tests for torch_concepts.nn.modules.mid.models

Tests for Variable, Factor, and ProbabilisticModel.
"""
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, Normal
from torch_concepts.nn.modules.mid.models.variable import Variable
from torch_concepts.nn.modules.mid.models.factor import Factor
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.distributions import Delta


class TestVariable(unittest.TestCase):
    """Test Variable class."""

    def test_single_concept_initialization(self):
        """Test creating a single concept variable."""
        var = Variable(
            concepts='color',
            parents=[],
            distribution=Bernoulli,
            size=1
        )
        self.assertEqual(var.concepts, ['color'])
        self.assertEqual(var.distribution, Bernoulli)

    def test_multiple_concepts_initialization(self):
        """Test creating multiple concept variables."""
        vars_list = Variable(
            concepts=['A', 'B', 'C'],
            parents=[],
            distribution=Bernoulli,
            size=1
        )
        self.assertEqual(len(vars_list), 3)
        self.assertEqual(vars_list[0].concepts, ['A'])
        self.assertEqual(vars_list[1].concepts, ['B'])
        self.assertEqual(vars_list[2].concepts, ['C'])

    def test_variable_with_delta_distribution(self):
        """Test variable with Delta distribution."""
        var = Variable(
            concepts=['feature'],
            parents=[],
            distribution=Delta,
            size=1
        )
        self.assertEqual(var.distribution, Delta)

    def test_variable_with_categorical_distribution(self):
        """Test variable with Categorical distribution."""
        var = Variable(
            concepts=['color'],
            parents=[],
            distribution=Categorical,
            size=3
        )
        self.assertEqual(var.distribution, Categorical)
        self.assertEqual(var.size, 3)

    def test_variable_with_normal_distribution(self):
        """Test variable with Normal distribution."""
        var = Variable(
            concepts=['continuous'],
            parents=[],
            distribution=Normal,
            size=1
        )
        self.assertEqual(var.distribution, Normal)

    def test_variable_with_parents(self):
        """Test variable with parent variables."""
        parent_var = Variable(
            concepts=['parent'],
            parents=[],
            distribution=Bernoulli,
            size=1
        )
        child_var = Variable(
            concepts=['child'],
            parents=[parent_var],
            distribution=Bernoulli,
            size=1
        )
        self.assertEqual(len(child_var.parents), 1)
        self.assertEqual(child_var.parents[0], parent_var)

    def test_variable_out_features(self):
        """Test out_features property."""
        var_binary = Variable(concepts=['binary'], parents=[], distribution=Bernoulli, size=1)
        self.assertEqual(var_binary.out_features, 1)

        var_cat = Variable(concepts=['category'], parents=[], distribution=Categorical, size=5)
        self.assertEqual(var_cat.out_features, 5)

    def test_variable_in_features(self):
        """Test in_features property with parents."""
        parent1 = Variable(concepts=['p1'], parents=[], distribution=Bernoulli, size=1)
        parent2 = Variable(concepts=['p2'], parents=[], distribution=Categorical, size=3)

        child = Variable(
            concepts=['child'],
            parents=[parent1, parent2],
            distribution=Bernoulli,
            size=1
        )
        self.assertEqual(child.in_features, 1 + 3)

    def test_variable_with_metadata(self):
        """Test variable with metadata."""
        metadata = {'description': 'test variable', 'importance': 0.8}
        var = Variable(
            concepts=['test'],
            parents=[],
            distribution=Bernoulli,
            size=1,
            metadata=metadata
        )
        self.assertEqual(var.metadata, metadata)

    def test_multiple_concepts_with_different_distributions(self):
        """Test multiple concepts with different distributions."""
        vars_list = Variable(
            concepts=['A', 'B', 'C'],
            parents=[],
            distribution=[Bernoulli, Categorical, Delta],
            size=[1, 3, 1]
        )
        self.assertEqual(vars_list[0].distribution, Bernoulli)
        self.assertEqual(vars_list[1].distribution, Categorical)
        self.assertEqual(vars_list[2].distribution, Delta)

    def test_multiple_concepts_with_different_sizes(self):
        """Test multiple concepts with different sizes."""
        vars_list = Variable(
            concepts=['A', 'B', 'C'],
            parents=[],
            distribution=Categorical,
            size=[2, 3, 4]
        )
        self.assertEqual(vars_list[0].size, 2)
        self.assertEqual(vars_list[1].size, 3)
        self.assertEqual(vars_list[2].size, 4)

    def test_variable_with_none_distribution(self):
        """Test variable with None distribution defaults to Delta."""
        vars_list = Variable(
            concepts=['A', 'B'],
            parents=[],
            distribution=None,
            size=1
        )
        self.assertEqual(vars_list[0].distribution, Delta)
        self.assertEqual(vars_list[1].distribution, Delta)

    def test_variable_validation_error(self):
        """Test validation error for mismatched list lengths."""
        with self.assertRaises(ValueError):
            Variable(
                concepts=['A', 'B', 'C'],
                parents=[],
                distribution=[Bernoulli, Categorical],  # Only 2, need 3
                size=1
            )


class TestFactor(unittest.TestCase):
    """Test Factor class."""

    def test_single_concept_factor(self):
        """Test creating a factor with single concept."""
        module = nn.Linear(10, 1)
        factor = Factor(concepts='concept_a', module_class=module)
        self.assertEqual(factor.concepts, ['concept_a'])
        self.assertIsNotNone(factor.module_class)

    def test_multiple_concepts_single_module(self):
        """Test multiple concepts with single module (replicated)."""
        module = nn.Linear(10, 1)
        factors = Factor(concepts=['A', 'B', 'C'], module_class=module)
        self.assertEqual(len(factors), 3)
        self.assertEqual(factors[0].concepts, ['A'])
        self.assertEqual(factors[1].concepts, ['B'])
        self.assertEqual(factors[2].concepts, ['C'])

    def test_multiple_concepts_multiple_modules(self):
        """Test multiple concepts with different modules."""
        module_a = nn.Linear(10, 1)
        module_b = nn.Linear(10, 2)
        module_c = nn.Linear(10, 3)

        factors = Factor(
            concepts=['A', 'B', 'C'],
            module_class=[module_a, module_b, module_c]
        )
        self.assertEqual(len(factors), 3)
        self.assertIsInstance(factors[0].module_class, nn.Linear)
        self.assertEqual(factors[1].module_class.out_features, 2)
        self.assertEqual(factors[2].module_class.out_features, 3)

    def test_factor_forward(self):
        """Test forward pass through factor."""
        module = nn.Linear(10, 1)
        factor = Factor(concepts='concept', module_class=module)

        x = torch.randn(4, 10)
        output = factor(input=x)
        self.assertEqual(output.shape, (4, 1))

    def test_factor_with_variable(self):
        """Test linking factor to variable."""
        module = nn.Linear(10, 1)
        factor = Factor(concepts='concept', module_class=module)

        var = Variable(concepts=['concept'], parents=[], distribution=Bernoulli, size=1)
        factor.variable = var

        self.assertEqual(factor.variable, var)

    def test_factor_with_parents(self):
        """Test factor with parent variables."""
        module = nn.Linear(10, 1)
        factor = Factor(concepts='child', module_class=module)

        parent_var = Variable(concepts=['parent'], parents=[], distribution=Bernoulli, size=1)
        factor.parents = [parent_var]

        self.assertEqual(len(factor.parents), 1)

    def test_factor_validation_error(self):
        """Test validation error for mismatched concept/module counts."""
        with self.assertRaises(ValueError):
            Factor(
                concepts=['A', 'B', 'C'],
                module_class=[nn.Linear(10, 1), nn.Linear(10, 1)]  # Only 2, need 3
            )

    def test_get_parent_combinations_no_parents(self):
        """Test _get_parent_combinations with no parents."""
        module = nn.Linear(10, 1)
        factor = Factor(concepts='concept', module_class=module)
        var = Variable(concepts=['concept'], parents=[], distribution=Bernoulli, size=1)
        factor.variable = var
        factor.parents = []

        inputs, states = factor._get_parent_combinations()
        self.assertEqual(inputs.shape[0], 1)
        self.assertEqual(states.shape[1], 0)

    def test_get_parent_combinations_bernoulli_parent(self):
        """Test _get_parent_combinations with Bernoulli parent."""
        parent_var = Variable(concepts=['parent'], parents=[], distribution=Bernoulli, size=1)
        module = nn.Linear(1, 1)
        factor = Factor(concepts='child', module_class=module)
        child_var = Variable(concepts=['child'], parents=[parent_var], distribution=Bernoulli, size=1)
        factor.variable = child_var
        factor.parents = [parent_var]

        inputs, states = factor._get_parent_combinations()
        # Bernoulli with size=1 should give 2 combinations: [0], [1]
        self.assertEqual(inputs.shape[0], 2)

    def test_get_parent_combinations_categorical_parent(self):
        """Test _get_parent_combinations with Categorical parent."""
        parent_var = Variable(concepts=['parent'], parents=[], distribution=Categorical, size=3)
        module = nn.Linear(3, 1)
        factor = Factor(concepts='child', module_class=module)
        child_var = Variable(concepts=['child'], parents=[parent_var], distribution=Bernoulli, size=1)
        factor.variable = child_var
        factor.parents = [parent_var]

        inputs, states = factor._get_parent_combinations()
        # Categorical with size=3 should give 3 combinations
        self.assertEqual(inputs.shape[0], 3)

    def test_get_parent_combinations_delta_parent(self):
        """Test _get_parent_combinations with Delta parent."""
        parent_var = Variable(concepts=['parent'], parents=[], distribution=Delta, size=2)
        module = nn.Linear(2, 1)
        factor = Factor(concepts='child', module_class=module)
        child_var = Variable(concepts=['child'], parents=[parent_var], distribution=Bernoulli, size=1)
        factor.variable = child_var
        factor.parents = [parent_var]

        inputs, states = factor._get_parent_combinations()
        self.assertIsNotNone(inputs)

    def test_build_cpt_without_variable(self):
        """Test build_cpt raises error when variable not linked."""
        module = nn.Linear(10, 1)
        factor = Factor(concepts='concept', module_class=module)

        with self.assertRaises(RuntimeError):
            factor.build_cpt()


class TestProbabilisticModel(unittest.TestCase):
    """Test ProbabilisticModel class."""

    def test_initialization(self):
        """Test probabilistic model initialization."""
        model = ProbabilisticModel(variables=[], factors=[])
        self.assertEqual(len(model.variables), 0)
        self.assertEqual(len(model.factors), 0)

    def test_add_single_variable(self):
        """Test adding a single variable."""
        var = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        model = ProbabilisticModel(variables=[var], factors=[])
        self.assertEqual(len(model.variables), 1)

    def test_add_multiple_variables(self):
        """Test adding multiple variables."""
        vars_list = [
            Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1),
            Variable(concepts=['B'], parents=[], distribution=Bernoulli, size=1),
            Variable(concepts=['C'], parents=[], distribution=Bernoulli, size=1)
        ]
        model = ProbabilisticModel(variables=vars_list, factors=[])
        self.assertEqual(len(model.variables), 3)

    def test_add_factors(self):
        """Test adding factors to model."""
        var = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        factor = Factor(concepts='A', module_class=nn.Linear(10, 1))

        model = ProbabilisticModel(variables=[var], factors=[factor])
        self.assertEqual(len(model.factors), 1)

    def test_variables_and_factors_linkage(self):
        """Test that variables and factors are properly linked."""
        var = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        factor = Factor(concepts='A', module_class=nn.Linear(10, 1))

        model = ProbabilisticModel(variables=[var], factors=[factor])
        self.assertIsNotNone(model)

    def test_hierarchical_structure(self):
        """Test hierarchical variable structure."""
        parent = Variable(concepts=['parent'], parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts=['child'], parents=[parent], distribution=Bernoulli, size=1)

        parent_factor = Factor(concepts='parent', module_class=nn.Linear(10, 1))
        child_factor = Factor(concepts='child', module_class=nn.Linear(1, 1))

        model = ProbabilisticModel(
            variables=[parent, child],
            factors=[parent_factor, child_factor]
        )
        self.assertEqual(len(model.variables), 2)
        self.assertEqual(len(model.factors), 2)

    def test_multiple_parents(self):
        """Test variable with multiple parents."""
        parent1 = Variable(concepts=['p1'], parents=[], distribution=Bernoulli, size=1)
        parent2 = Variable(concepts=['p2'], parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts=['child'], parents=[parent1, parent2], distribution=Bernoulli, size=1)

        model = ProbabilisticModel(variables=[parent1, parent2, child], factors=[])
        self.assertEqual(len(model.variables), 3)

    def test_categorical_variable(self):
        """Test with categorical variables."""
        var = Variable(concepts=['color'], parents=[], distribution=Categorical, size=3)
        factor = Factor(concepts='color', module_class=nn.Linear(10, 3))

        model = ProbabilisticModel(variables=[var], factors=[factor])
        self.assertIsNotNone(model)

    def test_delta_distribution(self):
        """Test with Delta (deterministic) distribution."""
        var = Variable(concepts=['feature'], parents=[], distribution=Delta, size=1)
        factor = Factor(concepts='feature', module_class=nn.Linear(10, 1))

        model = ProbabilisticModel(variables=[var], factors=[factor])
        self.assertIsNotNone(model)

    def test_concept_to_variable_mapping(self):
        """Test concept name to variable mapping."""
        vars_list = [
            Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1),
            Variable(concepts=['B'], parents=[], distribution=Categorical, size=3)
        ]
        model = ProbabilisticModel(variables=vars_list, factors=[])
        # Model should create mapping from concept names to variables
        self.assertEqual(len(model.variables), 2)

    def test_get_module_of_concept(self):
        """Test get_module_of_concept method."""
        var = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        factor = Factor(concepts='A', module_class=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], factors=[factor])

        module = model.get_module_of_concept('A')
        self.assertIsNotNone(module)
        self.assertEqual(module.concepts, ['A'])

    def test_get_module_of_nonexistent_concept(self):
        """Test get_module_of_concept with non-existent concept."""
        var = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        factor = Factor(concepts='A', module_class=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], factors=[factor])

        module = model.get_module_of_concept('B')
        self.assertIsNone(module)

    def test_build_cpt_bernoulli(self):
        """Test build_cpt for Bernoulli variable."""
        parent = Variable(concepts=['parent'], parents=[], distribution=Delta, size=2)
        child = Variable(concepts=['child'], parents=[parent], distribution=Bernoulli, size=1)

        parent_factor = Factor(concepts='parent', module_class=nn.Identity())
        child_factor = Factor(concepts='child', module_class=nn.Linear(2, 1))

        model = ProbabilisticModel(
            variables=[parent, child],
            factors=[parent_factor, child_factor]
        )

        # Get the linked factor and build CPT
        child_factor_linked = model.get_module_of_concept('child')
        cpt = child_factor_linked.build_cpt()
        self.assertIsNotNone(cpt)

    def test_build_potential_categorical(self):
        """Test build_potential for Categorical variable."""
        parent = Variable(concepts=['parent'], parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts=['child'], parents=[parent], distribution=Categorical, size=3)

        parent_factor = Factor(concepts='parent', module_class=nn.Linear(10, 1))
        child_factor = Factor(concepts='child', module_class=nn.Linear(1, 3))

        model = ProbabilisticModel(
            variables=[parent, child],
            factors=[parent_factor, child_factor]
        )

        child_factor_linked = model.get_module_of_concept('child')
        potential = child_factor_linked.build_potential()
        self.assertIsNotNone(potential)

    def test_multiple_parent_combinations(self):
        """Test factor with multiple parents."""
        parent1 = Variable(concepts=['p1'], parents=[], distribution=Bernoulli, size=1)
        parent2 = Variable(concepts=['p2'], parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts=['child'], parents=[parent1, parent2], distribution=Bernoulli, size=1)

        p1_factor = Factor(concepts='p1', module_class=nn.Linear(10, 1))
        p2_factor = Factor(concepts='p2', module_class=nn.Linear(10, 1))
        child_factor = Factor(concepts='child', module_class=nn.Linear(2, 1))

        model = ProbabilisticModel(
            variables=[parent1, parent2, child],
            factors=[p1_factor, p2_factor, child_factor]
        )

        self.assertEqual(len(model.variables), 3)


class TestVariableFactorIntegration(unittest.TestCase):
    """Test integration between Variables and Factors."""

    def test_factor_output_matches_variable_size(self):
        """Test that factor output size matches variable size."""
        var = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        factor = Factor(concepts='A', module_class=nn.Linear(10, 1))

        x = torch.randn(4, 10)
        output = factor(input=x)
        self.assertEqual(output.shape[1], var.out_features)

    def test_parent_child_feature_matching(self):
        """Test that child input features match parent output features."""
        parent = Variable(concepts=['parent'], parents=[], distribution=Categorical, size=3)
        child = Variable(concepts=['child'], parents=[parent], distribution=Bernoulli, size=1)

        child_factor = Factor(concepts='child', module_class=nn.Linear(3, 1))

        parent_output = torch.randn(4, 3)
        child_output = child_factor(input=parent_output)
        self.assertEqual(child_output.shape, (4, 1))

    def test_complex_hierarchy(self):
        """Test complex hierarchical structure."""
        var_a = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        var_b = Variable(concepts=['B'], parents=[var_a], distribution=Bernoulli, size=1)
        var_c = Variable(concepts=['C'], parents=[var_a], distribution=Bernoulli, size=1)
        var_d = Variable(concepts=['D'], parents=[var_b, var_c], distribution=Bernoulli, size=1)

        factor_a = Factor(concepts='A', module_class=nn.Linear(10, 1))
        factor_b = Factor(concepts='B', module_class=nn.Linear(1, 1))
        factor_c = Factor(concepts='C', module_class=nn.Linear(1, 1))
        factor_d = Factor(concepts='D', module_class=nn.Linear(2, 1))

        model = ProbabilisticModel(
            variables=[var_a, var_b, var_c, var_d],
            factors=[factor_a, factor_b, factor_c, factor_d]
        )
        self.assertEqual(len(model.variables), 4)
        self.assertEqual(var_d.in_features, 2)

    def test_mixed_distributions(self):
        """Test model with mixed distribution types."""
        var_delta = Variable(concepts=['emb'], parents=[], distribution=Delta, size=10)
        var_bern = Variable(concepts=['binary'], parents=[var_delta], distribution=Bernoulli, size=1)
        var_cat = Variable(concepts=['multi'], parents=[var_delta], distribution=Categorical, size=3)

        factor_delta = Factor(concepts='emb', module_class=nn.Identity())
        factor_bern = Factor(concepts='binary', module_class=nn.Linear(10, 1))
        factor_cat = Factor(concepts='multi', module_class=nn.Linear(10, 3))

        model = ProbabilisticModel(
            variables=[var_delta, var_bern, var_cat],
            factors=[factor_delta, factor_bern, factor_cat]
        )
        self.assertEqual(len(model.variables), 3)


if __name__ == '__main__':
    unittest.main()
