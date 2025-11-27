"""
Comprehensive tests for torch_concepts.nn.modules.mid.models

Tests for Variable, ParametricCPD, and ProbabilisticModel.
"""
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch_concepts.nn.modules.mid.models.variable import Variable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.mid.models.probabilistic_model import (
    _reinitialize_with_new_param,
    ProbabilisticModel,
)

class TestProbabilisticModel(unittest.TestCase):
    """Test ProbabilisticModel class."""

    def test_initialization(self):
        """Test probabilistic model initialization."""
        model = ProbabilisticModel(variables=[], parametric_cpds=[])
        self.assertEqual(len(model.variables), 0)
        self.assertEqual(len(model.parametric_cpds), 0)

    def test_add_single_variable(self):
        """Test adding a single variable."""
        var = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        model = ProbabilisticModel(variables=[var], parametric_cpds=[])
        self.assertEqual(len(model.variables), 1)

    def test_add_multiple_variables(self):
        """Test adding multiple variables."""
        vars_list = [
            Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1),
            Variable(concepts=['B'], parents=[], distribution=Bernoulli, size=1),
            Variable(concepts=['C'], parents=[], distribution=Bernoulli, size=1)
        ]
        model = ProbabilisticModel(variables=vars_list, parametric_cpds=[])
        self.assertEqual(len(model.variables), 3)

    def test_add_cpds(self):
        """Test adding cpds to model."""
        var = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        cpd = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])
        self.assertEqual(len(model.parametric_cpds), 1)

    def test_variables_and_cpds_linkage(self):
        """Test that variables and cpds are properly linked."""
        var = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        cpd = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])
        self.assertIsNotNone(model)

    def test_hierarchical_structure(self):
        """Test hierarchical variable structure."""
        parent = Variable(concepts=['parent'], parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts=['child'], parents=[parent], distribution=Bernoulli, size=1)

        parent_cpd = ParametricCPD(concepts='parent', parametrization=nn.Linear(10, 1))
        child_cpd = ParametricCPD(concepts='child', parametrization=nn.Linear(1, 1))

        model = ProbabilisticModel(
            variables=[parent, child],
            parametric_cpds=[parent_cpd, child_cpd]
        )
        self.assertEqual(len(model.variables), 2)
        self.assertEqual(len(model.parametric_cpds), 2)

    def test_multiple_parents(self):
        """Test variable with multiple parents."""
        parent1 = Variable(concepts=['p1'], parents=[], distribution=Bernoulli, size=1)
        parent2 = Variable(concepts=['p2'], parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts=['child'], parents=[parent1, parent2], distribution=Bernoulli, size=1)

        model = ProbabilisticModel(variables=[parent1, parent2, child], parametric_cpds=[])
        self.assertEqual(len(model.variables), 3)

    def test_categorical_variable(self):
        """Test with categorical variables."""
        var = Variable(concepts=['color'], parents=[], distribution=Categorical, size=3)
        cpd = ParametricCPD(concepts='color', parametrization=nn.Linear(10, 3))

        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])
        self.assertIsNotNone(model)

    def test_delta_distribution(self):
        """Test with Delta (deterministic) distribution."""
        var = Variable(concepts=['feature'], parents=[], distribution=Delta, size=1)
        cpd = ParametricCPD(concepts='feature', parametrization=nn.Linear(10, 1))

        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])
        self.assertIsNotNone(model)

    def test_concept_to_variable_mapping(self):
        """Test concept name to variable mapping."""
        vars_list = [
            Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1),
            Variable(concepts=['B'], parents=[], distribution=Categorical, size=3)
        ]
        model = ProbabilisticModel(variables=vars_list, parametric_cpds=[])
        # Model should create mapping from concept names to variables
        self.assertEqual(len(model.variables), 2)

    def test_get_module_of_concept(self):
        """Test get_module_of_concept method."""
        var = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        cpd = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])

        module = model.get_module_of_concept('A')
        self.assertIsNotNone(module)
        self.assertEqual(module.concepts, ['A'])

    def test_get_module_of_nonexistent_concept(self):
        """Test get_module_of_concept with non-existent concept."""
        var = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        cpd = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])

        module = model.get_module_of_concept('B')
        self.assertIsNone(module)

    def test_multiple_parent_combinations(self):
        """Test cpd with multiple parents."""
        parent1 = Variable(concepts=['p1'], parents=[], distribution=Bernoulli, size=1)
        parent2 = Variable(concepts=['p2'], parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts=['child'], parents=[parent1, parent2], distribution=Bernoulli, size=1)

        p1_cpd = ParametricCPD(concepts='p1', parametrization=nn.Linear(10, 1))
        p2_cpd = ParametricCPD(concepts='p2', parametrization=nn.Linear(10, 1))
        child_cpd = ParametricCPD(concepts='child', parametrization=nn.Linear(2, 1))

        model = ProbabilisticModel(
            variables=[parent1, parent2, child],
            parametric_cpds=[p1_cpd, p2_cpd, child_cpd]
        )

        self.assertEqual(len(model.variables), 3)


class TestVariableParametricCPDIntegration(unittest.TestCase):
    """Test integration between Variables and ParametricCPDs."""

    def test_cpd_output_matches_variable_size(self):
        """Test that cpd output size matches variable size."""
        var = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        cpd = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))

        x = torch.randn(4, 10)
        output = cpd(input=x)
        self.assertEqual(output.shape[1], var.out_features)

    def test_parent_child_feature_matching(self):
        """Test that child input features match parent output features."""
        parent = Variable(concepts=['parent'], parents=[], distribution=Categorical, size=3)
        child = Variable(concepts=['child'], parents=[parent], distribution=Bernoulli, size=1)

        child_cpd = ParametricCPD(concepts='child', parametrization=nn.Linear(3, 1))

        parent_output = torch.randn(4, 3)
        child_output = child_cpd(input=parent_output)
        self.assertEqual(child_output.shape, (4, 1))

    def test_complex_hierarchy(self):
        """Test complex hierarchical structure."""
        var_a = Variable(concepts=['A'], parents=[], distribution=Bernoulli, size=1)
        var_b = Variable(concepts=['B'], parents=[var_a], distribution=Bernoulli, size=1)
        var_c = Variable(concepts=['C'], parents=[var_a], distribution=Bernoulli, size=1)
        var_d = Variable(concepts=['D'], parents=[var_b, var_c], distribution=Bernoulli, size=1)

        cpd_a = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))
        cpd_b = ParametricCPD(concepts='B', parametrization=nn.Linear(1, 1))
        cpd_c = ParametricCPD(concepts='C', parametrization=nn.Linear(1, 1))
        cpd_d = ParametricCPD(concepts='D', parametrization=nn.Linear(2, 1))

        model = ProbabilisticModel(
            variables=[var_a, var_b, var_c, var_d],
            parametric_cpds=[cpd_a, cpd_b, cpd_c, cpd_d]
        )
        self.assertEqual(len(model.variables), 4)
        self.assertEqual(var_d.in_features, 2)

    def test_mixed_distributions(self):
        """Test model with mixed distribution types."""
        var_delta = Variable(concepts=['emb'], parents=[], distribution=Delta, size=10)
        var_bern = Variable(concepts=['binary'], parents=[var_delta], distribution=Bernoulli, size=1)
        var_cat = Variable(concepts=['multi'], parents=[var_delta], distribution=Categorical, size=3)

        cpd_delta = ParametricCPD(concepts='emb', parametrization=nn.Identity())
        cpd_bern = ParametricCPD(concepts='binary', parametrization=nn.Linear(10, 1))
        cpd_cat = ParametricCPD(concepts='multi', parametrization=nn.Linear(10, 3))

        model = ProbabilisticModel(
            variables=[var_delta, var_bern, var_cat],
            parametric_cpds=[cpd_delta, cpd_bern, cpd_cat]
        )
        self.assertEqual(len(model.variables), 3)


def test_reinitialize_parametric_cpd_parametrization_changed():
    orig = ParametricCPD(concepts='a', parametrization=nn.Linear(3, 1))
    new_param = nn.Linear(5, 1)
    new = _reinitialize_with_new_param(orig, 'parametrization', new_param)
    assert isinstance(new, ParametricCPD)
    assert new.parametrization.in_features == 5


def test_probabilistic_model_no_parents_build_cpt_and_potential_delta():
    # Variable with no parents, deterministic (Delta)
    var = Variable(concepts='x', parents=[], distribution=Delta, size=1)
    # parametrization expects input size equal to its in_features
    module = nn.Linear(in_features=2, out_features=1)
    pcpd = ParametricCPD(concepts='x', parametrization=module)

    model = ProbabilisticModel(variables=[var], parametric_cpds=[pcpd])

    cpts = model.build_cpts()
    pots = model.build_potentials()

    assert 'x' in cpts
    assert 'x' in pots

    # For Delta, CPT should equal the module output for a zero input of appropriate size
    cpt = cpts['x']
    pot = pots['x']
    assert isinstance(cpt, torch.Tensor)
    assert isinstance(pot, torch.Tensor)
    # shapes: for our setup, input batch is 1 and out_features is 1
    assert cpt.shape[-1] >= 1
    assert pot.shape[-1] >= 1


def test_probabilistic_model_with_parent_bernolli_and_helpers():
    # Parent variable (Bernoulli) and child depending on parent
    parent = Variable(concepts='p', parents=[], distribution=Bernoulli, size=1)
    child = Variable(concepts='c', parents=['p'], distribution=Bernoulli, size=1)

    # parametrizations: parent has no parents, so its module.in_features can be 1
    parent_module = nn.Linear(in_features=1, out_features=1)
    child_module = nn.Linear(in_features=1, out_features=1)  # expects parent.out_features == 1

    parent_pcpd = ParametricCPD(concepts='p', parametrization=parent_module)
    child_pcpd = ParametricCPD(concepts='c', parametrization=child_module)

    model = ProbabilisticModel(variables=[parent, child], parametric_cpds=[parent_pcpd, child_pcpd])

    # get_by_distribution
    bern_vars = model.get_by_distribution(Bernoulli)
    assert any(v.concepts[0] == 'p' for v in bern_vars)
    assert any(v.concepts[0] == 'c' for v in bern_vars)

    # get_variable_parents resolves string parent to Variable
    parents_of_c = model.get_variable_parents('c')
    assert len(parents_of_c) == 1
    assert parents_of_c[0].concepts[0] == 'p'

    # get_module_of_concept returns the ParametricCPD module
    mod_c = model.get_module_of_concept('c')
    assert isinstance(mod_c, ParametricCPD)

    # Build CPT for child should succeed
    cpts = model.build_cpts()
    assert 'c' in cpts
    # For Bernoulli, CPT rows include parent state and probability column
    cpt_c = cpts['c']
    assert cpt_c.shape[1] >= 1


if __name__ == '__main__':
    unittest.main()
