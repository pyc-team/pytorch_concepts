"""
Comprehensive tests for torch_concepts.nn.modules.mid.models

Tests for ProbabilisticModel (generic container and directed model),
ParametricFactor (base factor), and integration between Variables and CPDs.
"""
import unittest
import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch_concepts.nn.modules.mid.models.variable import Variable
from torch_concepts.nn.modules.mid.models.factor import ParametricFactor
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.mid.models.probabilistic_model import (
    ProbabilisticModel,
)


# ===========================================================================
# Tests for ParametricFactor (base, scope-aware, no parents)
# ===========================================================================

class TestParametricFactor(unittest.TestCase):
    """Test ParametricFactor base class."""

    def test_single_concept(self):
        """Test ParametricFactor with single concept string."""
        module = nn.Linear(5, 1)
        factor = ParametricFactor(concepts='c1', parametrization=module)
        self.assertEqual(factor.concept, 'c1')
        self.assertEqual(factor.concepts, ['c1'])
        self.assertIs(factor.parametrization, module)
        self.assertIsNone(factor.variable)

    def test_multi_concept_scope(self):
        """Test ParametricFactor with list of concepts creates a single factor with full scope."""
        module = nn.Linear(5, 2)
        factor = ParametricFactor(concepts=['c1', 'c2'], parametrization=module)
        self.assertIsInstance(factor, ParametricFactor)
        self.assertEqual(factor.concepts, ['c1', 'c2'])
        self.assertEqual(factor.concept, 'c1')
        self.assertIs(factor.parametrization, module)

    def test_forward_pass(self):
        """Test forward pass through ParametricFactor."""
        module = nn.Linear(3, 1)
        factor = ParametricFactor(concepts='c1', parametrization=module)
        x = torch.randn(2, 3)
        out = factor(input=x)
        self.assertEqual(out.shape, (2, 1))

    def test_forward_pass_multi_concept(self):
        """Test forward pass through ParametricFactor with multi-concept scope."""
        module = nn.Linear(3, 2)
        factor = ParametricFactor(concepts=['c1', 'c2'], parametrization=module)
        x = torch.randn(2, 3)
        out = factor(input=x)
        self.assertEqual(out.shape, (2, 2))

    def test_no_parents_attribute(self):
        """Test that ParametricFactor does NOT have a parents list."""
        module = nn.Linear(5, 1)
        factor = ParametricFactor(concepts='c1', parametrization=module)
        # ParametricFactor should not have parents (that's ParametricCPD's job)
        self.assertFalse(hasattr(factor, 'parents') and factor.parents)

    def test_repr_single_concept(self):
        """Test repr string for single-concept factor."""
        module = nn.Linear(5, 1)
        factor = ParametricFactor(concepts='c1', parametrization=module)
        self.assertIn('c1', repr(factor))
        self.assertIn('Linear', repr(factor))

    def test_repr_multi_concept(self):
        """Test repr string for multi-concept factor shows full scope."""
        module = nn.Linear(5, 2)
        factor = ParametricFactor(concepts=['c1', 'c2'], parametrization=module)
        r = repr(factor)
        self.assertIn('c1', r)
        self.assertIn('c2', r)
        self.assertIn('Linear', r)


# ===========================================================================
# Tests for ProbabilisticModel (generic container)
# ===========================================================================

class TestProbabilisticModel(unittest.TestCase):
    """Test ProbabilisticModel as a generic factor-graph container."""

    def test_initialization_with_factors_kwarg(self):
        """Test initialization using 'factors' keyword."""
        var = Variable(concepts='A', distribution=Bernoulli, size=1)
        factor = ParametricFactor(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], factors=[factor])
        self.assertEqual(len(model.variables), 1)
        self.assertEqual(len(model.factors), 1)

    def test_initialization_with_parametric_cpds_kwarg(self):
        """Test backward-compat initialization using 'parametric_cpds'."""
        var = Variable(concepts='A', distribution=Bernoulli, size=1)
        cpd = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])
        self.assertEqual(len(model.factors), 1)

    def test_initialization_empty_lists(self):
        """Test initialization with empty lists."""
        model = ProbabilisticModel(variables=[], factors=[])
        self.assertEqual(len(model.variables), 0)
        self.assertEqual(len(model.factors), 0)

    def test_no_factors_raises_type_error(self):
        """Test that omitting both factors and parametric_cpds raises TypeError."""
        with self.assertRaises(TypeError):
            ProbabilisticModel(variables=[])

    def test_add_single_variable(self):
        """Test adding a single variable."""
        var = Variable(concepts='A', distribution=Bernoulli, size=1)
        factor = ParametricFactor(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], factors=[factor])
        self.assertEqual(len(model.variables), 1)

    def test_add_multiple_variables(self):
        """Test adding multiple variables."""
        vars_list = [
            Variable(concepts='A', distribution=Bernoulli, size=1),
            Variable(concepts='B', distribution=Bernoulli, size=1),
            Variable(concepts='C', distribution=Bernoulli, size=1)
        ]
        model = ProbabilisticModel(variables=vars_list, factors=[])
        self.assertEqual(len(model.variables), 3)

    def test_concept_to_variable_mapping(self):
        """Test concept name to variable mapping."""
        var_a = Variable(concepts='A', distribution=Bernoulli, size=1)
        var_b = Variable(concepts='B', distribution=Categorical, size=3)
        model = ProbabilisticModel(variables=[var_a, var_b], factors=[])
        self.assertIn('A', model.concept_to_variable)
        self.assertIn('B', model.concept_to_variable)
        self.assertIs(model.concept_to_variable['A'], var_a)

    def test_get_module_of_concept(self):
        """Test get_module_of_concept method."""
        var = Variable(concepts='A', distribution=Bernoulli, size=1)
        factor = ParametricFactor(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], factors=[factor])

        module = model.get_module_of_concept('A')
        self.assertIsNotNone(module)
        self.assertEqual(module.concept, 'A')

    def test_get_module_of_nonexistent_concept(self):
        """Test get_module_of_concept with non-existent concept."""
        var = Variable(concepts='A', distribution=Bernoulli, size=1)
        factor = ParametricFactor(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], factors=[factor])

        module = model.get_module_of_concept('B')
        self.assertIsNone(module)

    def test_get_by_distribution(self):
        """Test get_by_distribution method."""
        var_bern = Variable(concepts='b', distribution=Bernoulli, size=1)
        var_cat = Variable(concepts='c', distribution=Categorical, size=3)
        model = ProbabilisticModel(variables=[var_bern, var_cat], factors=[])
        bern_vars = model.get_by_distribution(Bernoulli)
        self.assertEqual(len(bern_vars), 1)
        self.assertEqual(bern_vars[0].concept, 'b')

    def test_variable_linkage(self):
        """Test that factors are linked to their corresponding variables."""
        var = Variable(concepts='A', distribution=Bernoulli, size=1)
        factor = ParametricFactor(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], factors=[factor])

        retrieved = model.get_module_of_concept('A')
        self.assertIs(retrieved.variable, var)

    def test_factors_registered_as_modules(self):
        """Test that factors are properly registered as nn.Module submodules."""
        var = Variable(concepts='A', distribution=Bernoulli, size=1)
        factor = ParametricFactor(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], factors=[factor])
        params = list(model.parameters())
        self.assertGreater(len(params), 0)

    def test_mixed_distributions(self):
        """Test model with mixed distribution types."""
        var_delta = Variable(concepts='emb', distribution=Delta, size=10)
        var_bern = Variable(concepts='binary', distribution=Bernoulli, size=1)
        var_cat = Variable(concepts='multi', distribution=Categorical, size=3)

        f_delta = ParametricFactor(concepts='emb', parametrization=nn.Identity())
        f_bern = ParametricFactor(concepts='binary', parametrization=nn.Linear(10, 1))
        f_cat = ParametricFactor(concepts='multi', parametrization=nn.Linear(10, 3))

        model = ProbabilisticModel(
            variables=[var_delta, var_bern, var_cat],
            factors=[f_delta, f_bern, f_cat]
        )
        self.assertEqual(len(model.variables), 3)
        self.assertEqual(len(model.factors), 3)


# ===========================================================================
# Tests for ProbabilisticModel (directed model)
# ===========================================================================

class TestProbabilisticModel(unittest.TestCase):
    """Test ProbabilisticModel directed subclass."""

    def test_initialization(self):
        """Test basic ProbabilisticModel initialization."""
        var = Variable(concepts='A', distribution=Bernoulli, size=1)
        cpd = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])
        self.assertEqual(len(model.variables), 1)
        self.assertEqual(len(model.factors), 1)

    def test_parametric_cpds_alias(self):
        """Test that parametric_cpds property aliases factors."""
        var = Variable(concepts='A', distribution=Bernoulli, size=1)
        cpd = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])
        self.assertIs(model.parametric_cpds, model.factors)

    def test_hierarchical_structure(self):
        """Test parent-child structure with ProbabilisticModel."""
        parent = Variable(concepts='parent', distribution=Bernoulli, size=1)
        child = Variable(concepts='child', distribution=Bernoulli, size=1)

        parent_cpd = ParametricCPD(concepts='parent', parametrization=nn.Linear(10, 1))
        child_cpd = ParametricCPD(concepts='child', parametrization=nn.Linear(1, 1), parents=[parent])

        model = ProbabilisticModel(
            variables=[parent, child],
            parametric_cpds=[parent_cpd, child_cpd]
        )
        self.assertEqual(len(model.variables), 2)
        self.assertEqual(len(model.factors), 2)

    def test_string_parent_resolution(self):
        """Test that string parents are resolved to Variable objects."""
        parent = Variable(concepts='p', distribution=Bernoulli, size=1)
        child = Variable(concepts='c', distribution=Bernoulli, size=1)

        parent_cpd = ParametricCPD(concepts='p', parametrization=nn.Linear(1, 1))
        child_cpd = ParametricCPD(concepts='c', parametrization=nn.Linear(1, 1), parents=['p'])

        model = ProbabilisticModel(
            variables=[parent, child],
            parametric_cpds=[parent_cpd, child_cpd]
        )
        resolved_parents = model.get_variable_parents('c')
        self.assertEqual(len(resolved_parents), 1)
        self.assertIsInstance(resolved_parents[0], Variable)
        self.assertEqual(resolved_parents[0].concept, 'p')

    def test_get_variable_parents(self):
        """Test get_variable_parents method."""
        p1 = Variable(concepts='p1', distribution=Bernoulli, size=1)
        p2 = Variable(concepts='p2', distribution=Bernoulli, size=1)
        child = Variable(concepts='child', distribution=Bernoulli, size=1)

        p1_cpd = ParametricCPD(concepts='p1', parametrization=nn.Linear(10, 1))
        p2_cpd = ParametricCPD(concepts='p2', parametrization=nn.Linear(10, 1))
        child_cpd = ParametricCPD(concepts='child', parametrization=nn.Linear(2, 1), parents=[p1, p2])

        model = ProbabilisticModel(
            variables=[p1, p2, child],
            parametric_cpds=[p1_cpd, p2_cpd, child_cpd]
        )
        parents = model.get_variable_parents('child')
        self.assertEqual(len(parents), 2)
        parent_concepts = {p.concept for p in parents}
        self.assertIn('p1', parent_concepts)
        self.assertIn('p2', parent_concepts)

    def test_get_variable_parents_root(self):
        """Test get_variable_parents returns empty list for root."""
        var = Variable(concepts='A', distribution=Bernoulli, size=1)
        cpd = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])
        self.assertEqual(model.get_variable_parents('A'), [])

    def test_get_variable_parents_nonexistent(self):
        """Test get_variable_parents returns empty for nonexistent concept."""
        var = Variable(concepts='A', distribution=Bernoulli, size=1)
        cpd = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))
        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])
        self.assertEqual(model.get_variable_parents('Z'), [])

    def test_build_cpts_no_parents_delta(self):
        """Test build_cpts for Delta variable with no parents."""
        var = Variable(concepts='x', distribution=Delta, size=1)
        module = nn.Linear(in_features=2, out_features=1)
        cpd = ParametricCPD(concepts='x', parametrization=module)

        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])
        cpts = model.build_cpts()

        self.assertIn('x', cpts)
        self.assertIsInstance(cpts['x'], torch.Tensor)
        self.assertGreaterEqual(cpts['x'].shape[-1], 1)

    def test_build_potentials_no_parents_delta(self):
        """Test build_potentials for Delta variable with no parents."""
        var = Variable(concepts='x', distribution=Delta, size=1)
        module = nn.Linear(in_features=2, out_features=1)
        cpd = ParametricCPD(concepts='x', parametrization=module)

        model = ProbabilisticModel(variables=[var], parametric_cpds=[cpd])
        pots = model.build_potentials()

        self.assertIn('x', pots)
        self.assertIsInstance(pots['x'], torch.Tensor)

    def test_build_cpts_with_parent_bernoulli(self):
        """Test build_cpts with parent-child Bernoulli structure."""
        parent = Variable(concepts='p', distribution=Bernoulli, size=1)
        child = Variable(concepts='c', distribution=Bernoulli, size=1)

        parent_cpd = ParametricCPD(concepts='p', parametrization=nn.Linear(1, 1))
        child_cpd = ParametricCPD(concepts='c', parametrization=nn.Linear(1, 1), parents=['p'])

        model = ProbabilisticModel(
            variables=[parent, child],
            parametric_cpds=[parent_cpd, child_cpd]
        )

        cpts = model.build_cpts()
        self.assertIn('c', cpts)
        cpt_c = cpts['c']
        self.assertGreaterEqual(cpt_c.shape[1], 1)

    def test_get_by_distribution(self):
        """Test that get_by_distribution works on ProbabilisticModel."""
        parent = Variable(concepts='p', distribution=Bernoulli, size=1)
        child = Variable(concepts='c', distribution=Bernoulli, size=1)

        parent_cpd = ParametricCPD(concepts='p', parametrization=nn.Linear(1, 1))
        child_cpd = ParametricCPD(concepts='c', parametrization=nn.Linear(1, 1), parents=['p'])

        model = ProbabilisticModel(
            variables=[parent, child],
            parametric_cpds=[parent_cpd, child_cpd]
        )
        bern_vars = model.get_by_distribution(Bernoulli)
        self.assertEqual(len(bern_vars), 2)

    def test_complex_hierarchy(self):
        """Test complex hierarchical structure: A -> B, A -> C, B+C -> D."""
        var_a = Variable(concepts='A', distribution=Bernoulli, size=1)
        var_b = Variable(concepts='B', distribution=Bernoulli, size=1)
        var_c = Variable(concepts='C', distribution=Bernoulli, size=1)
        var_d = Variable(concepts='D', distribution=Bernoulli, size=1)

        cpd_a = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))
        cpd_b = ParametricCPD(concepts='B', parametrization=nn.Linear(1, 1), parents=['A'])
        cpd_c = ParametricCPD(concepts='C', parametrization=nn.Linear(1, 1), parents=['A'])
        cpd_d = ParametricCPD(concepts='D', parametrization=nn.Linear(2, 1), parents=['B', 'C'])

        model = ProbabilisticModel(
            variables=[var_a, var_b, var_c, var_d],
            parametric_cpds=[cpd_a, cpd_b, cpd_c, cpd_d]
        )
        self.assertEqual(len(model.variables), 4)
        d_parents = model.get_variable_parents('D')
        self.assertEqual(len(d_parents), 2)


# ===========================================================================
# Tests for integration between Variables and ParametricCPDs
# ===========================================================================

class TestVariableParametricCPDIntegration(unittest.TestCase):
    """Test integration between Variables and ParametricCPDs."""

    def test_cpd_output_matches_variable_size(self):
        """Test that cpd output size matches variable size."""
        var = Variable(concepts='A', distribution=Bernoulli, size=1)
        cpd = ParametricCPD(concepts='A', parametrization=nn.Linear(10, 1))

        x = torch.randn(4, 10)
        output = cpd(input=x)
        self.assertEqual(output.shape[1], var.out_features)

    def test_parent_child_feature_matching(self):
        """Test that child input features match parent output features."""
        parent = Variable(concepts='parent', distribution=Categorical, size=3)
        child = Variable(concepts='child', distribution=Bernoulli, size=1)

        child_cpd = ParametricCPD(concepts='child', parametrization=nn.Linear(3, 1), parents=[parent])

        parent_output = torch.randn(4, 3)
        child_output = child_cpd(input=parent_output)
        self.assertEqual(child_output.shape, (4, 1))

    def test_in_features_with_parents(self):
        """Test in_features property on ParametricCPD."""
        p1 = Variable(concepts='p1', distribution=Bernoulli, size=1)
        p2 = Variable(concepts='p2', distribution=Categorical, size=3)
        cpd = ParametricCPD(concepts='child', parametrization=nn.Linear(4, 1), parents=[p1, p2])
        self.assertEqual(cpd.in_features, 4)

    def test_in_features_no_parents(self):
        """Test in_features returns 0 for root CPD."""
        cpd = ParametricCPD(concepts='root', parametrization=nn.Linear(10, 1))
        self.assertEqual(cpd.in_features, 0)


if __name__ == '__main__':
    unittest.main()
