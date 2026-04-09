"""Comprehensive tests for ParametricCPD to increase coverage."""
import unittest

import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from torch_concepts.nn.modules.mid.models.parametric_cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.variable import Variable, ConceptVariable
from torch_concepts.distributions import Delta


class TestParametricCPDBasic:
    """Test basic ParametricCPD functionality."""

    def test_single_concept_initialization(self):
        """Test ParametricCPD with single concept."""
        module = nn.Linear(5, 1)
        cpd = ParametricCPD(concepts='c1', parametrization=module)
        assert cpd.concept == 'c1'
        assert cpd.parametrization is module

    def test_multi_concept_initialization_splits(self):
        """Test ParametricCPD splits into multiple CPDs for multiple concepts."""
        module = nn.Linear(5, 2)
        cpds = ParametricCPD(concepts=['c1', 'c2'], parametrization=module)
        assert isinstance(cpds, list)
        assert len(cpds) == 2
        assert cpds[0].concept == 'c1'
        assert cpds[1].concept == 'c2'

    def test_multi_concept_with_module_list(self):
        """Test ParametricCPD with list of modules."""
        mod1 = nn.Linear(5, 1)
        mod2 = nn.Linear(5, 1)
        cpds = ParametricCPD(concepts=['c1', 'c2'], parametrization=[mod1, mod2])
        assert len(cpds) == 2
        assert cpds[0].parametrization.in_features == 5
        assert cpds[1].parametrization.in_features == 5

    def test_forward_pass(self):
        """Test forward pass through ParametricCPD."""
        module = nn.Linear(3, 1)
        cpd = ParametricCPD(concepts='c1', parametrization=module)
        x = torch.randn(2, 3)
        out = cpd(input=x)
        assert out.shape == (2, 1)


class TestParametricCPDParentCombinations:
    """Test _get_parent_combinations method."""

    def test_no_parents(self):
        """Test _get_parent_combinations with no parents."""
        var = Variable(concepts='c1', distribution=Delta, size=1)
        module = nn.Linear(2, 1)
        cpd = ParametricCPD(concepts='c1', parametrization=module)
        cpd.variable = var
        cpd.parents = []

        all_inputs, discrete_states = cpd._get_parent_combinations()
        # No parents: should return placeholder with shape (1, in_features)
        assert all_inputs.shape == (1, 2)
        assert discrete_states.shape == (1, 0)

    def test_single_bernoulli_parent(self):
        """Test _get_parent_combinations with single Bernoulli parent."""
        parent_var = Variable(concepts='p', distribution=Bernoulli, size=1)
        child_var = Variable(concepts='c', distribution=Bernoulli, size=1)

        module = nn.Linear(1, 1)
        cpd = ParametricCPD(concepts='c', parametrization=module, parents=['p'])
        cpd.variable = child_var
        cpd.parents = [parent_var]

        all_inputs, discrete_states = cpd._get_parent_combinations()
        # Bernoulli parent: 2 states (0, 1)
        assert all_inputs.shape == (2, 1)
        assert discrete_states.shape == (2, 1)
        # Check values
        assert torch.allclose(all_inputs, torch.tensor([[0.0], [1.0]]))

    def test_single_categorical_parent(self):
        """Test _get_parent_combinations with Categorical parent."""
        parent_var = Variable(concepts='p', distribution=Categorical, size=3)
        child_var = Variable(concepts='c', distribution=Bernoulli, size=1)

        module = nn.Linear(3, 1)
        cpd = ParametricCPD(concepts='c', parametrization=module, parents=['p'])
        cpd.variable = child_var
        cpd.parents = [parent_var]

        all_inputs, discrete_states = cpd._get_parent_combinations()
        # Categorical with 3 classes: 3 one-hot states
        assert all_inputs.shape == (3, 3)
        assert discrete_states.shape == (3, 1)
        # Should contain one-hot vectors
        assert torch.allclose(all_inputs[0], torch.tensor([1.0, 0.0, 0.0]))
        assert torch.allclose(all_inputs[1], torch.tensor([0.0, 1.0, 0.0]))
        assert torch.allclose(all_inputs[2], torch.tensor([0.0, 0.0, 1.0]))

    def test_continuous_parent_only(self):
        """Test _get_parent_combinations with only continuous (Delta) parent."""
        parent_var = Variable(concepts='p', distribution=Delta, size=2)
        child_var = Variable(concepts='c', distribution=Delta, size=1)

        module = nn.Linear(2, 1)
        cpd = ParametricCPD(concepts='c', parametrization=module, parents=['p'])
        cpd.variable = child_var
        cpd.parents = [parent_var]

        all_inputs, discrete_states = cpd._get_parent_combinations()
        # Continuous parent: fixed zeros placeholder
        assert all_inputs.shape == (1, 2)
        assert discrete_states.shape == (1, 0)
        assert torch.allclose(all_inputs, torch.zeros((1, 2)))

    def test_mixed_discrete_and_continuous_parents(self):
        """Test _get_parent_combinations with mixed parents."""
        p1 = Variable(concepts='p1', distribution=Bernoulli, size=1)
        p2 = Variable(concepts='p2', distribution=Delta, size=2)
        child_var = Variable(concepts='c', distribution=Bernoulli, size=1)

        module = nn.Linear(3, 1)  # 1 from Bernoulli + 2 from Delta
        cpd = ParametricCPD(concepts='c', parametrization=module, parents=['p'])
        cpd.variable = child_var
        cpd.parents = [p1, p2]

        all_inputs, discrete_states = cpd._get_parent_combinations()
        # Bernoulli: 2 states, continuous fixed at zeros
        assert all_inputs.shape == (2, 3)
        assert discrete_states.shape == (2, 1)
        # First 2 rows should differ only in the discrete part
        assert torch.allclose(all_inputs[:, 1:], torch.zeros((2, 2)))


class TestParametricCPDRepr:
    """Test __repr__ method."""

    def test_repr_output(self):
        """Test string representation of ParametricCPD."""
        module = nn.Linear(5, 1)
        cpd = ParametricCPD(concepts='c1', parametrization=module)
        repr_str = repr(cpd)
        assert 'ParametricCPD' in repr_str
        assert 'c1' in repr_str
        assert 'Linear' in repr_str



class TestParametricCPD(unittest.TestCase):
    """Test ParametricCPD class."""

    def test_single_concept_cpd(self):
        """Test creating a cpd with single concept."""
        module = nn.Linear(10, 1)
        cpd = ParametricCPD(concepts='concept_a', parametrization=module)
        self.assertEqual(cpd.concept, 'concept_a')
        self.assertIsNotNone(cpd.modules)

    def test_multiple_concepts_single_module(self):
        """Test multiple concepts with single module (replicated)."""
        module = nn.Linear(10, 1)
        cpds = ParametricCPD(concepts=['A', 'B', 'C'], parametrization=module)
        self.assertEqual(len(cpds), 3)
        self.assertEqual(cpds[0].concept, 'A')
        self.assertEqual(cpds[1].concept, 'B')
        self.assertEqual(cpds[2].concept, 'C')

    def test_multiple_concepts_multiple_modules(self):
        """Test multiple concepts with different modules."""
        module_a = nn.Linear(10, 1)
        module_b = nn.Linear(10, 2)
        module_c = nn.Linear(10, 3)

        cpds = ParametricCPD(
            concepts=['A', 'B', 'C'],
            parametrization=[module_a, module_b, module_c]
        )
        self.assertEqual(len(cpds), 3)
        self.assertIsInstance(cpds[0].parametrization, nn.Linear)
        self.assertEqual(cpds[1].parametrization.out_features, 2)
        self.assertEqual(cpds[2].parametrization.out_features, 3)

    def test_cpd_forward(self):
        """Test forward pass through cpd."""
        module = nn.Linear(10, 1)
        cpd = ParametricCPD(concepts='concept', parametrization=module)

        x = torch.randn(4, 10)
        output = cpd(input=x)
        self.assertEqual(output.shape, (4, 1))

    def test_cpd_with_variable(self):
        """Test linking cpd to variable."""
        module = nn.Linear(10, 1)
        cpd = ParametricCPD(concepts='concept', parametrization=module)

        var = Variable(concepts='concept', distribution=Bernoulli, size=1)
        cpd.variable = var

        self.assertEqual(cpd.variable, var)

    def test_cpd_with_parents(self):
        """Test cpd with parent variables."""
        module = nn.Linear(10, 1)
        parent_var = Variable(concepts='parent', distribution=Bernoulli, size=1)
        cpd = ParametricCPD(concepts='child', parametrization=module, parents=[parent_var])

        self.assertEqual(len(cpd.parents), 1)

    def test_cpd_validation_error(self):
        """Test validation error for mismatched concept/module counts."""
        with self.assertRaises(ValueError):
            ParametricCPD(
                concepts=['A', 'B', 'C'],
                parametrization=[nn.Linear(10, 1), nn.Linear(10, 1)]  # Only 2, need 3
            )

    def test_get_parent_combinations_no_parents(self):
        """Test _get_parent_combinations with no parents."""
        module = nn.Linear(10, 1)
        cpd = ParametricCPD(concepts='concept', parametrization=module)
        var = Variable(concepts='concept', distribution=Bernoulli, size=1)
        cpd.variable = var
        cpd.parents = []

        inputs, states = cpd._get_parent_combinations()
        self.assertEqual(inputs.shape[0], 1)
        self.assertEqual(states.shape[1], 0)

    def test_get_parent_combinations_bernoulli_parent(self):
        """Test _get_parent_combinations with Bernoulli parent."""
        parent_var = Variable(concepts='parent', distribution=Bernoulli, size=1)
        module = nn.Linear(1, 1)
        cpd = ParametricCPD(concepts='child', parametrization=module, parents=[parent_var])
        child_var = Variable(concepts='child', distribution=Bernoulli, size=1)
        cpd.variable = child_var
        cpd.parents = [parent_var]

        inputs, states = cpd._get_parent_combinations()
        # Bernoulli with size=1 should give 2 combinations: [0], [1]
        self.assertEqual(inputs.shape[0], 2)

    def test_get_parent_combinations_categorical_parent(self):
        """Test _get_parent_combinations with Categorical parent."""
        parent_var = Variable(concepts='parent', distribution=Categorical, size=3)
        module = nn.Linear(3, 1)
        cpd = ParametricCPD(concepts='child', parametrization=module, parents=[parent_var])
        child_var = Variable(concepts='child', distribution=Bernoulli, size=1)
        cpd.variable = child_var
        cpd.parents = [parent_var]

        inputs, states = cpd._get_parent_combinations()
        # Categorical with size=3 should give 3 combinations
        self.assertEqual(inputs.shape[0], 3)

    def test_get_parent_combinations_delta_parent(self):
        """Test _get_parent_combinations with Delta parent."""
        parent_var = Variable(concepts='parent', distribution=Delta, size=2)
        module = nn.Linear(2, 1)
        cpd = ParametricCPD(concepts='child', parametrization=module, parents=[parent_var])
        child_var = Variable(concepts='child', distribution=Bernoulli, size=1)
        cpd.variable = child_var
        cpd.parents = [parent_var]

        inputs, states = cpd._get_parent_combinations()
        self.assertIsNotNone(inputs)


class TestParametricCPDParentCap(unittest.TestCase):
    """Test _get_parent_combinations caps exponential blowup."""

    def test_too_many_binary_parents_raises(self):
        """More than _MAX_DISCRETE_BITS binary parents should raise RuntimeError."""
        parents = [ConceptVariable(f'p{i}', distribution=Bernoulli, size=1) for i in range(21)]

        cpd = ParametricCPD(concepts='child', parametrization=nn.Linear(21, 1), parents=[p.concept for p in parents])
        cpd.parents = parents  # bypass string resolution

        with self.assertRaises(RuntimeError, msg="discrete parent bits"):
            cpd._get_parent_combinations()

    def test_within_cap_succeeds(self):
        """A small number of parents should not raise."""
        parents = [ConceptVariable(f'p{i}', distribution=Bernoulli, size=1) for i in range(3)]

        cpd = ParametricCPD(concepts='child', parametrization=nn.Linear(3, 1), parents=[p.concept for p in parents])
        cpd.parents = parents
        inputs, states = cpd._get_parent_combinations()
        self.assertEqual(inputs.shape[0], 8)  # 2^3 = 8


if __name__ == '__main__':
    unittest.main()
