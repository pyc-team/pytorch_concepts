"""Comprehensive tests for ParametricCPD to increase coverage."""
import unittest

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, OneHotCategorical

from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.variable import Variable
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

if __name__ == '__main__':
    unittest.main()
