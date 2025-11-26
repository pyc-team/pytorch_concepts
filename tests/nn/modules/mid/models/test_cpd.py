"""Comprehensive tests for ParametricCPD to increase coverage."""
import unittest

import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.variable import Variable
from torch_concepts.distributions import Delta


class TestParametricCPDBasic:
    """Test basic ParametricCPD functionality."""

    def test_single_concept_initialization(self):
        """Test ParametricCPD with single concept."""
        module = nn.Linear(5, 1)
        cpd = ParametricCPD(concepts='c1', parametrization=module)
        assert cpd.concepts == ['c1']
        assert cpd.parametrization is module

    def test_multi_concept_initialization_splits(self):
        """Test ParametricCPD splits into multiple CPDs for multiple concepts."""
        module = nn.Linear(5, 2)
        cpds = ParametricCPD(concepts=['c1', 'c2'], parametrization=module)
        assert isinstance(cpds, list)
        assert len(cpds) == 2
        assert cpds[0].concepts == ['c1']
        assert cpds[1].concepts == ['c2']

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
        var = Variable(concepts='c1', parents=[], distribution=Delta, size=1)
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
        parent_var = Variable(concepts='p', parents=[], distribution=Bernoulli, size=1)
        child_var = Variable(concepts='c', parents=['p'], distribution=Bernoulli, size=1)

        module = nn.Linear(1, 1)
        cpd = ParametricCPD(concepts='c', parametrization=module)
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
        parent_var = Variable(concepts='p', parents=[], distribution=Categorical, size=3)
        child_var = Variable(concepts='c', parents=['p'], distribution=Bernoulli, size=1)

        module = nn.Linear(3, 1)
        cpd = ParametricCPD(concepts='c', parametrization=module)
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
        parent_var = Variable(concepts='p', parents=[], distribution=Delta, size=2)
        child_var = Variable(concepts='c', parents=['p'], distribution=Delta, size=1)

        module = nn.Linear(2, 1)
        cpd = ParametricCPD(concepts='c', parametrization=module)
        cpd.variable = child_var
        cpd.parents = [parent_var]

        all_inputs, discrete_states = cpd._get_parent_combinations()
        # Continuous parent: fixed zeros placeholder
        assert all_inputs.shape == (1, 2)
        assert discrete_states.shape == (1, 0)
        assert torch.allclose(all_inputs, torch.zeros((1, 2)))

    def test_mixed_discrete_and_continuous_parents(self):
        """Test _get_parent_combinations with mixed parents."""
        p1 = Variable(concepts='p1', parents=[], distribution=Bernoulli, size=1)
        p2 = Variable(concepts='p2', parents=[], distribution=Delta, size=2)
        child_var = Variable(concepts='c', parents=['p1', 'p2'], distribution=Bernoulli, size=1)

        module = nn.Linear(3, 1)  # 1 from Bernoulli + 2 from Delta
        cpd = ParametricCPD(concepts='c', parametrization=module)
        cpd.variable = child_var
        cpd.parents = [p1, p2]

        all_inputs, discrete_states = cpd._get_parent_combinations()
        # Bernoulli: 2 states, continuous fixed at zeros
        assert all_inputs.shape == (2, 3)
        assert discrete_states.shape == (2, 1)
        # First 2 rows should differ only in the discrete part
        assert torch.allclose(all_inputs[:, 1:], torch.zeros((2, 2)))


class TestParametricCPDBuildCPT:
    """Test build_cpt method."""

    def test_build_cpt_delta_no_parents(self):
        """Test build_cpt for Delta variable with no parents."""
        var = Variable(concepts='c', parents=[], distribution=Delta, size=1)
        module = nn.Linear(2, 1)
        cpd = ParametricCPD(concepts='c', parametrization=module)
        cpd.variable = var
        cpd.parents = []

        cpt = cpd.build_cpt()
        # For Delta, CPT is just the output
        assert cpt.shape[0] == 1
        assert cpt.shape[1] == 1

    def test_build_cpt_bernoulli_no_parents(self):
        """Test build_cpt for Bernoulli variable with no parents."""
        var = Variable(concepts='c', parents=[], distribution=Bernoulli, size=1)
        module = nn.Linear(1, 1)
        cpd = ParametricCPD(concepts='c', parametrization=module)
        cpd.variable = var
        cpd.parents = []

        cpt = cpd.build_cpt()
        # For Bernoulli with no parents: [P(X=1)]
        assert cpt.shape[0] == 1
        # CPT should be [discrete_state_vectors (0 cols) | P(X=1) (1 col)]
        assert cpt.shape[1] == 1

    def test_build_cpt_bernoulli_with_parent(self):
        """Test build_cpt for Bernoulli variable with Bernoulli parent."""
        parent = Variable(concepts='p', parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts='c', parents=['p'], distribution=Bernoulli, size=1)

        module = nn.Linear(1, 1)
        cpd = ParametricCPD(concepts='c', parametrization=module)
        cpd.variable = child
        cpd.parents = [parent]

        cpt = cpd.build_cpt()
        # 2 parent states, CPT: [Parent State | P(C=1)]
        assert cpt.shape == (2, 2)
        # First column should be parent states [0, 1]
        assert torch.allclose(cpt[:, 0], torch.tensor([0.0, 1.0]))
        # Second column should be probabilities in [0, 1]
        assert torch.all((cpt[:, 1] >= 0.0) & (cpt[:, 1] <= 1.0))

    def test_build_cpt_categorical(self):
        """Test build_cpt for Categorical variable."""
        var = Variable(concepts='c', parents=[], distribution=Categorical, size=3)
        module = nn.Linear(2, 3)
        cpd = ParametricCPD(concepts='c', parametrization=module)
        cpd.variable = var
        cpd.parents = []

        cpt = cpd.build_cpt()
        # Categorical: CPT is softmax probabilities
        assert cpt.shape == (1, 3)
        # Probabilities should sum to 1
        assert torch.allclose(cpt.sum(dim=-1), torch.tensor([1.0]))

    def test_build_cpt_input_mismatch_raises_error(self):
        """Test build_cpt raises error when input dimensions mismatch."""
        parent = Variable(concepts='p', parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts='c', parents=['p'], distribution=Bernoulli, size=1)

        # Module expects 5 features but parent only provides 1
        module = nn.Linear(5, 1)
        cpd = ParametricCPD(concepts='c', parametrization=module)
        cpd.variable = child
        cpd.parents = [parent]

        with pytest.raises(RuntimeError, match="Input tensor dimension mismatch"):
            cpd.build_cpt()


class TestParametricCPDBuildPotential:
    """Test build_potential method."""

    def test_build_potential_bernoulli_no_parents(self):
        """Test build_potential for Bernoulli variable with no parents."""
        var = Variable(concepts='c', parents=[], distribution=Bernoulli, size=1)
        module = nn.Linear(1, 1)
        cpd = ParametricCPD(concepts='c', parametrization=module)
        cpd.variable = var
        cpd.parents = []

        pot = cpd.build_potential()
        # Potential for Bernoulli: [Parent States (0 cols) | Child State | P(X=state)]
        # Two rows: one for X=1, one for X=0
        assert pot.shape == (2, 2)
        # Child state column should have [1, 0]
        assert torch.allclose(pot[:, 0], torch.tensor([1.0, 0.0]))
        # Probabilities should sum to 1
        assert torch.allclose(pot[:, 1].sum(), torch.tensor(1.0), atol=1e-5)

    def test_build_potential_bernoulli_with_parent(self):
        """Test build_potential for Bernoulli with Bernoulli parent."""
        parent = Variable(concepts='p', parents=[], distribution=Bernoulli, size=1)
        child = Variable(concepts='c', parents=['p'], distribution=Bernoulli, size=1)

        module = nn.Linear(1, 1)
        cpd = ParametricCPD(concepts='c', parametrization=module)
        cpd.variable = child
        cpd.parents = [parent]

        pot = cpd.build_potential()
        # 2 parent states × 2 child states = 4 rows
        # [Parent State | Child State | P(C=child_state | P=parent_state)]
        assert pot.shape == (4, 3)
        # Child states should be [1, 1, 0, 0] (ordered by child first, then parent varies)
        # Actually the implementation does [c=1 for all parents], [c=0 for all parents]
        # So first 2 rows: child=1, last 2 rows: child=0
        assert torch.allclose(pot[:2, 1], torch.tensor([1.0, 1.0]))
        assert torch.allclose(pot[2:, 1], torch.tensor([0.0, 0.0]))

    def test_build_potential_categorical(self):
        """Test build_potential for Categorical variable."""
        var = Variable(concepts='c', parents=[], distribution=Categorical, size=3)
        module = nn.Linear(2, 3)
        cpd = ParametricCPD(concepts='c', parametrization=module)
        cpd.variable = var
        cpd.parents = []

        pot = cpd.build_potential()
        # 3 classes: 3 rows [Parent States (0) | Child State | P(X=i)]
        assert pot.shape == (3, 2)
        # Child state column should be [0, 1, 2]
        assert torch.allclose(pot[:, 0], torch.tensor([0.0, 1.0, 2.0]))
        # Probabilities should sum to 1 across all rows
        assert torch.allclose(pot[:, 1].sum(), torch.tensor(1.0), atol=1e-5)

    def test_build_potential_delta(self):
        """Test build_potential for Delta variable."""
        var = Variable(concepts='c', parents=[], distribution=Delta, size=2)
        module = nn.Linear(3, 2)
        cpd = ParametricCPD(concepts='c', parametrization=module)
        cpd.variable = var
        cpd.parents = []

        pot = cpd.build_potential()
        # Delta: [Parent States (0) | Child Value (2 dims)]
        assert pot.shape == (1, 2)


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
        self.assertEqual(cpd.concepts, ['concept_a'])
        self.assertIsNotNone(cpd.modules)

    def test_multiple_concepts_single_module(self):
        """Test multiple concepts with single module (replicated)."""
        module = nn.Linear(10, 1)
        cpds = ParametricCPD(concepts=['A', 'B', 'C'], parametrization=module)
        self.assertEqual(len(cpds), 3)
        self.assertEqual(cpds[0].concepts, ['A'])
        self.assertEqual(cpds[1].concepts, ['B'])
        self.assertEqual(cpds[2].concepts, ['C'])

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

        var = Variable(concepts=['concept'], parents=[], distribution=Bernoulli, size=1)
        cpd.variable = var

        self.assertEqual(cpd.variable, var)

    def test_cpd_with_parents(self):
        """Test cpd with parent variables."""
        module = nn.Linear(10, 1)
        cpd = ParametricCPD(concepts='child', parametrization=module)

        parent_var = Variable(concepts=['parent'], parents=[], distribution=Bernoulli, size=1)
        cpd.parents = [parent_var]

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
        var = Variable(concepts=['concept'], parents=[], distribution=Bernoulli, size=1)
        cpd.variable = var
        cpd.parents = []

        inputs, states = cpd._get_parent_combinations()
        self.assertEqual(inputs.shape[0], 1)
        self.assertEqual(states.shape[1], 0)

    def test_get_parent_combinations_bernoulli_parent(self):
        """Test _get_parent_combinations with Bernoulli parent."""
        parent_var = Variable(concepts=['parent'], parents=[], distribution=Bernoulli, size=1)
        module = nn.Linear(1, 1)
        cpd = ParametricCPD(concepts='child', parametrization=module)
        child_var = Variable(concepts=['child'], parents=[parent_var], distribution=Bernoulli, size=1)
        cpd.variable = child_var
        cpd.parents = [parent_var]

        inputs, states = cpd._get_parent_combinations()
        # Bernoulli with size=1 should give 2 combinations: [0], [1]
        self.assertEqual(inputs.shape[0], 2)

    def test_get_parent_combinations_categorical_parent(self):
        """Test _get_parent_combinations with Categorical parent."""
        parent_var = Variable(concepts=['parent'], parents=[], distribution=Categorical, size=3)
        module = nn.Linear(3, 1)
        cpd = ParametricCPD(concepts='child', parametrization=module)
        child_var = Variable(concepts=['child'], parents=[parent_var], distribution=Bernoulli, size=1)
        cpd.variable = child_var
        cpd.parents = [parent_var]

        inputs, states = cpd._get_parent_combinations()
        # Categorical with size=3 should give 3 combinations
        self.assertEqual(inputs.shape[0], 3)

    def test_get_parent_combinations_delta_parent(self):
        """Test _get_parent_combinations with Delta parent."""
        parent_var = Variable(concepts=['parent'], parents=[], distribution=Delta, size=2)
        module = nn.Linear(2, 1)
        cpd = ParametricCPD(concepts='child', parametrization=module)
        child_var = Variable(concepts=['child'], parents=[parent_var], distribution=Bernoulli, size=1)
        cpd.variable = child_var
        cpd.parents = [parent_var]

        inputs, states = cpd._get_parent_combinations()
        self.assertIsNotNone(inputs)

    def test_build_cpt_without_variable(self):
        """Test build_cpt raises error when variable not linked."""
        module = nn.Linear(10, 1)
        cpd = ParametricCPD(concepts='concept', parametrization=module)

        with self.assertRaises(RuntimeError):
            cpd.build_cpt()


if __name__ == '__main__':
    unittest.main()
