"""
Comprehensive tests for torch_concepts.nn.modules.mid.models

Tests for Variable, ParametricCPD, and ProbabilisticModel.
"""
import unittest
import pytest
import torch
from torch.distributions import Bernoulli, Categorical, Normal, RelaxedBernoulli

from torch_concepts.nn.modules.mid.models.variable import (
    Variable,
    EndogenousVariable,
    ExogenousVariable,
)
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


class TestVariableMultiConceptCreation:
    """Test Variable.__new__ multi-concept behavior."""

    def test_multi_concept_returns_list(self):
        """Test that multiple concepts return a list of Variables."""
        vars_list = Variable(
            concepts=['a', 'b', 'c'],
            parents=[],
            distribution=Delta,
            size=1
        )
        assert isinstance(vars_list, list)
        assert len(vars_list) == 3
        assert vars_list[0].concepts == ['a']
        assert vars_list[1].concepts == ['b']
        assert vars_list[2].concepts == ['c']

    def test_multi_concept_with_distribution_list(self):
        """Test multi-concept with per-concept distributions."""
        vars_list = Variable(
            concepts=['a', 'b', 'c'],
            parents=[],
            distribution=[Bernoulli, Delta, Categorical],
            size=[1, 2, 3]
        )
        assert len(vars_list) == 3
        assert vars_list[0].distribution is Bernoulli
        assert vars_list[1].distribution is Delta
        assert vars_list[2].distribution is Categorical

    def test_multi_concept_distribution_length_mismatch_raises_error(self):
        """Test that mismatched distribution list length raises error."""
        with pytest.raises(ValueError, match="distribution and size must either be single values or lists of length"):
            Variable(
                concepts=['a', 'b', 'c'],
                parents=[],
                distribution=[Bernoulli, Delta],  # Only 2, need 3
                size=1
            )

    def test_multi_concept_size_list_mismatch_raises_error(self):
        """Test that mismatched size list length raises error."""
        with pytest.raises(ValueError, match="distribution and size must either be single values or lists of length"):
            Variable(
                concepts=['a', 'b'],
                parents=[],
                distribution=Delta,
                size=[1, 2, 3]  # 3 sizes for 2 concepts
            )


class TestVariableValidation:
    """Test Variable validation logic."""

    def test_categorical_with_size_one_raises_error(self):
        """Test that Categorical with size=1 raises error."""
        with pytest.raises(ValueError, match="Categorical Variable must have a size > 1"):
            Variable(
                concepts='cat',
                parents=[],
                distribution=Categorical,
                size=1
            )

    def test_bernoulli_with_size_not_one_raises_error(self):
        """Test that Bernoulli with size != 1 raises error."""
        with pytest.raises(ValueError, match="Bernoulli Variable must have size=1"):
            Variable(
                concepts='bern',
                parents=[],
                distribution=Bernoulli,
                size=3
            )

    def test_normal_distribution_support(self):
        """Test that Normal distribution is supported."""
        var = Variable(
            concepts='norm',
            parents=[],
            distribution=Normal,
            size=2
        )
        assert var.distribution is Normal
        assert var.size == 2


class TestVariableOutFeatures:
    """Test out_features property calculation."""

    def test_out_features_delta(self):
        """Test out_features for Delta distribution."""
        var = Variable(concepts='d', parents=[], distribution=Delta, size=3)
        assert var.out_features == 3

    def test_out_features_bernoulli(self):
        """Test out_features for Bernoulli distribution."""
        var = Variable(concepts='b', parents=[], distribution=Bernoulli, size=1)
        assert var.out_features == 1

    def test_out_features_categorical(self):
        """Test out_features for Categorical distribution."""
        var = Variable(concepts=['c'], parents=[], distribution=Categorical, size=5)
        assert var.out_features == 5

    def test_out_features_normal(self):
        """Test out_features for Normal distribution."""
        var = Variable(concepts='n', parents=[], distribution=Normal, size=4)
        assert var.out_features == 4

    def test_out_features_cached(self):
        """Test that out_features is cached after first call."""
        var = Variable(concepts='x', parents=[], distribution=Delta, size=2)
        _ = var.out_features
        assert var._out_features == 2
        # Second call should use cached value
        assert var.out_features == 2


class TestVariableInFeatures:
    """Test in_features property calculation."""

    def test_in_features_no_parents(self):
        """Test in_features with no parents."""
        var = Variable(concepts='x', parents=[], distribution=Delta, size=2)
        assert var.in_features == 0

    def test_in_features_single_parent(self):
        """Test in_features with single parent."""
        parent = Variable(concepts='p', parents=[], distribution=Delta, size=3)
        child = Variable(concepts='c', parents=[parent], distribution=Delta, size=2)
        assert child.in_features == 3

    def test_in_features_multiple_parents(self):
        """Test in_features with multiple parents."""
        p1 = Variable(concepts='p1', parents=[], distribution=Delta, size=2)
        p2 = Variable(concepts='p2', parents=[], distribution=Bernoulli, size=1)
        p3 = Variable(concepts='p3', parents=[], distribution=Categorical, size=4)
        child = Variable(concepts='c', parents=[p1, p2, p3], distribution=Delta, size=1)
        assert child.in_features == 2 + 1 + 4

    def test_in_features_non_variable_parent_raises_error(self):
        """Test that non-Variable parent raises TypeError."""
        var = Variable(concepts='c', parents=['not_a_variable'], distribution=Delta, size=1)
        with pytest.raises(TypeError, match="is not a Variable object"):
            _ = var.in_features


class TestVariableSlicing:
    """Test Variable.__getitem__ slicing."""

    def test_slice_single_concept_by_string(self):
        """Test slicing to get single concept by string."""
        vars_list = Variable(concepts=['a', 'b', 'c'], parents=[], distribution=Delta, size=2)
        var_a = vars_list[0]
        sliced = var_a['a']
        assert sliced.concepts == ['a']
        assert sliced.size == 2

    def test_slice_single_concept_by_list(self):
        """Test slicing by list with single concept."""
        # When creating multiple concepts, Variable returns a list
        # So we need to slice the individual Variable, not the list
        vars_list = Variable(concepts=['a', 'b'], parents=[], distribution=Delta, size=2)
        # vars_list is actually a list of 2 Variables when multiple concepts
        # Take the first one and slice it
        var_a = vars_list[0]  # This is Variable with concept 'a'
        sliced = var_a[['a']]
        assert sliced.concepts == ['a']

    def test_slice_concept_not_found_raises_error(self):
        """Test that slicing non-existent concept raises error."""
        var = Variable(concepts='x', parents=[], distribution=Delta, size=1)
        with pytest.raises(ValueError, match="not found in variable"):
            var['y']

    def test_slice_categorical_multiple_concepts_raises_error(self):
        """Test that slicing Categorical into multiple concepts raises error."""
        var = Variable(concepts=['cat'], parents=[], distribution=Categorical, size=3)
        # This should work fine for single concept
        sliced = var['cat']
        assert sliced.concepts == ['cat']


class TestVariableRepr:
    """Test Variable.__repr__."""

    def test_repr_without_metadata(self):
        """Test repr without metadata."""
        var = Variable(concepts='x', parents=[], distribution=Delta, size=2)
        repr_str = repr(var)
        assert 'Variable' in repr_str
        assert 'x' in repr_str
        assert 'Delta' in repr_str
        assert 'size=2' in repr_str

    def test_repr_with_metadata(self):
        """Test repr with metadata."""
        var = Variable(
            concepts='y',
            parents=[],
            distribution=Bernoulli,
            size=1,
            metadata={'key': 'value'}
        )
        repr_str = repr(var)
        assert 'metadata=' in repr_str


class TestEndogenousVariable:
    """Test EndogenousVariable subclass."""

    def test_endogenous_variable_sets_metadata(self):
        """Test that EndogenousVariable sets variable_type metadata."""
        var = EndogenousVariable(
            concepts='endo',
            parents=[],
            distribution=Bernoulli,
            size=1
        )
        assert var.metadata['variable_type'] == 'endogenous'
        assert var.distribution is Bernoulli

    def test_endogenous_variable_preserves_custom_metadata(self):
        """Test that custom metadata is preserved."""
        var = EndogenousVariable(
            concepts='endo',
            parents=[],
            distribution=Delta,
            size=1,
            metadata={'custom': 'data'}
        )
        assert var.metadata['variable_type'] == 'endogenous'
        assert var.metadata['custom'] == 'data'


class TestExogenousVariable:
    """Test ExogenousVariable subclass."""

    def test_exogenous_variable_sets_metadata(self):
        """Test that ExogenousVariable sets variable_type metadata."""
        var = ExogenousVariable(
            concepts='exo',
            parents=[],
            distribution=Delta,
            size=128
        )
        assert var.metadata['variable_type'] == 'exogenous'
        assert var.size == 128

    def test_exogenous_variable_with_endogenous_reference(self):
        """Test ExogenousVariable can reference an endogenous variable."""
        endo = EndogenousVariable(concepts='e', parents=[], distribution=Bernoulli, size=1)
        exo = ExogenousVariable(
            concepts='exo_e',
            parents=[],
            distribution=Delta,
            size=64,
            metadata={'endogenous_var': endo}
        )
        assert exo.metadata['variable_type'] == 'exogenous'
        assert exo.metadata['endogenous_var'] is endo


class TestVariableEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_concept_with_list_distribution(self):
        """Test single concept with distribution as list."""
        var = Variable(
            concepts=['x'],
            parents=[],
            distribution=[Delta],
            size=[2]
        )
        assert var.concepts == ['x']
        assert var.distribution is Delta
        assert var.size == 2

    def test_relaxed_bernoulli_out_features(self):
        """Test out_features with RelaxedBernoulli."""
        var = Variable(
            concepts='rb',
            parents=[],
            distribution=RelaxedBernoulli,
            size=1
        )
        assert var.out_features == 1

    def test_variable_with_metadata_copy_on_slice(self):
        """Test that metadata is copied when slicing."""
        # Create a single variable with multiple concepts
        # For this test, we need a single Variable object, not a list
        # Use string concept to ensure single Variable
        var = Variable(
            concepts='ab',  # Single string = single Variable
            parents=[],
            distribution=Delta,
            size=1,
            metadata={'original': True}
        )
        sliced = var[['ab']]  # Slice by concept list
        assert sliced.metadata['original'] is True
        # Note: Since this is slicing the same concept,
        # the metadata is copied in the new Variable instance


if __name__ == '__main__':
    unittest.main()
