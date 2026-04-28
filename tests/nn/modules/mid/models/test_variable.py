"""
Comprehensive tests for torch_concepts.nn.modules.mid.models

Tests for Variable, ParametricCPD, and ProbabilisticModel.
"""
import unittest
import pytest
import torch
from torch.distributions import Bernoulli, Categorical, OneHotCategorical, RelaxedBernoulli

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
            distribution=Bernoulli,
            size=1
        )
        self.assertEqual(var.concept, 'color')
        self.assertEqual(var.distribution, Bernoulli)

    def test_multiple_concepts_initialization(self):
        """Test creating multiple concept variables."""
        vars_list = Variable(
            concepts=['A', 'B', 'C'],
            distribution=Bernoulli,
            size=1
        )
        self.assertEqual(len(vars_list), 3)
        self.assertEqual(vars_list[0].concept, 'A')
        self.assertEqual(vars_list[1].concept, 'B')
        self.assertEqual(vars_list[2].concept, 'C')

    def test_variable_with_delta_distribution(self):
        """Test variable with Delta distribution."""
        var = Variable(
            concepts='feature',
            distribution=Delta,
            size=1
        )
        self.assertEqual(var.distribution, Delta)

    def test_variable_with_categorical_distribution(self):
        """Test variable with OneHotCategorical distribution."""
        var = Variable(
            concepts='color',
            distribution=OneHotCategorical,
            size=3
        )
        self.assertEqual(var.distribution, OneHotCategorical)
        self.assertEqual(var.size, 3)

    def test_variable_out_features(self):
        """Test out_features property."""
        var_binary = Variable(concepts='binary', distribution=Bernoulli, size=1)
        self.assertEqual(var_binary.out_features, 1)

        var_cat = Variable(concepts='category', distribution=OneHotCategorical, size=5)
        self.assertEqual(var_cat.out_features, 5)

    def test_variable_with_metadata(self):
        """Test variable with metadata."""
        metadata = {'description': 'test variable', 'importance': 0.8}
        var = Variable(
            concepts='test',
            distribution=Bernoulli,
            size=1,
            metadata=metadata
        )
        self.assertEqual(var.metadata, metadata)

    def test_multiple_concepts_with_different_distributions(self):
        """Test multiple concepts with different distributions."""
        vars_list = Variable(
            concepts=['A', 'B', 'C'],
            distribution=[Bernoulli, OneHotCategorical, Delta],
            size=[1, 3, 1]
        )
        self.assertEqual(vars_list[0].distribution, Bernoulli)
        self.assertEqual(vars_list[1].distribution, OneHotCategorical)
        self.assertEqual(vars_list[2].distribution, Delta)

    def test_multiple_concepts_with_different_sizes(self):
        """Test multiple concepts with different sizes."""
        vars_list = Variable(
            concepts=['A', 'B', 'C'],
            distribution=OneHotCategorical,
            size=[2, 3, 4]
        )
        self.assertEqual(vars_list[0].size, 2)
        self.assertEqual(vars_list[1].size, 3)
        self.assertEqual(vars_list[2].size, 4)

    def test_variable_with_none_distribution(self):
        """Test variable with None distribution defaults to Delta."""
        vars_list = Variable(
            concepts=['A', 'B'],
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
                distribution=[Bernoulli, OneHotCategorical],  # Only 2, need 3
                size=1
            )


class TestVariableMultiConceptCreation:
    """Test Variable.__new__ multi-concept behavior."""

    def test_multi_concept_returns_list(self):
        """Test that multiple concepts return a list of Variables."""
        vars_list = Variable(
            concepts=['a', 'b', 'c'],
            distribution=Delta,
            size=1
        )
        assert isinstance(vars_list, list)
        assert len(vars_list) == 3
        assert vars_list[0].concept == 'a'
        assert vars_list[1].concept == 'b'
        assert vars_list[2].concept == 'c'

    def test_multi_concept_with_distribution_list(self):
        """Test multi-concept with per-concept distributions."""
        vars_list = Variable(
            concepts=['a', 'b', 'c'],
            distribution=[Bernoulli, Delta, OneHotCategorical],
            size=[1, 2, 3]
        )
        assert len(vars_list) == 3
        assert vars_list[0].distribution is Bernoulli
        assert vars_list[1].distribution is Delta
        assert vars_list[2].distribution is OneHotCategorical

    def test_multi_concept_distribution_length_mismatch_raises_error(self):
        """Test that mismatched distribution list length raises error."""
        with pytest.raises(ValueError, match="distribution and size must either be single values or lists of length"):
            Variable(
                concepts=['a', 'b', 'c'],
                distribution=[Bernoulli, Delta],  # Only 2, need 3
                size=1
            )

    def test_multi_concept_size_list_mismatch_raises_error(self):
        """Test that mismatched size list length raises error."""
        with pytest.raises(ValueError, match="distribution and size must either be single values or lists of length"):
            Variable(
                concepts=['a', 'b'],
                distribution=Delta,
                size=[1, 2, 3]  # 3 sizes for 2 concepts
            )


class TestVariableValidation:
    """Test Variable validation logic."""

    def test_categorical_raises_error(self):
        """Test that Categorical always raises error (use OneHotCategorical)."""
        with pytest.raises(ValueError, match="Use OneHotCategorical"):
            Variable(
                concepts='cat',
                distribution=Categorical,
                size=3
            )

    def test_bernoulli_with_size_not_one_raises_error(self):
        """Test that Bernoulli with size != 1 raises error."""
        with pytest.raises(ValueError, match="must have size=1"):
            Variable(
                concepts='bern',
                distribution=Bernoulli,
                size=3
            )


class TestVariableOutFeatures:
    """Test out_features property calculation."""

    def test_out_features_delta(self):
        """Test out_features for Delta distribution."""
        var = Variable(concepts='d', distribution=Delta, size=3)
        assert var.out_features == 3

    def test_out_features_bernoulli(self):
        """Test out_features for Bernoulli distribution."""
        var = Variable(concepts='b', distribution=Bernoulli, size=1)
        assert var.out_features == 1

    def test_out_features_categorical(self):
        """Test out_features for OneHotCategorical distribution."""
        var = Variable(concepts='c', distribution=OneHotCategorical, size=5)
        assert var.out_features == 5

    def test_out_features_equals_size(self):
        """Test that out_features is always equal to size."""
        var = Variable(concepts='x', distribution=Delta, size=2)
        assert var.out_features == var.size
        assert var.out_features == 2


class TestVariableRepr:
    """Test Variable.__repr__."""

    def test_repr_without_metadata(self):
        """Test repr without metadata."""
        var = Variable(concepts='x', distribution=Delta, size=2)
        repr_str = repr(var)
        assert 'Variable' in repr_str
        assert 'x' in repr_str
        assert 'Delta' in repr_str
        assert 'size=2' in repr_str

    def test_repr_with_metadata(self):
        """Test repr with metadata."""
        var = Variable(
            concepts='y',
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
            distribution=Bernoulli,
            size=1
        )
        assert var.metadata['variable_type'] == 'concept'
        assert var.distribution is Bernoulli

    def test_endogenous_variable_preserves_custom_metadata(self):
        """Test that custom metadata is preserved."""
        var = EndogenousVariable(
            concepts='endo',
            distribution=Delta,
            size=1,
            metadata={'custom': 'data'}
        )
        assert var.metadata['variable_type'] == 'concept'
        assert var.metadata['custom'] == 'data'


class TestExogenousVariable:
    """Test ExogenousVariable subclass."""

    def test_exogenous_variable_sets_metadata(self):
        """Test that ExogenousVariable sets variable_type metadata."""
        var = ExogenousVariable(
            concepts='exo',
            distribution=Delta,
            size=128
        )
        assert var.metadata['variable_type'] == 'exogenous'
        assert var.size == 128

    def test_exogenous_variable_with_endogenous_reference(self):
        """Test ExogenousVariable can reference an endogenous variable."""
        endo = EndogenousVariable(concepts='e', distribution=Bernoulli, size=1)
        exo = ExogenousVariable(
            concepts='exo_e',
            distribution=Delta,
            size=64,
            metadata={'endogenous_var': endo}
        )
        assert exo.metadata['variable_type'] == 'exogenous'
        assert exo.metadata['endogenous_var'] is endo


class TestVariableEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_concept_with_list_distribution_raises_error(self):
        """Test that single concept (str) with distribution as list raises error."""
        with pytest.raises(ValueError, match="must be a single value, not a list"):
            Variable(
                concepts='x',
                distribution=[Delta],
                size=1
            )

    def test_single_concept_with_list_size_raises_error(self):
        """Test that single concept (str) with size as list raises error."""
        with pytest.raises(ValueError, match="must be a single value, not a list"):
            Variable(
                concepts='x',
                distribution=Delta,
                size=[2]
            )

    def test_single_concept_in_list_returns_list(self):
        """Test that single concept in list returns list with one Variable."""
        vars_list = Variable(
            concepts=['x'],
            distribution=Delta,
            size=2
        )
        assert isinstance(vars_list, list)
        assert len(vars_list) == 1
        assert vars_list[0].concept == 'x'
        assert vars_list[0].distribution is Delta
        assert vars_list[0].size == 2

    def test_relaxed_bernoulli_out_features(self):
        """Test out_features with RelaxedBernoulli."""
        var = Variable(
            concepts='rb',
            distribution=RelaxedBernoulli,
            size=1
        )
        assert var.out_features == 1


class TestVariableDeepCopy:
    """Test that multi-variable creation uses deep copies of metadata/dist_kwargs."""

    def test_metadata_deep_copy(self):
        """Mutating one variable's metadata must not affect siblings."""
        meta = {'config': {'temp': 0.5}}
        vars_ = Variable(['A', 'B'], distribution=Bernoulli, size=1, metadata=meta)
        vars_[0].metadata['config']['temp'] = 999
        assert vars_[1].metadata['config']['temp'] == 0.5

    def test_dist_kwargs_deep_copy(self):
        """Mutating one variable's dist_kwargs must not affect siblings."""
        dk = {'extra': [1, 2, 3]}
        vars_ = Variable(['A', 'B'], distribution=Bernoulli, size=1, dist_kwargs=dk)
        vars_[0].dist_kwargs['extra'].append(4)
        assert vars_[1].dist_kwargs['extra'] == [1, 2, 3]


if __name__ == '__main__':
    # Use pytest to run all tests (including non-unittest classes)
    import sys
    sys.exit(pytest.main([__file__, '-v', '-s']))
