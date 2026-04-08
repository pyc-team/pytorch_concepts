"""
Tests for Variable and ParametricCPD initialization behavior.

This module tests the str vs list input handling for concepts parameter,
ensuring the correct return types (single instance vs list) and proper
validation of other parameters.
"""
import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from torch_concepts.nn.modules.mid.models.variable import (
    Variable,
    ConceptVariable,
    ExogenousVariable,
    LatentVariable,
)
from torch_concepts.nn.modules.mid.models.parametric_cpd import ParametricCPD
from torch_concepts.distributions import Delta


class TestVariableInitializationContract:
    """Test Variable initialization contract: str -> single, list -> list."""

    # --- String concepts: returns single Variable ---

    def test_str_concept_returns_single_variable(self):
        """String concept returns a single Variable instance, not a list."""
        var = Variable(concepts='x', distribution=Delta, size=1)
        assert isinstance(var, Variable)
        assert not isinstance(var, list)
        assert var.concept == 'x'

    def test_str_concept_with_list_distribution_raises(self):
        """String concept with list distribution raises ValueError."""
        with pytest.raises(ValueError, match="must be a single value, not a list"):
            Variable(concepts='x', distribution=[Delta], size=1)

    def test_str_concept_with_list_size_raises(self):
        """String concept with list size raises ValueError."""
        with pytest.raises(ValueError, match="must be a single value, not a list"):
            Variable(concepts='x', distribution=Delta, size=[1])

    def test_str_concept_with_both_list_params_raises(self):
        """String concept with both list distribution and size raises ValueError."""
        with pytest.raises(ValueError, match="must be a single value, not a list"):
            Variable(concepts='x', distribution=[Delta], size=[1])

    # --- List concepts: always returns list ---

    def test_list_concept_single_element_returns_list(self):
        """List with single concept returns list with one Variable."""
        result = Variable(concepts=['x'], distribution=Delta, size=1)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Variable)
        assert result[0].concept == 'x'

    def test_list_concept_multiple_elements_returns_list(self):
        """List with multiple concepts returns list of Variables."""
        result = Variable(concepts=['a', 'b', 'c'], distribution=Delta, size=1)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0].concept == 'a'
        assert result[1].concept == 'b'
        assert result[2].concept == 'c'

    def test_list_concept_with_single_distribution_broadcasts(self):
        """Single distribution is broadcast to all concepts in list."""
        result = Variable(concepts=['a', 'b'], distribution=Bernoulli, size=1)
        assert all(v.distribution is Bernoulli for v in result)

    def test_list_concept_with_single_size_broadcasts(self):
        """Single size is broadcast to all concepts in list."""
        result = Variable(concepts=['a', 'b', 'c'], distribution=Delta, size=5)
        assert all(v.size == 5 for v in result)

    def test_list_concept_with_distribution_list(self):
        """Distribution list assigns per-concept distributions."""
        result = Variable(
            concepts=['a', 'b', 'c'],
            distribution=[Bernoulli, Categorical, Delta],
            size=[1, 3, 1]
        )
        assert result[0].distribution is Bernoulli
        assert result[1].distribution is Categorical
        assert result[2].distribution is Delta

    def test_list_concept_with_size_list(self):
        """Size list assigns per-concept sizes."""
        result = Variable(
            concepts=['a', 'b', 'c'],
            distribution=Categorical,
            size=[2, 3, 4]
        )
        assert result[0].size == 2
        assert result[1].size == 3
        assert result[2].size == 4

    def test_list_concept_mismatched_distribution_length_raises(self):
        """Mismatched distribution list length raises ValueError."""
        with pytest.raises(ValueError, match="must either be single values or lists of length"):
            Variable(
                concepts=['a', 'b', 'c'],
                distribution=[Bernoulli, Delta],  # 2, but need 3
                size=1
            )

    def test_list_concept_mismatched_size_length_raises(self):
        """Mismatched size list length raises ValueError."""
        with pytest.raises(ValueError, match="must either be single values or lists of length"):
            Variable(
                concepts=['a', 'b', 'c'],
                distribution=Delta,
                size=[1, 2]  # 2, but need 3
            )


class TestVariableSubclassesInitialization:
    """Test that Variable subclasses inherit the same initialization behavior."""

    @pytest.mark.parametrize("cls", [ConceptVariable, ExogenousVariable, LatentVariable])
    def test_subclass_str_concept_returns_single(self, cls):
        """Subclass with str concept returns single instance."""
        var = cls(concepts='x', distribution=Delta, size=1)
        assert isinstance(var, cls)
        assert not isinstance(var, list)

    @pytest.mark.parametrize("cls", [ConceptVariable, ExogenousVariable, LatentVariable])
    def test_subclass_list_concept_returns_list(self, cls):
        """Subclass with list concept returns list of instances."""
        result = cls(concepts=['a', 'b'], distribution=Delta, size=1)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(v, cls) for v in result)

    @pytest.mark.parametrize("cls", [ConceptVariable, ExogenousVariable, LatentVariable])
    def test_subclass_str_concept_with_list_param_raises(self, cls):
        """Subclass with str concept and list param raises ValueError."""
        with pytest.raises(ValueError, match="must be a single value, not a list"):
            cls(concepts='x', distribution=[Delta], size=1)


class TestParametricCPDInitializationContract:
    """Test ParametricCPD initialization contract: str -> single, list -> list."""

    # --- String concepts: returns single ParametricCPD ---

    def test_str_concept_returns_single_cpd(self):
        """String concept returns a single ParametricCPD instance, not a list."""
        module = nn.Linear(5, 1)
        cpd = ParametricCPD(concepts='x', parametrization=module)
        assert isinstance(cpd, ParametricCPD)
        assert not isinstance(cpd, list)
        assert cpd.concept == 'x'

    def test_str_concept_with_list_parametrization_raises(self):
        """String concept with list parametrization raises ValueError."""
        modules = [nn.Linear(5, 1)]
        with pytest.raises(ValueError, match="must be a single module, not a list"):
            ParametricCPD(concepts='x', parametrization=modules)

    # --- List concepts: always returns list ---

    def test_list_concept_single_element_returns_list(self):
        """List with single concept returns list with one ParametricCPD."""
        module = nn.Linear(5, 1)
        result = ParametricCPD(concepts=['x'], parametrization=module)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ParametricCPD)
        assert result[0].concept == 'x'

    def test_list_concept_multiple_elements_returns_list(self):
        """List with multiple concepts returns list of ParametricCPDs."""
        module = nn.Linear(5, 1)
        result = ParametricCPD(concepts=['a', 'b', 'c'], parametrization=module)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0].concept == 'a'
        assert result[1].concept == 'b'
        assert result[2].concept == 'c'

    def test_list_concept_with_single_module_broadcasts(self):
        """Single module is deep-copied to all concepts in list."""
        module = nn.Linear(5, 1)
        result = ParametricCPD(concepts=['a', 'b'], parametrization=module)
        # Each should have its own copy (not same object)
        assert result[0].parametrization is not result[1].parametrization
        # But same structure
        assert result[0].parametrization.in_features == 5
        assert result[1].parametrization.in_features == 5

    def test_list_concept_with_module_list(self):
        """Module list assigns per-concept modules."""
        mod1 = nn.Linear(5, 1)
        mod2 = nn.Linear(10, 2)
        result = ParametricCPD(concepts=['a', 'b'], parametrization=[mod1, mod2])
        assert result[0].parametrization.in_features == 5
        assert result[1].parametrization.in_features == 10

    def test_list_concept_mismatched_module_length_raises(self):
        """Mismatched module list length raises ValueError."""
        modules = [nn.Linear(5, 1), nn.Linear(5, 1)]  # 2, but need 3
        with pytest.raises(ValueError, match="must either be a single module or a list of length"):
            ParametricCPD(concepts=['a', 'b', 'c'], parametrization=modules)


class TestInitializationWithMetadata:
    """Test initialization behavior with metadata."""

    def test_str_concept_preserves_metadata(self):
        """String concept Variable preserves metadata."""
        meta = {'key': 'value', 'num': 42}
        var = Variable(concepts='x', distribution=Delta, size=1, metadata=meta)
        assert var.metadata['key'] == 'value'
        assert var.metadata['num'] == 42

    def test_list_concept_each_gets_copy_of_metadata(self):
        """Each Variable in list gets its own copy of metadata."""
        meta = {'shared': True}
        result = Variable(concepts=['a', 'b'], distribution=Delta, size=1, metadata=meta)
        # Each should have its own copy
        result[0].metadata['new_key'] = 'only_a'
        assert 'new_key' not in result[1].metadata

    def test_list_concept_none_metadata_creates_empty_dict(self):
        """None metadata creates empty dict for each Variable."""
        result = Variable(concepts=['a', 'b'], distribution=Delta, size=1, metadata=None)
        assert result[0].metadata == {}
        assert result[1].metadata == {}


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
