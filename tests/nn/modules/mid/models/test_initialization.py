"""Tests for Variable and ParametricCPD initialization contracts.

Covers the str-vs-list dispatch, plate (members=) path, parametrization
shorthand expansion, and the LazyConstructor integration in CPD.
"""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import (
    ConceptVariable,
    EmbeddingVariable,
)
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.low.priors import LearnablePrior
from torch_concepts.nn.modules.low.lazy import LazyConstructor
from torch_concepts.nn.modules.low.predictors.linear import LinearConceptToConcept
from torch_concepts.distributions import Delta


# ===========================================================================
# 1. Variable initialization contract
# ===========================================================================

class TestVariableInitializationContract:
    def test_str_returns_single_instance(self):
        v = ConceptVariable("x", distribution=Delta, size=1)
        assert isinstance(v, ConceptVariable)
        assert not isinstance(v, list)

    def test_list_single_element_returns_list(self):
        result = ConceptVariable(["x"], distribution=Delta, size=1)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_list_multiple_elements_returns_list(self):
        result = ConceptVariable(["a", "b", "c"], distribution=dist.Bernoulli, size=1)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_list_single_distribution_broadcast(self):
        result = ConceptVariable(["a", "b"], distribution=dist.Bernoulli, size=1)
        assert all(v.distribution is dist.Bernoulli for v in result)

    def test_list_single_size_broadcast(self):
        result = ConceptVariable(["a", "b", "c"], distribution=Delta, size=5)
        assert all(v.size == 5 for v in result)

    def test_list_per_variable_distribution(self):
        result = ConceptVariable(
            ["a", "b", "c"],
            distribution=[dist.Bernoulli, dist.OneHotCategorical, Delta],
            size=[1, 3, 2],
        )
        assert result[0].distribution is dist.Bernoulli
        assert result[1].distribution is dist.OneHotCategorical
        assert result[2].distribution is Delta

    def test_list_distribution_mismatch_raises(self):
        with pytest.raises(ValueError):
            ConceptVariable(["a", "b", "c"], distribution=[dist.Bernoulli, Delta], size=1)

    def test_list_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            ConceptVariable(["a", "b"], distribution=Delta, size=[1, 2, 3])

    def test_plate_path_members(self):
        v = ConceptVariable("g", members=["c1", "c2"], distribution=dist.Bernoulli)
        assert v.name == "g"
        assert v.members == ["c1", "c2"]
        assert v.is_plate

    def test_plate_and_list_exclusive(self):
        with pytest.raises(TypeError):
            ConceptVariable(["a", "b"], distribution=dist.Bernoulli, members=["x"])

    def test_embedding_str_path(self):
        e = EmbeddingVariable("e", distribution=Delta, size=16)
        assert isinstance(e, EmbeddingVariable)
        assert e.name == "e"

    def test_embedding_list_path(self):
        es = EmbeddingVariable(["e1", "e2"], distribution=Delta, size=8)
        assert isinstance(es, list)
        assert all(isinstance(e, EmbeddingVariable) for e in es)


# ===========================================================================
# 2. ParametricCPD initialization contract
# ===========================================================================

class TestParametricCPDInitializationContract:
    def test_single_variable_returns_single_cpd(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        cpd = ParametricCPD(variable=v, parametrization={"logits": LearnablePrior(1)})
        assert isinstance(cpd, ParametricCPD)
        assert not isinstance(cpd, list)

    def test_list_of_variables_returns_list(self):
        vs = ConceptVariable(["c1", "c2", "c3"], distribution=dist.Bernoulli, size=1)
        result = ParametricCPD(variable=vs, parametrization=nn.Linear(4, 1))
        assert isinstance(result, list)
        assert len(result) == 3

    def test_module_shorthand_expands_to_probs(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        cpd = ParametricCPD(variable=v, parametrization=nn.Sigmoid())
        assert "probs" in cpd.parametrization

    def test_module_shorthand_expands_to_value_for_delta(self):
        v = ConceptVariable("x", distribution=Delta, size=4)
        cpd = ParametricCPD(variable=v, parametrization=nn.Identity())
        assert "value" in cpd.parametrization

    def test_dict_parametrization_accepted(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        cpd = ParametricCPD(variable=v, parametrization={"logits": nn.Linear(4, 1)})
        assert "logits" in cpd.parametrization

    def test_single_module_for_multi_param_dist_raises(self):
        v = ConceptVariable("n", distribution=dist.Normal, size=2)
        with pytest.raises(ValueError, match="multiple parameters"):
            ParametricCPD(variable=v, parametrization=nn.Linear(4, 2))

    def test_list_cpds_modules_deep_copied(self):
        vs = ConceptVariable(["c1", "c2"], distribution=dist.Bernoulli, size=1)
        m = nn.Linear(4, 1)
        result = ParametricCPD(variable=vs, parametrization=m)
        ids = set()
        for c in result:
            ids.add(id(list(c.parametrization.values())[0]))
        assert len(ids) == 2  # each has its own copy

    def test_parents_stored(self):
        x = ConceptVariable("x", distribution=Delta, size=4)
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        cpd = ParametricCPD(variable=c, parametrization=nn.Linear(4, 1), parents=[x])
        assert len(cpd.parents) == 1
        assert cpd.parents[0] is x


# ===========================================================================
# 3. LazyConstructor integration
# ===========================================================================

class TestLazyConstructorCPD:
    def test_lazy_built_at_cpd_construction(self):
        x = ConceptVariable("x", distribution=Delta, size=4)
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=2)
        lazy = LazyConstructor(LinearConceptToConcept)
        cpd = ParametricCPD(variable=c, parametrization=lazy, parents=[x])
        # After construction the lazy module must be resolved to a concrete layer.
        module = list(cpd.parametrization.values())[0]
        assert not (isinstance(module, LazyConstructor) and module.module is None), \
            "LazyConstructor should be resolved after CPD construction"

    def test_lazy_cpd_forward_works(self):
        x = ConceptVariable("x", distribution=Delta, size=8)
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=3)
        lazy = LazyConstructor(LinearConceptToConcept)
        cpd = ParametricCPD(variable=c, parametrization=lazy, parents=[x])
        B = 5
        out = cpd(parent_values={"x": torch.randn(B, 8)})
        assert "probs" in out
        assert out["probs"].shape == (B, 3)


# ===========================================================================
# 4. BayesianNetwork initialization (integration)
# ===========================================================================

class TestBayesianNetworkInitContract:
    def test_one_factor_per_variable_required(self):
        x = ConceptVariable("x", distribution=Delta, size=4)
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
        cpd_c = ParametricCPD(variable=c, parametrization=nn.Linear(4, 1), parents=[x])
        m = BayesianNetwork(variables=[x, c], factors=[cpd_x, cpd_c])
        assert set(m.variables.keys()) == {"x", "c"}
        assert set(m.factors.keys()) == {"x", "c"}

    def test_plate_variable_accepted(self):
        plate = ConceptVariable("g", members=["m1", "m2"], distribution=dist.Bernoulli)
        cpd_g = ParametricCPD(variable=plate, parametrization={"probs": LearnablePrior(plate.size)})
        m = BayesianNetwork(variables=[plate], factors=[cpd_g])
        assert "g" in m.variables
        assert "g" in m.factors
