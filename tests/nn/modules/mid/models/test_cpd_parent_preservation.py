"""Tests for ParametricCPD parent preservation in BayesianNetwork.

Verifies that parent references on CPDs remain intact after model construction,
including when a LazyConstructor is used for parametrization (the lazy module
is built at CPD construction time, not model construction time).
"""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable, EmbeddingVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.low.priors import LearnablePrior
from torch_concepts.nn.modules.low.lazy import LazyConstructor
from torch_concepts.nn.modules.low.predictors.linear import LinearConceptToConcept
from torch_concepts.distributions import Delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chain_model():
    """x (delta, root) -> a (bernoulli) -> b (bernoulli) -> c (bernoulli)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    a = ConceptVariable("a", distribution=dist.Bernoulli, size=1)
    b = ConceptVariable("b", distribution=dist.Bernoulli, size=1)
    c = ConceptVariable("c", distribution=dist.Bernoulli, size=1)

    cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
    cpd_a = ParametricCPD(variable=a, parametrization=nn.Linear(4, 1), parents=[x])
    cpd_b = ParametricCPD(variable=b, parametrization=nn.Linear(1, 1), parents=[a])
    cpd_c = ParametricCPD(variable=c, parametrization=nn.Linear(1, 1), parents=[b])
    m = BayesianNetwork(variables=[x, a, b, c], factors=[cpd_x, cpd_a, cpd_b, cpd_c])
    return m, (x, a, b, c)


# ===========================================================================
# 1. Parent references intact after BayesianNetwork construction
# ===========================================================================

class TestCPDParentsAfterModelConstruction:
    def test_root_has_no_parents(self):
        m, (x, a, b, c) = _build_chain_model()
        assert m.factors["x"].parents == []

    def test_a_parent_is_x(self):
        m, (x, a, b, c) = _build_chain_model()
        assert len(m.factors["a"].parents) == 1
        assert m.factors["a"].parents[0] is x

    def test_b_parent_is_a(self):
        m, (x, a, b, c) = _build_chain_model()
        assert len(m.factors["b"].parents) == 1
        assert m.factors["b"].parents[0] is a

    def test_c_parent_is_b(self):
        m, (x, a, b, c) = _build_chain_model()
        assert len(m.factors["c"].parents) == 1
        assert m.factors["c"].parents[0] is b

    def test_multiple_parents_preserved(self):
        p1 = ConceptVariable("p1", distribution=dist.Bernoulli, size=2)
        p2 = ConceptVariable("p2", distribution=dist.Bernoulli, size=3)
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        cpd_p1 = ParametricCPD(variable=p1, parametrization={"logits": LearnablePrior(2)})
        cpd_p2 = ParametricCPD(variable=p2, parametrization={"logits": LearnablePrior(3)})
        cpd_c = ParametricCPD(variable=c, parametrization=nn.Linear(5, 1), parents=[p1, p2])
        m = BayesianNetwork(variables=[p1, p2, c], factors=[cpd_p1, cpd_p2, cpd_c])
        parent_names = [p.name for p in m.factors["c"].parents]
        assert set(parent_names) == {"p1", "p2"}


# ===========================================================================
# 2. Variable references intact
# ===========================================================================

class TestCPDVariableAfterModelConstruction:
    def test_variable_references_correct_object(self):
        m, (x, a, b, c) = _build_chain_model()
        assert m.factors["a"].variable is a
        assert m.factors["b"].variable is b

    def test_is_root_correct(self):
        m, (x, a, b, c) = _build_chain_model()
        assert m.factors["x"].is_root is True
        assert m.factors["a"].is_root is False


# ===========================================================================
# 3. LazyConstructor: resolved at CPD construction, parents preserved
# ===========================================================================

class TestLazyCPDParentsPreserved:
    def test_lazy_cpd_parents_preserved(self):
        x = ConceptVariable("x", distribution=Delta, size=8)
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=2)
        lazy = LazyConstructor(LinearConceptToConcept)
        cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(8)})
        cpd_c = ParametricCPD(variable=c, parametrization=lazy, parents=[x])
        m = BayesianNetwork(variables=[x, c], factors=[cpd_x, cpd_c])
        # parents must be preserved
        assert len(m.factors["c"].parents) == 1
        assert m.factors["c"].parents[0] is x

    def test_lazy_cpd_variable_preserved(self):
        x = ConceptVariable("x", distribution=Delta, size=8)
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=2)
        lazy = LazyConstructor(LinearConceptToConcept)
        cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(8)})
        cpd_c = ParametricCPD(variable=c, parametrization=lazy, parents=[x])
        m = BayesianNetwork(variables=[x, c], factors=[cpd_x, cpd_c])
        assert m.factors["c"].variable is c

    def test_lazy_cpd_forward_after_construction(self):
        x = ConceptVariable("x", distribution=Delta, size=8)
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=2)
        lazy = LazyConstructor(LinearConceptToConcept)
        cpd_c = ParametricCPD(variable=c, parametrization=lazy, parents=[x])
        B = 4
        out = cpd_c(parent_values={"x": torch.randn(B, 8)})
        assert "probs" in out
        assert out["probs"].shape == (B, 2)


# ===========================================================================
# 4. Forward pass consistency
# ===========================================================================

class TestForwardPassParentConsistency:
    def test_chain_forward_pass(self):
        m, _ = _build_chain_model()
        B = 3
        # Manually run the forward pass following the chain
        params_x = m.factors["x"].root_params(B)
        val_x = params_x["value"]
        assert val_x.shape == (B, 4)

        params_a = m.factors["a"](parent_values={"x": val_x})
        val_a = params_a["probs"]
        assert val_a.shape == (B, 1)

        params_b = m.factors["b"](parent_values={"a": val_a})
        val_b = params_b["probs"]
        assert val_b.shape == (B, 1)

        params_c = m.factors["c"](parent_values={"b": val_b})
        assert params_c["probs"].shape == (B, 1)

    def test_multi_parent_feature_sizes(self):
        p1 = ConceptVariable("p1", distribution=dist.Bernoulli, size=2)
        p2 = ConceptVariable("p2", distribution=dist.Bernoulli, size=3)
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        cpd_p1 = ParametricCPD(variable=p1, parametrization={"logits": LearnablePrior(2)})
        cpd_p2 = ParametricCPD(variable=p2, parametrization={"logits": LearnablePrior(3)})
        cpd_c = ParametricCPD(variable=c, parametrization=nn.Linear(5, 1), parents=[p1, p2])
        m = BayesianNetwork(variables=[p1, p2, c], factors=[cpd_p1, cpd_p2, cpd_c])
        B = 4
        v_p1 = torch.rand(B, 2)
        v_p2 = torch.rand(B, 3)
        out = m.factors["c"](parent_values={"p1": v_p1, "p2": v_p2})
        assert out["probs"].shape == (B, 1)
