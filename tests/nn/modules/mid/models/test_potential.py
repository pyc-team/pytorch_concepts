"""ParametricPotential (undirected, energy-based factor) and the unified factor
interface (scope, name, log_potential) on both CPDs and potentials."""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.potential import (
    ParametricPotential,
    enumerable_cardinality,
)
from torch_concepts.nn.modules.mid.inference.utils import build_distribution
from torch_concepts.nn.modules.low.priors import LearnablePrior


def _bin(name):
    return ConceptVariable(name, distribution=dist.Bernoulli, size=1)


def _cat(name, k):
    return ConceptVariable(name, distribution=dist.OneHotCategorical, size=k)


def _mlp(in_dim, hidden=16):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))


class TestEnumerableCardinality:
    def test_binary(self):
        assert enumerable_cardinality(_bin("a")) == 2

    def test_categorical(self):
        assert enumerable_cardinality(_cat("b", 4)) == 4

    def test_multibit_bernoulli_raises(self):
        v = ConceptVariable("v", distribution=dist.Bernoulli, size=3)
        with pytest.raises(ValueError, match="independent bits"):
            enumerable_cardinality(v)

    def test_continuous_raises(self):
        v = ConceptVariable("v", distribution=dist.Normal, size=2)
        with pytest.raises(ValueError, match="not discretely enumerable"):
            enumerable_cardinality(v)


class TestParametricPotential:
    def test_energy_shape(self):
        a, b = _bin("a"), _bin("b")
        pot = ParametricPotential(scope=[a, b], parametrization=nn.Linear(2, 1), name="phi")
        e = pot.energy({a: torch.tensor([[1.0], [0.0]]), b: torch.tensor([[0.0], [1.0]])})
        assert e.shape == (2,)

    def test_energy_categorical_scope(self):
        a, b = _bin("a"), _cat("b", 3)
        pot = ParametricPotential(scope=[a, b], parametrization=nn.Linear(1 + 3, 1), name="phi")
        av = torch.tensor([[1.0], [0.0]])
        bv = torch.nn.functional.one_hot(torch.tensor([2, 0]), 3).float()
        assert pot.energy({a: av, b: bv}).shape == (2,)

    def test_log_potential_equals_neg_energy(self):
        a, b = _bin("a"), _bin("b")
        pot = ParametricPotential(scope=[a, b], parametrization=nn.Linear(2, 1), name="phi")
        av = torch.tensor([[1.0], [0.0]])
        bv = torch.tensor([[0.0], [1.0]])
        lp = pot.log_potential({a: av, b: bv})
        e = pot.energy({a: av, b: bv})
        assert torch.allclose(lp, -e)

    def test_conditional_energy_uses_embedding(self):
        a = _bin("a")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=4)
        pot = ParametricPotential(scope=[a], parametrization=nn.Linear(1 + 4, 1),
                                  conditioning=[emb], name="phi")
        e = pot.energy({a: torch.tensor([[1.0], [0.0], [1.0]])},
                       conditioning={"emb": torch.randn(3, 4)})
        assert e.shape == (3,)

    def test_scope_name_conditioning(self):
        a, b = _bin("a"), _bin("b")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=2)
        pot = ParametricPotential(scope=[a, b], parametrization=nn.Linear(4, 1),
                                  conditioning=[emb], name="phi")
        assert [v.name for v in pot.scope] == ["a", "b"]
        assert [v.name for v in pot.conditioning] == ["emb"]
        assert pot.name == "phi"

    def test_default_name(self):
        a, b = _bin("a"), _bin("b")
        pot = ParametricPotential(scope=[a, b], parametrization=nn.Linear(2, 1))
        assert pot.name == "phi(a,b)"

    def test_bad_param_key_raises(self):
        a = _bin("a")
        with pytest.raises(ValueError, match="'energy'"):
            ParametricPotential(scope=[a], parametrization={"table": nn.Linear(1, 1)}, name="phi")

    def test_linear_energy_is_additive(self):
        # A linear energy over the concatenated inputs cannot represent a coupling:
        # E(1,1) - E(1,0) - E(0,1) + E(0,0) == 0.
        a, b = _bin("a"), _bin("b")
        pot = ParametricPotential(scope=[a, b], parametrization=nn.Linear(2, 1), name="phi")

        def E(ia, ib):
            return pot.energy({a: torch.tensor([[float(ia)]]), b: torch.tensor([[float(ib)]])})[0]

        interaction = E(1, 1) - E(1, 0) - E(0, 1) + E(0, 0)
        assert torch.allclose(interaction, torch.zeros(()), atol=1e-6)

    def test_mlp_energy_has_interaction(self):
        torch.manual_seed(0)
        a, b = _bin("a"), _bin("b")
        pot = ParametricPotential(scope=[a, b], parametrization=_mlp(2), name="phi")

        def E(ia, ib):
            return pot.energy({a: torch.tensor([[float(ia)]]), b: torch.tensor([[float(ib)]])})[0]

        interaction = E(1, 1) - E(1, 0) - E(0, 1) + E(0, 0)
        assert interaction.abs() > 1e-4


class TestCPDFactorInterface:
    def test_scope_and_name(self):
        x = ConceptVariable("x", distribution=dist.Normal, size=2)
        c = _bin("c")
        cpd = ParametricCPD(variable=c, parametrization={"logits": nn.Linear(2, 1)}, parents=[x])
        assert cpd.name == "c"
        assert [v.name for v in cpd.scope] == ["c", "x"]

    def test_log_potential_equals_logprob(self):
        x = ConceptVariable("x", distribution=dist.Normal, size=2)
        c = _bin("c")
        cpd = ParametricCPD(variable=c, parametrization={"logits": nn.Linear(2, 1)}, parents=[x])
        xv = torch.randn(4, 2)
        cv = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        lp = cpd.log_potential({c: cv, x: xv})
        params = cpd(parent_values={"x": xv, "c": cv})
        expected = build_distribution(c, params).log_prob(cv)
        assert torch.allclose(lp, expected)

    def test_root_cpd_log_potential(self):
        a = _bin("a")
        cpd = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        av = torch.tensor([[1.0], [0.0]])
        lp = cpd.log_potential({a: av})
        expected = build_distribution(a, cpd.root_params(2)).log_prob(av)
        assert torch.allclose(lp, expected)
