"""ParametricPotential / TabularPotential and the unified factor interface
(scope, name, log_potential) on both CPDs and potentials
(FACTOR_GRAPH_INSTRUCTIONS.md §5.2-5.4)."""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.potential import (
    ParametricPotential,
    TabularPotential,
    enumerable_cardinality,
)
from torch_concepts.nn.modules.mid.inference.utils import build_distribution
from torch_concepts.nn.modules.low.priors import LearnablePrior


def _bin(name):
    return ConceptVariable(name, distribution=dist.Bernoulli, size=1)


def _cat(name, k):
    return ConceptVariable(name, distribution=dist.OneHotCategorical, size=k)


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


class TestTabularPotential:
    def test_table_shape_unconditional(self):
        a, b = _bin("a"), _cat("b", 3)
        pot = TabularPotential(scope=[a, b], parametrization=LearnablePrior(2 * 3), name="phi")
        assert pot.cardinalities == [2, 3]
        table = pot.log_potential_table(batch_size=5)
        assert table.shape == (5, 2, 3)

    def test_table_shape_conditional(self):
        a, b = _bin("a"), _bin("b")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=4)
        pot = TabularPotential(scope=[a, b], parametrization=nn.Linear(4, 4), conditioning=[emb], name="phi")
        table = pot.log_potential_table(conditioning={"emb": torch.randn(6, 4)})
        assert table.shape == (6, 2, 2)

    def test_log_potential_equals_neg_energy(self):
        a, b = _bin("a"), _bin("b")
        pot = TabularPotential(scope=[a, b], parametrization=LearnablePrior(4), name="phi")
        av = torch.tensor([[1.0], [0.0]])
        bv = torch.tensor([[0.0], [1.0]])
        lp = pot.log_potential({a: av, b: bv})
        e = pot.energy({a: av, b: bv})
        assert torch.allclose(lp, -e)

    def test_log_potential_matches_table_gather(self):
        a, b = _bin("a"), _cat("b", 3)
        pot = TabularPotential(scope=[a, b], parametrization=LearnablePrior(6), name="phi")
        table = pot.log_potential_table(batch_size=4)  # (4, 2, 3)
        av = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        bv = torch.nn.functional.one_hot(torch.tensor([2, 0, 1, 2]), 3).float()
        lp = pot.log_potential({a: av, b: bv})
        manual = table[torch.arange(4), av.reshape(4).long(), bv.argmax(-1)]
        assert torch.allclose(lp, manual)

    def test_scope_and_name_and_conditioning(self):
        a, b = _bin("a"), _bin("b")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=2)
        pot = TabularPotential(scope=[a, b], parametrization=nn.Linear(2, 4), conditioning=[emb], name="phi")
        assert [v.name for v in pot.scope] == ["a", "b"]
        assert [v.name for v in pot.conditioning] == ["emb"]
        assert pot.name == "phi"

    def test_table_blowup_guard(self):
        big = [_cat(f"v{i}", 50) for i in range(4)]  # 50**4 = 6.25M > 1M
        with pytest.raises(ValueError, match="MAX_TABLE_SIZE"):
            TabularPotential(scope=big, parametrization=LearnablePrior(1), name="phi")

    def test_bad_param_key_raises(self):
        a = _bin("a")
        with pytest.raises(ValueError, match="'table'"):
            TabularPotential(scope=[a], parametrization={"probs": LearnablePrior(2)}, name="phi")


class TestParametricPotentialBaseIsAbstract:
    def test_cannot_instantiate_base_energy(self):
        # ParametricPotential.energy is abstract; a bare subclass without it fails.
        a = _bin("a")
        with pytest.raises(TypeError):
            ParametricPotential(scope=[a], parametrization={"e": LearnablePrior(1)})


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
