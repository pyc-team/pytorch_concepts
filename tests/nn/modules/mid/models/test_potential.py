"""ParametricPotential (undirected, energy-based factor) and the unified factor
interface (scope, name, log_potential) on both CPDs and potentials."""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.variable import ConceptVariable, EmbeddingVariable
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.factors.potential import ParametricPotential
from torch_concepts.nn.modules.mid.inference.utils import (
    build_distribution,
    enumerable_cardinality,
)
from torch_concepts.nn.modules.low.priors import LearnablePrior
from torch_concepts.nn.modules.low.lazy import LazyConstructor
from torch_concepts.nn.modules.low.predictors.linear import LinearConceptToConcept
from torch_concepts.nn.modules.low.encoders.linear import LinearEmbeddingToConcept


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

    def test_energy_uses_embedding_in_scope(self):
        """An embedding is an ordinary scope variable — no separate conditioning."""
        a = _bin("a")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=4)
        pot = ParametricPotential(scope=[a, emb], parametrization=nn.Linear(1 + 4, 1),
                                  name="phi")
        e = pot.energy({a: torch.tensor([[1.0], [0.0], [1.0]]),
                        emb: torch.randn(3, 4)})
        assert e.shape == (3,)

    def test_log_potential_requires_complete_assignment(self):
        """Every scope variable must be in the assignment — the observed ones
        (an embedding) at their evidence value, not passed separately."""
        a = _bin("a")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=4)
        pot = ParametricPotential(scope=[a, emb], parametrization=nn.Linear(1 + 4, 1),
                                  name="phi")
        av = torch.tensor([[1.0], [0.0], [1.0]])
        lp = pot.log_potential({a: av, emb: torch.randn(3, 4)})
        assert lp.shape == (3,)
        with pytest.raises(KeyError):
            pot.log_potential({a: av})

    def test_scope_and_name(self):
        a, b = _bin("a"), _bin("b")
        emb = ConceptVariable("emb", distribution=dist.Normal, size=2)
        pot = ParametricPotential(scope=[a, b, emb], parametrization=nn.Linear(4, 1),
                                  name="phi")
        assert [v.name for v in pot.scope] == ["a", "b", "emb"]
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


class TestLazyConstructorPotential:
    """LazyConstructor support in ParametricPotential (mirrors
    TestLazyConstructorCPD in test_initialization.py)."""

    def test_lazy_built_at_potential_construction(self):
        a, b = _bin("a"), _bin("b")
        lazy = LazyConstructor(LinearConceptToConcept)
        pot = ParametricPotential(scope=[a, b], parametrization=lazy, name="phi")
        module = pot.parametrization["energy"]
        assert not (isinstance(module, LazyConstructor) and module.module is None), \
            "LazyConstructor should be resolved after Potential construction"
        assert isinstance(module, LinearConceptToConcept)

    def test_lazy_sizes_from_scope(self):
        """in_concepts is the summed size of the concept-typed scope variables;
        out_concepts is always 1, regardless of the variables' own sizes."""
        a = ConceptVariable("a", distribution=dist.Bernoulli, size=3)
        b = ConceptVariable("b", distribution=dist.Bernoulli, size=5)
        lazy = LazyConstructor(LinearConceptToConcept)
        pot = ParametricPotential(scope=[a, b], parametrization=lazy, name="phi")
        module = pot.parametrization["energy"]
        assert module.predictor.in_features == 8  # 3 + 5
        assert module.predictor.out_features == 1

    def test_lazy_sizes_from_embedding_scope(self):
        """in_embeddings is the summed size of the embedding-typed scope
        variables, wired through LazyConstructor.build's PyTorch-standard
        `in_features` alias."""
        e1 = EmbeddingVariable("e1", distribution=dist.Normal, size=4)
        e2 = EmbeddingVariable("e2", distribution=dist.Normal, size=6)
        lazy = LazyConstructor(LinearEmbeddingToConcept)
        pot = ParametricPotential(scope=[e1, e2], parametrization=lazy, name="phi")
        module = pot.parametrization["energy"]
        assert module.encoder.in_features == 10  # 4 + 6
        assert module.encoder.out_features == 1

    def test_lazy_potential_forward_works(self):
        a, b = _bin("a"), _bin("b")
        lazy = LazyConstructor(LinearConceptToConcept)
        pot = ParametricPotential(scope=[a, b], parametrization=lazy, name="phi")
        B = 5
        out = pot(inputs={"a": torch.rand(B, 1), "b": torch.rand(B, 1)})
        assert out.shape == (B,)

    def test_lazy_potential_energy_and_log_potential_work(self):
        a, b = _bin("a"), _bin("b")
        lazy = LazyConstructor(LinearConceptToConcept)
        pot = ParametricPotential(scope=[a, b], parametrization=lazy, name="phi")
        av = torch.tensor([[1.0], [0.0]])
        bv = torch.tensor([[0.0], [1.0]])
        e = pot.energy({a: av, b: bv})
        lp = pot.log_potential({a: av, b: bv})
        assert e.shape == (2,)
        assert torch.allclose(lp, -e)

    def test_already_built_lazy_is_unwrapped(self):
        a, b = _bin("a"), _bin("b")
        lazy = LazyConstructor(LinearConceptToConcept)
        lazy.build(out_concepts=1, in_concepts=2)
        built_module = lazy.module
        pot = ParametricPotential(scope=[a, b], parametrization=lazy, name="phi")
        module = pot.parametrization["energy"]
        assert not isinstance(module, LazyConstructor)
        assert module is built_module

    def test_lazy_dict_form_parametrization(self):
        a, b = _bin("a"), _bin("b")
        lazy = LazyConstructor(LinearConceptToConcept)
        pot = ParametricPotential(
            scope=[a, b], parametrization={"energy": lazy}, name="phi"
        )
        module = pot.parametrization["energy"]
        assert isinstance(module, LinearConceptToConcept)
        assert module.predictor.in_features == 2

    def test_no_lazy_entries_is_a_fast_path_noop(self):
        """_instantiate_lazy returns the very same dict object when nothing
        needs building, skipping the sizing/build work entirely."""
        params = {"energy": nn.Linear(2, 1)}
        a, b = _bin("a"), _bin("b")
        result = ParametricPotential._instantiate_lazy(params, [a, b])
        assert result is params


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


class TestBuildDistributionPlates:
    """A plate of a whole-event family (``OneHotCategorical``) is ``k`` independent
    per-member distributions, not one over the flattened ``k * member_size``
    classes — otherwise most valid plate values score as impossible."""

    def test_categorical_plate_scores_each_member_independently(self):
        h = ConceptVariable(
            "h", members=["h1", "h2"], distribution=dist.OneHotCategorical, size=3
        )
        logits = torch.randn(4, 6)
        d = build_distribution(h, {"logits": logits})
        # A value where *both* members are one-hot must be in support and score
        # as the sum of the two members' log-probs.
        value = torch.tensor([[1.0, 0, 0, 0, 0, 1]]).expand(4, 6)
        per_member = dist.OneHotCategorical(logits=logits.reshape(4, 2, 3))
        expected = per_member.log_prob(value.reshape(4, 2, 3)).sum(-1)
        assert torch.allclose(d.log_prob(value), expected, atol=1e-5)

    def test_categorical_plate_samples_are_per_member_one_hot(self):
        h = ConceptVariable(
            "h", members=["h1", "h2"], distribution=dist.OneHotCategorical, size=3
        )
        s = build_distribution(h, {"logits": torch.zeros(64, 6)}).sample()
        assert s.shape == (64, 6)
        assert torch.allclose(s[..., :3].sum(-1), torch.ones(64))
        assert torch.allclose(s[..., 3:].sum(-1), torch.ones(64))
        assert set(s.unique().tolist()) <= {0.0, 1.0}

    def test_bernoulli_plate_unchanged(self):
        g = ConceptVariable("g", members=["g1", "g2"], distribution=dist.Bernoulli)
        d = build_distribution(g, {"logits": torch.zeros(1, 2)})
        assert isinstance(d, dist.Independent)
        assert d.log_prob(torch.tensor([[1.0, 0.0]])).shape == (1,)
