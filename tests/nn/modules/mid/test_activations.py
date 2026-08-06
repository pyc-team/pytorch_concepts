"""Tests for DefaultActivation (torch_concepts.nn.modules.mid.activations).

A ``ParametricCPD`` applies no activation, so a head must already emit a value in
its parameter's domain. ``DefaultActivation`` reads the family's standard mapping
off its ``DistributionSpec`` instead of making the caller remember it.
"""
import pytest
import torch
import torch.nn as nn
from torch.distributions import (
    Bernoulli,
    Categorical,
    MultivariateNormal,
    Normal,
    OneHotCategorical,
    RelaxedBernoulli,
    RelaxedOneHotCategorical,
)

from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    BayesianNetwork,
    ConceptVariable,
    DefaultActivation,
    EmbeddingVariable,
    LearnablePrior,
    ParametricCPD,
    Sequential,
    TrilActivation,
)
from torch_concepts.nn.modules.mid.inference.utils import build_distribution


class TestResolution:
    """Which module each (family, parameter) pair resolves to."""

    @pytest.mark.parametrize(
        "param, distribution, expected",
        [
            ("probs", Bernoulli, nn.Sigmoid),
            ("probs", RelaxedBernoulli, nn.Sigmoid),
            ("logits", Bernoulli, nn.Identity),
            ("logits", OneHotCategorical, nn.Identity),
            ("probs", Categorical, nn.Softmax),
            ("probs", OneHotCategorical, nn.Softmax),
            ("probs", RelaxedOneHotCategorical, nn.Softmax),
            ("scale", Normal, nn.Softplus),
            ("loc", Normal, nn.Identity),
            ("loc", MultivariateNormal, nn.Identity),
            ("value", Delta, nn.Identity),
        ],
    )
    def test_resolves_the_families_standard_activation(self, param, distribution, expected):
        assert isinstance(DefaultActivation(param, distribution).activation, expected)

    def test_scale_tril_resolves_to_the_cholesky_assembly(self):
        act = DefaultActivation("scale_tril", MultivariateNormal, size=3)
        assert isinstance(act.activation, TrilActivation)
        tril = act(torch.randn(8, 6))
        assert tril.shape == (8, 3, 3)
        assert bool((tril.diagonal(dim1=-2, dim2=-1) > 0).all())

    def test_relaxed_families_resolve_like_their_exact_counterparts(self):
        exact = DefaultActivation("probs", Bernoulli, size=4, member_size=4)
        relaxed = DefaultActivation("probs", RelaxedBernoulli, size=4, member_size=4)
        raw = torch.randn(8, 4)
        assert torch.equal(exact(raw), relaxed(raw))

    def test_an_unconstrained_parameter_is_a_no_op(self):
        raw = torch.randn(8, 3)
        assert torch.equal(DefaultActivation("logits", Bernoulli)(raw), raw)


class TestDomains:
    """The output really does land in the parameter's domain."""

    def test_bernoulli_probs_are_in_the_unit_interval(self):
        probs = DefaultActivation("probs", Bernoulli)(torch.randn(8, 5) * 10)
        assert bool(((probs >= 0) & (probs <= 1)).all())

    def test_normal_scale_is_positive(self):
        scale = DefaultActivation("scale", Normal)(torch.randn(8, 5) * 10)
        assert bool((scale > 0).all())

    def test_a_lone_categorical_normalises_over_the_whole_row(self):
        act = DefaultActivation("probs", OneHotCategorical, size=4, member_size=4)
        probs = act(torch.randn(8, 4))
        assert probs.shape == (8, 4)
        assert torch.allclose(probs.sum(-1), torch.ones(8))

    def test_a_categorical_plate_normalises_per_member(self):
        # 2 members x 3 states: each member's block sums to 1, not the whole row.
        act = DefaultActivation("probs", OneHotCategorical, size=6, member_size=3)
        probs = act(torch.randn(8, 6))
        assert probs.shape == (8, 6)  # stays flat
        assert torch.allclose(probs.reshape(8, 2, 3).sum(-1), torch.ones(8, 2))
        # The naive whole-row softmax would have made this sum to 1, not 2.
        assert torch.allclose(probs.sum(-1), torch.full((8,), 2.0))

    def test_leading_dimensions_are_preserved(self):
        act = DefaultActivation("probs", OneHotCategorical, size=6, member_size=3)
        probs = act(torch.randn(4, 5, 6))
        assert probs.shape == (4, 5, 6)
        assert torch.allclose(probs.reshape(4, 5, 2, 3).sum(-1), torch.ones(4, 5, 2))


class TestForVariable:
    """``for_variable`` pulls family, size and member width off a Variable."""

    def test_plate_of_bernoullis(self):
        v = ConceptVariable("c", members=["c1", "c2", "c3"], distribution=Bernoulli)
        act = DefaultActivation.for_variable(v, "probs")
        assert isinstance(act.activation, nn.Sigmoid)

    def test_plate_of_categoricals_normalises_per_member(self):
        v = ConceptVariable("c", members=["c1", "c2"], distribution=OneHotCategorical, size=3)
        act = DefaultActivation.for_variable(v, "probs")
        probs = act(torch.randn(8, v.size))
        assert torch.allclose(probs.reshape(8, 2, 3).sum(-1), torch.ones(8, 2))

    def test_lone_categorical_matches_the_explicit_form(self):
        v = ConceptVariable("c", distribution=OneHotCategorical, size=5)
        raw = torch.randn(8, 5)
        assert torch.equal(
            DefaultActivation.for_variable(v, "probs")(raw),
            DefaultActivation("probs", OneHotCategorical, size=5, member_size=5)(raw),
        )

    def test_supplies_the_size_scale_tril_needs(self):
        v = ConceptVariable("c", distribution=MultivariateNormal, size=3)
        assert DefaultActivation.for_variable(v, "scale_tril")(torch.randn(8, 6)).shape == (8, 3, 3)


class TestErrors:
    def test_a_parameter_the_family_does_not_have_is_rejected(self):
        with pytest.raises(ValueError, match=r"Normal has no parameter 'probs'"):
            DefaultActivation("probs", Normal)

    def test_the_error_lists_the_valid_parameter_names(self):
        with pytest.raises(ValueError, match=r"\['loc', 'scale'\]"):
            DefaultActivation("probs", Normal)

    def test_scale_tril_without_a_size_is_rejected(self):
        with pytest.raises(ValueError, match="needs the event `size`"):
            DefaultActivation("scale_tril", MultivariateNormal)

    def test_an_unregistered_family_is_rejected(self):
        with pytest.raises(ValueError, match="not a supported family"):
            DefaultActivation("probs", torch.distributions.Poisson)


class TestModuleHygiene:
    def test_the_activation_is_a_child_module(self):
        act = DefaultActivation("scale_tril", MultivariateNormal, size=3)
        assert "activation" in dict(act.named_children())

    def test_buffers_travel_with_the_module(self):
        act = DefaultActivation("scale_tril", MultivariateNormal, size=3)
        # TrilActivation's index buffers are non-persistent, so they stay out of
        # state_dict but must still follow a dtype/device cast.
        act = act.to(torch.float64)
        assert act(torch.randn(8, 6, dtype=torch.float64)).dtype == torch.float64

    def test_repr_names_the_parameter_and_the_family(self):
        text = repr(DefaultActivation("probs", Bernoulli))
        assert "param='probs'" in text and "distribution=Bernoulli" in text


class TestInsideACPD:
    """The whole point: composing a raw head into a valid parametrization."""

    def test_a_plain_sequential_head_parametrizes_its_variable(self):
        x = EmbeddingVariable("x", distribution=Delta, size=4)
        c = ConceptVariable("c", members=["c1", "c2"], distribution=OneHotCategorical, size=3)
        cpd = ParametricCPD(
            c,
            parents=[x],
            parametrization={
                "probs": nn.Sequential(
                    nn.Linear(4, c.size), DefaultActivation.for_variable(c, "probs")
                )
            },
        )
        params = cpd(parent_values={"x": torch.randn(8, 4)})
        # build_distribution would reject probs that do not normalise per member.
        build_distribution(c, params)
        assert torch.allclose(params["probs"].reshape(8, 2, 3).sum(-1), torch.ones(8, 2))

    def test_a_pyc_sequential_head_works_the_same(self):
        x = EmbeddingVariable("x", distribution=Delta, size=4)
        c = ConceptVariable("c", distribution=Bernoulli, size=3)
        cpd = ParametricCPD(
            c,
            parents=[x],
            parametrization=Sequential(
                nn.Linear(4, 3), DefaultActivation.for_variable(c, "probs")
            ),
        )
        probs = cpd(parent_values={"x": torch.randn(8, 4)})["probs"]
        assert bool(((probs >= 0) & (probs <= 1)).all())

    def test_a_normal_cpd_gets_a_positive_scale(self):
        x = EmbeddingVariable("x", distribution=Delta, size=4)
        z = ConceptVariable("z", distribution=Normal, size=2)
        cpd = ParametricCPD(
            z,
            parents=[x],
            parametrization={
                "loc": nn.Linear(4, 2),
                "scale": nn.Sequential(nn.Linear(4, 2), DefaultActivation("scale", Normal)),
            },
        )
        params = cpd(parent_values={"x": torch.randn(8, 4)})
        assert bool((params["scale"] > 0).all())
        build_distribution(z, params)

    def test_gradients_flow_through_the_activation(self):
        head = nn.Sequential(nn.Linear(4, 3), DefaultActivation("probs", Bernoulli))
        head(torch.randn(8, 4)).sum().backward()
        assert head[0].weight.grad is not None

    def test_it_survives_the_deepcopy_of_a_broadcast_parametrization(self):
        # ParametricCPD.__new__ deep-copies one parametrization per variable.
        x = EmbeddingVariable("x", distribution=Delta, size=4)
        cs = ConceptVariable(["c1", "c2"], distribution=Bernoulli)
        cpds = ParametricCPD(
            cs,
            parents=[x],
            parametrization=nn.Sequential(nn.Linear(4, 1), DefaultActivation("probs", Bernoulli)),
        )
        assert len(cpds) == 2
        model = BayesianNetwork(variables=[x, *cs], factors=[
            ParametricCPD(x, parametrization=LearnablePrior(4)), *cpds
        ])
        # The copies must not share weights.
        assert cpds[0].parametrization["probs"][0].weight is not cpds[1].parametrization["probs"][0].weight
        for cpd in cpds:
            probs = cpd(parent_values={"x": torch.randn(8, 4)})["probs"]
            assert bool(((probs >= 0) & (probs <= 1)).all())
        assert model is not None
