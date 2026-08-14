"""Soft vs hard discrete draws, selected by the variable's **declared family**.

A variable declared ``Bernoulli`` / ``RelaxedBernoulli`` samples a soft Concrete
value; declaring ``RelaxedBernoulliStraightThrough`` gives an exact bit with a
soft gradient. No engine carries a flag for this.

The registry entries these rely on also fix a real trap: a straight-through class
is a *subclass* of its plain relaxed base, so without its own ``SPECS`` key it
resolves to the base spec and silently samples soft
(``test_straight_through_families_are_not_shadowed_by_their_base``).
"""
import pyro.distributions as pyro_dist
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.low.priors import FixedPrior
from torch_concepts.nn.modules.mid.distributions import spec_for
from torch_concepts.nn.modules.mid.variable import ConceptVariable
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.torch.ancestral import AncestralSamplingInference
from torch_concepts.nn.modules.mid.inference.torch.utils import sample_from

ST_BERNOULLI = pyro_dist.RelaxedBernoulliStraightThrough
ST_CATEGORICAL = pyro_dist.RelaxedOneHotCategoricalStraightThrough


def _bernoulli_model(family):
    """x (delta) -> c (binary, size=4)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    c = ConceptVariable("c", distribution=family, size=4)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_c = ParametricCPD(
        variable=c,
        # Mid-range probs (near 0.5) so the relaxed draw isn't already pinned
        # near 0/1 by construction.
        parametrization=nn.Sequential(nn.Linear(4, 4), nn.Sigmoid()),
        parents=[x],
    )
    return BayesianNetwork(variables=[x, c], factors=[cpd_x, cpd_c])


def _categorical_plate_model(family):
    """x (delta) -> g (plate: [m1, m2], categorical, cardinality 3)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    g = ConceptVariable("g", members=["m1", "m2"], distribution=family, size=3)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_g = ParametricCPD(
        variable=g,
        # `probs` must come back flat (*leading, size=6); softmax per member
        # (3-wide chunk) before flattening, so each member is its own simplex.
        parametrization={"probs": nn.Sequential(
            nn.Linear(4, 6), nn.Unflatten(-1, (2, 3)), nn.Softmax(dim=-1), nn.Flatten(start_dim=-2),
        )},
        parents=[x],
    )
    return BayesianNetwork(variables=[x, g], factors=[cpd_x, cpd_g])


def _normal_model():
    """x (delta) -> n (normal, size=3); x is Delta itself."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    n = ConceptVariable("n", distribution=dist.Normal, size=3)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_n = ParametricCPD(
        variable=n,
        parametrization={
            "loc": nn.Linear(4, 3),
            "scale": nn.Sequential(nn.Linear(4, 3), nn.Softplus()),
        },
        parents=[x],
    )
    return BayesianNetwork(variables=[x, n], factors=[cpd_x, cpd_n])


class TestRegistry:
    def test_straight_through_families_are_not_shadowed_by_their_base(self):
        """The bug these entries exist to prevent.

        ``_lookup`` falls back to the nearest registered base class, and the
        straight-through classes subclass their plain relaxed counterparts — so
        without their own keys they would resolve to the soft spec.
        """
        assert spec_for(ST_BERNOULLI).relaxed is not spec_for(dist.RelaxedBernoulli).relaxed
        assert spec_for(ST_CATEGORICAL).relaxed is not spec_for(dist.RelaxedOneHotCategorical).relaxed


class TestSampleFrom:
    def test_plain_bernoulli_is_soft(self):
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=8)
        params = {"probs": torch.full((5, 8), 0.5)}
        torch.manual_seed(0)
        soft = sample_from(c, params, temperature=torch.tensor(1.0))
        assert not torch.all((soft == 0.0) | (soft == 1.0))

    def test_straight_through_bernoulli_is_exact(self):
        c = ConceptVariable("c", distribution=ST_BERNOULLI, size=8)
        params = {"probs": torch.full((5, 8), 0.5)}
        hard = sample_from(c, params, temperature=torch.tensor(1.0))
        assert torch.all((hard == 0.0) | (hard == 1.0))

    def test_straight_through_categorical_is_exact(self):
        g = ConceptVariable("g", distribution=ST_CATEGORICAL, size=5)
        params = {"probs": torch.softmax(torch.randn(6, 5), dim=-1)}
        hard = sample_from(g, params, temperature=torch.tensor(1.0))
        assert torch.all(hard.sum(-1) == 1.0)
        assert torch.all((hard == 0.0) | (hard == 1.0))

    def test_straight_through_gradient_is_nonzero(self):
        c = ConceptVariable("c", distribution=ST_BERNOULLI, size=8)
        logits = torch.zeros(5, 8, requires_grad=True)
        params = {"probs": torch.sigmoid(logits)}
        hard = sample_from(c, params, temperature=torch.tensor(1.0))
        hard.sum().backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0

    def test_continuous_and_delta_are_unaffected(self):
        n = ConceptVariable("n", distribution=dist.Normal, size=3)
        params = {"loc": torch.randn(4, 3), "scale": torch.rand(4, 3) + 0.1}
        torch.manual_seed(0)
        first = sample_from(n, params, temperature=torch.tensor(1.0))
        torch.manual_seed(0)
        second = sample_from(n, params, temperature=torch.tensor(1.0))
        assert torch.equal(first, second)

        d = ConceptVariable("d", distribution=Delta, size=3)
        value = torch.randn(4, 3)
        assert torch.equal(
            sample_from(d, {"value": value}, temperature=torch.tensor(1.0)), value
        )


class TestAncestralSampling:
    def test_no_engine_level_switch(self):
        assert not hasattr(AncestralSamplingInference(_bernoulli_model(dist.Bernoulli)), "hard")

    def test_plain_family_draws_intermediate_values(self):
        eng = AncestralSamplingInference(_bernoulli_model(dist.Bernoulli), initial_temperature=1.0)
        torch.manual_seed(0)
        out = eng.query(query=["c"], evidence={"x": torch.zeros(20, 4)})
        c = out.samples["c"]
        assert not torch.all((c == 0.0) | (c == 1.0))

    def test_straight_through_family_draws_exact_bits(self):
        eng = AncestralSamplingInference(_bernoulli_model(ST_BERNOULLI), initial_temperature=1.0)
        out = eng.query(query=["c"], evidence={"x": torch.zeros(20, 4)})
        c = out.samples["c"]
        assert torch.all((c == 0.0) | (c == 1.0))

    def test_straight_through_plate_is_one_hot_per_member(self):
        eng = AncestralSamplingInference(
            _categorical_plate_model(ST_CATEGORICAL), initial_temperature=1.0
        )
        B = 6
        out = eng.query(query=["g"], evidence={"x": torch.randn(B, 4)})
        g = out.samples["g"].reshape(B, 2, 3)
        assert torch.all(g.sum(-1) == 1.0)
        assert torch.all((g == 0.0) | (g == 1.0))

    def test_continuous_draws_are_reproducible(self):
        torch.manual_seed(0)
        first = AncestralSamplingInference(_normal_model(), initial_temperature=1.0).query(
            query=["n"], evidence={}, n_samples=5
        )
        torch.manual_seed(0)
        second = AncestralSamplingInference(_normal_model(), initial_temperature=1.0).query(
            query=["n"], evidence={}, n_samples=5
        )
        assert torch.equal(first.samples["n"], second.samples["n"])
