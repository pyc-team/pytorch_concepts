"""``AncestralSamplingInference(hard=True)``: straight-through hard sampling.

Covers the new ``hard`` constructor flag and the ``sample_from(..., hard=...)``
mechanism it delegates to: exact quantization of discrete draws, the per-member
split for a categorical plate, no effect on continuous/``Delta`` families, a
live straight-through gradient, and that the default stays soft.
"""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts import seed_everything
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.low.priors import FixedPrior, LearnablePrior
from torch_concepts.nn.modules.mid.variable import ConceptVariable
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.torch.ancestral import AncestralSamplingInference
from torch_concepts.nn.modules.mid.inference.torch.utils import sample_from


def _bernoulli_model():
    """x (delta) -> c (bernoulli, size=4)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    c = ConceptVariable("c", distribution=dist.Bernoulli, size=4)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_c = ParametricCPD(
        variable=c,
        # Mid-range probs (near 0.5) so the relaxed draw isn't already pinned
        # near 0/1 by construction.
        parametrization=nn.Sequential(nn.Linear(4, 4), nn.Sigmoid()),
        parents=[x],
    )
    return BayesianNetwork(variables=[x, c], factors=[cpd_x, cpd_c])


def _categorical_plate_model():
    """x (delta) -> g (plate: [m1, m2], OneHotCategorical, cardinality 3)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    g = ConceptVariable("g", members=["m1", "m2"], distribution=dist.OneHotCategorical, size=3)
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


def _normal_delta_model():
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


class TestSampleFromHard:
    def test_default_is_soft(self):
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=8)
        params = {"probs": torch.full((5, 8), 0.5)}
        torch.manual_seed(0)
        soft = sample_from(c, params, temperature=torch.tensor(1.0))
        assert not torch.all((soft == 0.0) | (soft == 1.0))

    def test_hard_bernoulli_is_exact(self):
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=8)
        params = {"probs": torch.full((5, 8), 0.5)}
        hard = sample_from(c, params, temperature=torch.tensor(1.0), hard=True)
        assert torch.all((hard == 0.0) | (hard == 1.0))

    def test_hard_one_hot_categorical_is_exact(self):
        g = ConceptVariable("g", distribution=dist.OneHotCategorical, size=5)
        params = {"probs": torch.softmax(torch.randn(6, 5), dim=-1)}
        hard = sample_from(g, params, temperature=torch.tensor(1.0), hard=True)
        assert torch.all((hard.sum(-1) == 1.0))
        assert torch.all((hard == 0.0) | (hard == 1.0))

    def test_hard_normal_matches_soft(self):
        n = ConceptVariable("n", distribution=dist.Normal, size=3)
        params = {"loc": torch.randn(4, 3), "scale": torch.rand(4, 3) + 0.1}
        torch.manual_seed(0)
        soft = sample_from(n, params, temperature=torch.tensor(1.0))
        torch.manual_seed(0)
        hard = sample_from(n, params, temperature=torch.tensor(1.0), hard=True)
        assert torch.equal(soft, hard)

    def test_hard_delta_matches_soft(self):
        d = ConceptVariable("d", distribution=Delta, size=3)
        params = {"value": torch.randn(4, 3)}
        soft = sample_from(d, params, temperature=torch.tensor(1.0))
        hard = sample_from(d, params, temperature=torch.tensor(1.0), hard=True)
        assert torch.equal(soft, hard)

    def test_straight_through_gradient_is_nonzero(self):
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=8)
        logits = torch.zeros(5, 8, requires_grad=True)
        params = {"probs": torch.sigmoid(logits)}
        hard = sample_from(c, params, temperature=torch.tensor(1.0), hard=True)
        hard.sum().backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0


class TestAncestralSamplingInferenceHardFlag:
    def test_default_hard_is_false(self):
        m = _bernoulli_model()
        eng = AncestralSamplingInference(m)
        assert eng.hard is False

    def test_soft_by_default_has_intermediate_values(self):
        m = _bernoulli_model()
        eng = AncestralSamplingInference(m, initial_temperature=1.0)
        torch.manual_seed(0)
        out = eng.query(query=["c"], evidence={"x": torch.zeros(20, 4)})
        c = out.samples["c"]
        assert not torch.all((c == 0.0) | (c == 1.0))

    def test_hard_true_gives_exact_bernoulli_values(self):
        m = _bernoulli_model()
        eng = AncestralSamplingInference(m, initial_temperature=1.0, hard=True)
        out = eng.query(query=["c"], evidence={"x": torch.zeros(20, 4)})
        c = out.samples["c"]
        assert torch.all((c == 0.0) | (c == 1.0))

    def test_hard_true_categorical_plate_one_hot_per_member(self):
        m = _categorical_plate_model()
        eng = AncestralSamplingInference(m, initial_temperature=1.0, hard=True)
        B = 6
        out = eng.query(query=["g"], evidence={"x": torch.randn(B, 4)})
        g = out.samples["g"].reshape(B, 2, 3)
        assert torch.all(g.sum(-1) == 1.0)
        assert torch.all((g == 0.0) | (g == 1.0))

    def test_hard_true_leaves_normal_and_delta_bit_identical_to_soft(self):
        m = _normal_delta_model()
        eng_soft = AncestralSamplingInference(m, initial_temperature=1.0, hard=False)
        torch.manual_seed(0)
        out_soft = eng_soft.query(query=["n"], evidence={}, n_samples=5)

        eng_hard = AncestralSamplingInference(m, initial_temperature=1.0, hard=True)
        torch.manual_seed(0)
        out_hard = eng_hard.query(query=["n"], evidence={}, n_samples=5)

        assert torch.equal(out_soft.samples["n"], out_hard.samples["n"])
