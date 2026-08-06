"""Unconditional sampling from a model's priors.

Generating from a model means drawing every root from its own prior — no
evidence, no guide. The ancestral engine already does that; what it lacked was a
way to ask for more than one draw, because with no tensor anywhere there is
nothing to read a batch size from. ``n_samples`` supplies it.
"""
import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal

from torch_concepts.nn import (
    AncestralSamplingInference,
    BayesianNetwork,
    ConceptVariable,
    DefaultActivation,
    DeterministicInference,
    EmbeddingVariable,
    FixedPrior,
    ParametricCPD,
)


@pytest.fixture
def pgm():
    """``z ~ N(0, I)`` -> a Bernoulli concept. Both roots are fixed priors, so an
    unconditional pass is fully determined by the sampling."""
    z = EmbeddingVariable("z", distribution=Normal, size=4)
    c = ConceptVariable("c", distribution=Bernoulli, size=3)
    torch.manual_seed(0)
    return BayesianNetwork(
        variables=[z, c],
        factors=[
            ParametricCPD(z, parents=[], parametrization={
                "loc": FixedPrior(torch.zeros(4)),
                "scale": FixedPrior(torch.ones(4)),
            }),
            ParametricCPD(c, parents=[z], parametrization={
                "probs": nn.Sequential(
                    nn.Linear(4, 3), DefaultActivation.for_variable(c, "probs")
                )
            }),
        ],
    )


class TestNSamples:
    def test_it_sets_the_batch_size_of_an_unconditional_pass(self, pgm):
        engine = AncestralSamplingInference(pgm, p_int=0.0)
        out = engine.query(["z", "c"], evidence={}, n_samples=16)
        assert out.probs["c"].shape == (16, 3)
        assert out.params["loc"]["z"].shape == (16, 4)

    def test_without_it_an_unconditional_pass_is_a_single_draw(self, pgm):
        # The historical fallback, unchanged.
        engine = AncestralSamplingInference(pgm, p_int=0.0)
        assert engine.query(["z", "c"], evidence={}).probs["c"].shape == (1, 3)

    def test_z_really_is_resampled_from_the_prior(self, pgm):
        engine = AncestralSamplingInference(pgm, p_int=0.0)
        first = engine.query(["z", "c"], evidence={}, n_samples=8).samples["z"]
        second = engine.query(["z", "c"], evidence={}, n_samples=8).samples["z"]
        assert not torch.allclose(first, second)

    def test_the_draws_follow_the_declared_prior(self, pgm):
        engine = AncestralSamplingInference(pgm, p_int=0.0)
        z = engine.query(["z"], evidence={}, n_samples=20000).samples["z"]
        assert abs(float(z.mean())) < 0.05
        assert abs(float(z.std()) - 1.0) < 0.05

    def test_evidence_still_wins_over_n_samples(self, pgm):
        # The leading shape comes from the supplied tensor; n_samples only fills
        # in when there is nothing to read one from.
        engine = AncestralSamplingInference(pgm, p_int=0.0)
        out = engine.query(
            ["c"], evidence={"z": torch.randn(5, 4)}, n_samples=16
        )
        assert out.probs["c"].shape == (5, 3)

    def test_it_works_for_the_other_torch_engines(self, pgm):
        out = DeterministicInference(pgm).query(["z", "c"], evidence={}, n_samples=7)
        assert out.probs["c"].shape == (7, 3)
