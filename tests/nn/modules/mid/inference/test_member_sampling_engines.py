"""Uniform plate-member support in the sampling engines: member names as query
and evidence in RejectionSampling and torch ImportanceSampling, checked against
known probabilities on a small plate DAG (PLATE_REFACTOR_INSTRUCTIONS.md §5.2,
§5.3, §7, §8.3)."""
import warnings

import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts import seed_everything
from torch_concepts.nn.modules.mid.models.variable import ConceptVariable, EmbeddingVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.low.priors import FixedPrior
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.mid.inference.torch.rejection import RejectionSampling
from torch_concepts.nn.modules.mid.inference.torch.importance_sampling.importance_sampling import (
    ImportanceSampling,
)
from torch_concepts.nn.modules.mid.inference.torch.importance_sampling.mutilated_network import (
    MutilatedNetworkProposal,
)


def _independent_plate():
    """Root plate g = {a, b}, independent Bernoulli members with probs [0.8, 0.3]."""
    g = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli)
    cpd = ParametricCPD(g, parametrization={"probs": FixedPrior(torch.tensor([0.8, 0.3]))})
    return BayesianNetwork(variables=[g], factors=[cpd])


def _forcing_chain():
    """x -> concepts{c1, c2} -> y, with y depending sharply on c1."""
    x = EmbeddingVariable("x", distribution=Delta, size=2)
    concepts = ConceptVariable("concepts", members=["c1", "c2"], distribution=dist.Bernoulli)
    y = ConceptVariable("y", distribution=dist.Bernoulli)
    y_lin = nn.Linear(2, 1)
    with torch.no_grad():
        y_lin.weight.copy_(torch.tensor([[6.0, 0.0]]))
        y_lin.bias.copy_(torch.tensor([-3.0]))
    factors = [
        ParametricCPD(x, parametrization={"value": FixedPrior(torch.zeros(2))}),
        ParametricCPD(concepts, parents=[x],
                      parametrization=nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())),
        ParametricCPD(y, parents=[concepts],
                      parametrization=nn.Sequential(y_lin, nn.Sigmoid())),
    ]
    return BayesianNetwork(variables=[x, concepts, y], factors=factors)


class TestRejectionMemberSupport:
    def test_member_marginal_matches_prior(self):
        seed_everything(0)
        eng = RejectionSampling(_independent_plate(), n_samples=8000)
        one = torch.ones(1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert abs(float(eng.query({"a": one}).probabilities[0]) - 0.8) < 0.05
            assert abs(float(eng.query({"b": one}).probabilities[0]) - 0.3) < 0.05

    def test_member_evidence_on_independent_member_is_exact(self):
        seed_everything(1)
        eng = RejectionSampling(_independent_plate(), n_samples=8000)
        one = torch.ones(1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(eng.query({"a": one}, evidence={"b": one}).probabilities[0])
        assert abs(p - 0.8) < 0.06  # b independent of a -> conditioning does not move a


class TestImportanceMemberSupport:
    def test_member_marginal_matches_prior(self):
        seed_everything(0)
        eng = ImportanceSampling(
            _independent_plate(), proposal=MutilatedNetworkProposal(_independent_plate()),
            n_samples=6000, initial_temperature=0.1,
        )
        one = torch.ones(1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert abs(float(eng.query({"a": one}).probabilities[0]) - 0.8) < 0.07

    def test_member_evidence_forcing_propagates_to_task(self):
        seed_everything(0)
        pgm = _forcing_chain()
        eng = ImportanceSampling(
            pgm, proposal=MutilatedNetworkProposal(pgm),
            n_samples=4000, initial_temperature=0.1,
        )
        one, zero = torch.ones(1, 1), torch.zeros(1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_hi = float(eng.query({"y": one}, evidence={"c1": one}).probabilities[0])
            p_lo = float(eng.query({"y": one}, evidence={"c1": zero}).probabilities[0])
        assert p_hi > 0.8 and p_lo < 0.2  # forcing c1 flips y
