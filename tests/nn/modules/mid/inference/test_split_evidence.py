"""BaseInference._split_evidence: grouping evidence into whole-variable and
per-owner member entries (PLATE_REFACTOR_INSTRUCTIONS.md §4.1, §8.3)."""
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.variable import ConceptVariable, EmbeddingVariable
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.torch.deterministic import DeterministicInference
from torch_concepts.nn.modules.low.priors import LearnablePrior
from torch_concepts.distributions import Delta


def _engine():
    x = EmbeddingVariable("x", distribution=Delta, size=4)
    concepts = ConceptVariable("concepts", members=["c1", "c2", "c3"], distribution=dist.Bernoulli)
    y = ConceptVariable("y", distribution=dist.Bernoulli)
    factors = [
        ParametricCPD(x, parametrization=LearnablePrior(4)),
        ParametricCPD(concepts, parents=[x],
                      parametrization=nn.Sequential(nn.Linear(4, 3), nn.Sigmoid())),
        ParametricCPD(y, parents=[concepts],
                      parametrization=nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())),
    ]
    return DeterministicInference(BayesianNetwork(variables=[x, concepts, y], factors=factors))


def test_mixed_evidence_split():
    eng = _engine()
    x, c1, c3 = torch.randn(2, 4), torch.ones(2, 1), torch.zeros(2, 1)
    whole, member = eng._split_evidence({"x": x, "concepts": torch.rand(2, 3), "c1": c1, "c3": c3})
    assert set(whole) == {"x", "concepts"}
    assert set(member) == {"concepts"}
    assert set(member["concepts"]) == {"c1", "c3"}
    assert torch.equal(member["concepts"]["c1"], c1)


def test_all_ordinary_evidence_has_empty_member_dict():
    eng = _engine()
    whole, member = eng._split_evidence({"x": torch.randn(2, 4), "concepts": torch.rand(2, 3)})
    assert member == {}
    assert set(whole) == {"x", "concepts"}


def test_empty_evidence():
    eng = _engine()
    whole, member = eng._split_evidence({})
    assert whole == {} and member == {}
