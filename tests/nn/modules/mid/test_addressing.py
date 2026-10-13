"""Addressable names: every variable and plate-member name is unique.

Evidence, queries and interventions address the model by name, so a repeated
name must fail at construction rather than silently resolve to whichever
variable was indexed last.
"""
import pytest
import torch.distributions as dist

from torch_concepts.nn.modules.low.priors import LearnablePrior
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.utils import unpack_plates
from torch_concepts.nn.modules.mid.variable import ConceptVariable


def _model(*variables):
    return BayesianNetwork(
        variables=list(variables),
        factors=[ParametricCPD(v, {"logits": LearnablePrior(v.size)}) for v in variables],
    )


def _plate(name, members):
    return ConceptVariable(name, members=members, distribution=dist.Bernoulli)


@pytest.mark.parametrize("variables", [
    # c_t and c_{t+1} of a time-indexed model, sharing member names
    lambda: [_plate("a", ["m1", "m2"]), _plate("b", ["m1", "m2"])],
    # an ordinary variable shadowed by a plate member
    lambda: [ConceptVariable("m1", distribution=dist.Bernoulli), _plate("a", ["m1", "m2"])],
    # a member named like its own plate
    lambda: [_plate("g", ["g", "h"])],
])
def test_a_repeated_name_raises(variables):
    with pytest.raises(ValueError, match="must be unique"):
        _model(*variables())


def test_distinct_names_resolve_to_their_owner():
    a, b = _plate("a", ["a1", "a2"]), _plate("b", ["b1", "b2"])
    model = _model(a, b, ConceptVariable("y", distribution=dist.Bernoulli))
    assert model.resolve("a2") is a
    assert model.resolve("b1") is b
    # unpacking turns members into variables; their names are unique by construction
    assert set(unpack_plates(model).variables) == {"a1", "a2", "b1", "b2", "y"}
