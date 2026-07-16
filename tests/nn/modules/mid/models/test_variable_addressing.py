"""Member addressing now lives on ``Variable``; ``ParametricCPD`` keeps thin
delegators. This checks the moved logic is correct and the delegators agree
(FACTOR_GRAPH_INSTRUCTIONS.md §5.1)."""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD


def _plate():
    return ConceptVariable("g", members=["m1", "m2", "m3"], distribution=dist.Bernoulli)


class TestVariableAddressing:
    def test_select_value_whole(self):
        g = _plate()
        v = torch.randn(4, 3)
        assert torch.equal(g.select_value(v, "g"), v)

    def test_select_value_member_is_view(self):
        g = _plate()
        v = torch.randn(4, 3)
        m2 = g.select_value(v, "m2")
        assert m2.shape == (4, 1)
        assert torch.equal(m2, v[:, 1:2])
        assert m2.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()

    def test_select_params_member(self):
        g = _plate()
        params = {"probs": torch.rand(2, 3)}
        out = g.select(params, "m3")
        assert torch.equal(out["probs"], params["probs"][:, 2:3])

    def test_clamp_members_overwrites_only_observed(self):
        g = _plate()
        v = torch.zeros(2, 3)
        obs = {"m1": torch.ones(2, 1), "m3": torch.full((2, 1), 5.0)}
        out = g.clamp_members(v, obs)
        assert torch.equal(out[:, 0:1], torch.ones(2, 1))
        assert torch.equal(out[:, 1:2], torch.zeros(2, 1))  # m2 untouched
        assert torch.equal(out[:, 2:3], torch.full((2, 1), 5.0))
        assert torch.equal(v, torch.zeros(2, 3))  # input not mutated

    def test_clamp_members_empty_is_noop(self):
        g = _plate()
        v = torch.randn(2, 3)
        assert g.clamp_members(v, {}) is v


class TestCPDDelegatorsAgree:
    def test_delegators_match_variable(self):
        g = _plate()
        cpd = ParametricCPD(
            variable=g,
            parametrization={"probs": nn.Sequential(nn.Linear(2, 3), nn.Sigmoid())},
            parents=[ConceptVariable("x", distribution=dist.Normal, size=2)],
        )
        v = torch.randn(5, 3)
        params = {"probs": torch.rand(5, 3)}
        assert torch.equal(cpd.select_value(v, "m2"), g.select_value(v, "m2"))
        assert torch.equal(cpd.select(params, "m2")["probs"], g.select(params, "m2")["probs"])
        obs = {"m1": torch.ones(5, 1)}
        assert torch.equal(cpd.clamp_members(v, obs), g.clamp_members(v, obs))
