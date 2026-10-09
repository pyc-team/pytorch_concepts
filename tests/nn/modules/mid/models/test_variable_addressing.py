"""Member addressing lives on ``Variable``: ``member_of`` reads one member's slice
in either layout, ``clamp_members`` splices observed members into a value in
member layout. ``ParametricCPD`` keeps a single thin delegator,
``clamp_members``, which must agree with the variable's."""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.variable import ConceptVariable
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD


def _plate():
    return ConceptVariable("g", members=["m1", "m2", "m3"], distribution=dist.Bernoulli)


class TestVariableAddressing:
    def test_member_of_flat_value_is_view(self):
        g = _plate()
        v = torch.randn(4, 3)
        m2 = g.member_of(v, "m2")
        assert m2.shape == (4, 1)
        assert torch.equal(m2, v[:, 1:2])
        assert m2.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()

    def test_member_of_reads_either_layout(self):
        """A flat row and the same value in member layout give the same member."""
        g = _plate()
        flat = torch.randn(4, 3)
        assert torch.equal(g.member_of(flat, "m3"), g.member_of(flat.reshape(4, 3, 1), "m3"))

    def test_member_of_param(self):
        g = _plate()
        params = {"probs": torch.rand(2, 3)}
        assert torch.equal(g.member_of(params["probs"], "m3", "probs"), params["probs"][:, 2:3])

    def test_clamp_members_overwrites_only_observed(self):
        g = _plate()
        v = torch.zeros(2, 3, 1)  # member layout: (batch, n_members, member_size)
        obs = {"m1": torch.ones(2, 1), "m3": torch.full((2, 1), 5.0)}
        out = g.clamp_members(v, obs)
        assert torch.equal(out[:, 0], torch.ones(2, 1))
        assert torch.equal(out[:, 1], torch.zeros(2, 1))  # m2 untouched
        assert torch.equal(out[:, 2], torch.full((2, 1), 5.0))
        assert torch.equal(v, torch.zeros(2, 3, 1))  # input not mutated

    def test_clamp_members_empty_is_noop(self):
        g = _plate()
        v = torch.randn(2, 3, 1)
        assert g.clamp_members(v, {}) is v

    def test_clamp_members_rejects_flat_value(self):
        g = _plate()
        with pytest.raises(ValueError, match="member layout"):
            g.clamp_members(torch.zeros(4, 3), {"m1": torch.ones(4, 1)})

    def test_clamp_members_rejects_flat_value_when_batch_equals_members(self):
        """The case that used to go through silently: with batch size == number
        of members the member axis landed on the batch axis and the write
        overwrote a *row* instead of a member."""
        g = _plate()
        with pytest.raises(ValueError, match="member layout"):
            g.clamp_members(torch.zeros(3, 3), {"m1": torch.ones(3, 1)})


class TestCPDDelegatorAgrees:
    def test_clamp_members_matches_variable(self):
        g = _plate()
        cpd = ParametricCPD(
            variable=g,
            parametrization={"probs": nn.Sequential(nn.Linear(2, 3), nn.Sigmoid())},
            parents=[ConceptVariable("x", distribution=dist.Normal, size=2)],
        )
        v = torch.randn(5, 3, 1)
        obs = {"m1": torch.ones(5, 1)}
        assert torch.equal(cpd.clamp_members(v, obs), g.clamp_members(v, obs))
