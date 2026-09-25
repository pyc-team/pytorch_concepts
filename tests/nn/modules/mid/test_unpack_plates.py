"""``unpack_plates``: one ordinary variable per plate member, exactly.

The correctness pivot is separability — a plate's members are conditionally
independent given the parents, so its CPD splits into k factors whose
log-potentials sum to the plate's. Everything else here guards the ways that can
silently go wrong: member order, shared weights, and the undirected case where
separability does *not* hold.
"""
import pytest
import torch
import torch.distributions as dist
import torch.nn as nn

from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.low.priors import FixedPrior
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.factors.potential import ParametricPotential
from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.graph.markov_network import MarkovNetwork
from torch_concepts.nn.modules.mid.inference.utils import unpack_plates
from torch_concepts.nn.modules.mid.variable import ConceptVariable, EmbeddingVariable


def _chain(k=3):
    """x -> c{c0..ck-1} -> y, so the plate is both a child and a parent."""
    torch.manual_seed(0)
    x = EmbeddingVariable("x", distribution=Delta, size=2)
    c = ConceptVariable("c", members=[f"c{i}" for i in range(k)],
                        distribution=dist.Bernoulli)
    y = ConceptVariable("y", distribution=dist.Bernoulli)
    return x, c, y, BayesianNetwork(
        variables=[x, c, y],
        factors=[
            ParametricCPD(x, parametrization={"value": FixedPrior(torch.zeros(2))}),
            ParametricCPD(c, parents=[x],
                          parametrization=nn.Sequential(nn.Linear(2, k), nn.Sigmoid())),
            ParametricCPD(y, parents=[c],
                          parametrization=nn.Sequential(nn.Linear(k, 1), nn.Sigmoid())),
        ],
    )


class TestStructure:
    def test_plate_becomes_one_variable_and_one_factor_per_member(self):
        _, _, _, pgm = _chain(k=3)
        out = unpack_plates(pgm)
        assert sorted(out.variables) == ["c0", "c1", "c2", "x", "y"]
        assert sorted(out.factors) == ["c0", "c1", "c2", "x", "y"]
        for i in range(3):
            assert [v.name for v in out.factors[f"c{i}"].scope] == [f"c{i}", "x"]

    def test_a_plate_parent_becomes_its_members_in_order(self):
        _, c, _, pgm = _chain(k=3)
        out = unpack_plates(pgm)
        # One factor still, but its inputs are the members — in plate order,
        # which is what makes the concatenated row mean the same thing.
        assert [v.name for v in out.factors["y"].scope] == ["y", "c0", "c1", "c2"]
        assert [v.name for v in out.factors["y"].parents] == list(c.members)

    def test_a_model_without_plates_is_returned_unchanged(self):
        torch.manual_seed(0)
        a = ConceptVariable("a", distribution=dist.Bernoulli)
        pgm = BayesianNetwork(
            variables=[a],
            factors=[ParametricCPD(a, parametrization={"probs": FixedPrior(torch.tensor([0.3]))})],
        )
        assert unpack_plates(pgm) is pgm

    def test_the_head_is_shared_not_copied(self):
        _, _, _, pgm = _chain(k=3)
        head = pgm.factors["c"].parametrization["probs"]
        out = unpack_plates(pgm)
        for i in range(3):
            assert out.factors[f"c{i}"].parametrization["probs"].head is head
        # Same weights, so training the unpacked model trains the packed one.
        assert len(set(id(p) for p in out.parameters())) == len(list(pgm.parameters()))


class TestSeparability:
    def test_member_log_potentials_sum_to_the_plate_s(self):
        x, c, _, pgm = _chain(k=3)
        out = unpack_plates(pgm)
        xv = torch.randn(5, 2)
        cv = torch.randint(0, 2, (5, 3)).float()

        whole = pgm.factors["c"].log_potential({x: xv, c: cv})
        parts = sum(
            out.factors[f"c{i}"].log_potential(
                {out.variables["x"]: xv, out.variables[f"c{i}"]: cv[:, i:i + 1]}
            )
            for i in range(3)
        )
        assert torch.allclose(whole.flatten(), parts.flatten(), atol=1e-6)

    def test_a_plate_parent_factor_sees_the_same_row(self):
        x, c, y, pgm = _chain(k=3)
        out = unpack_plates(pgm)
        cv = torch.randint(0, 2, (4, 3)).float()
        yv = torch.randint(0, 2, (4, 1)).float()

        whole = pgm.factors["y"].log_potential({c: cv, y: yv})
        unpacked = out.factors["y"].log_potential(
            {**{out.variables[f"c{i}"]: cv[:, i:i + 1] for i in range(3)},
             out.variables["y"]: yv}
        )
        assert torch.allclose(whole.flatten(), unpacked.flatten(), atol=1e-6)


class TestUndirectedIsRejected:
    def test_a_plate_in_a_potential_scope_raises(self):
        torch.manual_seed(0)
        g = ConceptVariable("g", members=["g1", "g2"], distribution=dist.Bernoulli)
        b = ConceptVariable("b", distribution=dist.Bernoulli)
        mrf = MarkovNetwork(
            variables=[g, b],
            factors=[ParametricPotential(scope=[g, b], parametrization=nn.Linear(3, 1))],
        )
        with pytest.raises(ValueError, match="not separable"):
            unpack_plates(mrf)

    def test_a_potential_on_a_member_handle_is_rebuilt(self):
        """A handle is separable — it is one member — but it must be swapped out.

        A member handle keeps a back-reference to its plate, and a graph rejects
        that as "not the registered variable", so reusing the factor as-is would
        not build.
        """
        torch.manual_seed(0)
        g = ConceptVariable("g", members=["g1", "g2"], distribution=dist.Bernoulli)
        b = ConceptVariable("b", distribution=dist.Bernoulli)
        energy = nn.Linear(2, 1)
        mrf = MarkovNetwork(
            variables=[g, b],
            factors=[ParametricPotential(scope=[g.member("g1"), b], parametrization=energy)],
        )
        out = unpack_plates(mrf)
        [pot] = out.factors.values()
        assert [v.name for v in pot.scope] == ["g1", "b"]
        assert all(v.plate is v for v in pot.scope)  # no dangling plate back-ref
        assert list(pot.parametrization.values())[0] is energy  # same weights
