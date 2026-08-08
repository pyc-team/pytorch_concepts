"""Tests for ParametricCPD's shared trunk.

A CPD runs one module per distribution parameter, each on its own aggregation of
the parents. When those parameters share an expensive front end — the ``loc`` and
``scale`` of a Normal behind a pretrained backbone — that front end runs once per
parameter. A ``trunk`` aggregates and extracts once, and the parameter modules
become cheap heads over its output.
"""
import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal

from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ConceptVariable,
    DefaultActivation,
    EmbeddingVariable,
    LazyConstructor,
    LearnablePrior,
    LinearConceptToConcept,
    LinearEmbeddingToConcept,
    ParametricCPD,
    intervention,
    DoIntervention,
    UniformPolicy,
)


class CountingTrunk(nn.Module):
    """A trunk that records how many times it ran."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.out_features = out_features
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.linear(x)


@pytest.fixture
def parent():
    return EmbeddingVariable("x", distribution=Delta, size=4)


@pytest.fixture
def normal_child():
    return ConceptVariable("z", distribution=Normal, size=3)


class TestSharedForward:
    def test_the_trunk_runs_once_for_a_two_parameter_cpd(self, parent, normal_child):
        trunk = CountingTrunk(4, 8)
        cpd = ParametricCPD(
            normal_child,
            parents=[parent],
            trunk=trunk,
            parametrization={
                "loc": nn.Linear(8, 3),
                "scale": nn.Sequential(
                    nn.Linear(8, 3), DefaultActivation.for_variable(normal_child, "scale")
                ),
            },
        )
        out = cpd(parent_values={"x": torch.randn(6, 4)})
        assert trunk.calls == 1
        assert out["loc"].shape == (6, 3)
        assert bool((out["scale"] > 0).all())

    def test_without_a_trunk_each_parameter_runs_its_own_head(self, parent, normal_child):
        # The historical behaviour, unchanged: two heads, two forwards.
        head = CountingTrunk(4, 3)
        cpd = ParametricCPD(
            normal_child,
            parents=[parent],
            parametrization={"loc": head, "scale": nn.Sequential(head, nn.Softplus())},
        )
        cpd(parent_values={"x": torch.randn(6, 4)})
        assert head.calls == 2

    def test_it_matches_the_equivalent_trunk_less_composition(self, parent, normal_child):
        torch.manual_seed(0)
        trunk, loc_head, scale_head = nn.Linear(4, 8), nn.Linear(8, 3), nn.Linear(8, 3)
        with_trunk = ParametricCPD(
            normal_child, parents=[parent], trunk=trunk,
            parametrization={"loc": loc_head, "scale": scale_head},
        )
        without = ParametricCPD(
            normal_child, parents=[parent],
            parametrization={
                "loc": nn.Sequential(trunk, loc_head),
                "scale": nn.Sequential(trunk, scale_head),
            },
        )
        x = torch.randn(5, 4)
        a = with_trunk(parent_values={"x": x})
        b = without(parent_values={"x": x})
        assert torch.allclose(a["loc"], b["loc"])
        assert torch.allclose(a["scale"], b["scale"])

    def test_gradients_reach_the_trunk_from_both_heads(self, parent, normal_child):
        trunk = CountingTrunk(4, 8)
        cpd = ParametricCPD(
            normal_child, parents=[parent], trunk=trunk,
            parametrization={"loc": nn.Linear(8, 3), "scale": nn.Linear(8, 3)},
        )
        out = cpd(parent_values={"x": torch.randn(6, 4)})
        (out["loc"].sum() + out["scale"].sum()).backward()
        assert trunk.linear.weight.grad is not None

    def test_the_trunk_is_a_registered_submodule(self, parent, normal_child):
        trunk = CountingTrunk(4, 8)
        cpd = ParametricCPD(
            normal_child, parents=[parent], trunk=trunk,
            parametrization={"loc": nn.Linear(8, 3), "scale": nn.Linear(8, 3)},
        )
        assert any(m is trunk for m in cpd.modules())
        assert any(k.startswith("trunk.") for k in cpd.state_dict())


class TestLazySizing:
    @pytest.mark.parametrize(
        "head_cls", [LinearConceptToConcept, LinearEmbeddingToConcept, nn.Linear]
    )
    def test_a_lazy_head_is_sized_from_the_trunks_out_features(
        self, parent, normal_child, head_cls
    ):
        trunk = CountingTrunk(4, 16)
        cpd = ParametricCPD(
            normal_child, parents=[parent], trunk=trunk,
            parametrization={
                "loc": LazyConstructor(head_cls),
                "scale": LazyConstructor(head_cls),
            },
        )
        assert cpd(parent_values={"x": torch.randn(5, 4)})["loc"].shape == (5, 3)
        assert trunk.calls == 1

    def test_a_trunk_without_out_features_is_rejected_for_a_lazy_head(
        self, parent, normal_child
    ):
        with pytest.raises(ValueError, match="does not declare `out_features`"):
            ParametricCPD(
                normal_child, parents=[parent], trunk=nn.Identity(),
                parametrization={
                    "loc": LazyConstructor(nn.Linear),
                    "scale": LazyConstructor(nn.Linear),
                },
            )


class TestErrors:
    def test_a_root_cpd_rejects_a_trunk(self, parent):
        with pytest.raises(ValueError, match="needs parents"):
            ParametricCPD(
                parent, parametrization={"value": LearnablePrior(4)},
                trunk=nn.Linear(4, 4),
            )

    def test_a_non_module_trunk_is_rejected(self, parent, normal_child):
        with pytest.raises(TypeError, match="`trunk` must be an nn.Module"):
            ParametricCPD(
                normal_child, parents=[parent], trunk="not a module",
                parametrization={"loc": nn.Linear(4, 3), "scale": nn.Linear(4, 3)},
            )


class TestInteractions:
    def test_broadcast_cpds_do_not_share_a_trunk(self, parent):
        # A list of variables builds independent CPDs, so the trunk is copied
        # per CPD exactly as the parametrization is.
        cs = ConceptVariable(["c1", "c2"], distribution=Bernoulli)
        cpds = ParametricCPD(
            cs, parents=[parent], trunk=CountingTrunk(4, 8),
            parametrization={"probs": nn.Sequential(
                nn.Linear(8, 1), DefaultActivation("probs", Bernoulli))},
        )
        assert cpds[0].trunk is not cpds[1].trunk
        assert cpds[0].trunk.linear.weight is not cpds[1].trunk.linear.weight

    def test_intervention_swaps_the_head_not_the_trunk(self, parent):
        c = ConceptVariable("c", distribution=Bernoulli, size=3)
        trunk = CountingTrunk(4, 8)
        cpd = ParametricCPD(
            c, parents=[parent], trunk=trunk,
            parametrization={"probs": nn.Sequential(
                nn.Linear(8, 3), DefaultActivation.for_variable(c, "probs"))},
        )

        class Pgm:
            factors = {"c": cpd}

        original = cpd.parametrization["probs"]
        with intervention(
            Pgm(), DoIntervention(constants=0.0), UniformPolicy(),
            variable_to_intervene_on="c", parameter_to_intervene_on="probs",
        ):
            assert cpd.parametrization["probs"] is not original
            assert cpd.trunk is trunk  # the expensive part is untouched
            out = cpd(parent_values={"x": torch.randn(4, 4)})["probs"]
            assert torch.allclose(out, torch.zeros(4, 3), atol=1e-5)
        assert cpd.parametrization["probs"] is original
