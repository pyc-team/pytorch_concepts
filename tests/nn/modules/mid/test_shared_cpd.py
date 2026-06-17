"""Tests for plate variables.

Plate variables are created via `ConceptVariable('name', members=['m1', 'm2', ...], ...)`.
These tests cover the full plate lifecycle: construction, forward pass,
member slicing, and plate-aware inference.
"""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.torch.deterministic import DeterministicInference
from torch_concepts.nn.modules.mid.inference.torch.ancestral import AncestralInference
from torch_concepts.nn.modules.low.priors import LearnablePrior, FixedPrior
from torch_concepts.distributions import Delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plate2(name="g"):
    return ConceptVariable(name, members=["m1", "m2"], distribution=dist.Bernoulli)


def _plate3(name="g"):
    return ConceptVariable(name, members=["c1", "c2", "c3"], distribution=dist.Bernoulli)


def _root_model_with_plate(plate):
    cpd = ParametricCPD(variable=plate, parametrization={"probs": LearnablePrior(plate.size)})
    return BayesianNetwork(variables=[plate], factors=[cpd])


# ===========================================================================
# 1. Plate variable construction
# ===========================================================================

class TestPlateVariableConstruction:
    def test_is_plate_flag(self):
        g = _plate2()
        assert g.is_plate is True

    def test_members_list(self):
        g = _plate2()
        assert g.members == ["m1", "m2"]

    def test_name(self):
        g = _plate2("concepts")
        assert g.name == "concepts"

    def test_member_size_default_one(self):
        g = _plate2()
        assert g.member_size == 1

    def test_total_size_equals_n_times_member_size(self):
        g = _plate3()
        assert g.size == 3 * g.member_size

    def test_plate_size_three_members_size_three_total(self):
        g = _plate3()
        assert g.size == 3

    def test_not_plate_for_normal_variable(self):
        v = ConceptVariable("v", distribution=dist.Bernoulli, size=1)
        assert v.is_plate is False

    def test_distribution_bernoulli(self):
        g = _plate2()
        assert g.distribution is dist.Bernoulli

    def test_mvn_plate_not_allowed(self):
        with pytest.raises(ValueError):
            ConceptVariable("g", members=["a", "b"], distribution=dist.MultivariateNormal)


# ===========================================================================
# 2. Plate member addressing
# ===========================================================================

class TestPlateAddressing:
    def test_column_of_returns_slice(self):
        g = _plate2()
        sl = g.column_of("m1")
        assert isinstance(sl, slice)

    def test_column_of_m1_starts_at_0(self):
        g = _plate2()
        sl = g.column_of("m1")
        assert sl.start == 0

    def test_column_of_m2_starts_at_1(self):
        g = _plate2()
        sl = g.column_of("m2")
        assert sl.start == 1

    def test_column_of_unknown_raises(self):
        g = _plate2()
        with pytest.raises((KeyError, ValueError)):
            g.column_of("unknown")

    def test_member_returns_handle(self):
        g = _plate2()
        h = g.member("m1")
        assert h.name == "m1"

    def test_member_handle_plate_backref(self):
        g = _plate2()
        h = g.member("m1")
        assert h._plate is g

    def test_member_size_from_handle(self):
        g = ConceptVariable("g", members=["c1", "c2"], distribution=dist.Bernoulli, size=2)
        sl = g.column_of("c1")
        assert sl.stop - sl.start == 2

    def test_actual_tensor_slicing(self):
        g = _plate3()
        B = 4
        full = torch.rand(B, 3)
        sl_c2 = g.column_of("c2")
        sliced = full[:, sl_c2]
        assert sliced.shape == (B, 1)
        assert (sliced == full[:, 1:2]).all()


# ===========================================================================
# 3. CPD with plate variable
# ===========================================================================

class TestPlateCPD:
    def test_root_plate_cpd_construction(self):
        g = _plate2()
        cpd = ParametricCPD(variable=g, parametrization={"probs": LearnablePrior(g.size)})
        assert cpd.variable is g

    def test_root_plate_cpd_is_root(self):
        g = _plate2()
        cpd = ParametricCPD(variable=g, parametrization={"probs": LearnablePrior(g.size)})
        assert cpd.is_root is True

    def test_root_plate_cpd_forward(self):
        g = _plate2()
        cpd = ParametricCPD(variable=g, parametrization={"probs": LearnablePrior(g.size)})
        B = 5
        out = cpd.root_params(B)
        assert out["probs"].shape == (B, 2)

    def test_nonroot_plate_cpd_forward(self):
        x = ConceptVariable("x", distribution=Delta, size=4)
        g = _plate3()
        cpd = ParametricCPD(variable=g, parametrization={"probs": nn.Sequential(nn.Linear(4, 3), nn.Sigmoid())}, parents=[x])
        B = 3
        out = cpd(parent_values={"x": torch.randn(B, 4)})
        assert out["probs"].shape == (B, 3)

    def test_cpd_select_plate_name(self):
        g = _plate2()
        cpd = ParametricCPD(variable=g, parametrization={"probs": LearnablePrior(g.size)})
        B = 4
        params = cpd.root_params(B)
        selected = cpd.select(params, "g")
        assert selected["probs"].shape == (B, 2)

    def test_cpd_select_member_name(self):
        g = _plate2()
        cpd = ParametricCPD(variable=g, parametrization={"probs": LearnablePrior(g.size)})
        B = 4
        params = cpd.root_params(B)
        selected = cpd.select(params, "m1")
        assert selected["probs"].shape == (B, 1)

    def test_cpd_select_both_members_different(self):
        g = _plate2()
        cpd = ParametricCPD(variable=g, parametrization={"probs": LearnablePrior(g.size)})
        B = 2
        params = cpd.root_params(B)
        m1 = cpd.select(params, "m1")["probs"]
        m2 = cpd.select(params, "m2")["probs"]
        assert m1.shape == (B, 1)
        assert m2.shape == (B, 1)
        # Together they reconstruct the full tensor
        assert torch.allclose(torch.cat([m1, m2], dim=1), params["probs"])


# ===========================================================================
# 4. clamp_members
# ===========================================================================

class TestClampMembers:
    def test_empty_observed_no_op(self):
        g = _plate2()
        cpd = ParametricCPD(variable=g, parametrization={"probs": LearnablePrior(g.size)})
        B = 3
        value = torch.rand(B, 2)
        clamped = cpd.clamp_members(value, {})
        assert torch.equal(clamped, value)

    def test_clamp_m1_replaces_column(self):
        g = _plate2()
        cpd = ParametricCPD(variable=g, parametrization={"probs": LearnablePrior(g.size)})
        B = 3
        value = torch.zeros(B, 2)
        obs_m1 = torch.ones(B, 1)
        clamped = cpd.clamp_members(value, {"m1": obs_m1})
        assert (clamped[:, 0:1] == 1.0).all()
        assert (clamped[:, 1:2] == 0.0).all()

    def test_clamp_m2_replaces_column(self):
        g = _plate2()
        cpd = ParametricCPD(variable=g, parametrization={"probs": LearnablePrior(g.size)})
        B = 3
        value = torch.zeros(B, 2)
        obs_m2 = torch.ones(B, 1)
        clamped = cpd.clamp_members(value, {"m2": obs_m2})
        assert (clamped[:, 1:2] == 1.0).all()
        assert (clamped[:, 0:1] == 0.0).all()

    def test_clamp_both_members(self):
        g = _plate2()
        cpd = ParametricCPD(variable=g, parametrization={"probs": LearnablePrior(g.size)})
        B = 2
        value = torch.zeros(B, 2)
        clamped = cpd.clamp_members(value, {
            "m1": torch.ones(B, 1),
            "m2": torch.full((B, 1), 0.5),
        })
        assert (clamped[:, 0:1] == 1.0).all()
        assert (clamped[:, 1:2] == 0.5).all()

    def test_clamp_does_not_mutate_original(self):
        g = _plate2()
        cpd = ParametricCPD(variable=g, parametrization={"probs": LearnablePrior(g.size)})
        B = 2
        value = torch.zeros(B, 2)
        value_copy = value.clone()
        cpd.clamp_members(value, {"m1": torch.ones(B, 1)})
        assert torch.equal(value, value_copy)


# ===========================================================================
# 5. BayesianNetwork with plate
# ===========================================================================

class TestBayesianNetworkPlate:
    def test_plate_variable_in_variables_dict(self):
        g = _plate2()
        m = _root_model_with_plate(g)
        assert "g" in m.variables

    def test_plate_factor_in_factors_dict(self):
        g = _plate2()
        m = _root_model_with_plate(g)
        assert "g" in m.factors

    def test_queryable_names_includes_members(self):
        g = _plate2()
        m = _root_model_with_plate(g)
        assert "m1" in m.queryable_names
        assert "m2" in m.queryable_names

    def test_resolve_member_returns_plate_variable(self):
        g = _plate2()
        m = _root_model_with_plate(g)
        assert m.resolve("m1") is g
        assert m.resolve("m2") is g

    def test_plate_as_parent(self):
        x = ConceptVariable("x", distribution=Delta, size=4)
        g = _plate2()
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
        cpd_g = ParametricCPD(variable=g, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        cpd_c = ParametricCPD(variable=c, parametrization=nn.Sequential(nn.Linear(2, 1), nn.Sigmoid()), parents=[g])
        m = BayesianNetwork(variables=[x, g, c], factors=[cpd_x, cpd_g, cpd_c])
        assert m is not None

    def test_member_as_parent(self):
        x = ConceptVariable("x", distribution=Delta, size=4)
        g = _plate2()
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
        cpd_g = ParametricCPD(variable=g, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        m1_handle = g.member("m1")
        cpd_c = ParametricCPD(variable=c, parametrization=nn.Sequential(nn.Linear(1, 1), nn.Sigmoid()), parents=[m1_handle])
        m = BayesianNetwork(variables=[x, g, c], factors=[cpd_x, cpd_g, cpd_c])
        assert m is not None


# ===========================================================================
# 6. Inference with plate variables
# ===========================================================================

class TestInferenceWithPlate:
    def _make_model(self):
        x = ConceptVariable("x", distribution=Delta, size=4)
        g = _plate3()
        cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
        cpd_g = ParametricCPD(variable=g, parametrization={"probs": nn.Sequential(nn.Linear(4, 3), nn.Sigmoid())}, parents=[x])
        return BayesianNetwork(variables=[x, g], factors=[cpd_x, cpd_g])

    def test_deterministic_query_plate_name(self):
        m = self._make_model()
        eng = DeterministicInference(m)
        B = 3
        out = eng.query(query=["g"], evidence={"x": torch.randn(B, 4)})
        assert out.params["g"]["probs"].shape == (B, 3)

    def test_deterministic_query_member_c1(self):
        m = self._make_model()
        eng = DeterministicInference(m)
        B = 3
        out = eng.query(query=["c1"], evidence={"x": torch.randn(B, 4)})
        assert out.params["c1"]["probs"].shape == (B, 1)

    def test_deterministic_query_all_members(self):
        m = self._make_model()
        eng = DeterministicInference(m)
        B = 3
        out = eng.query(query=["c1", "c2", "c3"], evidence={"x": torch.randn(B, 4)})
        for name in ["c1", "c2", "c3"]:
            assert name in out.params
            assert out.params[name]["probs"].shape == (B, 1)

    def test_ancestral_samples_plate(self):
        m = self._make_model()
        eng = AncestralInference(m)
        B = 2
        out = eng.query(query=["g"], evidence={"x": torch.randn(B, 4)})
        assert "g" in out.samples
        assert out.samples["g"].shape == (B, 3)

    def test_ancestral_samples_member(self):
        m = self._make_model()
        eng = AncestralInference(m)
        B = 2
        out = eng.query(query=["c1"], evidence={"x": torch.randn(B, 4)})
        assert "c1" in out.samples

    def test_member_evidence_partial_plate(self):
        m = self._make_model()
        eng = DeterministicInference(m)
        B = 2
        c1_obs = torch.ones(B, 1)
        out = eng.query(query=["c2", "c3"], evidence={"x": torch.randn(B, 4), "c1": c1_obs})
        assert "c2" in out.params
        assert "c3" in out.params

    def test_whole_plate_evidence_skips_cpd(self):
        m = self._make_model()
        eng = DeterministicInference(m)
        B = 2
        g_obs = torch.rand(B, 3)
        out = eng.query(query=[], evidence={"x": torch.randn(B, 4), "g": g_obs})
        # g is evidence; no params for it
        assert "g" not in out.params
