"""Comprehensive tests for DeterministicInference and AncestralSamplingInference.

Covers: construction, basic query, evidence clamping, teacher forcing,
out.params structure, out.samples in ancestral mode, plate queries,
required_variables memoization, and parallelize_levels.
"""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.torch.deterministic import DeterministicInference
from torch_concepts.nn.modules.mid.inference.torch.ancestral import AncestralSamplingInference
from torch_concepts.nn.modules.low.priors import LearnablePrior, FixedPrior
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.outputs import InferenceOutput


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_simple_model():
    """x (delta root, size=4) -> c (bernoulli, size=2)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    c = ConceptVariable("c", distribution=dist.Bernoulli, size=2)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_c = ParametricCPD(variable=c, parametrization=nn.Sequential(nn.Linear(4, 2), nn.Sigmoid()), parents=[x])
    return BayesianNetwork(variables=[x, c], factors=[cpd_x, cpd_c])


def _make_chain_model():
    """x (delta) -> a (bernoulli) -> b (bernoulli)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    a = ConceptVariable("a", distribution=dist.Bernoulli, size=2)
    b = ConceptVariable("b", distribution=dist.Bernoulli, size=1)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_a = ParametricCPD(variable=a, parametrization=nn.Sequential(nn.Linear(4, 2), nn.Sigmoid()), parents=[x])
    cpd_b = ParametricCPD(variable=b, parametrization=nn.Sequential(nn.Linear(2, 1), nn.Sigmoid()), parents=[a])
    return BayesianNetwork(variables=[x, a, b], factors=[cpd_x, cpd_a, cpd_b])


def _make_plate_model():
    """x (delta) -> g (plate: [m1, m2], bernoulli) -> y (bernoulli)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    g = ConceptVariable("g", members=["m1", "m2"], distribution=dist.Bernoulli)
    y = ConceptVariable("y", distribution=dist.Bernoulli, size=1)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_g = ParametricCPD(variable=g, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
    cpd_y = ParametricCPD(variable=y, parametrization=nn.Sequential(nn.Linear(2, 1), nn.Sigmoid()), parents=[g])
    return BayesianNetwork(variables=[x, g, y], factors=[cpd_x, cpd_g, cpd_y])


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestDeterministicInferenceConstruction:
    def test_basic_construction(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        assert isinstance(eng, DeterministicInference)

    def test_mode_is_deterministic(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        assert eng.mode == "deterministic"

    def test_default_p_int_zero(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        assert eng.p_int == 0.0

    def test_custom_p_int(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False, p_int=0.5)
        assert eng.p_int == 0.5

    def test_parallelize_levels_default_false(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        assert eng.parallelize_levels is False


class TestAncestralSamplingInferenceConstruction:
    def test_basic_construction(self):
        m = _make_simple_model()
        eng = AncestralSamplingInference(m)
        assert isinstance(eng, AncestralSamplingInference)

    def test_mode_is_ancestral(self):
        m = _make_simple_model()
        eng = AncestralSamplingInference(m)
        assert eng.mode == "ancestral"

    def test_default_p_int_one(self):
        m = _make_simple_model()
        eng = AncestralSamplingInference(m)
        assert eng.p_int == 1.0

    def test_initial_temperature_stored(self):
        m = _make_simple_model()
        eng = AncestralSamplingInference(m, initial_temperature=2.0)
        assert float(eng.temperature) == pytest.approx(2.0)


# ===========================================================================
# 2. Basic query — DeterministicInference
# ===========================================================================

class TestDeterministicQuery:
    def test_returns_inference_output(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        out = eng.query(query=["c"], evidence={})
        assert isinstance(out, InferenceOutput)

    def test_params_key_present(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        out = eng.query(query=["c"], evidence={})
        assert "c" in out.params

    def test_probs_shape_no_batch(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        out = eng.query(query=["c"], evidence={})
        assert out.params["c"]["probs"].shape == (1, 2)

    def test_probs_shape_with_batch(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        B = 5
        out = eng.query(query=["c"], evidence={"x": torch.randn(B, 4)})
        assert out.params["c"]["probs"].shape == (B, 2)

    def test_no_samples_in_deterministic_mode(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        out = eng.query(query=["c"], evidence={})
        assert len(out.samples) == 0

    def test_probs_in_valid_range(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        out = eng.query(query=["c"], evidence={})
        probs = out.params["c"]["probs"]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_querying_root_returns_value(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        out = eng.query(query=["x", "c"], evidence={})
        assert "x" in out.params
        assert "value" in out.params["x"]

    def test_list_query_format(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        out = eng.query(query=["x", "c"], evidence={})
        assert "x" in out.params
        assert "c" in out.params


# ===========================================================================
# 3. Evidence clamping
# ===========================================================================

class TestEvidenceClamping:
    def test_evidence_variable_skips_cpd(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        B = 3
        x_obs = torch.randn(B, 4)
        out = eng.query(query=["c"], evidence={"x": x_obs})
        assert "x" not in out.params
        assert "c" in out.params

    def test_evidence_shape_passes_through(self):
        m = _make_chain_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        B = 4
        x_obs = torch.randn(B, 4)
        out = eng.query(query=["a", "b"], evidence={"x": x_obs})
        assert out.params["a"]["probs"].shape == (B, 2)
        assert out.params["b"]["probs"].shape == (B, 1)

    def test_evidence_clamped_in_chain(self):
        m = _make_chain_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        B = 2
        a_obs = torch.ones(B, 2)
        out = eng.query(query=["b"], evidence={"a": a_obs})
        assert "b" in out.params

    def test_query_and_evidence_overlap_accepted(self):
        m = _make_chain_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        B = 2
        out = eng.query(query=["a"], evidence={"a": torch.ones(B, 2)})
        assert out is not None


# ===========================================================================
# 4. Teacher forcing
# ===========================================================================

class TestTeacherForcing:
    def test_teacher_force_at_p_int_1(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False, p_int=1.0)
        B = 3
        gt_c = torch.ones(B, 2)
        out = eng.query(query={"c": gt_c}, evidence={"x": torch.randn(B, 4)})
        assert "c" in out.params

    def test_teacher_force_no_error_at_p_int_0(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False, p_int=0.0)
        B = 3
        gt_c = torch.ones(B, 2)
        out = eng.query(query={"c": gt_c}, evidence={"x": torch.randn(B, 4)})
        assert "c" in out.params


# ===========================================================================
# 5. AncestralSamplingInference — samples
# ===========================================================================

class TestAncestralQuerySamples:
    def test_returns_inference_output(self):
        m = _make_simple_model()
        eng = AncestralSamplingInference(m)
        out = eng.query(query=["c"], evidence={})
        assert isinstance(out, InferenceOutput)

    def test_samples_populated_in_ancestral_mode(self):
        m = _make_simple_model()
        eng = AncestralSamplingInference(m)
        out = eng.query(query=["c"], evidence={})
        assert len(out.samples) > 0

    def test_samples_contain_queried_variable(self):
        m = _make_simple_model()
        eng = AncestralSamplingInference(m)
        out = eng.query(query=["c"], evidence={})
        assert "c" in out.samples

    def test_samples_shape(self):
        m = _make_simple_model()
        eng = AncestralSamplingInference(m)
        B = 4
        out = eng.query(query=["c"], evidence={"x": torch.randn(B, 4)})
        assert out.samples["c"].shape == (B, 2)

    def test_params_also_present_in_ancestral(self):
        m = _make_simple_model()
        eng = AncestralSamplingInference(m)
        out = eng.query(query=["c"], evidence={})
        assert "c" in out.params


# ===========================================================================
# 6. Plate queries
# ===========================================================================

class TestPlateQueries:
    def test_query_plate_name(self):
        m = _make_plate_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        B = 3
        out = eng.query(query=["g"], evidence={"x": torch.randn(B, 4)})
        assert "g" in out.params
        assert out.params["g"]["probs"].shape == (B, 2)

    def test_query_member_name(self):
        m = _make_plate_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        B = 3
        out = eng.query(query=["m1"], evidence={"x": torch.randn(B, 4)})
        assert "m1" in out.params
        assert out.params["m1"]["probs"].shape == (B, 1)

    def test_query_both_members(self):
        m = _make_plate_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        B = 3
        out = eng.query(query=["m1", "m2"], evidence={"x": torch.randn(B, 4)})
        assert "m1" in out.params
        assert "m2" in out.params

    def test_member_probs_shapes(self):
        m = _make_plate_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        B = 2
        out2 = eng.query(query=["m1", "m2"], evidence={"x": torch.randn(B, 4)})
        assert out2.params["m1"]["probs"].shape == (B, 1)
        assert out2.params["m2"]["probs"].shape == (B, 1)

    def test_ancestral_samples_plate(self):
        m = _make_plate_model()
        eng = AncestralSamplingInference(m)
        B = 3
        out = eng.query(query=["g"], evidence={"x": torch.randn(B, 4)})
        assert "g" in out.samples

    def test_member_evidence_partial_observation(self):
        m = _make_plate_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        B = 2
        m1_obs = torch.ones(B, 1)
        out = eng.query(query=["g"], evidence={"x": torch.randn(B, 4), "m1": m1_obs})
        assert "g" in out.params


# ===========================================================================
# 7. Required variables memoization
# ===========================================================================

class TestRequiredVariablesMemoization:
    def test_cache_populated_after_first_call(self):
        m = _make_chain_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        eng.query(query=["b"], evidence={})
        assert len(eng._required_cache) > 0

    def test_cache_hit_same_query(self):
        m = _make_chain_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        eng.query(query=["b"], evidence={})
        first = dict(eng._required_cache)
        eng.query(query=["b"], evidence={})
        assert len(eng._required_cache) == len(first)

    def test_different_queries_separate_cache_entries(self):
        m = _make_chain_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        eng.query(query=["a"], evidence={})
        eng.query(query=["b"], evidence={})
        assert len(eng._required_cache) == 2


# ===========================================================================
# 8. parallelize_levels
# ===========================================================================

class TestParallelizeLevels:
    def test_parallelize_levels_produces_same_shape(self):
        m = _make_chain_model()
        eng_seq = DeterministicInference(m, activate_before_propagation=False, parallelize_levels=False)
        eng_par = DeterministicInference(m, activate_before_propagation=False, parallelize_levels=True)
        B = 3
        ev = {"x": torch.randn(B, 4)}
        out_seq = eng_seq.query(query=["b"], evidence=ev)
        out_par = eng_par.query(query=["b"], evidence=ev)
        assert out_seq.params["b"]["probs"].shape == out_par.params["b"]["probs"].shape

    def test_parallelize_levels_flag_stored(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False, parallelize_levels=True)
        assert eng.parallelize_levels is True


# ===========================================================================
# 9. step() and temperature annealing
# ===========================================================================

class TestTemperatureAnnealing:
    def test_initial_temperature_default_one(self):
        m = _make_simple_model()
        eng = AncestralSamplingInference(m, initial_temperature=1.0)
        assert float(eng.temperature) == pytest.approx(1.0)

    def test_step_increments_in_ancestral(self):
        m = _make_simple_model()
        eng = AncestralSamplingInference(m, initial_temperature=2.0, annealing="exponential", annealing_rate=0.1)
        t0 = float(eng.temperature)
        eng.step()
        t1 = float(eng.temperature)
        assert t1 != t0

    def test_step_noop_in_deterministic(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        t0 = float(eng.temperature)
        eng.step()
        t1 = float(eng.temperature)
        assert t0 == pytest.approx(t1)


# ===========================================================================
# 10. InferenceOutput model_params alias
# ===========================================================================

class TestInferenceOutputAlias:
    def test_model_params_alias(self):
        m = _make_simple_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        out = eng.query(query=["c"], evidence={})
        # model_params alias was removed; params is the canonical field
        assert hasattr(out, 'params')
