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


# ===========================================================================
# Additional imports for rejection / importance sampling tests
# ===========================================================================

from torch_concepts.nn.modules.mid.inference.torch.rejection import RejectionSampling
from torch_concepts.nn.modules.mid.inference.torch.importance_sampling.importance_sampling import ImportanceSampling
from torch_concepts.nn.modules.mid.inference.torch.importance_sampling.mutilated_network import MutilatedNetworkProposal
from torch_concepts.nn.modules.mid.inference.torch.importance_sampling.base_proposal import BaseProposal, _stabilize_relaxed
from torch_concepts.distributions import Delta


# ---------------------------------------------------------------------------
# Bernoulli-only model factory (required by rejection / importance sampling)
# ---------------------------------------------------------------------------

def _make_bernoulli_model():
    """a (bernoulli root, size=1) -> b (bernoulli, size=1)."""
    a = ConceptVariable("a", distribution=dist.Bernoulli, size=1)
    b = ConceptVariable("b", distribution=dist.Bernoulli, size=1)
    cpd_a = ParametricCPD(variable=a, parametrization={"probs": FixedPrior(torch.tensor([0.5]))})
    cpd_b = ParametricCPD(variable=b, parametrization=nn.Sequential(nn.Linear(1, 1), nn.Sigmoid()), parents=[a])
    return BayesianNetwork(variables=[a, b], factors=[cpd_a, cpd_b])


def _make_mixed_model():
    """x (delta root, size=4) -> c (bernoulli, size=1).
    Delta is continuous — useful for testing that query/evidence must be discrete.
    """
    x = ConceptVariable("x", distribution=Delta, size=4)
    c = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_c = ParametricCPD(variable=c, parametrization=nn.Sequential(nn.Linear(4, 1), nn.Sigmoid()), parents=[x])
    return BayesianNetwork(variables=[x, c], factors=[cpd_x, cpd_c])


# ===========================================================================
# 11. RejectionSampling — Construction
# ===========================================================================

class TestRejectionSamplingConstruction:
    def test_basic_construction(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=50)
        assert isinstance(eng, RejectionSampling)

    def test_n_samples_stored(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=100)
        assert eng.n_samples == 100

    def test_warn_low_acceptance_stored(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=50, warn_low_acceptance=0.05)
        assert eng.warn_low_acceptance == pytest.approx(0.05)

    def test_invalid_n_samples_raises(self):
        m = _make_bernoulli_model()
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            RejectionSampling(m, n_samples=0)

    def test_negative_n_samples_raises(self):
        m = _make_bernoulli_model()
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            RejectionSampling(m, n_samples=-5)

    def test_repr_contains_name(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=50)
        r = repr(eng)
        assert "RejectionSampling" in r

    def test_repr_contains_n_samples(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=77)
        r = repr(eng)
        assert "77" in r


# ===========================================================================
# 12. RejectionSampling — Query
# ===========================================================================

class TestRejectionSamplingQuery:
    def test_returns_inference_output(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=50)
        out = eng.query(query={"b": torch.tensor([[1.0]])}, evidence={})
        assert isinstance(out, InferenceOutput)

    def test_probabilities_shape_single_row(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=50)
        out = eng.query(query={"b": torch.tensor([[1.0]])}, evidence={})
        assert out.probabilities.shape == (1,)

    def test_probabilities_shape_batch(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=50)
        B = 3
        out = eng.query(query={"b": torch.ones(B, 1)}, evidence={})
        assert out.probabilities.shape == (B,)

    def test_probabilities_in_valid_range(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=100)
        out = eng.query(query={"b": torch.tensor([[1.0]])}, evidence={})
        assert float(out.probabilities[0]) >= 0.0
        assert float(out.probabilities[0]) <= 1.0

    def test_query_with_evidence(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=100)
        # Query b=1 given a=1
        out = eng.query(
            query={"b": torch.tensor([[1.0]])},
            evidence={"a": torch.tensor([[1.0]])},
        )
        assert out.probabilities.shape == (1,)
        assert float(out.probabilities[0]) >= 0.0

    def test_query_evidence_none_treated_as_empty(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=50)
        # Passing evidence=None should not raise
        out = eng.query(query={"b": torch.tensor([[1.0]])}, evidence=None)
        assert out.probabilities.shape == (1,)

    def test_zero_prob_evidence_warns(self):
        """P(E=e) ≈ 0 when the evidence value can never be matched."""
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=30)
        # 999.0 can never match a Bernoulli sample (0 or 1)
        with pytest.warns(UserWarning, match=r"P\(E=e\) ≈ 0"):
            eng.query(
                query={"a": torch.tensor([[1.0]])},
                evidence={"b": torch.tensor([[999.0]])},
            )

    def test_low_acceptance_rate_warns(self):
        """Acceptance rate warning when very few samples match."""
        m = _make_bernoulli_model()
        # Use tiny n_samples and impossible query so almost no samples match
        eng = RejectionSampling(m, n_samples=10, warn_low_acceptance=1.0)
        with pytest.warns(UserWarning, match="low joint acceptance rate"):
            eng.query(
                query={"b": torch.tensor([[1.0]])},
                evidence={},
            )

    def test_draw_joint_indirectly(self):
        """_draw_joint is exercised whenever query() is called."""
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=50)
        out = eng.query(query={"a": torch.tensor([[1.0]])}, evidence={})
        assert out.probabilities.shape == (1,)

    def test_build_mask_indirectly(self):
        """_build_mask is exercised by multi-variable evidence."""
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=50)
        out = eng.query(
            query={"b": torch.ones(2, 1)},
            evidence={"a": torch.ones(2, 1)},
        )
        assert out.probabilities.shape == (2,)


# ===========================================================================
# 13. RejectionSampling — Validation
# ===========================================================================

class TestRejectionSamplingValidation:
    def test_non_dict_query_raises(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=10)
        with pytest.raises(ValueError, match="requires 'query' to be a dict"):
            eng.query(query=["b"], evidence={})

    def test_non_tensor_query_value_raises(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=10)
        with pytest.raises(ValueError, match="must be a Tensor"):
            eng.query(query={"b": [[1.0]]}, evidence={})

    def test_non_tensor_evidence_value_raises(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=10)
        with pytest.raises(ValueError, match="must be a Tensor"):
            eng.query(query={"b": torch.tensor([[1.0]])}, evidence={"a": 1.0})

    def test_missing_batch_dim_raises(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=10)
        with pytest.raises(ValueError, match="leading batch dimension"):
            eng.query(query={"b": torch.tensor([1.0])}, evidence={})

    def test_mismatched_batch_sizes_raises(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=10)
        with pytest.raises(ValueError, match="mismatched batch sizes"):
            eng.query(
                query={"b": torch.ones(2, 1)},
                evidence={"a": torch.ones(3, 1)},
            )

    def test_unknown_variable_raises(self):
        m = _make_bernoulli_model()
        eng = RejectionSampling(m, n_samples=10)
        with pytest.raises(ValueError, match="unknown variable names"):
            eng.query(query={"z_unknown": torch.tensor([[1.0]])}, evidence={})

    def test_continuous_query_variable_raises(self):
        """Delta is continuous — querying it should raise ValueError."""
        m = _make_mixed_model()
        eng = RejectionSampling(m, n_samples=10)
        with pytest.raises(ValueError, match="continuous"):
            eng.query(
                query={"x": torch.zeros(1, 4)},
                evidence={},
            )

    def test_continuous_evidence_variable_raises(self):
        """Providing a continuous variable as evidence should raise ValueError."""
        m = _make_mixed_model()
        eng = RejectionSampling(m, n_samples=10)
        with pytest.raises(ValueError, match="continuous"):
            eng.query(
                query={"c": torch.tensor([[1.0]])},
                evidence={"x": torch.zeros(1, 4)},
            )


# ===========================================================================
# 14. ImportanceSampling — Construction
# ===========================================================================

class TestImportanceSamplingConstruction:
    def test_basic_construction(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=50)
        assert isinstance(eng, ImportanceSampling)

    def test_n_samples_stored(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=75)
        assert eng.n_samples == 75

    def test_warn_low_ess_stored(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=50, warn_low_ess=0.05)
        assert eng.warn_low_ess == pytest.approx(0.05)

    def test_invalid_n_samples_raises(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            ImportanceSampling(m, proposal=proposal, n_samples=0)

    def test_invalid_proposal_type_raises(self):
        m = _make_bernoulli_model()
        with pytest.raises(TypeError, match="BaseProposal subclass"):
            ImportanceSampling(m, proposal="not_a_proposal", n_samples=50)

    def test_repr_contains_name(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=50)
        r = repr(eng)
        assert "ImportanceSampling" in r

    def test_repr_contains_n_samples(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=66)
        r = repr(eng)
        assert "66" in r

    def test_initial_temperature_stored(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=50, initial_temperature=2.0)
        assert eng.initial_temperature == pytest.approx(2.0)

    def test_temperature_buffer_initial(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=50, initial_temperature=3.0)
        assert float(eng.temperature) == pytest.approx(3.0)


# ===========================================================================
# 15. ImportanceSampling — Query
# ===========================================================================

class TestImportanceSamplingQuery:
    def test_returns_inference_output(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=50)
        out = eng.query(query={"b": torch.tensor([[1.0]])}, evidence={})
        assert isinstance(out, InferenceOutput)

    def test_probabilities_shape_single_row(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=50)
        out = eng.query(query={"b": torch.tensor([[1.0]])}, evidence={})
        assert out.probabilities.shape == (1,)

    def test_probabilities_shape_batch(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=50)
        B = 3
        out = eng.query(query={"b": torch.ones(B, 1)}, evidence={})
        assert out.probabilities.shape == (B,)

    def test_probabilities_in_valid_range(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=100)
        out = eng.query(query={"b": torch.tensor([[1.0]])}, evidence={})
        p = float(out.probabilities[0])
        assert p >= 0.0
        assert p <= 1.0

    def test_step_advances_temperature(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(
            m, proposal=proposal, n_samples=50,
            initial_temperature=2.0, annealing="exponential", annealing_rate=0.5,
        )
        t0 = float(eng.temperature)
        eng.step()
        t1 = float(eng.temperature)
        assert t1 != t0

    def test_step_increments_counter(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=50)
        assert eng._step == 0
        eng.step()
        assert eng._step == 1

    def test_query_evidence_none_treated_as_empty(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=50)
        out = eng.query(query={"b": torch.tensor([[1.0]])}, evidence=None)
        assert out.probabilities.shape == (1,)

    def test_warn_ess_low(self):
        """_warn_ess fires when the ESS threshold exceeds n_samples."""
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        # warn_low_ess=2.0 means threshold = 2 * n_samples which always exceeds
        # the actual ESS (at most n_samples), so the warning is always emitted.
        eng = ImportanceSampling(m, proposal=proposal, n_samples=10, warn_low_ess=2.0)
        with pytest.warns(UserWarning, match="low effective sample size"):
            eng.query(query={"b": torch.tensor([[1.0]])}, evidence={})


# ===========================================================================
# 16. ImportanceSampling — Validation
# ===========================================================================

class TestImportanceSamplingValidation:
    def test_empty_query_raises(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=10)
        with pytest.raises(ValueError, match="non-empty"):
            eng.query(query={}, evidence={})

    def test_non_dict_query_raises(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=10)
        with pytest.raises(ValueError, match="non-empty"):
            eng.query(query=None, evidence={})

    def test_non_tensor_query_value_raises(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=10)
        with pytest.raises(ValueError, match="must be a Tensor"):
            eng.query(query={"b": [[1.0]]}, evidence={})

    def test_non_tensor_evidence_value_raises(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=10)
        with pytest.raises(ValueError, match="must be a Tensor"):
            eng.query(
                query={"b": torch.tensor([[1.0]])},
                evidence={"a": 1.0},
            )

    def test_overlap_query_evidence_raises(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=10)
        with pytest.raises(ValueError, match="appear in both query and evidence"):
            eng.query(
                query={"b": torch.tensor([[1.0]])},
                evidence={"b": torch.tensor([[1.0]])},
            )

    def test_missing_batch_dim_raises(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=10)
        with pytest.raises(ValueError, match="leading batch dimension"):
            eng.query(query={"b": torch.tensor([1.0])}, evidence={})

    def test_mismatched_batch_sizes_raises(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=10)
        with pytest.raises(ValueError, match="mismatched batch sizes"):
            eng.query(
                query={"b": torch.ones(2, 1)},
                evidence={"a": torch.ones(3, 1)},
            )

    def test_unknown_variable_raises(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=10)
        with pytest.raises(ValueError, match="unknown variable names"):
            eng.query(
                query={"z_unknown": torch.tensor([[1.0]])},
                evidence={},
            )

    def test_continuous_query_variable_raises(self):
        """Querying a continuous (Delta) variable should raise ValueError."""
        m = _make_mixed_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=10)
        with pytest.raises(ValueError, match="continuous"):
            eng.query(
                query={"x": torch.zeros(1, 4)},
                evidence={},
            )


# ===========================================================================
# 17. MutilatedNetworkProposal
# ===========================================================================

class TestMutilatedNetworkProposal:
    def test_construction(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        assert isinstance(proposal, MutilatedNetworkProposal)
        assert isinstance(proposal, BaseProposal)

    def test_pgm_property(self):
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        assert proposal.pgm is m

    def test_propose_root_variable(self):
        """propose() for a root variable broadcasts params to batch_size."""
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        var_a = m.variables["a"]
        cpd_a = m.factors["a"]
        batch_size = 4
        temperature = torch.tensor(1.0)
        params = proposal.propose(
            variable=var_a,
            parent_values={},
            evidence={},
            batch_size=batch_size,
            temperature=temperature,
            layer_kwargs={},
        )
        assert "probs" in params
        assert params["probs"].shape[0] == batch_size

    def test_propose_non_root_variable(self):
        """propose() for a non-root variable uses parent values."""
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        var_b = m.variables["b"]
        batch_size = 3
        temperature = torch.tensor(1.0)
        # Provide parent values for a (size=1)
        parent_vals = {"a": torch.full((batch_size, 1), 0.5)}
        params = proposal.propose(
            variable=var_b,
            parent_values=parent_vals,
            evidence={},
            batch_size=batch_size,
            temperature=temperature,
            layer_kwargs={},
        )
        assert "probs" in params
        assert params["probs"].shape[0] == batch_size

    def test_sample_via_importance_sampling(self):
        """BaseProposal.sample() is exercised through an IS query call."""
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        eng = ImportanceSampling(m, proposal=proposal, n_samples=20)
        out = eng.query(query={"b": torch.tensor([[1.0]])}, evidence={})
        assert out.probabilities.shape == (1,)

    def test_no_learnable_parameters(self):
        """MutilatedNetworkProposal has no parameters of its own."""
        m = _make_bernoulli_model()
        proposal = MutilatedNetworkProposal(m)
        assert list(proposal.parameters()) == []


# ===========================================================================
# 18. _stabilize_relaxed
# ===========================================================================

class TestStabilizeRelaxed:
    def test_bernoulli_clamps_to_eps(self):
        """Values at 0 / 1 should be nudged inward."""
        a = ConceptVariable("a", distribution=dist.Bernoulli, size=2)
        sample = torch.tensor([[0.0, 1.0]])  # at the boundary
        out = _stabilize_relaxed(a, sample, eps=1e-6)
        assert (out > 0).all()
        assert (out < 1).all()

    def test_bernoulli_interior_unchanged(self):
        """Values strictly inside (0, 1) should not move."""
        a = ConceptVariable("a", distribution=dist.Bernoulli, size=2)
        sample = torch.tensor([[0.3, 0.7]])
        out = _stabilize_relaxed(a, sample, eps=1e-6)
        assert torch.allclose(out, sample)

    def test_onehot_clamps_and_renorms(self):
        """A one-hot sample with a near-zero entry should be renormalised."""
        a = ConceptVariable("a", distribution=dist.OneHotCategorical, size=3)
        sample = torch.tensor([[0.0, 0.0, 1.0]])  # two zeros
        out = _stabilize_relaxed(a, sample, eps=1e-6)
        # All entries must be positive
        assert (out > 0).all()
        # Must sum to 1 along the last dim
        assert torch.allclose(out.sum(dim=-1), torch.ones(out.shape[:-1]))

    def test_continuous_passthrough(self):
        """A Delta (continuous) variable should pass through untouched."""
        x = ConceptVariable("x", distribution=Delta, size=3)
        sample = torch.tensor([[1.5, -2.0, 0.0]])
        out = _stabilize_relaxed(x, sample)
        assert torch.allclose(out, sample)


# ===========================================================================
# 11. BaseInference — direct tests
# ===========================================================================

from torch_concepts.nn.modules.mid.inference.base import BaseInference


class _ConcreteInference(BaseInference):
    """Minimal concrete subclass for testing BaseInference directly."""
    def query(self, query, evidence):
        return InferenceOutput()


class TestBaseInferenceDirect:
    def test_construction_stores_pgm(self):
        m = _make_simple_model()
        eng = _ConcreteInference(m)
        assert eng.pgm is m

    def test_validate_containers_unknown_query(self):
        m = _make_simple_model()
        eng = _ConcreteInference(m)
        with pytest.raises(ValueError, match="unknown query names"):
            eng._validate_containers({"unknown_var": torch.randn(2, 4)}, {})

    def test_validate_containers_unknown_evidence(self):
        m = _make_simple_model()
        eng = _ConcreteInference(m)
        with pytest.raises(ValueError, match="unknown evidence names"):
            eng._validate_containers({}, {"unknown_var": torch.randn(2, 4)})

    def test_validate_containers_non_tensor_evidence(self):
        m = _make_simple_model()
        eng = _ConcreteInference(m)
        with pytest.raises(ValueError, match="must be a Tensor"):
            eng._validate_containers({}, {"x": [1.0, 2.0]})

    def test_validate_containers_nothing_to_do(self):
        m = _make_simple_model()
        eng = _ConcreteInference(m)
        with pytest.raises(ValueError, match="nothing to do"):
            eng._validate_containers({}, {})

    def test_validate_containers_mismatched_batch_sizes(self):
        m = _make_chain_model()
        eng = _ConcreteInference(m)
        with pytest.raises(ValueError, match="mismatched batch sizes"):
            eng._validate_containers(
                {"a": torch.randn(2, 2)},
                {"x": torch.randn(3, 4)}
            )

    def test_normalize_query_list(self):
        result = BaseInference._normalize_query(["x", "c"])
        assert result == {"x": None, "c": None}

    def test_normalize_query_dict(self):
        d = {"x": torch.randn(2, 4), "c": None}
        result = BaseInference._normalize_query(d)
        assert result is d

    def test_call_delegates_to_query(self):
        m = _make_simple_model()
        eng = _ConcreteInference(m)
        out = eng(query=["c"], evidence={})
        assert isinstance(out, InferenceOutput)

    def test_format_repr(self):
        m = _make_simple_model()
        eng = _ConcreteInference(m)
        r = repr(eng)
        assert "_ConcreteInference" in r

    def test_format_repr_with_fields(self):
        m = _make_simple_model()
        eng = _ConcreteInference(m)
        r = eng._format_repr(n_samples=100, annealing="constant")
        assert "n_samples=100" in r
        assert "annealing='constant'" in r

    def test_warn_on_root_with_input_requirement(self):
        """BaseInference warns when a root CPD's parametrization takes inputs."""
        import warnings
        # _make_simple_model has x with FixedPrior(no input) — no warning
        m = _make_simple_model()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ConcreteInference(m)
        # FixedPrior forward() takes no args → no warning expected
        assert not any("root variables" in str(wi.message) for wi in w)


# ===========================================================================
# 12. make_temperature_schedule and build_distribution utility tests
# ===========================================================================

from torch_concepts.nn.modules.mid.inference.utils import make_temperature_schedule
from torch_concepts.nn.modules.mid.inference.torch.utils import (
    build_relaxed_distribution,
    propagated_value,
)


class TestMakeTemperatureSchedule:
    def test_callable_schedule(self):
        """A callable is returned unchanged."""
        fn = lambda step: 1.0 / (1 + step)
        schedule = make_temperature_schedule(1.0, fn, 0.0)
        assert schedule is fn

    def test_linear_schedule_decreases(self):
        schedule = make_temperature_schedule(2.0, "linear", 0.1)
        assert schedule(0) == pytest.approx(2.0)
        assert schedule(5) < 2.0

    def test_linear_schedule_floor(self):
        """Linear schedule floors at 1e-6."""
        schedule = make_temperature_schedule(1.0, "linear", 10.0)
        assert schedule(100) == pytest.approx(1e-6)

    def test_unknown_annealing_raises(self):
        with pytest.raises(ValueError, match="Unknown annealing schedule"):
            make_temperature_schedule(1.0, "cosine", 0.1)

    def test_exponential_schedule_returns_callable(self):
        schedule = make_temperature_schedule(1.0, "exponential", 0.1)
        assert callable(schedule)
        assert schedule(0) == pytest.approx(1.0)
        assert schedule(10) < 1.0

    def test_constant_schedule(self):
        schedule = make_temperature_schedule(0.5, "constant", 0.0)
        assert schedule(0) == pytest.approx(0.5)
        assert schedule(100) == pytest.approx(0.5)


# ===========================================================================
# 19. build_relaxed_distribution — relaxed-family variables
# ===========================================================================

class TestBuildRelaxedDistribution:
    _T = torch.tensor(0.5)

    def _var(self, family, size=2, members=None):
        return ConceptVariable("v", distribution=family, size=size, members=members)

    def test_bernoulli_returns_independent(self):
        v = self._var(dist.Bernoulli)
        d = build_relaxed_distribution(v, {"probs": torch.full((1, 2), 0.5)}, self._T)
        assert isinstance(d, dist.Independent)

    def test_relaxed_bernoulli_declared_returns_independent(self):
        """Variable declared as RelaxedBernoulli should yield the same surrogate."""
        v = self._var(dist.RelaxedBernoulli)
        d = build_relaxed_distribution(v, {"probs": torch.full((1, 2), 0.5)}, self._T)
        assert isinstance(d, dist.Independent)

    def test_onehot_categorical_returns_relaxed_onehot(self):
        v = self._var(dist.OneHotCategorical, size=3)
        d = build_relaxed_distribution(v, {"probs": torch.ones(1, 3) / 3}, self._T)
        assert isinstance(d, dist.RelaxedOneHotCategorical)

    def test_relaxed_onehot_declared_returns_relaxed_onehot(self):
        """Variable declared as RelaxedOneHotCategorical should resolve to the same family."""
        v = self._var(dist.RelaxedOneHotCategorical, size=3)
        d = build_relaxed_distribution(v, {"probs": torch.ones(1, 3) / 3}, self._T)
        assert isinstance(d, dist.RelaxedOneHotCategorical)

    def test_categorical_raises(self):
        v = self._var(dist.Categorical, size=3)
        with pytest.raises(ValueError, match="OneHotCategorical"):
            build_relaxed_distribution(v, {"probs": torch.ones(1, 3) / 3}, self._T)

    def test_rsample_differentiable_bernoulli(self):
        v = self._var(dist.Bernoulli)
        probs = torch.full((1, 2), 0.5, requires_grad=True)
        d = build_relaxed_distribution(v, {"probs": probs}, self._T)
        s = d.rsample()
        s.sum().backward()
        assert probs.grad is not None

    def test_rsample_differentiable_relaxed_bernoulli(self):
        v = self._var(dist.RelaxedBernoulli)
        probs = torch.full((1, 2), 0.5, requires_grad=True)
        d = build_relaxed_distribution(v, {"probs": probs}, self._T)
        s = d.rsample()
        s.sum().backward()
        assert probs.grad is not None


# ===========================================================================
# 20. propagated_value — new relaxed entries in _PRIMARY_PARAM
# ===========================================================================

class TestPropagatedValue:
    def test_bernoulli_probs(self):
        p = torch.tensor([[0.3, 0.7]])
        assert torch.allclose(propagated_value(dist.Bernoulli, {"probs": p}), p)

    def test_bernoulli_logits_fallback(self):
        lg = torch.zeros(1, 2)
        assert torch.allclose(propagated_value(dist.Bernoulli, {"logits": lg}), lg)

    def test_relaxed_bernoulli_probs(self):
        """RelaxedBernoulli is in _PRIMARY_PARAM — should resolve to probs."""
        p = torch.tensor([[0.4, 0.6]])
        assert torch.allclose(propagated_value(dist.RelaxedBernoulli, {"probs": p}), p)

    def test_relaxed_onehot_probs(self):
        """RelaxedOneHotCategorical is in _PRIMARY_PARAM — should resolve to probs."""
        p = torch.ones(1, 3) / 3
        assert torch.allclose(propagated_value(dist.RelaxedOneHotCategorical, {"probs": p}), p)

    def test_normal_loc(self):
        loc = torch.zeros(1, 2)
        assert torch.allclose(propagated_value(dist.Normal, {"loc": loc}), loc)

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported distribution"):
            propagated_value(dist.Poisson, {"rate": torch.ones(1, 1)})
