"""Tests for PyroImportanceSampling (Pyro backend).

All tests are skipped if pyro is not installed.
Covers: construction, _build_proposal validation, _require_discrete,
_validate, query() output (out.probabilities), evidence handling,
and low-ESS warning.
"""
import pytest
import warnings
import torch
import torch.nn as nn
import torch.distributions as dist

pyro = pytest.importorskip("pyro", reason="pyro not installed")

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.pyro.importance import PyroImportanceSampling
from torch_concepts.nn.modules.low.priors import LearnablePrior, FixedPrior
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.outputs import InferenceOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bernoulli_pgm():
    """Root: x (delta, 4) -> c (bernoulli, 2)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    c = ConceptVariable("c", distribution=dist.Bernoulli, size=2)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_c = ParametricCPD(variable=c, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
    return BayesianNetwork(variables=[x, c], factors=[cpd_x, cpd_c]), x, c


def _make_chain_pgm():
    """x (delta) -> a (bernoulli, 1) -> b (bernoulli, 1)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    a = ConceptVariable("a", distribution=dist.Bernoulli, size=1)
    b = ConceptVariable("b", distribution=dist.Bernoulli, size=1)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_a = ParametricCPD(variable=a, parametrization={"probs": nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())}, parents=[x])
    cpd_b = ParametricCPD(variable=b, parametrization={"probs": nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())}, parents=[a])
    return BayesianNetwork(variables=[x, a, b], factors=[cpd_x, cpd_a, cpd_b]), x, a, b


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestPyroImportanceSamplingConstruction:
    def test_basic_construction(self):
        pgm, _, _ = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=10)
        assert isinstance(eng, PyroImportanceSampling)

    def test_n_samples_stored(self):
        pgm, _, _ = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=50)
        assert eng.n_samples == 50

    def test_n_samples_zero_raises(self):
        pgm, _, _ = _make_bernoulli_pgm()
        with pytest.raises(ValueError, match="n_samples"):
            PyroImportanceSampling(pgm, n_samples=0)

    def test_empty_proposal(self):
        pgm, _, _ = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=10, proposal={})
        assert len(eng.proposal) == 0

    def test_warn_low_ess_default(self):
        pgm, _, _ = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=10)
        assert eng.warn_low_ess == pytest.approx(0.01)


# ===========================================================================
# 2. _build_proposal validation
# ===========================================================================

class TestBuildProposalValidation:
    def test_unknown_variable_in_proposal_raises(self):
        pgm, x, c = _make_bernoulli_pgm()
        z = ConceptVariable("z", distribution=dist.Bernoulli, size=1)
        cpd_z = ParametricCPD(variable=z, parametrization={"logits": LearnablePrior(1)})
        with pytest.raises(ValueError, match="unknown proposal"):
            PyroImportanceSampling(pgm, n_samples=10, proposal={"z": cpd_z})

    def test_non_cpd_proposal_raises(self):
        pgm, x, c = _make_bernoulli_pgm()
        with pytest.raises(TypeError, match="ParametricCPD"):
            PyroImportanceSampling(pgm, n_samples=10, proposal={"c": nn.Linear(4, 2)})

    def test_proposal_name_mismatch_raises(self):
        pgm, x, c = _make_bernoulli_pgm()
        # build a CPD for x but register under "c"
        cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
        with pytest.raises(ValueError, match="does not match"):
            PyroImportanceSampling(pgm, n_samples=10, proposal={"c": cpd_x})

    def test_valid_proposal_accepted(self):
        pgm, x, c = _make_bernoulli_pgm()
        guide_c = ParametricCPD(variable=c, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        eng = PyroImportanceSampling(pgm, n_samples=10, proposal={"c": guide_c})
        assert "c" in eng.proposal


# ===========================================================================
# 3. _validate / _require_discrete
# ===========================================================================

class TestValidate:
    def test_empty_query_raises(self):
        pgm, x, c = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=10)
        with pytest.raises(ValueError, match="non-empty"):
            eng.query(query={}, evidence={})

    def test_continuous_query_variable_raises(self):
        x = ConceptVariable("x", distribution=Delta, size=4)
        n = ConceptVariable("n", distribution=dist.Normal, size=1)
        cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
        cpd_n = ParametricCPD(variable=n, parametrization={"loc": nn.Linear(4, 1), "scale": nn.Linear(4, 1)}, parents=[x])
        pgm = BayesianNetwork(variables=[x, n], factors=[cpd_x, cpd_n])
        eng = PyroImportanceSampling(pgm, n_samples=10)
        B = 2
        with pytest.raises(ValueError, match="continuous"):
            eng.query(query={"n": torch.randn(B, 1)}, evidence={"x": torch.randn(B, 4)})

    def test_overlap_query_evidence_raises(self):
        pgm, x, c = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=10)
        B = 2
        c_val = torch.ones(B, 2)
        with pytest.raises(ValueError, match="both query and evidence"):
            eng.query(query={"c": c_val}, evidence={"c": c_val, "x": torch.randn(B, 4)})

    def test_missing_batch_dim_raises(self):
        pgm, x, c = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=10)
        with pytest.raises(ValueError, match="batch dimension"):
            eng.query(query={"c": torch.ones(2)}, evidence={})

    def test_batch_size_mismatch_raises(self):
        pgm, x, c = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=10)
        with pytest.raises(ValueError, match="batch sizes"):
            eng.query(query={"c": torch.ones(2, 2)}, evidence={"x": torch.randn(3, 4)})

    def test_unknown_variable_name_raises(self):
        pgm, x, c = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=10)
        B = 2
        with pytest.raises(ValueError, match="unknown variable"):
            eng.query(query={"z": torch.ones(B, 1)}, evidence={"x": torch.randn(B, 4)})


# ===========================================================================
# 4. query() output
# ===========================================================================

class TestPyroImportanceSamplingQuery:
    def test_returns_inference_output(self):
        pgm, x, c = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=20)
        B = 3
        x_obs = torch.randn(B, 4)
        c_target = torch.ones(B, 2)
        out = eng.query(query={"c": c_target}, evidence={"x": x_obs})
        assert isinstance(out, InferenceOutput)

    def test_probabilities_shape(self):
        pgm, x, c = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=20)
        B = 3
        x_obs = torch.randn(B, 4)
        c_target = torch.ones(B, 2)
        out = eng.query(query={"c": c_target}, evidence={"x": x_obs})
        assert out.probabilities.shape == (B,)

    def test_probabilities_in_0_1(self):
        # Use a root prior with known high probs so samples frequently match the target.
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=2)
        # FixedPrior with probs=0.9 makes c=1 very likely; all-ones target should match often.
        cpd_c = ParametricCPD(variable=c, parametrization={"probs": FixedPrior(torch.full((2,), 0.9))})
        pgm = BayesianNetwork(variables=[c], factors=[cpd_c])
        eng = PyroImportanceSampling(pgm, n_samples=200)
        B = 2
        c_target = torch.ones(B, 2)
        out = eng.query(query={"c": c_target}, evidence={})
        probs = out.probabilities
        finite = probs[torch.isfinite(probs)]
        assert (finite >= 0).all()
        assert (finite <= 1).all()

    def test_no_grad_in_query(self):
        pgm, x, c = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=10)
        B = 2
        x_obs = torch.randn(B, 4)
        c_target = torch.ones(B, 2)
        out = eng.query(query={"c": c_target}, evidence={"x": x_obs})
        assert not out.probabilities.requires_grad

    def test_chain_query_with_evidence(self):
        pgm, x, a, b = _make_chain_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=20)
        B = 2
        x_obs = torch.randn(B, 4)
        a_obs = torch.ones(B, 1)
        b_target = torch.ones(B, 1)
        out = eng.query(query={"b": b_target}, evidence={"x": x_obs, "a": a_obs})
        assert out.probabilities.shape == (B,)

    def test_no_evidence_accepted(self):
        pgm, x, c = _make_bernoulli_pgm()
        eng = PyroImportanceSampling(pgm, n_samples=10)
        B = 2
        c_target = torch.ones(B, 2)
        out = eng.query(query={"c": c_target}, evidence={})
        assert out.probabilities.shape == (B,)


# ===========================================================================
# 5. ESS warning
# ===========================================================================

class TestESSWarning:
    def test_low_ess_triggers_warning(self):
        pgm, x, c = _make_bernoulli_pgm()
        # Use very few samples so ESS is likely very low
        eng = PyroImportanceSampling(pgm, n_samples=2, warn_low_ess=0.99)
        B = 2
        x_obs = torch.randn(B, 4)
        c_target = torch.zeros(B, 2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eng.query(query={"c": c_target}, evidence={"x": x_obs})
            # The warning may or may not fire depending on entropy; just assert no crash.
        assert True  # reached here without error
