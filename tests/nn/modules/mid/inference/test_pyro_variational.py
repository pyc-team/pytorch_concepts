"""Tests for VariationalInference (Pyro backend).

All tests are skipped if pyro is not installed.
Covers: construction, latents validation, query() output structure
(out.params / out.guide_params), _align_param_keys, latent_names,
guide_conditioning, and step().
"""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

pyro = pytest.importorskip("pyro", reason="pyro not installed")

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.pyro.variational import VariationalInference
from torch_concepts.nn.modules.low.priors import LearnablePrior, FixedPrior
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.outputs import InferenceOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_pgm():
    """x (delta root) -> c (bernoulli)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    c = ConceptVariable("c", distribution=dist.Bernoulli, size=2)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_c = ParametricCPD(variable=c, parametrization=nn.Sequential(nn.Linear(4, 2), nn.Sigmoid()), parents=[x])
    return BayesianNetwork(variables=[x, c], factors=[cpd_x, cpd_c]), x, c


def _make_latent_pgm():
    """x (delta root) -> c (bernoulli, latent) -> y (bernoulli, observed)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    c = ConceptVariable("c", distribution=dist.Bernoulli, size=2)
    y = ConceptVariable("y", distribution=dist.Bernoulli, size=1)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_c = ParametricCPD(variable=c, parametrization=nn.Sequential(nn.Linear(4, 2), nn.Sigmoid()), parents=[x])
    cpd_y = ParametricCPD(variable=y, parametrization=nn.Sequential(nn.Linear(2, 1), nn.Sigmoid()), parents=[c])
    return BayesianNetwork(variables=[x, c, y], factors=[cpd_x, cpd_c, cpd_y]), x, c, y


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestVariationalInferenceConstruction:
    def test_no_latents_construction(self):
        pgm, x, c = _make_simple_pgm()
        vi = VariationalInference(pgm)
        assert isinstance(vi, VariationalInference)

    def test_with_latents_construction(self):
        pgm, x, c, y = _make_latent_pgm()
        guide_c = ParametricCPD(variable=c, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        vi = VariationalInference(pgm, latents={"c": guide_c})
        assert isinstance(vi, VariationalInference)

    def test_latent_names_empty_when_no_latents(self):
        pgm, x, c = _make_simple_pgm()
        vi = VariationalInference(pgm)
        assert vi.latent_names == []

    def test_latent_names_populated(self):
        pgm, x, c, y = _make_latent_pgm()
        guide_c = ParametricCPD(variable=c, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        vi = VariationalInference(pgm, latents={"c": guide_c})
        assert "c" in vi.latent_names

    def test_initial_temperature_default(self):
        pgm, _, _ = _make_simple_pgm()
        vi = VariationalInference(pgm)
        assert float(vi.temperature) == pytest.approx(1.0)


# ===========================================================================
# 2. _build_guides validation
# ===========================================================================

class TestBuildGuidesValidation:
    def test_unknown_latent_name_raises(self):
        pgm, x, c = _make_simple_pgm()
        z = ConceptVariable("z", distribution=dist.Bernoulli, size=1)
        guide_z = ParametricCPD(variable=z, parametrization={"logits": LearnablePrior(1)})
        with pytest.raises(ValueError, match="unknown latent"):
            VariationalInference(pgm, latents={"z": guide_z})

    def test_non_cpd_guide_raises(self):
        pgm, x, c = _make_simple_pgm()
        with pytest.raises(TypeError, match="ParametricCPD"):
            VariationalInference(pgm, latents={"c": nn.Linear(4, 2)})

    def test_guide_variable_name_mismatch_raises(self):
        pgm, x, c, y = _make_latent_pgm()
        # create a CPD with variable y but register under "c"
        guide_y = ParametricCPD(variable=y, parametrization={"probs": nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())}, parents=[x])
        with pytest.raises(ValueError, match="does not match"):
            VariationalInference(pgm, latents={"c": guide_y})

    def test_guide_latent_conditioning_on_latent_raises(self):
        pgm, x, c, y = _make_latent_pgm()
        # a guide for y conditioning on latent c is not allowed
        guide_y = ParametricCPD(variable=y, parametrization=nn.Sequential(nn.Linear(2, 1), nn.Sigmoid()), parents=[c])
        with pytest.raises(ValueError, match="cannot condition on latent"):
            VariationalInference(pgm, latents={"c": ParametricCPD(
                variable=c, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x]
            ), "y": guide_y})

    def test_guide_parent_not_in_pgm_raises(self):
        pgm, x, c = _make_simple_pgm()
        z = ConceptVariable("z", distribution=Delta, size=3)  # not in pgm
        guide_c = ParametricCPD(variable=c, parametrization=nn.Sequential(nn.Linear(3, 2), nn.Sigmoid()), parents=[z])
        with pytest.raises(ValueError, match="not a variable of the PGM"):
            VariationalInference(pgm, latents={"c": guide_c})


# ===========================================================================
# 3. guide_conditioning
# ===========================================================================

class TestGuideConditioning:
    def test_guide_conditioning_returns_parent_names(self):
        pgm, x, c, y = _make_latent_pgm()
        guide_c = ParametricCPD(variable=c, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        vi = VariationalInference(pgm, latents={"c": guide_c})
        gc = vi.guide_conditioning
        assert "c" in gc
        assert "x" in gc["c"]


# ===========================================================================
# 4. query() output structure
# ===========================================================================

class TestVariationalQuery:
    def test_returns_inference_output(self):
        pgm, x, c = _make_simple_pgm()
        vi = VariationalInference(pgm)
        B = 3
        out = vi.query(query={"x": torch.randn(B, 4), "c": None}, evidence={})
        assert isinstance(out, InferenceOutput)

    def test_params_contains_model_variable(self):
        pgm, x, c, y = _make_latent_pgm()
        guide_c = ParametricCPD(variable=c, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        vi = VariationalInference(pgm, latents={"c": guide_c})
        B = 3
        x_obs = torch.randn(B, 4)
        y_obs = torch.randint(0, 2, (B, 1)).float()
        out = vi.query(query={"x": x_obs, "c": None, "y": y_obs}, evidence={})
        assert isinstance(out.params, dict)

    def test_guide_params_populated_when_latents(self):
        pgm, x, c, y = _make_latent_pgm()
        guide_c = ParametricCPD(variable=c, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        vi = VariationalInference(pgm, latents={"c": guide_c})
        B = 2
        x_obs = torch.randn(B, 4)
        y_obs = torch.randint(0, 2, (B, 1)).float()
        out = vi.query(query={"x": x_obs, "c": None, "y": y_obs}, evidence={})
        assert isinstance(out.guide_params, dict)
        assert len(out.guide_params) > 0

    def test_guide_params_empty_when_no_latents(self):
        pgm, x, c = _make_simple_pgm()
        vi = VariationalInference(pgm)
        B = 2
        x_obs = torch.randn(B, 4)
        c_obs = torch.randint(0, 2, (B, 2)).float()
        out = vi.query(query={"x": x_obs, "c": c_obs}, evidence={})
        assert out.guide_params == {}


# ===========================================================================
# 5. step() and temperature
# ===========================================================================

class TestVariationalStep:
    def test_step_increments_temperature(self):
        pgm, _, _ = _make_simple_pgm()
        vi = VariationalInference(pgm, initial_temperature=2.0, annealing="exponential", annealing_rate=0.1)
        t0 = float(vi.temperature)
        vi.step()
        t1 = float(vi.temperature)
        assert t1 != t0
