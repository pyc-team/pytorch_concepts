"""Tests for VariationalInference (Pyro backend).

Covers: construction, latents validation, query() output structure
(out.params / out.guide_params), latent_names,
guide_conditioning, step(), categorical latents, and engines sharing a PGM.
"""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist
from pyro.distributions import RelaxedOneHotCategoricalStraightThrough

from torch_concepts.nn.modules.mid.variable import ConceptVariable
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
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


#: Probabilities of the categorical latent's prior, per member.
PRIOR = torch.tensor([0.5, 0.3, 0.2])


def _make_categorical_latent_pgm(family=dist.OneHotCategorical, members=None):
    """x (delta root) -> z (categorical, latent) -> y (bernoulli, observed).

    ``z``'s prior is written as ``logits`` and is ``PRIOR`` for every member, so
    a reported parameter can be checked against a known distribution.
    """
    n_members = 1 if members is None else len(members)
    x = ConceptVariable("x", distribution=Delta, size=4)
    z = ConceptVariable("z", distribution=family, size=3, members=members)
    y = ConceptVariable("y", distribution=dist.Bernoulli, size=1)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_z = ParametricCPD(variable=z, parametrization={"logits": FixedPrior(PRIOR.log().repeat(n_members))})
    cpd_y = ParametricCPD(variable=y, parametrization=nn.Sequential(nn.Linear(3 * n_members, 1), nn.Sigmoid()), parents=[z])
    return BayesianNetwork(variables=[x, z, y], factors=[cpd_x, cpd_z, cpd_y]), x, z


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
        vi.train()
        t0 = float(vi.temperature)
        vi.temperature_step()
        t1 = float(vi.temperature)
        assert t1 != t0

    def test_step_is_a_no_op_outside_training(self):
        """Evaluation must read the temperature training left behind."""
        pgm, _, _ = _make_simple_pgm()
        vi = VariationalInference(pgm, initial_temperature=2.0, annealing="exponential", annealing_rate=0.1)
        vi.eval()
        t0 = float(vi.temperature)
        vi.temperature_step()
        assert float(vi.temperature) == t0

    def test_annealing_stops_at_the_final_temperature(self):
        pgm, _, _ = _make_simple_pgm()
        vi = VariationalInference(
            pgm, initial_temperature=2.0, annealing="exponential",
            annealing_rate=0.5, final_temperature=0.1,
        )
        vi.train()
        for _ in range(500):
            vi.temperature_step()
        assert float(vi.temperature) == pytest.approx(0.1)


# ===========================================================================
# 6. Categorical latents
# ===========================================================================

class TestCategoricalLatent:
    """A sampled categorical site reports its parameters under the CPD's key.

    Pyro rebuilds a relaxed site's distribution while tracing, so the key the
    CPD was written with is lost. A CPD written with ``logits`` must get a
    categorical's logits back (a softmax recovers its probabilities), not a
    Bernoulli's ``log p - log(1 - p)``, which names a different distribution.
    """

    @pytest.mark.parametrize("family", [
        dist.OneHotCategorical,
        dist.RelaxedOneHotCategorical,
        RelaxedOneHotCategoricalStraightThrough,
    ])
    def test_logits_name_the_cpds_distribution(self, family):
        pgm, x, z = _make_categorical_latent_pgm(family)
        head = nn.Linear(4, 3)
        guide_z = ParametricCPD(variable=z, parametrization={"logits": head}, parents=[x])
        vi = VariationalInference(pgm, latents={"z": guide_z})
        x_obs = torch.randn(5, 4)
        out = vi.query(query={"x": x_obs, "z": None, "y": torch.ones(5, 1)}, evidence={})

        model_probs = torch.softmax(torch.as_tensor(out.params["z"]["logits"]), -1)
        guide_probs = torch.softmax(torch.as_tensor(out.guide_params["z"]["logits"]), -1)
        assert torch.allclose(model_probs, PRIOR.expand(5, 3), atol=1e-6)
        assert torch.allclose(guide_probs, torch.softmax(head(x_obs), -1), atol=1e-6)

    def test_root_guide(self):
        pgm, x, z = _make_categorical_latent_pgm()
        prior = LearnablePrior(3)
        vi = VariationalInference(pgm, latents={"z": ParametricCPD(variable=z, parametrization={"logits": prior})})
        out = vi.query(query={"x": torch.randn(5, 4), "z": None, "y": torch.ones(5, 1)}, evidence={})
        guide_probs = torch.softmax(torch.as_tensor(out.guide_params["z"]["logits"]), -1)
        assert torch.allclose(guide_probs, torch.softmax(prior(), -1).expand(5, 3), atol=1e-6)

    def test_plate_normalises_each_member(self):
        pgm, x, z = _make_categorical_latent_pgm(members=["z0", "z1"])
        head = nn.Linear(4, 6)
        guide_z = ParametricCPD(variable=z, parametrization={"logits": head}, parents=[x])
        vi = VariationalInference(pgm, latents={"z": guide_z})
        x_obs = torch.randn(5, 4)
        out = vi.query(query={"x": x_obs, "z": None, "y": torch.ones(5, 1)}, evidence={})

        def per_member(logits):
            return torch.softmax(torch.as_tensor(logits).reshape(5, 2, 3), -1)

        assert torch.allclose(per_member(out.params["z"]["logits"]), PRIOR.expand(5, 2, 3), atol=1e-6)
        assert torch.allclose(per_member(out.guide_params["z"]["logits"]), per_member(head(x_obs)), atol=1e-6)

    def test_bernoulli_logits_are_unchanged(self):
        pgm, x, c, y = _make_latent_pgm()
        head = nn.Linear(4, 2)
        vi = VariationalInference(pgm, latents={"c": ParametricCPD(variable=c, parametrization={"logits": head}, parents=[x])})
        x_obs = torch.randn(5, 4)
        out = vi.query(query={"x": x_obs, "c": None, "y": torch.ones(5, 1)}, evidence={})
        assert torch.allclose(torch.as_tensor(out.guide_params["c"]["logits"]), head(x_obs), atol=1e-6)


# ===========================================================================
# 7. Engines sharing a PGM
# ===========================================================================

class TestSharedPGM:
    """Guides live on the PGM, so a train and an eval engine share them."""

    @staticmethod
    def _guide(x, c):
        return ParametricCPD(variable=c, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])

    @staticmethod
    def _query(vi):
        return vi.query(query={"x": torch.randn(3, 4), "c": None, "y": torch.ones(3, 1)}, evidence={})

    def test_an_engine_without_latents_keeps_the_others_guides(self):
        pgm, x, c, y = _make_latent_pgm()
        train = VariationalInference(pgm, latents={"c": self._guide(x, c)})
        VariationalInference(pgm)
        assert set(self._query(train).guide_params["c"]) == {"probs"}

    def test_an_engine_without_latents_runs_no_guide(self):
        pgm, x, c, y = _make_latent_pgm()
        VariationalInference(pgm, latents={"c": self._guide(x, c)})
        assert self._query(VariationalInference(pgm)).guide_params == {}

    def test_the_same_guide_can_be_shared(self):
        pgm, x, c, y = _make_latent_pgm()
        guide_c = self._guide(x, c)
        VariationalInference(pgm, latents={"c": guide_c})
        VariationalInference(pgm, latents={"c": guide_c})
        assert pgm.guides["c"] is guide_c

    def test_a_different_guide_for_the_same_latent_is_rejected(self):
        pgm, x, c, y = _make_latent_pgm()
        VariationalInference(pgm, latents={"c": self._guide(x, c)})
        with pytest.raises(ValueError, match="already has a different guide"):
            VariationalInference(pgm, latents={"c": self._guide(x, c)})
