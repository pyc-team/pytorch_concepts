"""Golden regression: the exact ``InferenceOutput`` contract of the forward
engines on a plate model.

Written *before* the plate-containment refactor of ``forward.py`` (see
PLATE_REFACTOR_INSTRUCTIONS.md §8.1): the current output — which params/samples
keys appear, and that member entries are *views* (not copies) of their plate's
tensor — is the spec the refactor must preserve byte-for-byte.
"""
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.variable import ConceptVariable
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.graph.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.torch.deterministic import DeterministicInference
from torch_concepts.nn.modules.mid.inference.torch.ancestral import AncestralSamplingInference
from torch_concepts.nn.modules.low.priors import FixedPrior
from torch_concepts.distributions import Delta


def _plate_model():
    """x (delta root) -> g (plate: [m1, m2], bernoulli) -> y (bernoulli)."""
    x = ConceptVariable("x", distribution=Delta, size=4)
    g = ConceptVariable("g", members=["m1", "m2"], distribution=dist.Bernoulli)
    y = ConceptVariable("y", distribution=dist.Bernoulli, size=1)
    cpd_x = ParametricCPD(variable=x, parametrization={"value": FixedPrior(torch.zeros(4))})
    cpd_g = ParametricCPD(variable=g, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
    cpd_y = ParametricCPD(variable=y, parametrization=nn.Sequential(nn.Linear(2, 1), nn.Sigmoid()), parents=[g])
    return BayesianNetwork(variables=[x, g, y], factors=[cpd_x, cpd_g, cpd_y])


def _is_view_of(sub: torch.Tensor, whole: torch.Tensor) -> bool:
    """Whether ``sub`` slices into ``whole`` (shares its storage, no copy)."""
    return sub.untyped_storage().data_ptr() == whole.untyped_storage().data_ptr()


class TestForwardGoldenContract:
    def test_deterministic_params_keys_are_queried_names_only(self):
        m = _plate_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        out = eng.query(query=["g", "y", "m1"], evidence={"x": torch.randn(3, 4)})
        assert set(out.params) == {"g", "y", "m1"}
        assert out.samples == {}  # deterministic mode never emits samples

    def test_deterministic_member_params_are_views_of_plate(self):
        m = _plate_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        out = eng.query(query=["g", "m1", "m2"], evidence={"x": torch.randn(2, 4)})
        g = out.params["g"]["probs"]
        assert out.params["m1"]["probs"].shape == (2, 1)
        assert _is_view_of(out.params["m1"]["probs"], g)
        assert _is_view_of(out.params["m2"]["probs"], g)
        assert torch.allclose(out.params["m1"]["probs"], g[:, 0:1])
        assert torch.allclose(out.params["m2"]["probs"], g[:, 1:2])

    def test_whole_plate_evidence_emits_no_params_for_plate(self):
        m = _plate_model()
        eng = DeterministicInference(m, activate_before_propagation=False)
        out = eng.query(query=["y"], evidence={"x": torch.randn(2, 4), "g": torch.rand(2, 2)})
        assert "g" not in out.params
        assert set(out.params) == {"y"}

    def test_ancestral_samples_expose_computed_vars_plus_queried_members(self):
        m = _plate_model()
        eng = AncestralSamplingInference(m)
        out = eng.query(query=["g", "y", "m1"], evidence={"x": torch.randn(3, 4)})
        # params: only queried names; samples: every computed var + queried member.
        assert set(out.params) == {"g", "y", "m1"}
        assert set(out.samples) == {"g", "y", "m1"}
        assert out.samples["g"].shape == (3, 2)
        assert out.samples["y"].shape == (3, 1)
        assert _is_view_of(out.samples["m1"], out.samples["g"])
        assert torch.allclose(out.samples["m1"], out.samples["g"][:, 0:1])
        assert _is_view_of(out.params["m1"]["probs"], out.params["g"]["probs"])

    def test_ancestral_samples_include_unqueried_ancestor(self):
        m = _plate_model()
        eng = AncestralSamplingInference(m)
        out = eng.query(query=["y"], evidence={"x": torch.randn(3, 4)})
        # y's ancestor g is sampled (hence in samples) but not queried (not in params).
        assert set(out.params) == {"y"}
        assert set(out.samples) == {"g", "y"}
