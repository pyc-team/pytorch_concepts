"""Plate-member support in the Pyro VariationalInference engine: member evidence
is value forcing that reaches a downstream task (fixing leak 10), and a guide may
condition on a member-handle parent of an observed plate (new capability from
§6.1). See PLATE_REFACTOR_INSTRUCTIONS.md §6.1, §6.2, §8.3."""
import warnings

import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

pyro = pytest.importorskip("pyro", reason="pyro not installed")

from torch_concepts import seed_everything
from torch_concepts.nn.modules.mid.models.variable import ConceptVariable, EmbeddingVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.pyro.variational import VariationalInference
from torch_concepts.nn.modules.low.priors import LearnablePrior
from torch_concepts.distributions import Delta


def _plate_task_model():
    """x -> concepts{c1,c2,c3} -> y."""
    x = EmbeddingVariable("x", distribution=Delta, size=8)
    concepts = ConceptVariable("concepts", members=["c1", "c2", "c3"], distribution=dist.Bernoulli)
    y = ConceptVariable("y", distribution=dist.Bernoulli)
    factors = [
        ParametricCPD(x, parametrization=LearnablePrior(8)),
        ParametricCPD(concepts, parents=[x],
                      parametrization=nn.Sequential(nn.Linear(8, 3), nn.Sigmoid())),
        ParametricCPD(y, parents=[concepts],
                      parametrization=nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())),
    ]
    return BayesianNetwork(variables=[x, concepts, y], factors=factors)


def test_member_evidence_forcing_moves_downstream_task():
    pgm = _plate_task_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vi = VariationalInference(pgm)
    B = 8
    xt = torch.randn(B, 8)
    ones, zeros = torch.ones(B, 1), torch.zeros(B, 1)
    # Re-seed so the only difference between the two runs is the forced member.
    seed_everything(0)
    y_hi = vi.query(query=["y"], evidence={"x": xt, "c1": ones}).params["y"]["probs"]
    seed_everything(0)
    y_lo = vi.query(query=["y"], evidence={"x": xt, "c1": zeros}).params["y"]["probs"]
    assert not torch.allclose(y_hi, y_lo)


def _latent_guide_model():
    """x -> concepts{c1,c2} -> z (latent); guide q(z | c1)."""
    x = EmbeddingVariable("x", distribution=Delta, size=6)
    concepts = ConceptVariable("concepts", members=["c1", "c2"], distribution=dist.Bernoulli)
    z = ConceptVariable("z", distribution=dist.Bernoulli)
    factors = [
        ParametricCPD(x, parametrization=LearnablePrior(6)),
        ParametricCPD(concepts, parents=[x],
                      parametrization=nn.Sequential(nn.Linear(6, 2), nn.Sigmoid())),
        ParametricCPD(z, parents=[concepts],
                      parametrization=nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())),
    ]
    pgm = BayesianNetwork(variables=[x, concepts, z], factors=factors)
    guide = ParametricCPD(z, parents=[concepts.member("c1")],
                          parametrization=nn.Sequential(nn.Linear(1, 1), nn.Sigmoid()))
    return pgm, guide


def test_guide_conditions_on_member_handle_parent():
    pgm, guide = _latent_guide_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vi = VariationalInference(pgm, latents={"z": guide})
    B = 4
    out = vi.query(query=["z"], evidence={"x": torch.randn(B, 6), "concepts": torch.rand(B, 2)})
    assert "z" in out.guide_params
    assert out.guide_params["z"]["probs"].shape == (B, 1)
