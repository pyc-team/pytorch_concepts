"""Tests for the detach parameter.

Covers:
- Forward values are identical regardless of detach
- Gradient flow through concept parents is blocked when detach=True
- Gradient flow through exogenous/latent parents is preserved even when detached
- detach=False preserves full gradient flow (control)
"""

import warnings

import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from torch_concepts import LatentVariable, ConceptVariable, ExogenousVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import DeterministicInference
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.low.predictors.linear import LinearConceptToConcept


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chain_pgm():
    """Build a simple chain PGM: input -> A -> B (all Bernoulli / size=1).

    Returns (pgm, linear_A, linear_B) so tests can inspect parameter grads.
    """
    input_var = LatentVariable("input", distribution=Delta, size=4)
    var_A = ConceptVariable("A", distribution=Bernoulli, size=1)
    var_B = ConceptVariable("B", distribution=Bernoulli, size=1)

    linear_A = nn.Linear(4, 1)
    linear_B = LinearConceptToConcept(1, 1)

    cpd_input = ParametricCPD("input", parametrization=nn.Identity())
    cpd_A = ParametricCPD("A", parametrization=linear_A, parents=["input"])
    cpd_B = ParametricCPD("B", parametrization=linear_B, parents=["A"])

    pgm = ProbabilisticModel(
        variables=[input_var, var_A, var_B],
        factors=[cpd_input, cpd_A, cpd_B],
    )
    return pgm, linear_A, linear_B


def _build_exogenous_pgm():
    """Build PGM with concept + exogenous parent: input -> A, exog -> A -> B.

    Returns (pgm, linear_exog, linear_A, linear_B).
    """
    input_var = LatentVariable("input", distribution=Delta, size=4)
    exog_var = ExogenousVariable("exog", distribution=Delta, size=2)
    var_A = ConceptVariable("A", distribution=Bernoulli, size=1)
    var_B = ConceptVariable("B", distribution=Bernoulli, size=1)

    linear_exog = nn.Linear(2, 2)  # just pass-through-ish
    linear_A = nn.Linear(4 + 2, 1)  # input(4) + exog(2) -> 1
    linear_B = LinearConceptToConcept(1, 1)

    cpd_input = ParametricCPD("input", parametrization=nn.Identity())
    cpd_exog = ParametricCPD("exog", parametrization=linear_exog)
    cpd_A = ParametricCPD("A", parametrization=linear_A, parents=["input", "exog"])
    cpd_B = ParametricCPD("B", parametrization=linear_B, parents=["A"])

    pgm = ProbabilisticModel(
        variables=[input_var, exog_var, var_A, var_B],
        factors=[cpd_input, cpd_exog, cpd_A, cpd_B],
    )
    return pgm, linear_exog, linear_A, linear_B


# ---------------------------------------------------------------------------
# Tests: forward values
# ---------------------------------------------------------------------------

class TestDetachConceptsForwardValues:
    """Root concept outputs should match regardless of detach flag.
    Child concepts may differ because detach=True normalises parent
    predictions (logits → probabilities) before propagation."""

    def test_deterministic_root_output_matches(self):
        """Root concept A is computed identically whether detach is on or off."""
        pgm, _, _ = _build_chain_pgm()
        x = torch.randn(8, 4)

        inf_normal = DeterministicInference(pgm, detach=False)
        inf_detach = DeterministicInference(pgm, detach=True)

        out_normal = inf_normal.query(["A"], evidence={"input": x})
        out_detach = inf_detach.query(["A"], evidence={"input": x})

        torch.testing.assert_close(out_normal, out_detach)

    def test_deterministic_child_output_differs(self):
        """Child concept B sees different parent representations
        (logits vs probabilities), so its output may differ."""
        pgm, _, _ = _build_chain_pgm()
        x = torch.randn(8, 4)

        inf_normal = DeterministicInference(pgm, detach=False)
        inf_detach = DeterministicInference(pgm, detach=True)

        out_normal = inf_normal.query(["A", "B"], evidence={"input": x})
        out_detach = inf_detach.query(["A", "B"], evidence={"input": x})

        # A column (idx 0) should match
        torch.testing.assert_close(out_normal[:, :1], out_detach[:, :1])
        # B column (idx 1) is allowed to differ
        assert out_normal.shape == out_detach.shape

    def test_output_shape(self):
        pgm, _, _ = _build_chain_pgm()
        x = torch.randn(8, 4)
        inf = DeterministicInference(pgm, detach=True)
        out = inf.query(["A", "B"], evidence={"input": x})
        assert out.shape == (8, 2)

    def test_single_query_concept(self):
        pgm, _, _ = _build_chain_pgm()
        x = torch.randn(4, 4)
        inf = DeterministicInference(pgm, detach=True)
        out = inf.query(["B"], evidence={"input": x})
        assert out.shape == (4, 1)


# ---------------------------------------------------------------------------
# Tests: gradient flow
# ---------------------------------------------------------------------------

class TestDetachConceptsGradientFlow:
    """When detach=True, concept parents must NOT propagate gradients
    to upstream encoders, while exogenous/latent parents should still flow."""

    def test_no_gradient_through_concept_parent(self):
        """Loss on B should NOT give gradients to linear_A when detached."""
        pgm, linear_A, linear_B = _build_chain_pgm()
        x = torch.randn(8, 4)

        inf = DeterministicInference(pgm, detach=True)
        out = inf.query(["B"], evidence={"input": x})

        loss = out.sum()
        loss.backward()

        # linear_B should have gradients (it directly produces B)
        b_params = list(linear_B.parameters())
        assert any(p.grad is not None and (p.grad != 0).any() for p in b_params)

        # linear_A should have NO gradient from B's loss (detached path)
        assert linear_A.weight.grad is None or (linear_A.weight.grad == 0).all()

    def test_gradient_flows_without_detach(self):
        """Control: loss on B SHOULD give gradients to linear_A when not detached."""
        pgm, linear_A, linear_B = _build_chain_pgm()
        x = torch.randn(8, 4)

        inf = DeterministicInference(pgm, detach=False)
        out = inf.query(["B"], evidence={"input": x})

        loss = out.sum()
        loss.backward()

        # Both should have gradients
        assert linear_A.weight.grad is not None
        assert (linear_A.weight.grad != 0).any()
        b_params = list(linear_B.parameters())
        assert any(p.grad is not None and (p.grad != 0).any() for p in b_params)

    def test_own_concept_loss_still_gives_gradient(self):
        """Even when detached, A's own loss should produce gradients for linear_A."""
        pgm, linear_A, _ = _build_chain_pgm()
        x = torch.randn(8, 4)

        inf = DeterministicInference(pgm, detach=True)
        out = inf.query(["A"], evidence={"input": x})

        loss = out.sum()
        loss.backward()

        assert linear_A.weight.grad is not None
        assert (linear_A.weight.grad != 0).any()

    def test_exogenous_gradient_preserved_when_detached(self):
        """Exogenous variable path should keep gradients even with detach=True."""
        pgm, linear_exog, linear_A, linear_B = _build_exogenous_pgm()
        x = torch.randn(8, 4)
        u = torch.randn(8, 2)

        inf = DeterministicInference(pgm, detach=True)
        out = inf.query(["A", "B"], evidence={"input": x, "exog": u})

        loss_A = out[:, :1].sum()
        loss_A.backward(retain_graph=True)

        # exogenous encoder should receive gradient through A
        assert linear_exog.weight.grad is not None
        assert (linear_exog.weight.grad != 0).any()

# ---------------------------------------------------------------------------
# Tests: default behaviour unaffected
# ---------------------------------------------------------------------------

class TestDefaultBehaviourUnchanged:
    """Verify that the default (detach=False) matches the original behaviour."""

    def test_default_is_false(self):
        pgm, _, _ = _build_chain_pgm()
        inf = DeterministicInference(pgm)
        assert inf.detach is False

    def test_full_gradient_chain_by_default(self):
        pgm, linear_A, linear_B = _build_chain_pgm()
        x = torch.randn(8, 4)

        inf = DeterministicInference(pgm)
        out = inf.query(["A", "B"], evidence={"input": x})
        out.sum().backward()

        assert linear_A.weight.grad is not None
        assert (linear_A.weight.grad != 0).any()
