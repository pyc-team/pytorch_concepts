"""
Comprehensive tests for DeterministicInference.

Tests cover:
- activate() with each distribution type (Bernoulli, RelaxedBernoulli,
  Categorical, RelaxedOneHotCategorical, Delta, unknown)
- ground_truth_to_evidence() with binary and categorical cardinalities,
  1D and 2D inputs, and invalid shapes
- Integration: DeterministicInference query with Categorical variables,
  return_logits, gradient flow, and detach mode
"""
import pytest
import torch
import torch.nn as nn
from torch.distributions import (
    Bernoulli, Categorical, Normal,
    RelaxedBernoulli, RelaxedOneHotCategorical,
)

from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.mid.inference.deterministic import DeterministicInference
from torch_concepts.nn.modules.mid.models.variable import (
    Variable, ConceptVariable, LatentVariable,
)
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.low.predictors.linear import LinearConceptToConcept


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_variable(name, distribution, size=1):
    """Create a standalone ConceptVariable with the given distribution."""
    return ConceptVariable(name, distribution=distribution, size=size)


def _make_chain_pgm(dist_A, size_A, dist_B, size_B, latent=10):
    """Build a simple chain: input -> A -> B with specified distributions."""
    input_var = LatentVariable('input', distribution=Delta, size=latent)
    var_A = ConceptVariable('A', distribution=dist_A, size=size_A)
    var_B = ConceptVariable('B', distribution=dist_B, size=size_B)

    cpd_input = ParametricCPD('input', parametrization=nn.Identity())
    cpd_A = ParametricCPD('A', parametrization=nn.Linear(latent, size_A), parents=['input'])
    cpd_B = ParametricCPD('B', parametrization=LinearConceptToConcept(size_A, size_B), parents=['A'])

    pgm = ProbabilisticModel(
        variables=[input_var, var_A, var_B],
        factors=[cpd_input, cpd_A, cpd_B],
    )
    return pgm


# ===========================================================================
# Unit tests for activate()
# ===========================================================================

class TestActivateBernoulli:
    """activate with Bernoulli → sigmoid."""

    def test_output_range(self):
        var = _make_variable('c', Bernoulli)
        inf = DeterministicInference.__new__(DeterministicInference)
        pred = torch.randn(8, 1)
        out = inf.activate(pred, var)
        assert out.shape == pred.shape
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_matches_sigmoid(self):
        var = _make_variable('c', Bernoulli)
        inf = DeterministicInference.__new__(DeterministicInference)
        pred = torch.randn(4, 1)
        expected = torch.sigmoid(pred)
        torch.testing.assert_close(inf.activate(pred, var), expected)


class TestActivateRelaxedBernoulli:
    """activate with RelaxedBernoulli → sigmoid (same as Bernoulli)."""

    def test_output_range(self):
        var = _make_variable('c', RelaxedBernoulli)
        inf = DeterministicInference.__new__(DeterministicInference)
        pred = torch.randn(8, 1)
        out = inf.activate(pred, var)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_matches_sigmoid(self):
        var = _make_variable('c', RelaxedBernoulli)
        inf = DeterministicInference.__new__(DeterministicInference)
        pred = torch.randn(4, 1)
        torch.testing.assert_close(inf.activate(pred, var), torch.sigmoid(pred))


class TestActivateCategorical:
    """activate with Categorical → softmax."""

    def test_output_sums_to_one(self):
        var = _make_variable('c', Categorical, size=5)
        inf = DeterministicInference.__new__(DeterministicInference)
        pred = torch.randn(8, 5)
        out = inf.activate(pred, var)
        torch.testing.assert_close(out.sum(dim=-1), torch.ones(8))

    def test_matches_softmax(self):
        var = _make_variable('c', Categorical, size=3)
        inf = DeterministicInference.__new__(DeterministicInference)
        pred = torch.randn(4, 3)
        torch.testing.assert_close(inf.activate(pred, var), torch.softmax(pred, dim=-1))


class TestActivateRelaxedOneHotCategorical:
    """activate with RelaxedOneHotCategorical → softmax."""

    def test_output_sums_to_one(self):
        var = _make_variable('c', RelaxedOneHotCategorical, size=4)
        inf = DeterministicInference.__new__(DeterministicInference)
        pred = torch.randn(8, 4)
        out = inf.activate(pred, var)
        torch.testing.assert_close(out.sum(dim=-1), torch.ones(8))

    def test_matches_softmax(self):
        var = _make_variable('c', RelaxedOneHotCategorical, size=3)
        inf = DeterministicInference.__new__(DeterministicInference)
        pred = torch.randn(4, 3)
        torch.testing.assert_close(inf.activate(pred, var), torch.softmax(pred, dim=-1))


class TestActivateDelta:
    """activate with Delta → identity pass-through."""

    def test_identity(self):
        var = _make_variable('c', Delta)
        inf = DeterministicInference.__new__(DeterministicInference)
        pred = torch.randn(4, 3)
        torch.testing.assert_close(inf.activate(pred, var), pred)


class TestActivateUnknownDistribution:
    """activate with unsupported distribution → raises ValueError."""

    def test_raises_for_unknown_distribution(self):
        class _CustomDist(torch.distributions.Distribution):
            pass
        with pytest.raises(ValueError, match="No default activation"):
            _make_variable('c', _CustomDist)

    def test_identity_when_distribution_is_none(self):
        """Variable whose distribution is None."""
        var = _make_variable('c', Bernoulli)
        var.distribution = None  # override to None
        var.activation = lambda x: x  # match the fallback
        inf = DeterministicInference.__new__(DeterministicInference)
        pred = torch.randn(4, 1)
        result = inf.activate(pred, var)
        torch.testing.assert_close(result, pred)


# ===========================================================================
# Unit tests for ground_truth_to_evidence()
# ===========================================================================

class TestGroundTruthToEvidenceBinary:
    """ground_truth_to_evidence with cardinality=1 (binary)."""

    def test_2d_input(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        value = torch.tensor([[0.], [1.], [1.], [0.]])
        result = inf.ground_truth_to_evidence(value, cardinality=1)
        expected = torch.tensor([[0.], [1.], [1.], [0.]])
        torch.testing.assert_close(result, expected)

    def test_1d_input_gets_unsqueezed(self):
        """1D tensor (batch,) should be auto-unsqueezed to (batch, 1)."""
        inf = DeterministicInference.__new__(DeterministicInference)
        value = torch.tensor([1., 0., 1.])
        result = inf.ground_truth_to_evidence(value, cardinality=1)
        assert result.shape == (3, 1)
        expected = torch.tensor([[1.], [0.], [1.]])
        torch.testing.assert_close(result, expected)

    def test_output_is_float(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        value = torch.tensor([0, 1, 1])  # int input
        result = inf.ground_truth_to_evidence(value, cardinality=1)
        assert result.dtype == torch.float32


class TestGroundTruthToEvidenceCategorical:
    """ground_truth_to_evidence with cardinality>1 (categorical)."""

    def test_one_hot_3_classes(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        value = torch.tensor([[0], [1], [2]])
        result = inf.ground_truth_to_evidence(value, cardinality=3)
        expected = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        torch.testing.assert_close(result, expected)

    def test_one_hot_5_classes(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        value = torch.tensor([[4], [0], [2]])
        result = inf.ground_truth_to_evidence(value, cardinality=5)
        assert result.shape == (3, 5)
        assert result.sum(dim=-1).allclose(torch.ones(3))

    def test_1d_input_categorical(self):
        """1D class indices should be auto-unsqueezed."""
        inf = DeterministicInference.__new__(DeterministicInference)
        value = torch.tensor([0, 2, 1])
        result = inf.ground_truth_to_evidence(value, cardinality=3)
        assert result.shape == (3, 3)
        expected = torch.tensor([
            [1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
        ])
        torch.testing.assert_close(result, expected)

    def test_output_is_float(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        value = torch.tensor([0, 1])  # int
        result = inf.ground_truth_to_evidence(value, cardinality=2)
        assert result.dtype == torch.float32


class TestGroundTruthToEvidenceInvalidShape:
    """ground_truth_to_evidence with invalid input shapes."""

    def test_3d_input_raises(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        value = torch.tensor([[[1.0]]])  # 3D
        with pytest.raises(ValueError, match="Expected shape"):
            inf.ground_truth_to_evidence(value, cardinality=1)

    def test_2d_multi_column_raises(self):
        """Shape (batch, >1) should raise ValueError."""
        inf = DeterministicInference.__new__(DeterministicInference)
        value = torch.tensor([[1., 0.], [0., 1.]])  # (2, 2) — not (batch, 1)
        with pytest.raises(ValueError, match="Expected shape"):
            inf.ground_truth_to_evidence(value, cardinality=1)


# ===========================================================================
# Integration tests: DeterministicInference.query with various distributions
# ===========================================================================

class TestDeterministicQueryBernoulli:
    """End-to-end query with all-Bernoulli chain."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pgm = _make_chain_pgm(Bernoulli, 1, Bernoulli, 1)
        self.inf = DeterministicInference(self.pgm)
        self.x = torch.randn(4, 10)

    def test_shape(self):
        out = self.inf.query(['A', 'B'], evidence={'input': self.x})
        assert out.shape == (4, 2)

    def test_probabilities(self):
        out = self.inf.query(['A', 'B'], evidence={'input': self.x})
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_return_logits(self):
        logits = self.inf.query(['A', 'B'], evidence={'input': self.x}, return_logits=True)
        probs = self.inf.query(['A', 'B'], evidence={'input': self.x})
        assert not torch.allclose(logits, probs)


class TestDeterministicQueryCategorical:
    """End-to-end query with Categorical variables."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pgm = _make_chain_pgm(Categorical, 5, Bernoulli, 1, latent=10)
        self.inf = DeterministicInference(self.pgm)
        self.x = torch.randn(4, 10)

    def test_shape(self):
        out = self.inf.query(['A', 'B'], evidence={'input': self.x})
        # A (softmax, 5) + B (sigmoid, 1) = 6
        assert out.shape == (4, 6)

    def test_softmax_for_A(self):
        out = self.inf.query(['A'], evidence={'input': self.x})
        # Each row sums to 1 (softmax)
        torch.testing.assert_close(out.sum(dim=-1), torch.ones(4))


class TestDeterministicQueryRelaxedBernoulli:
    """End-to-end query with RelaxedBernoulli variables."""

    def test_query_shape_and_range(self):
        pgm = _make_chain_pgm(RelaxedBernoulli, 1, RelaxedBernoulli, 1)
        inf = DeterministicInference(pgm)
        x = torch.randn(4, 10)
        out = inf.query(['A', 'B'], evidence={'input': x})
        assert out.shape == (4, 2)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestDeterministicQueryRelaxedOneHotCategorical:
    """End-to-end query with RelaxedOneHotCategorical variables."""

    def test_query_shape_and_softmax(self):
        pgm = _make_chain_pgm(RelaxedOneHotCategorical, 4, Bernoulli, 1)
        inf = DeterministicInference(pgm)
        x = torch.randn(4, 10)
        out = inf.query(['A'], evidence={'input': x})
        assert out.shape == (4, 4)
        torch.testing.assert_close(out.sum(dim=-1), torch.ones(4))


class TestDeterministicQueryDelta:
    """End-to-end query where a concept uses Delta (pass-through)."""

    def test_identity_activation(self):
        pgm = _make_chain_pgm(Delta, 3, Bernoulli, 1)
        inf = DeterministicInference(pgm)
        x = torch.randn(4, 10)
        # Delta concept A: raw logits passed through unchanged
        logits = inf.query(['A'], evidence={'input': x}, return_logits=True)
        activated = inf.query(['A'], evidence={'input': x})
        # For Delta, activate is identity, so logits == activated
        torch.testing.assert_close(logits, activated)


class TestDeterministicQueryMixed:
    """Chain with mixed dist types: Categorical → Bernoulli."""

    def test_gradient_flow(self):
        pgm = _make_chain_pgm(Categorical, 3, Bernoulli, 1)
        inf = DeterministicInference(pgm)
        x = torch.randn(4, 10, requires_grad=True)
        out = inf.query(['B'], evidence={'input': x})
        out.sum().backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_detach_mode(self):
        pgm = _make_chain_pgm(Categorical, 3, Bernoulli, 1)
        inf = DeterministicInference(pgm, detach=True)
        x = torch.randn(4, 10, requires_grad=True)
        out = inf.query(['A', 'B'], evidence={'input': x})
        assert out.shape == (4, 4)  # A(3) + B(1)

    def test_return_logits_with_detach(self):
        pgm = _make_chain_pgm(Categorical, 3, Bernoulli, 1)
        inf = DeterministicInference(pgm, detach=True)
        x = torch.randn(4, 10)
        logits = inf.query(['A', 'B'], evidence={'input': x}, return_logits=True)
        assert logits.shape == (4, 4)


class TestDeterministicQueryLazy:
    """Lazy mode with DeterministicInference."""

    def test_lazy_query_subset(self):
        pgm = _make_chain_pgm(Bernoulli, 1, Bernoulli, 1)
        inf = DeterministicInference(pgm, lazy=True)
        x = torch.randn(4, 10)
        out = inf.query(['A'], evidence={'input': x})
        assert out.shape == (4, 1)

    def test_lazy_matches_non_lazy(self):
        torch.manual_seed(42)
        pgm = _make_chain_pgm(Bernoulli, 1, Bernoulli, 1)
        pgm2 = _make_chain_pgm(Bernoulli, 1, Bernoulli, 1)
        pgm2.load_state_dict(pgm.state_dict())

        inf_full = DeterministicInference(pgm)
        inf_lazy = DeterministicInference(pgm2, lazy=True)

        x = torch.randn(4, 10)
        torch.testing.assert_close(
            inf_full.query(['B'], evidence={'input': x}),
            inf_lazy.query(['B'], evidence={'input': x}),
        )


class TestGroundTruthRangeValidation:
    """Test that ground_truth_to_evidence warns on out-of-range binary values."""

    def test_binary_gt_out_of_range_warns(self):
        """Passing a value not in {0, 1} for binary should warn."""
        import warnings
        inference = DeterministicInference.__new__(DeterministicInference)
        value = torch.tensor([[0.5]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = inference.ground_truth_to_evidence(value, cardinality=1)
            matching = [x for x in w if "outside {0, 1}" in str(x.message)]
            assert len(matching) > 0, "Expected warning for out-of-range binary GT"
        assert result.shape == (1, 1)

    def test_binary_gt_valid_no_warning(self):
        """Values in {0, 1} should not produce a warning."""
        import warnings
        inference = DeterministicInference.__new__(DeterministicInference)
        value = torch.tensor([[0.0], [1.0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = inference.ground_truth_to_evidence(value, cardinality=1)
            matching = [x for x in w if "outside {0, 1}" in str(x.message)]
            assert len(matching) == 0, "Should not warn for valid binary GT"
        assert result.shape == (2, 1)
