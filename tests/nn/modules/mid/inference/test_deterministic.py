"""
Comprehensive tests for DeterministicInference.

Tests cover:
- constructor coercion of non-Delta variables to Delta with identity activation
- activate() as identity for every distribution type
- ground_truth_to_evidence() with binary and categorical cardinalities,
  1D and 2D inputs, and invalid shapes
- Integration: DeterministicInference query with non-Delta variables coerced
  to raw identity propagation, return_logits, gradient flow, and detach mode
"""
import pytest
import warnings
import torch
import torch.nn as nn
from torch.distributions import (
    Bernoulli, OneHotCategorical,
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
# Unit tests for constructor coercion and activate()
# ===========================================================================

class TestConstructorCoercion:
    """Constructor converts probabilistic variables to deterministic ones."""

    def test_warns_and_coerces_non_delta_variables(self):
        pgm = _make_chain_pgm(Bernoulli, 1, OneHotCategorical, 3)

        with pytest.warns(UserWarning, match="All Variables will be changed to Delta"):
            inference = DeterministicInference(pgm, log_probs=True)

        assert set(inference.variable_map) == {'input', 'A', 'B'}
        for variable in pgm.variables:
            assert variable.distribution is Delta
            pred = torch.randn(4, variable.size)
            torch.testing.assert_close(variable.activation(pred), pred)

    def test_delta_variables_do_not_warn(self, recwarn):
        pgm = _make_chain_pgm(Delta, 3, Delta, 1)

        DeterministicInference(pgm)

        matching = [
            warning for warning in recwarn
            if "DeterministicInference propagates logits/activations" in str(warning.message)
        ]
        assert matching == []


class TestActivateIdentity:
    """With log_probs=True, activate() returns raw predictions unchanged for all distribution types."""

    @pytest.mark.parametrize(
        ("distribution", "size"),
        [
            (Bernoulli, 1),
            (RelaxedBernoulli, 1),
            (OneHotCategorical, 5),
            (RelaxedOneHotCategorical, 4),
            (Delta, 3),
        ],
    )
    def test_identity_for_supported_distributions(self, distribution, size):
        pgm = _make_chain_pgm(distribution, size, Delta, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            inf = DeterministicInference(pgm, log_probs=True)
        var_A = inf.variable_map['A']
        pred = torch.randn(8, size)
        torch.testing.assert_close(inf.activate(pred, var_A), pred)


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
        with pytest.raises(ValueError, match="is not supported"):
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
        inf.log_probs = False
        value = torch.tensor([[0.], [1.], [1.], [0.]])
        result = inf.ground_truth_to_evidence(value, size=1, type='binary')
        expected = torch.tensor([[0.], [1.], [1.], [0.]])
        torch.testing.assert_close(result, expected)

    def test_1d_input_gets_unsqueezed(self):
        """1D tensor (batch,) should be auto-unsqueezed to (batch, 1)."""
        inf = DeterministicInference.__new__(DeterministicInference)
        inf.log_probs = False
        value = torch.tensor([1., 0., 1.])
        result = inf.ground_truth_to_evidence(value, size=1, type='binary')
        assert result.shape == (3, 1)
        expected = torch.tensor([[1.], [0.], [1.]])
        torch.testing.assert_close(result, expected)

    def test_output_is_float(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        inf.log_probs = False
        value = torch.tensor([0, 1, 1])  # int input
        result = inf.ground_truth_to_evidence(value, size=1, type='binary')
        assert result.dtype == torch.float32


class TestGroundTruthToEvidenceCategorical:
    """ground_truth_to_evidence with cardinality>1 (categorical)."""

    def test_one_hot_3_classes(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        inf.log_probs = False
        value = torch.tensor([[0], [1], [2]])
        result = inf.ground_truth_to_evidence(value, size=3, type='categorical')
        expected = torch.tensor([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        torch.testing.assert_close(result, expected)

    def test_one_hot_5_classes(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        inf.log_probs = False
        value = torch.tensor([[4], [0], [2]])
        result = inf.ground_truth_to_evidence(value, size=5, type='categorical')
        assert result.shape == (3, 5)
        assert result.sum(dim=-1).allclose(torch.ones(3))

    def test_1d_input_categorical(self):
        """1D class indices should be auto-unsqueezed."""
        inf = DeterministicInference.__new__(DeterministicInference)
        inf.log_probs = False
        value = torch.tensor([0, 2, 1])
        result = inf.ground_truth_to_evidence(value, size=3, type='categorical')
        assert result.shape == (3, 3)
        expected = torch.tensor([
            [1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
        ])
        torch.testing.assert_close(result, expected)

    def test_output_is_float(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        inf.log_probs = False
        value = torch.tensor([0, 1])  # int
        result = inf.ground_truth_to_evidence(value, size=2, type='categorical')
        assert result.dtype == torch.float32


class TestGroundTruthToEvidenceInvalidShape:
    """ground_truth_to_evidence with invalid input shapes."""

    def test_3d_input_raises(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        inf.log_probs = False
        value = torch.tensor([[[1.0]]])  # 3D
        with pytest.raises(ValueError, match="Expected shape"):
            inf.ground_truth_to_evidence(value, size=1, type='binary')

    def test_2d_multi_column_raises(self):
        """Shape (batch, >1) should raise ValueError."""
        inf = DeterministicInference.__new__(DeterministicInference)
        inf.log_probs = False
        value = torch.tensor([[1., 0.], [0., 1.]])  # (2, 2) — not (batch, 1)
        with pytest.raises(ValueError, match="Expected shape"):
            inf.ground_truth_to_evidence(value, size=1, type='binary')


# ===========================================================================
# Integration tests: DeterministicInference.query with various distributions
# ===========================================================================

class TestDeterministicQueryBernoulli:
    """End-to-end query with all-Bernoulli chain."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pgm = _make_chain_pgm(Bernoulli, 1, Bernoulli, 1)
        with pytest.warns(UserWarning, match="All Variables will be changed to Delta"):
            self.inf = DeterministicInference(self.pgm, log_probs=True)
        self.x = torch.randn(4, 10)

    def test_shape(self):
        out = self.inf.query(['A', 'B'], evidence={'input': self.x})
        assert out.probs.shape == (4, 2)

    def test_outputs_are_raw(self):
        out = self.inf.query(['A', 'B'], evidence={'input': self.x}, return_logits=True)
        torch.testing.assert_close(out.logits, out.probs)

    def test_return_logits(self):
        logits = self.inf.query(['A', 'B'], evidence={'input': self.x}, return_logits=True)
        probs = self.inf.query(['A', 'B'], evidence={'input': self.x})
        torch.testing.assert_close(logits.logits, probs.probs)


class TestDeterministicQueryCategorical:
    """End-to-end query with OneHotCategorical variables."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pgm = _make_chain_pgm(OneHotCategorical, 5, Bernoulli, 1, latent=10)
        with pytest.warns(UserWarning, match="All Variables will be changed to Delta"):
            self.inf = DeterministicInference(self.pgm, log_probs=True)
        self.x = torch.randn(4, 10)

    def test_shape(self):
        out = self.inf.query(['A', 'B'], evidence={'input': self.x})
        # A (raw, 5) + B (raw, 1) = 6
        assert out.probs.shape == (4, 6)

    def test_A_is_not_softmaxed(self):
        out = self.inf.query(['A'], evidence={'input': self.x}, return_logits=True)
        torch.testing.assert_close(out.logits, out.probs)


class TestDeterministicQueryRelaxedBernoulli:
    """End-to-end query with RelaxedBernoulli variables."""

    def test_query_shape_and_identity(self):
        pgm = _make_chain_pgm(RelaxedBernoulli, 1, RelaxedBernoulli, 1)
        with pytest.warns(UserWarning, match="All Variables will be changed to Delta"):
            inf = DeterministicInference(pgm, log_probs=True)
        x = torch.randn(4, 10)
        out = inf.query(['A', 'B'], evidence={'input': x}, return_logits=True)
        assert out.probs.shape == (4, 2)
        torch.testing.assert_close(out.logits, out.probs)


class TestDeterministicQueryRelaxedOneHotCategorical:
    """End-to-end query with RelaxedOneHotCategorical variables."""

    def test_query_shape_and_identity(self):
        pgm = _make_chain_pgm(RelaxedOneHotCategorical, 4, Bernoulli, 1)
        with pytest.warns(UserWarning, match="All Variables will be changed to Delta"):
            inf = DeterministicInference(pgm, log_probs=True)
        x = torch.randn(4, 10)
        out = inf.query(['A'], evidence={'input': x}, return_logits=True)
        assert out.probs.shape == (4, 4)
        torch.testing.assert_close(out.logits, out.probs)


class TestDeterministicQueryDelta:
    """End-to-end query where a concept uses Delta (pass-through)."""

    def test_identity_activation(self):
        pgm = _make_chain_pgm(Delta, 3, Bernoulli, 1)
        with pytest.warns(UserWarning, match="All Variables will be changed to Delta"):
            inf = DeterministicInference(pgm)
        x = torch.randn(4, 10)
        # Delta concept A: raw logits passed through unchanged
        logits = inf.query(['A'], evidence={'input': x}, return_logits=True)
        activated = inf.query(['A'], evidence={'input': x})
        # For Delta, activate is identity, so logits == activated
        torch.testing.assert_close(logits.logits, activated.probs)


class TestDeterministicQueryMixed:
    """Chain with mixed dist types: OneHotCategorical → Bernoulli."""

    def test_gradient_flow(self):
        pgm = _make_chain_pgm(OneHotCategorical, 3, Bernoulli, 1)
        with pytest.warns(UserWarning, match="All Variables will be changed to Delta"):
            inf = DeterministicInference(pgm)
        x = torch.randn(4, 10, requires_grad=True)
        out = inf.query(['B'], evidence={'input': x})
        out.probs.sum().backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_detach_mode(self):
        pgm = _make_chain_pgm(OneHotCategorical, 3, Bernoulli, 1)
        with pytest.warns(UserWarning, match="All Variables will be changed to Delta"):
            inf = DeterministicInference(pgm, detach=True)
        x = torch.randn(4, 10, requires_grad=True)
        out = inf.query(['A', 'B'], evidence={'input': x})
        assert out.probs.shape == (4, 4)  # A(3) + B(1)

    def test_return_logits_with_detach(self):
        pgm = _make_chain_pgm(OneHotCategorical, 3, Bernoulli, 1)
        with pytest.warns(UserWarning, match="All Variables will be changed to Delta"):
            inf = DeterministicInference(pgm, detach=True)
        x = torch.randn(4, 10)
        logits = inf.query(['A', 'B'], evidence={'input': x}, return_logits=True)
        assert logits.logits.shape == (4, 4)


class TestDeterministicQueryLazy:
    """Lazy mode with DeterministicInference."""

    def test_lazy_query_subset(self):
        pgm = _make_chain_pgm(Bernoulli, 1, Bernoulli, 1)
        with pytest.warns(UserWarning, match="All Variables will be changed to Delta"):
            inf = DeterministicInference(pgm, lazy=True)
        x = torch.randn(4, 10)
        out = inf.query(['A'], evidence={'input': x})
        assert out.probs.shape == (4, 1)

    def test_lazy_matches_non_lazy(self):
        torch.manual_seed(42)
        pgm = _make_chain_pgm(Bernoulli, 1, Bernoulli, 1)
        pgm2 = _make_chain_pgm(Bernoulli, 1, Bernoulli, 1)
        pgm2.load_state_dict(pgm.state_dict())

        with pytest.warns(UserWarning, match="All Variables will be changed to Delta"):
            inf_full = DeterministicInference(pgm)
        with pytest.warns(UserWarning, match="All Variables will be changed to Delta"):
            inf_lazy = DeterministicInference(pgm2, lazy=True)

        x = torch.randn(4, 10)
        torch.testing.assert_close(
            inf_full.query(['B'], evidence={'input': x}).probs,
            inf_lazy.query(['B'], evidence={'input': x}).probs,
        )


class TestGroundTruthRangeValidation:
    """Test that ground_truth_to_evidence warns on out-of-range binary values."""

    def test_binary_gt_out_of_range_warns(self):
        """Passing a value not in {0, 1} for binary should warn."""
        import warnings
        inference = DeterministicInference.__new__(DeterministicInference)
        inference.log_probs = False
        value = torch.tensor([[0.5]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = inference.ground_truth_to_evidence(value, size=1, type='binary')
            matching = [x for x in w if "outside {0, 1}" in str(x.message)]
            assert len(matching) > 0, "Expected warning for out-of-range binary GT"
        assert result.shape == (1, 1)

    def test_binary_gt_valid_no_warning(self):
        """Values in {0, 1} should not produce a warning."""
        import warnings
        inference = DeterministicInference.__new__(DeterministicInference)
        inference.log_probs = False
        value = torch.tensor([[0.0], [1.0]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = inference.ground_truth_to_evidence(value, size=1, type='binary')
            matching = [x for x in w if "outside {0, 1}" in str(x.message)]
            assert len(matching) == 0, "Should not warn for valid binary GT"
        assert result.shape == (2, 1)


# ===========================================================================
# Tests for log_probs=False (default): activate() applies sigmoid/softmax
# ===========================================================================

class TestActivateProbs:
    """With log_probs=False (default), activate() applies sigmoid/softmax to discrete variables."""

    def test_binary_gets_sigmoid(self):
        pgm = _make_chain_pgm(Bernoulli, 1, Delta, 1)
        with pytest.warns(UserWarning):
            inf = DeterministicInference(pgm, log_probs=False)
        var_A = inf.variable_map['A']
        pred = torch.randn(8, 1)
        torch.testing.assert_close(inf.activate(pred, var_A), torch.sigmoid(pred))

    def test_categorical_gets_softmax(self):
        pgm = _make_chain_pgm(OneHotCategorical, 4, Delta, 1)
        with pytest.warns(UserWarning):
            inf = DeterministicInference(pgm, log_probs=False)
        var_A = inf.variable_map['A']
        pred = torch.randn(8, 4)
        torch.testing.assert_close(inf.activate(pred, var_A), torch.softmax(pred, dim=-1))

    def test_delta_always_identity(self):
        """Delta variables use identity regardless of log_probs."""
        pgm = _make_chain_pgm(Delta, 3, Delta, 1)
        inf = DeterministicInference(pgm, log_probs=False)
        var_A = inf.variable_map['A']
        pred = torch.randn(8, 3)
        torch.testing.assert_close(inf.activate(pred, var_A), pred)


class TestDeterministicQueryProbs:
    """With log_probs=False (default), query() returns activated probabilities."""

    def test_binary_probs_are_sigmoid(self):
        pgm = _make_chain_pgm(Bernoulli, 1, Bernoulli, 1)
        with pytest.warns(UserWarning):
            inf = DeterministicInference(pgm, log_probs=False)
        x = torch.randn(4, 10)
        out = inf.query(['A'], evidence={'input': x}, return_logits=True)
        torch.testing.assert_close(out.probs, torch.sigmoid(out.logits))

    def test_categorical_probs_are_softmax(self):
        pgm = _make_chain_pgm(OneHotCategorical, 5, Delta, 1)
        with pytest.warns(UserWarning):
            inf = DeterministicInference(pgm, log_probs=False)
        x = torch.randn(4, 10)
        out = inf.query(['A'], evidence={'input': x}, return_logits=True)
        torch.testing.assert_close(out.probs, torch.softmax(out.logits, dim=-1))


class TestGroundTruthToEvidenceLogits:
    """ground_truth_to_evidence with log_probs=True converts discrete GT to logits."""

    def test_binary_returns_logits(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        inf.log_probs = True
        value = torch.tensor([[0.0], [1.0]])
        result = inf.ground_truth_to_evidence(value, size=1, type='binary')
        # logit(0) ≈ -16.1, logit(1) ≈ +16.1  (clamped by eps=1e-7)
        assert result[0, 0] < 0
        assert result[1, 0] > 0

    def test_categorical_returns_logits(self):
        inf = DeterministicInference.__new__(DeterministicInference)
        inf.log_probs = True
        value = torch.tensor([[0], [1], [2]])
        result = inf.ground_truth_to_evidence(value, size=3, type='categorical')
        # For each row: the selected class logit > 0, others < 0
        assert result[0, 0] > 0 and result[0, 1] < 0 and result[0, 2] < 0
        assert result[1, 1] > 0 and result[1, 0] < 0 and result[1, 2] < 0
        assert result[2, 2] > 0 and result[2, 0] < 0 and result[2, 1] < 0

    def test_continuous_unaffected_by_log_probs(self):
        """Continuous variables are passed through regardless of log_probs."""
        for lp in (True, False):
            inf = DeterministicInference.__new__(DeterministicInference)
            inf.log_probs = lp
            value = torch.randn(4, 3)
            result = inf.ground_truth_to_evidence(value, size=3, type='continuous')
            torch.testing.assert_close(result, value.float())
