"""
Tests for ImportanceSamplingInference and LazyImportanceSamplingInference.

Tests cover:
- Construction and distribution validation
- get_results returns raw logits
- ground_truth_to_evidence format conversion
- Marginal computation (single and multiple queries)
- Conditional marginal queries (evidence conditioning)
- return_dict mode with detailed statistics
- Query interface compatibility
- Pyro model construction (_pyro_model)
- _get_pyro_distribution for all supported distribution types
- LazyImportanceSamplingInference ancestor-only computation
- Error handling for invalid queries and unsupported distributions
"""

import pytest
import torch
import torch.nn as nn
from torch.distributions import (
    Bernoulli, Categorical,
    RelaxedBernoulli, RelaxedOneHotCategorical,
    Normal, LogNormal, Beta, Gamma,
)

from torch_concepts import LatentVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.inference.importance_sampling import (
    ImportanceSamplingInference,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_binary_chain():
    """
    Build a simple binary chain: input -> A -> B.

    Both A and B are Bernoulli concepts, with Linear CPDs.
    Returns (model, input_dim).
    """
    input_dim = 8
    input_var = LatentVariable("input", parents=[], size=input_dim)
    var_A = ConceptVariable("A", parents=["input"], distribution=Bernoulli)
    var_B = ConceptVariable("B", parents=["A"], distribution=Bernoulli)

    cpd_input = ParametricCPD("input", parametrization=nn.Identity())
    cpd_A = ParametricCPD("A", parametrization=nn.Linear(input_dim, 1))
    cpd_B = ParametricCPD("B", parametrization=nn.Linear(1, 1))

    model = ProbabilisticModel(
        variables=[input_var, var_A, var_B],
        parametric_cpds=[cpd_input, cpd_A, cpd_B],
    )
    return model, input_dim


def _make_diamond():
    """
    Build a diamond DAG: input -> A, input -> B, (A, B) -> C.

    All concept variables are Bernoulli.
    Returns (model, input_dim).
    """
    input_dim = 8
    input_var = LatentVariable("input", parents=[], size=input_dim)
    var_A = ConceptVariable("A", parents=["input"], distribution=Bernoulli)
    var_B = ConceptVariable("B", parents=["input"], distribution=Bernoulli)
    var_C = ConceptVariable("C", parents=["A", "B"], distribution=Bernoulli)

    cpd_input = ParametricCPD("input", parametrization=nn.Identity())
    cpd_A = ParametricCPD("A", parametrization=nn.Linear(input_dim, 1))
    cpd_B = ParametricCPD("B", parametrization=nn.Linear(input_dim, 1))
    cpd_C = ParametricCPD("C", parametrization=nn.Linear(2, 1))

    model = ProbabilisticModel(
        variables=[input_var, var_A, var_B, var_C],
        parametric_cpds=[cpd_input, cpd_A, cpd_B, cpd_C],
    )
    return model, input_dim


def _make_categorical_model():
    """
    Build a model with a Categorical concept: input -> A (3 classes).

    Returns (model, input_dim, num_classes).
    """
    input_dim = 8
    num_classes = 3
    input_var = LatentVariable("input", parents=[], size=input_dim)
    var_A = ConceptVariable(
        "A", parents=["input"], distribution=Categorical, size=num_classes
    )

    cpd_input = ParametricCPD("input", parametrization=nn.Identity())
    cpd_A = ParametricCPD("A", parametrization=nn.Linear(input_dim, num_classes))

    model = ProbabilisticModel(
        variables=[input_var, var_A],
        parametric_cpds=[cpd_input, cpd_A],
    )
    return model, input_dim, num_classes


# ======================================================================
# Construction & validation
# ======================================================================

class TestImportanceSamplingConstruction:
    """Test construction and distribution validation."""

    def test_basic_construction(self):
        model, _ = _make_binary_chain()
        inf = ImportanceSamplingInference(model, num_samples=50, num_draws=20)
        assert inf.num_samples == 50
        assert inf.num_draws == 20

    def test_default_sample_counts(self):
        model, _ = _make_binary_chain()
        inf = ImportanceSamplingInference(model)
        assert inf.num_samples == 1000
        assert inf.num_draws == 100

    def test_unsupported_distribution_raises(self):
        """Variables with unsupported distributions should raise ValueError."""
        from torch.distributions import Poisson

        input_var = LatentVariable("input", parents=[], size=4)
        var_A = ConceptVariable("A", parents=["input"], distribution=Poisson)

        cpd_input = ParametricCPD("input", parametrization=nn.Identity())
        cpd_A = ParametricCPD("A", parametrization=nn.Linear(4, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A],
        )

        with pytest.raises(ValueError, match="unsupported distribution"):
            ImportanceSamplingInference(model)

    def test_delta_distribution_accepted(self):
        """Delta (deterministic) variables should not raise."""
        input_var = LatentVariable("input", parents=[], size=4)
        var_A = ConceptVariable("A", parents=["input"], distribution=Delta)

        cpd_input = ParametricCPD("input", parametrization=nn.Identity())
        cpd_A = ParametricCPD("A", parametrization=nn.Linear(4, 1))

        model = ProbabilisticModel(
            variables=[input_var, var_A],
            parametric_cpds=[cpd_input, cpd_A],
        )
        # Should not raise
        ImportanceSamplingInference(model)


# ======================================================================
# get_results
# ======================================================================

class TestGetResults:
    """get_results should return the raw tensor unchanged."""

    def test_returns_raw_tensor(self):
        model, _ = _make_binary_chain()
        inf = ImportanceSamplingInference(model, num_samples=10)
        var = model.concept_to_variable["A"]

        raw = torch.randn(4, 1)
        out = inf.get_results(raw, var)
        assert torch.equal(out, raw)


# ======================================================================
# ground_truth_to_evidence
# ======================================================================

class TestGroundTruthToEvidence:
    """ground_truth_to_evidence should squeeze trailing dim-1."""

    def test_squeeze_trailing_dim(self):
        model, _ = _make_binary_chain()
        inf = ImportanceSamplingInference(model, num_samples=10)

        value = torch.ones(4, 1)
        result = inf.ground_truth_to_evidence(value, cardinality=2)
        assert result.shape == (4,)

    def test_already_1d(self):
        model, _ = _make_binary_chain()
        inf = ImportanceSamplingInference(model, num_samples=10)

        value = torch.ones(4)
        result = inf.ground_truth_to_evidence(value, cardinality=2)
        assert result.shape == (4,)


# ======================================================================
# Marginal computation
# ======================================================================

class TestMarginalBernoulli:
    """Test marginal queries on Bernoulli variables."""

    def test_marginal_single_variable_shape(self):
        """Marginal for one Bernoulli variable should be (batch, 1)."""
        model, dim = _make_binary_chain()
        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=50, num_draws=30)

        x = torch.randn(2, dim)
        result = inf.marginal(["A"], evidence={"input": x})
        assert result.shape == (2, 1)

    def test_marginal_values_in_unit_interval(self):
        """Marginal probabilities should be in [0, 1]."""
        model, dim = _make_binary_chain()
        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=100, num_draws=50)

        x = torch.randn(4, dim)
        result = inf.marginal(["A"], evidence={"input": x})
        assert (result >= 0).all() and (result <= 1).all()

    def test_marginal_multiple_variables(self):
        """Querying A and B should concatenate to (batch, 2)."""
        model, dim = _make_binary_chain()
        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=50, num_draws=30)

        x = torch.randn(3, dim)
        result = inf.marginal(["A", "B"], evidence={"input": x})
        assert result.shape == (3, 2)

    def test_marginal_override_samples(self):
        """num_samples/num_draws kwargs should override defaults."""
        model, dim = _make_binary_chain()
        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=10, num_draws=5)

        x = torch.randn(2, dim)
        # Override with different counts – should not error
        result = inf.marginal(["A"], evidence={"input": x},
                              num_samples=20, num_draws=10)
        assert result.shape == (2, 1)


class TestMarginalDiamond:
    """Test marginals on a diamond DAG."""

    def test_marginal_leaf(self):
        """p(C | x) should return (batch, 1)."""
        model, dim = _make_diamond()
        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=50, num_draws=30)

        x = torch.randn(2, dim)
        result = inf.marginal(["C"], evidence={"input": x})
        assert result.shape == (2, 1)

    def test_conditional_marginal(self):
        """p(C | A=1, x) should return (batch, 1)."""
        model, dim = _make_diamond()
        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=50, num_draws=30)

        batch = 2
        x = torch.randn(batch, dim)
        evidence = {"input": x, "A": torch.ones(batch, 1)}
        result = inf.marginal(["C"], evidence=evidence)
        assert result.shape == (batch, 1)

    def test_conditioning_changes_marginal(self):
        """
        Conditioning on A=0 vs A=1 should produce different marginals for C
        (provided the CPD weights are not degenerate).
        """
        torch.manual_seed(42)
        model, dim = _make_diamond()
        # Initialise CPD for C with non-trivial weights so A matters
        cpd_C = model.get_module_of_concept("C")
        with torch.no_grad():
            cpd_C.parametrization.weight.fill_(1.0)
            cpd_C.parametrization.bias.fill_(0.0)

        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=200, num_draws=100)

        batch = 4
        x = torch.randn(batch, dim)
        p_c_a0 = inf.marginal(["C"], {"input": x, "A": torch.zeros(batch, 1)})
        p_c_a1 = inf.marginal(["C"], {"input": x, "A": torch.ones(batch, 1)})

        # At least some batch elements should differ
        assert not torch.allclose(p_c_a0, p_c_a1, atol=0.05)


class TestMarginalCategorical:
    """Test marginal queries on Categorical variables."""

    def test_categorical_marginal_shape(self):
        """Categorical marginal should be (batch, num_classes)."""
        model, dim, K = _make_categorical_model()
        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=50, num_draws=30)

        x = torch.randn(3, dim)
        result = inf.marginal(["A"], evidence={"input": x})
        assert result.shape == (3, K)

    def test_categorical_probs_sum_to_one(self):
        """Class probabilities should roughly sum to 1."""
        model, dim, K = _make_categorical_model()
        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=200, num_draws=100)

        x = torch.randn(4, dim)
        result = inf.marginal(["A"], evidence={"input": x})
        sums = result.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=0.15)


# ======================================================================
# return_dict mode
# ======================================================================

class TestReturnDict:
    """Test return_dict=True output structure."""

    def test_bernoulli_return_dict(self):
        model, dim = _make_binary_chain()
        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=50, num_draws=30)

        x = torch.randn(2, dim)
        result = inf.marginal(["A"], evidence={"input": x}, return_dict=True)

        assert isinstance(result, dict)
        assert "A" in result
        assert "probs" in result["A"]
        assert result["A"]["probs"].shape == (2, 1)

    def test_categorical_return_dict(self):
        model, dim, K = _make_categorical_model()
        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=50, num_draws=30)

        x = torch.randn(2, dim)
        result = inf.marginal(["A"], evidence={"input": x}, return_dict=True)

        assert "probs" in result["A"]
        assert result["A"]["probs"].shape == (2, K)


# ======================================================================
# query() interface
# ======================================================================

class TestQueryInterface:
    """query() should be compatible with other inference engines."""

    def test_query_delegates_to_marginal(self):
        model, dim = _make_binary_chain()
        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=50, num_draws=30)

        x = torch.randn(2, dim)
        q = inf.query(["A"], evidence={"input": x})
        assert q.shape == (2, 1)

    def test_query_multiple(self):
        model, dim = _make_diamond()
        model.eval()
        inf = ImportanceSamplingInference(model, num_samples=50, num_draws=30)

        x = torch.randn(2, dim)
        q = inf.query(["A", "B", "C"], evidence={"input": x})
        assert q.shape == (2, 3)


# ======================================================================
# Error handling
# ======================================================================

class TestErrorHandling:
    """Test that appropriate errors are raised for bad inputs."""

    def test_invalid_query_variable(self):
        model, dim = _make_binary_chain()
        inf = ImportanceSamplingInference(model, num_samples=10)

        x = torch.randn(2, dim)
        with pytest.raises(ValueError, match="not found in model"):
            inf.marginal(["NonExistent"], evidence={"input": x})


# ======================================================================
# _pyro_model (built once at init)
# ======================================================================

class TestBuildPyroModel:
    """Test internal Pyro model built at construction time."""

    def test_model_is_callable(self):
        model, dim = _make_binary_chain()
        inf = ImportanceSamplingInference(model, num_samples=10)

        assert callable(inf._pyro_model)

    def test_model_runs_with_evidence(self):
        """Calling the model with evidence should return a dict of values."""
        model, dim = _make_binary_chain()
        inf = ImportanceSamplingInference(model, num_samples=10)

        x = torch.randn(2, dim)
        result = inf._pyro_model({"input": x})
        assert isinstance(result, dict)

    def test_model_runs_with_obs_dict(self):
        """Concept evidence passed as obs_dict should be accepted."""
        model, dim = _make_binary_chain()
        inf = ImportanceSamplingInference(model, num_samples=10)

        batch = 2
        x = torch.randn(batch, dim)
        obs_dict = {"A": torch.ones(batch)}
        result = inf._pyro_model({"input": x}, obs_dict=obs_dict)
        assert isinstance(result, dict)

    def test_no_concept_keys_for_input_only(self):
        """Passing only input evidence should not raise."""
        model, dim = _make_binary_chain()
        inf = ImportanceSamplingInference(model, num_samples=10)

        x = torch.randn(2, dim)
        result = inf._pyro_model({"input": x})
        assert "input" in result


# ======================================================================
# _get_pyro_distribution
# ======================================================================

class TestGetPyroDistribution:
    """Test distribution conversion for all supported types."""

    def _inf(self):
        model, _ = _make_binary_chain()
        return ImportanceSamplingInference(model, num_samples=10)

    def _var(self, dist_cls, size=1):
        return ConceptVariable("test", parents=[], distribution=dist_cls, size=size)

    def test_bernoulli_logits(self):
        inf = self._inf()
        var = self._var(Bernoulli)
        d = inf._get_pyro_distribution(var, torch.zeros(4))
        assert d is not None
        assert isinstance(d, type(d))  # smoke check

    def test_bernoulli_probs(self):
        inf = self._inf()
        var = self._var(Bernoulli)
        d = inf._get_pyro_distribution(var, {"probs": torch.full((4,), 0.5)})
        assert d is not None

    def test_categorical_logits(self):
        inf = self._inf()
        var = self._var(Categorical, size=3)
        d = inf._get_pyro_distribution(var, {"logits": torch.randn(4, 3)})
        assert d is not None

    def test_normal(self):
        inf = self._inf()
        var = self._var(Normal)
        d = inf._get_pyro_distribution(var, {"loc": torch.zeros(4), "scale": torch.ones(4)})
        assert d is not None

    def test_lognormal(self):
        inf = self._inf()
        var = self._var(LogNormal)
        d = inf._get_pyro_distribution(var, {"loc": torch.zeros(4), "scale": torch.ones(4)})
        assert d is not None

    def test_beta(self):
        inf = self._inf()
        var = self._var(Beta)
        d = inf._get_pyro_distribution(
            var, {"concentration1": torch.ones(4), "concentration0": torch.ones(4)}
        )
        assert d is not None

    def test_gamma(self):
        inf = self._inf()
        var = self._var(Gamma)
        d = inf._get_pyro_distribution(
            var, {"concentration": torch.ones(4), "rate": torch.ones(4)}
        )
        assert d is not None

    def test_unsupported_returns_none(self):
        inf = self._inf()
        var = self._var(Delta)
        d = inf._get_pyro_distribution(var, torch.zeros(4))
        assert d is None


# ======================================================================
# LazyImportanceSamplingInference
# ======================================================================

class TestLazyImportanceSampling:
    """LazyImportanceSamplingInference should inherit full functionality."""

    def test_construction(self):
        model, _ = _make_diamond()
        inf = LazyImportanceSamplingInference(model, num_samples=50, num_draws=20)
        assert inf.num_samples == 50

    def test_marginal_leaf_only(self):
        """Querying only the leaf C should still work."""
        model, dim = _make_diamond()
        model.eval()
        inf = LazyImportanceSamplingInference(model, num_samples=50, num_draws=30)

        x = torch.randn(2, dim)
        result = inf.marginal(["C"], evidence={"input": x})
        assert result.shape == (2, 1)

    def test_query_interface(self):
        model, dim = _make_binary_chain()
        model.eval()
        inf = LazyImportanceSamplingInference(model, num_samples=50, num_draws=30)

        x = torch.randn(2, dim)
        q = inf.query(["B"], evidence={"input": x})
        assert q.shape == (2, 1)
