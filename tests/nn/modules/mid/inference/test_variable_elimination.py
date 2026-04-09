"""
Comprehensive tests for VariableEliminationInference.

Tests cover:
- Unbatched marginal queries (single + all variables)
- Batched queries (input-conditioned CPDs)
- Evidence conditioning (set_evidence path)
- return_logits mode
- return_log_joint mode
- Elimination ordering: min-degree heuristic and user-provided
- Order caching
- ground_truth_to_evidence pass-through
- _factor_to_tensor for binary and categorical variables
- Gradient flow through the full VE pipeline
- Numerical correctness against brute-force enumeration
"""
import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.mid.inference.variable_elimination import (
    VariableEliminationInference,
    _min_degree_order,
)
from torch_concepts.nn.modules.mid.models.variable import (
    ConceptVariable,
    LatentVariable,
)
from torch_concepts.nn.modules.mid.models.parametric_cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel
from torch_concepts.nn.modules.mid.models.factor import Factor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary_chain(latent_dim=4):
    """Build input -> A -> B, both Bernoulli."""
    input_var = LatentVariable('input', distribution=Delta, size=latent_dim)
    var_A = ConceptVariable('A', distribution=Bernoulli, size=1)
    var_B = ConceptVariable('B', distribution=Bernoulli, size=1)

    cpd_input = ParametricCPD('input', parametrization=nn.Identity())
    cpd_A = ParametricCPD('A', parametrization=nn.Linear(latent_dim, 1),
                          parents=['input'])
    cpd_B = ParametricCPD('B', parametrization=nn.Linear(1, 1),
                          parents=['A'])

    pgm = ProbabilisticModel(
        variables=[input_var, var_A, var_B],
        factors=[cpd_input, cpd_A, cpd_B],
    )
    return pgm


def _make_diamond(latent_dim=4):
    """Build input -> A -> C, input -> B -> C (diamond / v-structure)."""
    input_var = LatentVariable('input', distribution=Delta, size=latent_dim)
    var_A = ConceptVariable('A', distribution=Bernoulli, size=1)
    var_B = ConceptVariable('B', distribution=Bernoulli, size=1)
    var_C = ConceptVariable('C', distribution=Bernoulli, size=1)

    cpd_input = ParametricCPD('input', parametrization=nn.Identity())
    cpd_A = ParametricCPD('A', parametrization=nn.Linear(latent_dim, 1),
                          parents=['input'])
    cpd_B = ParametricCPD('B', parametrization=nn.Linear(latent_dim, 1),
                          parents=['input'])
    cpd_C = ParametricCPD('C', parametrization=nn.Linear(2, 1),
                          parents=['A', 'B'])

    pgm = ProbabilisticModel(
        variables=[input_var, var_A, var_B, var_C],
        factors=[cpd_input, cpd_A, cpd_B, cpd_C],
    )
    return pgm


def _make_categorical_chain(latent_dim=4, k=3):
    """Build input -> A (Categorical K) -> B (Bernoulli)."""
    input_var = LatentVariable('input', distribution=Delta, size=latent_dim)
    var_A = ConceptVariable('A', distribution=Categorical, size=k)
    var_B = ConceptVariable('B', distribution=Bernoulli, size=1)

    cpd_input = ParametricCPD('input', parametrization=nn.Identity())
    cpd_A = ParametricCPD('A', parametrization=nn.Linear(latent_dim, k),
                          parents=['input'])
    cpd_B = ParametricCPD('B', parametrization=nn.Linear(k, 1),
                          parents=['A'])

    pgm = ProbabilisticModel(
        variables=[input_var, var_A, var_B],
        factors=[cpd_input, cpd_A, cpd_B],
    )
    return pgm


# ===========================================================================
# _min_degree_order
# ===========================================================================

class TestMinDegreeOrder:
    """Tests for the min-degree elimination ordering heuristic."""

    def test_returns_all_variables(self):
        cards = {'A': 2, 'B': 2, 'C': 2}
        fa = Factor(torch.ones(2, 2), ['A', 'B'], cards)
        fb = Factor(torch.ones(2, 2), ['B', 'C'], cards)
        order = _min_degree_order([fa, fb], ['A', 'B', 'C'])
        assert set(order) == {'A', 'B', 'C'}
        assert len(order) == 3

    def test_leaf_eliminated_before_hub(self):
        """In a chain A-B-C, the leaves A and C have degree 1
        while B has degree 2 — B should not be eliminated first."""
        cards = {'A': 2, 'B': 2, 'C': 2}
        fa = Factor(torch.ones(2, 2), ['A', 'B'], cards)
        fb = Factor(torch.ones(2, 2), ['B', 'C'], cards)
        order = _min_degree_order([fa, fb], ['A', 'B', 'C'])
        # B has the highest degree, so it should not be first
        assert order[0] != 'B'

    def test_single_variable(self):
        f = Factor(torch.ones(2), ['A'], {'A': 2})
        order = _min_degree_order([f], ['A'])
        assert order == ['A']

    def test_empty_elimination(self):
        f = Factor(torch.ones(2), ['A'], {'A': 2})
        order = _min_degree_order([f], [])
        assert order == []


# ===========================================================================
# query — basic
# ===========================================================================

class TestVEQueryUnbatched:
    """Unbatched queries (no input tensor)."""

    def test_marginal_single_var_no_input(self):
        """Query P(A) from a chain A -> B without input."""
        var_A = ConceptVariable('A', distribution=Bernoulli, size=1)
        var_B = ConceptVariable('B', distribution=Bernoulli, size=1)
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(1, 1))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(1, 1),
                              parents=['A'])
        pgm = ProbabilisticModel(
            variables=[var_A, var_B],
            factors=[cpd_A, cpd_B],
        )
        ve = VariableEliminationInference(pgm)
        result = ve.query(['A'])
        assert result.ndim == 1  # (n_features,)
        assert result.shape[-1] == 1  # binary → 1 column

    def test_marginal_probs_sum_to_one(self):
        """P(A=0) + P(A=1) = 1 for Bernoulli."""
        var_A = ConceptVariable('A', distribution=Bernoulli, size=1)
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(1, 1))
        pgm = ProbabilisticModel(variables=[var_A], factors=[cpd_A])
        ve = VariableEliminationInference(pgm)
        result = ve.query(['A'])  # P(A=1)
        p1 = result.item()
        assert 0.0 <= p1 <= 1.0

    def test_conditional_with_evidence(self):
        """Query P(B | A=1) from A -> B."""
        var_A = ConceptVariable('A', distribution=Bernoulli, size=1)
        var_B = ConceptVariable('B', distribution=Bernoulli, size=1)
        cpd_A = ParametricCPD('A', parametrization=nn.Linear(1, 1))
        cpd_B = ParametricCPD('B', parametrization=nn.Linear(1, 1),
                              parents=['A'])
        pgm = ProbabilisticModel(
            variables=[var_A, var_B], factors=[cpd_A, cpd_B])
        ve = VariableEliminationInference(pgm)
        result = ve.query(['B'], evidence={'A': 1})
        p1 = result.item()
        assert 0.0 <= p1 <= 1.0


class TestVEQueryBatched:
    """Batched queries with input-conditioned CPDs."""

    def test_batched_output_shape(self):
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(8, 4)
        result = ve.query(['A', 'B'], evidence={'input': x})
        assert result.shape == (8, 2)  # 2 binary → 2 columns

    def test_batched_probs_valid(self):
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        result = ve.query(['A', 'B'], evidence={'input': x})
        assert (result >= 0).all() and (result <= 1).all()

    def test_batched_with_evidence(self):
        pgm = _make_diamond()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        result = ve.query(['C'], evidence={'input': x, 'A': 1})
        assert result.shape == (4, 1)
        assert (result >= 0).all() and (result <= 1).all()


# ===========================================================================
# query — return modes
# ===========================================================================

class TestVEReturnModes:
    """Test return_logits and return_log_joint."""

    def test_return_logits_binary(self):
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        logits = ve.query(['A', 'B'], evidence={'input': x},
                          return_logits=True)
        # Logits can be any real number
        assert logits.shape == (4, 2)
        # Applying sigmoid should recover probabilities
        probs = torch.sigmoid(logits)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_return_logits_matches_probs(self):
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        probs = ve.query(['A', 'B'], evidence={'input': x})
        logits = ve.query(['A', 'B'], evidence={'input': x},
                          return_logits=True)
        recovered = torch.sigmoid(logits)
        torch.testing.assert_close(recovered, probs, atol=1e-5, rtol=1e-5)

    def test_return_log_joint_keys(self):
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        out = ve.query(['A', 'B'], evidence={'input': x},
                       return_log_joint=True)
        assert isinstance(out, dict)
        assert 'log_joint' in out
        assert 'logits' in out

    def test_return_log_joint_shape(self):
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        out = ve.query(['A', 'B'], evidence={'input': x},
                       return_log_joint=True)
        assert out['log_joint'].shape == (4, 2, 2)  # (batch, card_A, card_B)
        assert out['logits'].shape == (4, 2)

    def test_log_joint_sums_to_one(self):
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        out = ve.query(['A', 'B'], evidence={'input': x},
                       return_log_joint=True)
        joint = torch.exp(out['log_joint'])
        sums = joint.sum(dim=(1, 2))
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_return_log_joint_takes_precedence(self):
        """return_log_joint=True should return dict even if return_logits=True."""
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        out = ve.query(['A', 'B'], evidence={'input': x},
                       return_logits=True, return_log_joint=True)
        assert isinstance(out, dict)


# ===========================================================================
# Categorical variables
# ===========================================================================

class TestVECategorical:
    """Tests with categorical variables."""

    def test_categorical_prob_shape(self):
        pgm = _make_categorical_chain(k=3)
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        result = ve.query(['A', 'B'], evidence={'input': x})
        # Categorical(3) → 3 columns + Bernoulli → 1 column = 4
        assert result.shape == (4, 4)

    def test_categorical_probs_sum_to_one(self):
        pgm = _make_categorical_chain(k=3)
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        result = ve.query(['A', 'B'], evidence={'input': x})
        cat_probs = result[:, :3]  # first 3 columns for A
        sums = cat_probs.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_categorical_logits(self):
        pgm = _make_categorical_chain(k=3)
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        logits = ve.query(['A', 'B'], evidence={'input': x},
                          return_logits=True)
        assert logits.shape == (4, 4)

    def test_log_joint_shape_categorical(self):
        pgm = _make_categorical_chain(k=3)
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        out = ve.query(['A', 'B'], evidence={'input': x},
                       return_log_joint=True)
        # A has 3 states, B has 2 states → (batch, 3, 2)
        assert out['log_joint'].shape == (4, 3, 2)


# ===========================================================================
# Elimination ordering
# ===========================================================================

class TestEliminationOrdering:
    """Test user-provided and cached orderings."""

    def test_user_provided_order(self):
        pgm = _make_diamond()
        ve = VariableEliminationInference(pgm,
                                          elimination_order=['A', 'B', 'C'])
        x = torch.randn(4, 4)
        result = ve.query(['C'], evidence={'input': x})
        assert result.shape == (4, 1)

    def test_order_caching(self):
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        # First call fills cache
        ve.query(['A', 'B'], evidence={'input': x})
        assert len(ve._order_cache) == 1
        # Second call with same query/evidence pattern uses cache
        ve.query(['A', 'B'], evidence={'input': x})
        assert len(ve._order_cache) == 1

    def test_different_queries_separate_cache(self):
        pgm = _make_diamond()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        ve.query(['A'], evidence={'input': x})
        ve.query(['C'], evidence={'input': x})
        assert len(ve._order_cache) == 2


# ===========================================================================
# ground_truth_to_evidence
# ===========================================================================

class TestGroundTruthToEvidence:
    """VE's ground_truth_to_evidence is identity."""

    def test_passthrough(self):
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        t = torch.tensor([0, 1, 0, 1])
        result = ve.ground_truth_to_evidence(t, cardinality=2)
        assert result is t


# ===========================================================================
# Gradient flow
# ===========================================================================

class TestVEGradientFlow:
    """Ensure VE is fully differentiable."""

    def test_grad_through_query(self):
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        result = ve.query(['A', 'B'], evidence={'input': x})
        result.sum().backward()
        # CPD parameters should have gradients
        for p in pgm.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_grad_through_log_joint(self):
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        out = ve.query(['A', 'B'], evidence={'input': x},
                       return_log_joint=True)
        out['log_joint'].sum().backward()
        for p in pgm.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_grad_through_logits(self):
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)
        logits = ve.query(['A', 'B'], evidence={'input': x},
                          return_logits=True)
        logits.sum().backward()
        for p in pgm.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ===========================================================================
# Numerical correctness — brute-force comparison
# ===========================================================================

class TestVENumericalCorrectness:
    """Compare VE results against brute-force enumeration."""

    def _brute_force_joint(self, pgm, x):
        """Compute joint by building all factors and multiplying them."""
        factors = pgm.build_factors(input=x)
        result = factors[0]
        for f in factors[1:]:
            result = result.product(f)
        _, normalised = result.normalize()
        return normalised

    def test_marginal_matches_brute_force(self):
        """P(A) from VE should match marginalising B from joint."""
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)

        # VE result
        p_a_ve = ve.query(['A'], evidence={'input': x})

        # Brute-force: build joint P(A,B), marginalise B
        joint = self._brute_force_joint(pgm, x)
        # joint has variables ['A', 'B'], shape (batch, 2, 2)
        p_a_bf = joint.values.sum(dim=2)[:, 1]  # P(A=1)

        torch.testing.assert_close(p_a_ve.squeeze(), p_a_bf,
                                   atol=1e-5, rtol=1e-5)

    def test_conditional_matches_brute_force(self):
        """P(B|A=1) from VE should match slicing and normalising joint."""
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)

        # VE
        p_b_given_a1 = ve.query(['B'], evidence={'input': x, 'A': 1})

        # Brute-force: P(A=1, B) / P(A=1)
        joint = self._brute_force_joint(pgm, x)
        p_a1_b = joint.values[:, 1, :]  # (batch, 2) — P(A=1, B=0) and P(A=1, B=1)
        p_a1 = p_a1_b.sum(dim=1, keepdim=True)
        p_b_given_a1_bf = (p_a1_b / p_a1)[:, 1]  # P(B=1 | A=1)

        torch.testing.assert_close(p_b_given_a1.squeeze(), p_b_given_a1_bf,
                                   atol=1e-5, rtol=1e-5)

    def test_log_joint_matches_brute_force(self):
        """log P(A,B) from return_log_joint should match log of brute-force joint."""
        pgm = _make_binary_chain()
        ve = VariableEliminationInference(pgm)
        x = torch.randn(4, 4)

        out = ve.query(['A', 'B'], evidence={'input': x},
                       return_log_joint=True)
        bf = self._brute_force_joint(pgm, x)
        expected_log = torch.log(bf.values.clamp(min=1e-10))

        torch.testing.assert_close(out['log_joint'], expected_log,
                                   atol=1e-5, rtol=1e-5)
