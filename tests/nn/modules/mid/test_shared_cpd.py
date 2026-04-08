"""Comprehensive tests for the shared CPD feature.

Tests cover three layers:
  1. ParametricCPD construction with shared=True
  2. ProbabilisticModel registration & lookup of shared factors
  3. ForwardInference execution, slicing, and concatenation
"""
import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from torch_concepts import ConceptVariable, LatentVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    DeterministicInference,
    LinearConceptToConcept,
    LinearLatentToConcept,
)
from torch_concepts.nn.modules.mid.models.parametric_cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import ProbabilisticModel


# ======================================================================
# Fixtures
# ======================================================================

LATENT_DIM = 8
N_CONCEPTS = 5
N_CLASSES = 3
CONCEPT_NAMES = [f"c{i}" for i in range(N_CONCEPTS)]
BATCH = 4


def _make_encoder():
    return LinearLatentToConcept(in_latent=LATENT_DIM, out_concepts=N_CONCEPTS)


def _make_task_head():
    return LinearConceptToConcept(in_concepts=N_CONCEPTS, out_concepts=N_CLASSES)


def _build_shared_pgm(encoder, task_head):
    """PGM with a single shared CPD for all concepts."""
    input_var = LatentVariable("input", distribution=Delta, size=LATENT_DIM)
    concept_vars = ConceptVariable(CONCEPT_NAMES, distribution=Bernoulli)
    task_var = ConceptVariable("task", distribution=Categorical, size=N_CLASSES)

    cpd_input = ParametricCPD("input", parametrization=nn.Identity())
    cpd_concepts = ParametricCPD(
        CONCEPT_NAMES, parametrization=encoder, parents=["input"], shared=True,
    )
    cpd_task = ParametricCPD("task", parametrization=task_head, parents=CONCEPT_NAMES)

    return ProbabilisticModel(
        variables=[input_var] + concept_vars + [task_var],
        factors=[cpd_input, cpd_concepts, cpd_task],
    )


def _build_single_pgm(encoder, task_head):
    """Equivalent PGM with one CPD per concept (shared=False).

    Copies weights row-by-row from the shared encoder so outputs match.
    """
    input_var = LatentVariable("input", distribution=Delta, size=LATENT_DIM)
    concept_vars = ConceptVariable(CONCEPT_NAMES, distribution=Bernoulli)
    task_var = ConceptVariable("task", distribution=Categorical, size=N_CLASSES)

    cpd_input = ParametricCPD("input", parametrization=nn.Identity())

    concept_cpds = []
    for i, name in enumerate(CONCEPT_NAMES):
        single_enc = LinearLatentToConcept(in_latent=LATENT_DIM, out_concepts=1)
        with torch.no_grad():
            single_enc.encoder.weight.copy_(encoder.encoder.weight[i : i + 1])
            single_enc.encoder.bias.copy_(encoder.encoder.bias[i : i + 1])
        concept_cpds.append(
            ParametricCPD(name, parametrization=single_enc, parents=["input"])
        )

    cpd_task = ParametricCPD("task", parametrization=task_head, parents=CONCEPT_NAMES)

    return ProbabilisticModel(
        variables=[input_var] + concept_vars + [task_var],
        factors=[cpd_input] + concept_cpds + [cpd_task],
    )


# ======================================================================
# 1. ParametricCPD construction
# ======================================================================

class TestParametricCPDSharedConstruction:
    """Test ParametricCPD with shared=True construction semantics."""

    def test_shared_returns_single_instance(self):
        cpd = ParametricCPD(
            CONCEPT_NAMES, parametrization=nn.Linear(8, 5),
            parents=["input"], shared=True,
        )
        assert isinstance(cpd, ParametricCPD)
        assert not isinstance(cpd, list)

    def test_shared_false_returns_list(self):
        cpds = ParametricCPD(
            CONCEPT_NAMES, parametrization=nn.Linear(8, 5),
            parents=["input"], shared=False,
        )
        assert isinstance(cpds, list)
        assert len(cpds) == N_CONCEPTS

    def test_shared_stores_all_concepts(self):
        cpd = ParametricCPD(
            CONCEPT_NAMES, parametrization=nn.Linear(8, 5),
            parents=["input"], shared=True,
        )
        assert cpd.concepts == CONCEPT_NAMES
        assert cpd.concept == CONCEPT_NAMES[0]

    def test_shared_flag_stored(self):
        cpd = ParametricCPD(
            CONCEPT_NAMES, parametrization=nn.Linear(8, 5),
            parents=["input"], shared=True,
        )
        assert cpd.shared is True

    def test_non_shared_flag_stored(self):
        cpds = ParametricCPD(
            CONCEPT_NAMES, parametrization=nn.Linear(8, 5),
            parents=["input"], shared=False,
        )
        for c in cpds:
            assert c.shared is False

    def test_shared_no_deepcopy(self):
        """Shared CPD should reference the original module, not a copy."""
        module = nn.Linear(8, 5)
        cpd = ParametricCPD(
            CONCEPT_NAMES, parametrization=module,
            parents=["input"], shared=True,
        )
        assert cpd.parametrization is module

    def test_non_shared_deepcopies(self):
        """Non-shared CPDs should each have their own copy."""
        module = nn.Linear(8, 5)
        cpds = ParametricCPD(
            CONCEPT_NAMES, parametrization=module,
            parents=["input"], shared=False,
        )
        ids = {id(c.parametrization) for c in cpds}
        assert len(ids) == N_CONCEPTS  # all distinct copies

    def test_shared_rejects_module_list(self):
        with pytest.raises(ValueError, match="single module"):
            ParametricCPD(
                CONCEPT_NAMES,
                parametrization=[nn.Linear(8, 1) for _ in CONCEPT_NAMES],
                parents=["input"],
                shared=True,
            )

    def test_shared_preserves_parents(self):
        cpd = ParametricCPD(
            CONCEPT_NAMES, parametrization=nn.Linear(8, 5),
            parents=["input", "extra"], shared=True,
        )
        parent_names = [p if isinstance(p, str) else p.concept for p in cpd.parents]
        assert parent_names == ["input", "extra"]


# ======================================================================
# 2. ProbabilisticModel registration & lookup
# ======================================================================

class TestProbabilisticModelSharedRegistration:
    """Test ProbabilisticModel with shared CPDs."""

    @pytest.fixture
    def pgm(self):
        return _build_shared_pgm(_make_encoder(), _make_task_head())

    def test_shared_cpd_map_populated(self, pgm):
        assert pgm._shared_cpd_map == {
            "c1": "c0", "c2": "c0", "c3": "c0", "c4": "c0",
        }

    def test_single_pgm_has_empty_shared_map(self):
        pgm = _build_single_pgm(_make_encoder(), _make_task_head())
        assert pgm._shared_cpd_map == {}

    def test_primary_registered_in_factors(self, pgm):
        assert "c0" in pgm.factors

    def test_secondary_not_in_factors(self, pgm):
        for name in CONCEPT_NAMES[1:]:
            assert name not in pgm.factors

    def test_get_module_primary(self, pgm):
        cpd = pgm.get_module_of_concept("c0")
        assert cpd is not None
        assert cpd.shared is True

    def test_get_module_secondary_redirects(self, pgm):
        primary_cpd = pgm.get_module_of_concept("c0")
        for name in CONCEPT_NAMES[1:]:
            assert pgm.get_module_of_concept(name) is primary_cpd

    def test_all_variables_present(self, pgm):
        var_names = {v.concept for v in pgm.variables}
        expected = {"input"} | set(CONCEPT_NAMES) | {"task"}
        assert var_names == expected

    def test_factor_count(self, pgm):
        # input + shared_concepts (1 entry) + task = 3
        assert len(pgm.factors) == 3


# ======================================================================
# 3. Inference: shared vs single equivalence
# ======================================================================

class TestSharedCPDInferenceEquivalence:
    """Verify shared CPD produces identical output to individual CPDs."""

    @pytest.fixture
    def models(self):
        encoder = _make_encoder()
        task_head = _make_task_head()
        pgm_shared = _build_shared_pgm(encoder, task_head)
        pgm_single = _build_single_pgm(encoder, task_head)
        return pgm_shared, pgm_single

    @pytest.fixture
    def x(self):
        torch.manual_seed(42)
        return torch.randn(BATCH, LATENT_DIM)

    def test_full_query_equivalence(self, models, x):
        pgm_shared, pgm_single = models
        inf_shared = DeterministicInference(pgm_shared)
        inf_single = DeterministicInference(pgm_single)

        query = CONCEPT_NAMES + ["task"]
        r_shared = inf_shared.query(query, evidence={"input": x}, debug=True)
        r_single = inf_single.query(query, evidence={"input": x}, debug=True)

        assert torch.allclose(r_shared, r_single, atol=1e-6)

    def test_concept_only_query(self, models, x):
        pgm_shared, pgm_single = models
        inf_shared = DeterministicInference(pgm_shared)
        inf_single = DeterministicInference(pgm_single)

        r_shared = inf_shared.query(CONCEPT_NAMES, evidence={"input": x}, debug=True)
        r_single = inf_single.query(CONCEPT_NAMES, evidence={"input": x}, debug=True)

        assert torch.allclose(r_shared, r_single, atol=1e-6)


# ======================================================================
# 4. Subset queries
# ======================================================================

class TestSharedCPDSubsetQuery:
    """Test querying subsets of shared concepts."""

    @pytest.fixture
    def inf(self):
        pgm = _build_shared_pgm(_make_encoder(), _make_task_head())
        return DeterministicInference(pgm)

    @pytest.fixture
    def x(self):
        torch.manual_seed(0)
        return torch.randn(BATCH, LATENT_DIM)

    def test_single_concept_query(self, inf, x):
        result = inf.query(["c2"], evidence={"input": x}, debug=True)
        assert result.shape == (BATCH, 1)

    def test_subset_query(self, inf, x):
        result = inf.query(["c1", "c3"], evidence={"input": x}, debug=True)
        assert result.shape == (BATCH, 2)

    def test_reordered_query(self, inf, x):
        """Querying in reversed order should return correct values."""
        fwd = inf.query(CONCEPT_NAMES, evidence={"input": x}, debug=True)
        rev = inf.query(list(reversed(CONCEPT_NAMES)), evidence={"input": x}, debug=True)
        assert torch.allclose(fwd, rev.flip(-1), atol=1e-6)

    def test_subset_matches_full(self, inf, x):
        full = inf.query(CONCEPT_NAMES + ["task"], evidence={"input": x}, debug=True)
        subset = inf.query(["c1", "c3"], evidence={"input": x}, debug=True)
        # c1 is index 1, c3 is index 3 in full output
        assert torch.allclose(subset[:, 0:1], full[:, 1:2], atol=1e-6)
        assert torch.allclose(subset[:, 1:2], full[:, 3:4], atol=1e-6)

    def test_task_only_query(self, inf, x):
        """Querying only the task still works (shared concepts computed as ancestors)."""
        result = inf.query(["task"], evidence={"input": x}, debug=True)
        assert result.shape == (BATCH, N_CLASSES)


# ======================================================================
# 5. Mixed shared and individual CPDs at the same level
# ======================================================================

class TestMixedSharedAndIndividualCPDs:
    """Test a PGM with both shared and individual CPDs at the same level."""

    @pytest.fixture
    def setup(self):
        torch.manual_seed(99)
        shared_names = ["s0", "s1", "s2"]
        indiv_names = ["a", "b"]
        all_names = shared_names + indiv_names

        input_var = LatentVariable("input", distribution=Delta, size=8)
        shared_vars = ConceptVariable(shared_names, distribution=Bernoulli)
        indiv_vars = ConceptVariable(indiv_names, distribution=Bernoulli)
        task_var = ConceptVariable("task", distribution=Categorical, size=3)

        cpd_input = ParametricCPD("input", parametrization=nn.Identity())
        cpd_shared = ParametricCPD(
            shared_names, parametrization=LinearLatentToConcept(8, 3),
            parents=["input"], shared=True,
        )
        cpd_a = ParametricCPD("a", parametrization=LinearLatentToConcept(8, 1), parents=["input"])
        cpd_b = ParametricCPD("b", parametrization=LinearLatentToConcept(8, 1), parents=["input"])
        cpd_task = ParametricCPD(
            "task", parametrization=LinearConceptToConcept(5, 3), parents=all_names,
        )

        pgm = ProbabilisticModel(
            variables=[input_var] + shared_vars + indiv_vars + [task_var],
            factors=[cpd_input, cpd_shared, cpd_a, cpd_b, cpd_task],
        )
        inf = DeterministicInference(pgm)
        x = torch.randn(BATCH, 8)
        return inf, pgm, x, shared_names, indiv_names, all_names

    def test_shared_map_only_shared(self, setup):
        inf, pgm, x, shared_names, indiv_names, _ = setup
        assert set(pgm._shared_cpd_map.keys()) == {"s1", "s2"}
        assert all(v == "s0" for v in pgm._shared_cpd_map.values())

    def test_individual_not_in_shared_map(self, setup):
        _, pgm, _, _, indiv_names, _ = setup
        for name in indiv_names:
            assert name not in pgm._shared_cpd_map

    def test_full_query(self, setup):
        inf, _, x, _, _, all_names = setup
        result = inf.query(all_names + ["task"], evidence={"input": x}, debug=True)
        assert result.shape == (BATCH, 5 + 3)  # 5 concepts (3 shared + 2 indiv) + 3 task classes

    def test_shared_only_query(self, setup):
        inf, _, x, shared_names, _, _ = setup
        result = inf.query(shared_names, evidence={"input": x}, debug=True)
        assert result.shape == (BATCH, 3)

    def test_individual_only_query(self, setup):
        inf, _, x, _, indiv_names, _ = setup
        result = inf.query(indiv_names, evidence={"input": x}, debug=True)
        assert result.shape == (BATCH, 2)

    def test_cross_query(self, setup):
        """Query mixing shared and individual concepts."""
        inf, _, x, _, _, all_names = setup
        full = inf.query(all_names, evidence={"input": x}, debug=True)
        mixed = inf.query(["s0", "a", "s2"], evidence={"input": x}, debug=True)
        # s0 = col 0, a = col 3, s2 = col 2 in full
        assert torch.allclose(mixed[:, 0:1], full[:, 0:1], atol=1e-6)
        assert torch.allclose(mixed[:, 1:2], full[:, 3:4], atol=1e-6)
        assert torch.allclose(mixed[:, 2:3], full[:, 2:3], atol=1e-6)

    def test_task_query_uses_all_parents(self, setup):
        """Task depends on all 5 concepts; querying task alone should work."""
        inf, _, x, _, _, _ = setup
        result = inf.query(["task"], evidence={"input": x}, debug=True)
        assert result.shape == (BATCH, 3)


# ======================================================================
# 6. Gradient flow
# ======================================================================

class TestSharedCPDGradientFlow:
    """Verify gradients flow through shared CPD correctly."""

    def test_gradient_reaches_shared_encoder(self):
        encoder = _make_encoder()
        task_head = _make_task_head()
        pgm = _build_shared_pgm(encoder, task_head)
        inf = DeterministicInference(pgm)

        x = torch.randn(BATCH, LATENT_DIM)
        result = inf.query(CONCEPT_NAMES + ["task"], evidence={"input": x}, debug=True)
        loss = result.sum()
        loss.backward()

        assert encoder.encoder.weight.grad is not None
        assert encoder.encoder.weight.grad.abs().sum() > 0

    def test_gradient_same_as_single(self):
        """Gradients through shared CPD match those through individual CPDs."""
        torch.manual_seed(7)
        encoder = _make_encoder()
        task_head = _make_task_head()

        pgm_shared = _build_shared_pgm(encoder, task_head)
        pgm_single = _build_single_pgm(encoder, task_head)

        inf_shared = DeterministicInference(pgm_shared)
        inf_single = DeterministicInference(pgm_single)

        x = torch.randn(BATCH, LATENT_DIM)
        query = CONCEPT_NAMES + ["task"]

        r_shared = inf_shared.query(query, evidence={"input": x}, debug=True)
        r_single = inf_single.query(query, evidence={"input": x}, debug=True)

        r_shared.sum().backward()
        r_single.sum().backward()

        # Task head is the same object in both PGMs, so gradients accumulate.
        # Compare encoder gradients: shared encoder grad vs sum of individual encoder grads.
        shared_grad = encoder.encoder.weight.grad.clone()

        individual_grads = []
        for cpd_name in CONCEPT_NAMES:
            cpd = pgm_single.get_module_of_concept(cpd_name)
            individual_grads.append(cpd.parametrization.encoder.weight.grad.clone())
        stacked = torch.cat(individual_grads, dim=0)

        assert torch.allclose(shared_grad, stacked, atol=1e-5)


# ======================================================================
# 7. Multiple shared groups
# ======================================================================

class TestMultipleSharedGroups:
    """Test PGM with two separate shared CPD groups."""

    @pytest.fixture
    def setup(self):
        torch.manual_seed(11)
        group_a = ["a0", "a1", "a2"]
        group_b = ["b0", "b1"]
        all_concepts = group_a + group_b

        input_var = LatentVariable("input", distribution=Delta, size=8)
        vars_a = ConceptVariable(group_a, distribution=Bernoulli)
        vars_b = ConceptVariable(group_b, distribution=Bernoulli)
        task_var = ConceptVariable("task", distribution=Categorical, size=3)

        cpd_input = ParametricCPD("input", parametrization=nn.Identity())
        cpd_a = ParametricCPD(
            group_a, parametrization=LinearLatentToConcept(8, 3),
            parents=["input"], shared=True,
        )
        cpd_b = ParametricCPD(
            group_b, parametrization=LinearLatentToConcept(8, 2),
            parents=["input"], shared=True,
        )
        cpd_task = ParametricCPD(
            "task", parametrization=LinearConceptToConcept(5, 3), parents=all_concepts,
        )

        pgm = ProbabilisticModel(
            variables=[input_var] + vars_a + vars_b + [task_var],
            factors=[cpd_input, cpd_a, cpd_b, cpd_task],
        )
        inf = DeterministicInference(pgm)
        x = torch.randn(BATCH, 8)
        return inf, pgm, x, group_a, group_b, all_concepts

    def test_two_primaries_in_map(self, setup):
        _, pgm, _, group_a, group_b, _ = setup
        assert pgm._shared_cpd_map == {
            "a1": "a0", "a2": "a0",
            "b1": "b0",
        }

    def test_separate_cpd_instances(self, setup):
        _, pgm, _, _, _, _ = setup
        cpd_a = pgm.get_module_of_concept("a0")
        cpd_b = pgm.get_module_of_concept("b0")
        assert cpd_a is not cpd_b

    def test_full_query(self, setup):
        inf, _, x, _, _, all_concepts = setup
        result = inf.query(all_concepts + ["task"], evidence={"input": x}, debug=True)
        assert result.shape == (BATCH, 5 + 3)

    def test_cross_group_query(self, setup):
        inf, _, x, _, _, all_concepts = setup
        full = inf.query(all_concepts, evidence={"input": x}, debug=True)
        cross = inf.query(["a2", "b0"], evidence={"input": x}, debug=True)
        assert torch.allclose(cross[:, 0:1], full[:, 2:3], atol=1e-6)
        assert torch.allclose(cross[:, 1:2], full[:, 3:4], atol=1e-6)


# ======================================================================
# 8. Categorical shared concepts (size > 1 per concept)
# ======================================================================

class TestSharedCPDCategoricalConcepts:
    """Test shared CPD where each concept has size > 1 (Categorical)."""

    @pytest.fixture
    def setup(self):
        torch.manual_seed(77)
        names = ["color", "shape"]  # each has 3 classes
        input_var = LatentVariable("input", distribution=Delta, size=8)
        concept_vars = ConceptVariable(names, distribution=Categorical, size=3)
        task_var = ConceptVariable("task", distribution=Categorical, size=2)

        encoder = nn.Linear(8, 6)  # 3 + 3 = 6
        cpd_input = ParametricCPD("input", parametrization=nn.Identity())
        cpd_concepts = ParametricCPD(
            names, parametrization=encoder, parents=["input"], shared=True,
        )
        cpd_task = ParametricCPD(
            "task", parametrization=LinearConceptToConcept(6, 2), parents=names,
        )

        pgm = ProbabilisticModel(
            variables=[input_var] + concept_vars + [task_var],
            factors=[cpd_input, cpd_concepts, cpd_task],
        )
        inf = DeterministicInference(pgm)
        x = torch.randn(BATCH, 8)
        return inf, pgm, x, names, encoder

    def test_output_shape(self, setup):
        inf, _, x, names, _ = setup
        result = inf.query(names + ["task"], evidence={"input": x}, debug=True)
        # color(3) + shape(3) + task(2) = 8
        assert result.shape == (BATCH, 8)

    def test_slicing_categorical(self, setup):
        inf, _, x, names, encoder = setup
        full = inf.query(names, evidence={"input": x}, debug=True)
        color_only = inf.query(["color"], evidence={"input": x}, debug=True)
        shape_only = inf.query(["shape"], evidence={"input": x}, debug=True)
        # color is first 3 cols, shape is next 3
        assert torch.allclose(color_only, full[:, :3], atol=1e-6)
        assert torch.allclose(shape_only, full[:, 3:6], atol=1e-6)


# ======================================================================
# 9. Parallel execution (non-debug mode)
# ======================================================================

class TestSharedCPDParallelExecution:
    """Test that shared CPDs work in non-debug (parallel) mode."""

    def test_parallel_matches_sequential(self):
        torch.manual_seed(33)
        encoder = _make_encoder()
        task_head = _make_task_head()
        pgm = _build_shared_pgm(encoder, task_head)
        inf = DeterministicInference(pgm)

        x = torch.randn(BATCH, LATENT_DIM)
        query = CONCEPT_NAMES + ["task"]

        r_debug = inf.query(query, evidence={"input": x}, debug=True)
        r_parallel = inf.query(query, evidence={"input": x}, debug=False)

        assert torch.allclose(r_debug, r_parallel, atol=1e-6)


# ======================================================================
# 10. Edge cases
# ======================================================================

class TestSharedCPDEdgeCases:
    """Edge cases for shared CPD."""

    def test_single_concept_shared_true(self):
        """shared=True with a single-element list should work like shared=False."""
        cpd = ParametricCPD(
            ["only"], parametrization=nn.Linear(8, 1),
            parents=["input"], shared=True,
        )
        assert isinstance(cpd, ParametricCPD)
        assert cpd.shared is True
        assert cpd.concepts == ["only"]

    def test_single_concept_shared_in_pgm(self):
        """A single-concept shared CPD in a full PGM."""
        input_var = LatentVariable("input", distribution=Delta, size=8)
        concept_var = ConceptVariable(["x"], distribution=Bernoulli)
        cpd_input = ParametricCPD("input", parametrization=nn.Identity())
        cpd_x = ParametricCPD(
            ["x"], parametrization=nn.Linear(8, 1), parents=["input"], shared=True,
        )
        pgm = ProbabilisticModel(
            variables=[input_var] + concept_var,
            factors=[cpd_input, cpd_x],
        )
        inf = DeterministicInference(pgm)
        result = inf.query(["x"], evidence={"input": torch.randn(2, 8)}, debug=True)
        assert result.shape == (2, 1)

    def test_integer_concept_names(self):
        """Shared CPD with integer concept names (requires str() conversion)."""
        names = [0, 1, 2]
        input_var = LatentVariable("input", distribution=Delta, size=4)
        concept_vars = ConceptVariable(names, distribution=Bernoulli)

        cpd_input = ParametricCPD("input", parametrization=nn.Identity())
        cpd = ParametricCPD(
            names, parametrization=nn.Linear(4, 3), parents=["input"], shared=True,
        )

        pgm = ProbabilisticModel(
            variables=[input_var] + concept_vars,
            factors=[cpd_input, cpd],
        )
        inf = DeterministicInference(pgm)
        result = inf.query(names, evidence={"input": torch.randn(2, 4)}, debug=True)
        assert result.shape == (2, 3)
