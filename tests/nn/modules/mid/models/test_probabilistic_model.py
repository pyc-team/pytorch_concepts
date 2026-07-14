"""Comprehensive tests for BayesianNetwork and ProbabilisticModel.

BayesianNetwork is the concrete subclass of ProbabilisticModel used throughout
the codebase. All public API goes through it.
"""
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable, EmbeddingVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.bayesian_network import BayesianNetwork
from torch_concepts.nn.modules.low.priors import LearnablePrior, FixedPrior
from torch_concepts.distributions import Delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bvar(name, size=1):
    return ConceptVariable(name, distribution=dist.Bernoulli, size=size)

def _dvar(name, size=4):
    return ConceptVariable(name, distribution=Delta, size=size)

def _root_cpd(var):
    return ParametricCPD(variable=var, parametrization={"logits": LearnablePrior(var.size)})

def _nonroot_cpd(var, parents, in_size=None):
    in_size = in_size or sum(p.size for p in parents)
    return ParametricCPD(variable=var, parametrization=nn.Linear(in_size, var.size), parents=parents)

def _simple_model():
    """x (root, delta) -> c (bernoulli)."""
    x = _dvar("x")
    c = _bvar("c")
    cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(x.size)})
    cpd_c = _nonroot_cpd(c, [x])
    return BayesianNetwork(variables=[x, c], factors=[cpd_x, cpd_c])

def _chain_model():
    """x -> a -> b -> c."""
    x = _dvar("x", size=4)
    a = _bvar("a")
    b = _bvar("b")
    c = _bvar("c")
    cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
    cpd_a = _nonroot_cpd(a, [x])
    cpd_b = _nonroot_cpd(b, [a])
    cpd_c = _nonroot_cpd(c, [b])
    return BayesianNetwork(variables=[x, a, b, c], factors=[cpd_x, cpd_a, cpd_b, cpd_c])


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestBayesianNetworkConstruction:
    def test_simple_construction(self):
        m = _simple_model()
        assert isinstance(m, BayesianNetwork)

    def test_variables_is_dict(self):
        m = _simple_model()
        assert isinstance(m.variables, dict)

    def test_variables_keys(self):
        m = _simple_model()
        assert set(m.variables.keys()) == {"x", "c"}

    def test_variables_values_are_variable_objects(self):
        m = _simple_model()
        for v in m.variables.values():
            assert isinstance(v, ConceptVariable)

    def test_factors_is_moduledict(self):
        m = _simple_model()
        assert isinstance(m.factors, nn.ModuleDict)

    def test_factors_keys(self):
        m = _simple_model()
        assert set(m.factors.keys()) == {"x", "c"}

    def test_factors_values_are_cpds(self):
        m = _simple_model()
        for f in m.factors.values():
            assert isinstance(f, ParametricCPD)

    def test_dict_access_variable(self):
        m = _simple_model()
        assert m.variables["x"].name == "x"
        assert m.variables["c"].name == "c"

    def test_dict_access_factor(self):
        m = _simple_model()
        assert m.factors["c"].variable.name == "c"

    def test_guides_registered_initially_empty(self):
        m = _simple_model()
        assert len(m.guides) == 0

    def test_parameters_include_cpd_weights(self):
        m = _simple_model()
        params = list(m.parameters())
        assert len(params) > 0

    def test_wrong_factor_count_raises(self):
        x = _dvar("x")
        with pytest.raises(ValueError, match="exactly one factor per variable"):
            BayesianNetwork(variables=[x], factors=[])

    def test_duplicate_variable_names_raise(self):
        x1 = _dvar("x")
        x2 = _dvar("x")
        cpd1 = ParametricCPD(variable=x1, parametrization={"value": LearnablePrior(4)})
        cpd2 = ParametricCPD(variable=x2, parametrization={"value": LearnablePrior(4)})
        with pytest.raises(ValueError, match="Duplicate"):
            BayesianNetwork(variables=[x1, x2], factors=[cpd1, cpd2])

    def test_factor_for_unregistered_variable_raises(self):
        x = _dvar("x")
        c = _bvar("c")
        cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
        cpd_c = _nonroot_cpd(c, [x])
        # 2 factors but only 1 variable registered — BN rejects mismatched counts
        with pytest.raises(ValueError):
            BayesianNetwork(variables=[x], factors=[cpd_x, cpd_c])

    def test_duplicate_factor_raises(self):
        x = _dvar("x")
        cpd1 = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
        cpd2 = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
        # 2 factors for 1 variable — BN rejects the count mismatch
        with pytest.raises(ValueError):
            BayesianNetwork(variables=[x], factors=[cpd1, cpd2])


# ===========================================================================
# 2. Topological sort and levels
# ===========================================================================

class TestBayesianNetworkTopologicalOrder:
    def test_sorted_variables_length(self):
        m = _chain_model()
        assert len(m.sorted_variables) == 4

    def test_sorted_variables_root_first(self):
        m = _chain_model()
        # x is the only root; it must come first
        assert m.sorted_variables[0].name == "x"

    def test_chain_sorted_order(self):
        m = _chain_model()
        names = [v.name for v in m.sorted_variables]
        # chain x -> a -> b -> c must be in topological order
        assert names.index("x") < names.index("a")
        assert names.index("a") < names.index("b")
        assert names.index("b") < names.index("c")

    def test_levels_chain(self):
        m = _chain_model()
        levels = m.levels
        assert len(levels) == 4
        assert levels[0][0].name == "x"
        assert levels[1][0].name == "a"
        assert levels[2][0].name == "b"
        assert levels[3][0].name == "c"

    def test_levels_diamond(self):
        x = _dvar("x", size=4)
        a = _bvar("a")
        b = _bvar("b")
        c = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
        cpd_a = _nonroot_cpd(a, [x])
        cpd_b = _nonroot_cpd(b, [x])
        cpd_c = ParametricCPD(variable=c, parametrization=nn.Linear(2, 1), parents=[a, b])
        m = BayesianNetwork(variables=[x, a, b, c], factors=[cpd_x, cpd_a, cpd_b, cpd_c])
        levels = m.levels
        assert len(levels) == 3
        assert levels[0][0].name == "x"
        assert {v.name for v in levels[1]} == {"a", "b"}
        assert levels[2][0].name == "c"

    def test_levels_parallel_roots(self):
        a = _bvar("a")
        b = _bvar("b")
        cpd_a = ParametricCPD(variable=a, parametrization={"logits": LearnablePrior(1)})
        cpd_b = ParametricCPD(variable=b, parametrization={"logits": LearnablePrior(1)})
        m = BayesianNetwork(variables=[a, b], factors=[cpd_a, cpd_b])
        levels = m.levels
        assert len(levels) == 1
        assert {v.name for v in levels[0]} == {"a", "b"}

    def test_levels_cached(self):
        m = _simple_model()
        lev1 = m.levels
        lev2 = m.levels
        assert lev1 is lev2  # same object (cached)

    def test_cycle_detection(self):
        a = _bvar("a")
        b = _bvar("b")
        cpd_a = _nonroot_cpd(a, [b])
        cpd_b = _nonroot_cpd(b, [a])
        with pytest.raises(ValueError, match="cycle"):
            BayesianNetwork(variables=[a, b], factors=[cpd_a, cpd_b])


# ===========================================================================
# 3. queryable_names and resolve
# ===========================================================================

class TestBayesianNetworkQueryable:
    def test_queryable_names_includes_variables(self):
        m = _simple_model()
        assert "x" in m.queryable_names
        assert "c" in m.queryable_names

    def test_queryable_names_is_frozenset(self):
        m = _simple_model()
        assert isinstance(m.queryable_names, frozenset)

    def test_resolve_returns_variable(self):
        m = _simple_model()
        v = m.resolve("c")
        assert v.name == "c"
        assert v is m.variables["c"]

    def test_resolve_plate_name(self):
        plate = ConceptVariable("g", members=["m1", "m2"], distribution=dist.Bernoulli)
        x = _dvar("x")
        cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
        cpd_g = ParametricCPD(variable=plate, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        m = BayesianNetwork(variables=[x, plate], factors=[cpd_x, cpd_g])
        assert m.resolve("g") is plate

    def test_resolve_member_name_returns_plate(self):
        plate = ConceptVariable("g", members=["m1", "m2"], distribution=dist.Bernoulli)
        x = _dvar("x")
        cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
        cpd_g = ParametricCPD(variable=plate, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        m = BayesianNetwork(variables=[x, plate], factors=[cpd_x, cpd_g])
        assert m.resolve("m1") is plate
        assert m.resolve("m2") is plate

    def test_queryable_names_includes_plate_members(self):
        plate = ConceptVariable("g", members=["m1", "m2"], distribution=dist.Bernoulli)
        x = _dvar("x")
        cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
        cpd_g = ParametricCPD(variable=plate, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        m = BayesianNetwork(variables=[x, plate], factors=[cpd_x, cpd_g])
        assert "m1" in m.queryable_names
        assert "m2" in m.queryable_names
        assert "g" in m.queryable_names


# ===========================================================================
# 4. _validate_graph — parent validation
# ===========================================================================

class TestBayesianNetworkValidation:
    def test_unknown_parent_raises(self):
        z = _bvar("z")
        c = _bvar("c")
        cpd_c = _nonroot_cpd(c, [z])  # z not in variables
        cpd_z = ParametricCPD(variable=z, parametrization={"logits": LearnablePrior(1)})
        with pytest.raises(ValueError, match="not in variables list"):
            BayesianNetwork(variables=[c], factors=[cpd_c])

    def test_wrong_variable_instance_raises(self):
        x1 = _dvar("x")
        x2 = _dvar("x")  # same name, different object
        c = _bvar("c")
        cpd_x1 = ParametricCPD(variable=x1, parametrization={"value": LearnablePrior(4)})
        cpd_c = _nonroot_cpd(c, [x2])  # x2 not the registered x1
        with pytest.raises(ValueError, match="different Variable instance"):
            BayesianNetwork(variables=[x1, c], factors=[cpd_x1, cpd_c])

    def test_member_handle_parent_valid(self):
        x = _dvar("x")  # create x first so the same object is used in cpd_plate
        plate = ConceptVariable("g", members=["m1", "m2"], distribution=dist.Bernoulli)
        c = _bvar("c")
        cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
        cpd_plate = ParametricCPD(variable=plate, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        # c depends on just m1 of the plate
        m1_handle = plate.member("m1")
        cpd_c = _nonroot_cpd(c, [m1_handle])
        m = BayesianNetwork(variables=[x, plate, c], factors=[cpd_x, cpd_plate, cpd_c])
        assert m is not None  # no exception

    def test_member_handle_unregistered_plate_raises(self):
        other_plate = ConceptVariable("other", members=["m1"], distribution=dist.Bernoulli)
        c = _bvar("c")
        # registered plate is g, but parent is a member of 'other'
        plate = ConceptVariable("g", members=["m1"], distribution=dist.Bernoulli)
        cpd_plate = ParametricCPD(variable=plate, parametrization={"probs": nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())})
        m1 = other_plate.member("m1")
        cpd_c = _nonroot_cpd(c, [m1])
        with pytest.raises(ValueError):
            BayesianNetwork(variables=[plate, c], factors=[cpd_plate, cpd_c])

    def test_duplicate_parent_deduped(self):
        x = _dvar("x")
        c = _bvar("c")
        cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
        cpd_c = ParametricCPD(variable=c, parametrization=nn.Linear(4, 1), parents=[x, x])
        m = BayesianNetwork(variables=[x, c], factors=[cpd_x, cpd_c])
        assert len(m.factors["c"].parents) == 1  # deduped


# ===========================================================================
# 5. Plate-aware levels
# ===========================================================================

class TestBayesianNetworkPlateHandling:
    def test_plate_treated_as_single_variable_in_levels(self):
        x = _dvar("x")
        plate = ConceptVariable("g", members=["a", "b", "c"], distribution=dist.Bernoulli)
        y = _bvar("y")
        cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
        cpd_g = ParametricCPD(variable=plate, parametrization={"probs": nn.Sequential(nn.Linear(4, 3), nn.Sigmoid())}, parents=[x])
        cpd_y = ParametricCPD(variable=y, parametrization=nn.Linear(3, 1), parents=[plate])
        m = BayesianNetwork(variables=[x, plate, y], factors=[cpd_x, cpd_g, cpd_y])
        levels = m.levels
        assert len(levels) == 3
        assert levels[1][0].name == "g"
        assert levels[2][0].name == "y"

    def test_plate_member_as_parent_in_levels(self):
        x = _dvar("x")
        plate = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli)
        y = _bvar("y")
        cpd_x = ParametricCPD(variable=x, parametrization={"value": LearnablePrior(4)})
        cpd_g = ParametricCPD(variable=plate, parametrization={"probs": nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())}, parents=[x])
        # y depends on member 'a' only
        a_handle = plate.member("a")
        cpd_y = _nonroot_cpd(y, [a_handle])
        m = BayesianNetwork(variables=[x, plate, y], factors=[cpd_x, cpd_g, cpd_y])
        levels = m.levels
        assert len(levels) == 3  # x, g, y
