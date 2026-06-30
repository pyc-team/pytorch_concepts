"""Comprehensive tests for ParametricCPD."""
import copy
import pytest
import torch
import torch.nn as nn
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import ConceptVariable, EmbeddingVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.low.priors import LearnablePrior, FixedPrior
from torch_concepts.distributions import Delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bernoulli_var(name="c", size=1):
    return ConceptVariable(name, distribution=dist.Bernoulli, size=size)

def _delta_var(name="x", size=4):
    return ConceptVariable(name, distribution=Delta, size=size)

def _normal_var(name="n", size=2):
    return ConceptVariable(name, distribution=dist.Normal, size=size)

def _cat_var(name="k", size=3):
    return ConceptVariable(name, distribution=dist.OneHotCategorical, size=size)

def _plate_var(name="concepts", members=None, size=1):
    members = members or ["c1", "c2", "c3"]
    return ConceptVariable(name, members=members, distribution=dist.Bernoulli, size=size)


# ===========================================================================
# 1. Construction: single Variable path
# ===========================================================================

class TestParametricCPDConstruction:
    def test_module_shorthand_bernoulli(self):
        v = _bernoulli_var()
        cpd = ParametricCPD(variable=v, parametrization=nn.Sigmoid())
        assert isinstance(cpd, ParametricCPD)
        assert "probs" in cpd.parametrization

    def test_module_shorthand_delta(self):
        v = _delta_var()
        cpd = ParametricCPD(variable=v, parametrization=nn.Identity())
        assert "value" in cpd.parametrization

    def test_dict_parametrization(self):
        v = _normal_var()
        cpd = ParametricCPD(
            variable=v,
            parametrization={"loc": nn.Linear(4, 2), "scale": nn.Softplus()},
        )
        assert set(cpd.parametrization.keys()) == {"loc", "scale"}

    def test_dict_logits_for_bernoulli(self):
        v = _bernoulli_var()
        cpd = ParametricCPD(variable=v, parametrization={"logits": nn.Linear(4, 1)})
        assert "logits" in cpd.parametrization

    def test_variable_stored(self):
        v = _bernoulli_var("x")
        cpd = ParametricCPD(variable=v, parametrization={"probs": nn.Sigmoid()})
        assert cpd.variable is v

    def test_parents_stored(self):
        p = _delta_var()
        c = _bernoulli_var()
        cpd = ParametricCPD(variable=c, parametrization=nn.Linear(4, 1), parents=[p])
        assert cpd.parents == [p]

    def test_parents_default_empty(self):
        v = _bernoulli_var()
        cpd = ParametricCPD(variable=v, parametrization={"logits": LearnablePrior(1)})
        assert cpd.parents == []

    def test_is_root_true_when_no_parents(self):
        v = _bernoulli_var()
        cpd = ParametricCPD(variable=v, parametrization={"logits": LearnablePrior(1)})
        assert cpd.is_root is True

    def test_is_root_false_when_parents(self):
        p = _delta_var()
        c = _bernoulli_var()
        cpd = ParametricCPD(variable=c, parametrization=nn.Linear(4, 1), parents=[p])
        assert cpd.is_root is False

    def test_parametrization_is_nn_moduledict(self):
        v = _bernoulli_var()
        cpd = ParametricCPD(variable=v, parametrization={"probs": nn.Sigmoid()})
        # parametrization stored as an nn.ModuleDict (from ParametricFactor)
        assert hasattr(cpd, "parametrization")

    def test_parametrization_none_raises(self):
        v = _bernoulli_var()
        with pytest.raises(ValueError, match="parametrization.*required"):
            ParametricCPD(variable=v)

    def test_invalid_param_keys_raise(self):
        v = _bernoulli_var()
        with pytest.raises(ValueError, match="invalid parametrization keys"):
            ParametricCPD(variable=v, parametrization={"invalid_key": nn.Linear(4, 1)})

    def test_empty_param_dict_raises(self):
        v = _bernoulli_var()
        with pytest.raises(ValueError, match="must not be empty"):
            ParametricCPD(variable=v, parametrization={})

    def test_parametrization_value_must_be_module(self):
        v = _bernoulli_var()
        with pytest.raises(TypeError):
            ParametricCPD(variable=v, parametrization={"probs": "not_a_module"})

    def test_parent_must_be_variable(self):
        v = _bernoulli_var()
        with pytest.raises(TypeError):
            ParametricCPD(variable=v, parametrization=nn.Linear(4, 1), parents=["not_a_var"])

    def test_multi_param_module_shorthand_raises_for_normal(self):
        v = _normal_var()
        with pytest.raises(ValueError, match="multiple parameters"):
            ParametricCPD(variable=v, parametrization=nn.Linear(4, 2))

    def test_plate_variable_accepted(self):
        plate = _plate_var()
        cpd = ParametricCPD(
            variable=plate,
            parametrization={"probs": LearnablePrior(plate.size)},
        )
        assert cpd.variable is plate


# ===========================================================================
# 2. Construction: list-of-Variables path
# ===========================================================================

class TestParametricCPDListPath:
    def test_list_returns_list(self):
        vs = ConceptVariable(["c1", "c2", "c3"], distribution=dist.Bernoulli, size=1)
        result = ParametricCPD(variable=vs, parametrization=nn.Linear(4, 1))
        assert isinstance(result, list)
        assert len(result) == 3

    def test_list_each_is_cpd(self):
        vs = ConceptVariable(["c1", "c2"], distribution=dist.Bernoulli, size=1)
        result = ParametricCPD(variable=vs, parametrization=nn.Linear(4, 1))
        assert all(isinstance(c, ParametricCPD) for c in result)

    def test_list_modules_deep_copied(self):
        vs = ConceptVariable(["c1", "c2", "c3"], distribution=dist.Bernoulli, size=1)
        module = nn.Linear(4, 1)
        result = ParametricCPD(variable=vs, parametrization=module)
        ids = {id(list(c.parametrization.values())[0]) for c in result}
        assert len(ids) == 3  # all distinct copies

    def test_list_per_variable_module_dicts(self):
        vs = ConceptVariable(["c1", "c2"], distribution=dist.Bernoulli, size=1)
        mods = [{"probs": nn.Linear(4, 1)}, {"logits": nn.Linear(4, 1)}]
        result = ParametricCPD(variable=vs, parametrization=mods)
        assert "probs" in result[0].parametrization
        assert "logits" in result[1].parametrization

    def test_list_module_list_length_mismatch_raises(self):
        vs = ConceptVariable(["c1", "c2", "c3"], distribution=dist.Bernoulli, size=1)
        with pytest.raises(ValueError, match="length"):
            ParametricCPD(variable=vs, parametrization=[nn.Linear(4, 1), nn.Linear(4, 1)])

    def test_list_dict_elements_required_when_list(self):
        vs = ConceptVariable(["c1", "c2"], distribution=dist.Bernoulli, size=1)
        with pytest.raises(TypeError, match="every element must be a dict"):
            ParametricCPD(variable=vs, parametrization=[nn.Linear(4, 1), nn.Linear(4, 1)])


# ===========================================================================
# 3. forward() — root CPDs
# ===========================================================================

class TestParametricCPDForwardRoot:
    def test_root_bernoulli_logits(self):
        v = _bernoulli_var(size=3)
        prior = LearnablePrior(3)
        cpd = ParametricCPD(variable=v, parametrization={"logits": prior})
        out = cpd(parent_values={})
        assert "logits" in out
        assert out["logits"].shape == (3,)

    def test_root_fixed_probs(self):
        v = _bernoulli_var(size=2)
        cpd = ParametricCPD(variable=v, parametrization={"probs": FixedPrior([0.3, 0.7])})
        out = cpd(parent_values={})
        assert "probs" in out
        assert torch.allclose(out["probs"], torch.tensor([0.3, 0.7]))

    def test_root_delta(self):
        v = _delta_var(size=4)
        prior = LearnablePrior(4)
        cpd = ParametricCPD(variable=v, parametrization={"value": prior})
        out = cpd(parent_values={})
        assert "value" in out
        assert out["value"].shape == (4,)


# ===========================================================================
# 4. forward() — non-root CPDs
# ===========================================================================

class TestParametricCPDForwardNonRoot:
    def test_nonroot_bernoulli(self):
        x = _delta_var(size=4)
        c = _bernoulli_var(size=1)
        cpd = ParametricCPD(variable=c, parametrization={"probs": nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())}, parents=[x])
        B = 5
        out = cpd(parent_values={"x": torch.randn(B, 4)})
        assert "probs" in out
        assert out["probs"].shape == (B, 1)

    def test_nonroot_normal(self):
        x = _delta_var(size=6)
        n = _normal_var(size=2)
        cpd = ParametricCPD(
            variable=n,
            parametrization={"loc": nn.Linear(6, 2), "scale": nn.Sequential(nn.Linear(6, 2), nn.Softplus())},
            parents=[x],
        )
        B = 3
        out = cpd(parent_values={"x": torch.randn(B, 6)})
        assert set(out.keys()) == {"loc", "scale"}
        assert out["loc"].shape == (B, 2)
        assert out["scale"].shape == (B, 2)

    def test_nonroot_categorical(self):
        x = _delta_var(size=8)
        k = _cat_var(size=4)
        cpd = ParametricCPD(variable=k, parametrization={"probs": nn.Sequential(nn.Linear(8, 4), nn.Softmax(dim=-1))}, parents=[x])
        B = 2
        out = cpd(parent_values={"x": torch.randn(B, 8)})
        assert out["probs"].shape == (B, 4)

    def test_nonroot_multi_parents_concatenated(self):
        p1 = ConceptVariable("p1", distribution=dist.Bernoulli, size=2)
        p2 = ConceptVariable("p2", distribution=dist.Bernoulli, size=3)
        c = _bernoulli_var(size=1)
        # parents concatenated: 2 + 3 = 5 input features
        cpd = ParametricCPD(variable=c, parametrization=nn.Linear(5, 1), parents=[p1, p2])
        B = 4
        out = cpd(parent_values={"p1": torch.randn(B, 2), "p2": torch.randn(B, 3)})
        assert out["probs"].shape == (B, 1)


# ===========================================================================
# 5. root_params()
# ===========================================================================

class TestRootParams:
    def test_root_params_expands_batch(self):
        v = _bernoulli_var(size=3)
        prior = FixedPrior([0.1, 0.5, 0.9])
        cpd = ParametricCPD(variable=v, parametrization={"probs": prior})
        params = cpd.root_params(batch_size=7)
        assert "probs" in params
        assert params["probs"].shape == (7, 3)

    def test_root_params_values_correct(self):
        v = _bernoulli_var(size=2)
        cpd = ParametricCPD(variable=v, parametrization={"probs": FixedPrior([0.3, 0.7])})
        params = cpd.root_params(batch_size=4)
        expected = torch.tensor([0.3, 0.7]).expand(4, 2)
        assert torch.allclose(params["probs"], expected)


# ===========================================================================
# 6. select() — member addressing
# ===========================================================================

class TestCPDSelect:
    def _make_plate_cpd(self, members=("c1", "c2", "c3"), member_size=1):
        plate = ConceptVariable(
            "concepts", members=list(members),
            distribution=dist.Bernoulli, size=member_size,
        )
        x = _delta_var(size=4)
        N = len(members) * member_size
        cpd = ParametricCPD(
            variable=plate,
            parametrization={"probs": nn.Sequential(nn.Linear(4, N), nn.Sigmoid())},
            parents=[x],
        )
        return cpd, plate

    def test_select_plate_name_returns_full(self):
        cpd, plate = self._make_plate_cpd()
        B = 5
        params = {"probs": torch.rand(B, 3)}
        result = cpd.select(params, "concepts")
        assert result is params  # identity, no copy

    def test_select_member_name_returns_slice(self):
        cpd, plate = self._make_plate_cpd(["c1", "c2", "c3"])
        B = 4
        probs = torch.arange(B * 3, dtype=torch.float).reshape(B, 3)
        params = {"probs": probs}
        r_c1 = cpd.select(params, "c1")
        assert r_c1["probs"].shape == (B, 1)
        assert torch.allclose(r_c1["probs"], probs[:, 0:1])

    def test_select_member_c3(self):
        cpd, plate = self._make_plate_cpd(["c1", "c2", "c3"])
        B = 3
        probs = torch.arange(B * 3, dtype=torch.float).reshape(B, 3)
        params = {"probs": probs}
        r_c3 = cpd.select(params, "c3")
        assert torch.allclose(r_c3["probs"], probs[:, 2:3])

    def test_select_multi_size_member(self):
        cpd, _ = self._make_plate_cpd(["a", "b"], member_size=3)
        B = 2
        probs = torch.randn(B, 6)
        params = {"probs": probs}
        ra = cpd.select(params, "a")
        rb = cpd.select(params, "b")
        assert ra["probs"].shape == (B, 3)
        assert rb["probs"].shape == (B, 3)
        assert torch.allclose(torch.cat([ra["probs"], rb["probs"]], dim=-1), probs)

    def test_select_nonplate_returns_same(self):
        v = _bernoulli_var()
        x = _delta_var()
        cpd = ParametricCPD(variable=v, parametrization=nn.Linear(4, 1), parents=[x])
        B, params = 3, {"probs": torch.rand(3, 1)}
        assert cpd.select(params, "c") is params


# ===========================================================================
# 7. select_value() — value addressing
# ===========================================================================

class TestCPDSelectValue:
    def test_select_value_plate_name_returns_full(self):
        plate = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli)
        x = _delta_var()
        cpd = ParametricCPD(variable=plate, parametrization=nn.Linear(4, 2), parents=[x])
        B = 3
        value = torch.rand(B, 2)
        assert cpd.select_value(value, "g") is value

    def test_select_value_member(self):
        plate = ConceptVariable("g", members=["a", "b", "c"], distribution=dist.Bernoulli)
        x = _delta_var()
        cpd = ParametricCPD(variable=plate, parametrization={"probs": nn.Sequential(nn.Linear(4, 3), nn.Sigmoid())}, parents=[x])
        B = 4
        value = torch.arange(B * 3, dtype=torch.float).reshape(B, 3)
        va = cpd.select_value(value, "a")
        assert va.shape == (B, 1)
        assert torch.allclose(va, value[:, 0:1])

    def test_select_value_nonplate(self):
        v = _bernoulli_var()
        x = _delta_var()
        cpd = ParametricCPD(variable=v, parametrization=nn.Linear(4, 1), parents=[x])
        value = torch.rand(3, 1)
        assert cpd.select_value(value, "c") is value


# ===========================================================================
# 8. clamp_members()
# ===========================================================================

class TestCPDClampMembers:
    def _make_cpd(self, members=("c1", "c2", "c3")):
        plate = ConceptVariable("g", members=list(members), distribution=dist.Bernoulli)
        x = _delta_var()
        cpd = ParametricCPD(variable=plate, parametrization={"probs": nn.Sequential(nn.Linear(4, len(members)), nn.Sigmoid())}, parents=[x])
        return cpd

    def test_empty_observed_returns_original(self):
        cpd = self._make_cpd()
        B = 3
        value = torch.rand(B, 3)
        result = cpd.clamp_members(value, {})
        assert result is value  # no-op, same object

    def test_clamp_single_member(self):
        cpd = self._make_cpd(["a", "b", "c"])
        B = 4
        value = torch.rand(B, 3)
        obs = torch.ones(B, 1)
        result = cpd.clamp_members(value, {"b": obs})
        # column 1 should be all-ones
        assert torch.allclose(result[:, 1:2], torch.ones(B, 1))
        # columns 0 and 2 unchanged
        assert torch.allclose(result[:, 0:1], value[:, 0:1])
        assert torch.allclose(result[:, 2:3], value[:, 2:3])

    def test_clamp_does_not_mutate_original(self):
        cpd = self._make_cpd(["a", "b"])
        B = 3
        value = torch.zeros(B, 2)
        obs = torch.ones(B, 1)
        result = cpd.clamp_members(value, {"a": obs})
        assert torch.allclose(value, torch.zeros(B, 2))  # original unchanged

    def test_clamp_multiple_members(self):
        cpd = self._make_cpd(["a", "b", "c"])
        B = 2
        value = torch.zeros(B, 3)
        result = cpd.clamp_members(value, {
            "a": torch.ones(B, 1),
            "c": 2.0 * torch.ones(B, 1),
        })
        assert torch.allclose(result[:, 0:1], torch.ones(B, 1))
        assert torch.allclose(result[:, 1:2], torch.zeros(B, 1))
        assert torch.allclose(result[:, 2:3], 2.0 * torch.ones(B, 1))


# ===========================================================================
# 9. Integration: forward + select
# ===========================================================================

class TestCPDIntegration:
    def test_plate_forward_then_select_member(self):
        plate = ConceptVariable("concepts", members=["c1", "c2", "c3"], distribution=dist.Bernoulli)
        x = _delta_var(size=8)
        cpd = ParametricCPD(
            variable=plate,
            parametrization={"probs": nn.Sequential(nn.Linear(8, 3), nn.Sigmoid())},
            parents=[x],
        )
        B = 5
        params = cpd(parent_values={"x": torch.randn(B, 8)})
        full = cpd.select(params, "concepts")
        c2 = cpd.select(params, "c2")
        assert full["probs"].shape == (B, 3)
        assert c2["probs"].shape == (B, 1)
        assert torch.allclose(c2["probs"], full["probs"][:, 1:2])

    def test_root_params_then_select(self):
        plate = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli)
        cpd = ParametricCPD(variable=plate, parametrization={"probs": FixedPrior([0.3, 0.7])})
        params = cpd.root_params(batch_size=4)
        a_params = cpd.select(params, "a")
        assert torch.allclose(a_params["probs"], torch.full((4, 1), 0.3))


# ===========================================================================
# 10. ParametricCPD.__new__ edge cases (factor.py + cpd.py missing lines)
# ===========================================================================

class TestParametricCPDNewEdgeCases:
    """Cover __new__ paths not yet exercised."""

    def test_non_variable_non_list_raises(self):
        # cpd.py line 112: variable is not a Variable and not a list
        with pytest.raises(TypeError, match="must be a Variable or a list"):
            ParametricCPD(variable="not_a_variable", parametrization=nn.Linear(4, 1))

    def test_list_with_dict_parametrization_deep_copies(self):
        # cpd.py line 131: list path with dict parametrization
        vs = ConceptVariable(["c1", "c2"], distribution=dist.Bernoulli, size=1)
        param_dict = {"probs": nn.Linear(4, 1)}
        result = ParametricCPD(variable=vs, parametrization=param_dict)
        assert isinstance(result, list)
        assert len(result) == 2
        # Modules should be deep-copied (distinct objects)
        mod0 = list(result[0].parametrization.values())[0]
        mod1 = list(result[1].parametrization.values())[0]
        assert id(mod0) != id(mod1)

    def test_list_with_invalid_parametrization_type_raises(self):
        # cpd.py line 135: list path with parametrization that is not Module/dict/list
        vs = ConceptVariable(["c1", "c2"], distribution=dist.Bernoulli, size=1)
        with pytest.raises(TypeError, match="must be an nn.Module"):
            ParametricCPD(variable=vs, parametrization="invalid")

    def test_single_variable_with_invalid_parametrization_type_raises(self):
        # cpd.py line 192: single Variable path with non-None/non-Module/non-dict
        v = _bernoulli_var()
        with pytest.raises(TypeError, match="must be None, an nn.Module"):
            ParametricCPD(variable=v, parametrization=42)


# ===========================================================================
# 11. ParametricFactor aggregate dict and edge cases (factor.py missing lines)
# ===========================================================================

class TestParametricFactorAggregate:
    """Cover aggregate=dict and error paths in ParametricFactor.__init__."""

    def test_aggregate_dict_valid(self):
        # factor.py lines 125-132: aggregate as a valid dict mapping param -> callable
        x = _delta_var(size=4)
        c = _bernoulli_var(size=1)
        custom_agg = lambda inputs: torch.cat(list(inputs.values()), dim=-1)
        cpd = ParametricCPD(
            variable=c,
            parametrization={"probs": nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())},
            parents=[x],
            aggregate={"probs": custom_agg},
        )
        B = 3
        out = cpd(parent_values={"x": torch.randn(B, 4)})
        assert "probs" in out
        assert out["probs"].shape == (B, 1)

    def test_aggregate_dict_non_callable_raises(self):
        # factor.py line 126-131: aggregate dict contains non-callable value
        x = _delta_var(size=4)
        c = _bernoulli_var(size=1)
        with pytest.raises(TypeError, match="non-callable"):
            ParametricCPD(
                variable=c,
                parametrization={"probs": nn.Linear(4, 1)},
                parents=[x],
                aggregate={"probs": "not_callable"},
            )

    def test_aggregate_invalid_type_raises(self):
        # factor.py lines 133-137: aggregate is not None/callable/dict
        x = _delta_var(size=4)
        c = _bernoulli_var(size=1)
        with pytest.raises(TypeError, match="must be None, a callable, or a dict"):
            ParametricCPD(
                variable=c,
                parametrization={"probs": nn.Linear(4, 1)},
                parents=[x],
                aggregate=42,
            )

    def test_aggregate_callable_for_standard_module(self):
        # factor.py line 199: _resolve_aggregator returns user_aggregate for standard module
        x = _delta_var(size=4)
        c = _bernoulli_var(size=1)
        # A standard (non-pyc) module with a callable aggregate
        custom_agg = lambda inputs: torch.cat(list(inputs.values()), dim=-1)
        cpd = ParametricCPD(
            variable=c,
            parametrization={"probs": nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())},
            parents=[x],
            aggregate=custom_agg,
        )
        B = 2
        out = cpd(parent_values={"x": torch.randn(B, 4)})
        assert "probs" in out

    def test_split_by_type_invalid_variable_type_raises(self):
        # factor.py lines 222-228: _split_by_type raises for invalid variable_type
        import torch.distributions as dist
        from torch_concepts.nn.modules.mid.models.variable import ConceptVariable

        # Create a CPD with a parent, then monkey-patch parent's variable_type
        x = _delta_var(size=4)
        c = _bernoulli_var(size=1)

        # Need a pyc-style module to trigger _pyc_aggregate -> _split_by_type.
        # We'll use an EmbeddingVariable with a modified variable_type.
        from torch_concepts.nn.modules.mid.models.variable import EmbeddingVariable
        e = EmbeddingVariable("e", distribution=dist.Normal, size=4)
        # Monkey-patch to an invalid type so _split_by_type raises
        e.__class__ = type("BrokenVar", (EmbeddingVariable,), {"variable_type": property(lambda self: "invalid")})

        # Build CPD with embedding parent to get pyc-style aggregation
        from torch_concepts.nn.modules.low.predictors.linear import LinearConceptToConcept
        from torch_concepts.nn.modules.low.lazy import LazyConstructor
        cpd = ParametricCPD(
            variable=c,
            parametrization={"probs": nn.Sequential(nn.Linear(4, 1), nn.Sigmoid())},
            parents=[x],
        )
        # Directly call _split_by_type with the patched parent
        cpd.parents = [e]
        with pytest.raises(ValueError, match="invalid type"):
            cpd._split_by_type({e: torch.randn(2, 4)})


# ===========================================================================
# 12. LazyConstructor already-built path (factor.py line 162)
# ===========================================================================

class TestLazyConstructorAlreadyBuilt:
    def test_already_built_lazy_is_unwrapped(self):
        # factor.py line 162: LazyConstructor with module already set is unwrapped
        from torch_concepts.nn.modules.low.lazy import LazyConstructor
        from torch_concepts.nn.modules.low.predictors.linear import LinearConceptToConcept

        x = _delta_var(size=4)
        c = _bernoulli_var(size=1)

        lazy = LazyConstructor(LinearConceptToConcept)
        # Pre-build the lazy constructor so module is not None
        lazy.build(out_concepts=1, in_concepts=4)
        assert lazy.module is not None  # pre-condition

        # Now pass the already-built lazy to CPD — it should unwrap to the inner module
        cpd = ParametricCPD(variable=c, parametrization={"probs": lazy}, parents=[x])
        # The parametrization should contain the concrete module, not the LazyConstructor
        mod = cpd.parametrization["probs"]
        assert not isinstance(mod, LazyConstructor)
