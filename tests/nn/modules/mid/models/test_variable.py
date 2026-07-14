"""Comprehensive tests for Variable, ConceptVariable, and EmbeddingVariable."""
import math
import copy
import pytest
import torch
import torch.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import (
    Variable,
    ConceptVariable,
    EmbeddingVariable,
    PARAM_DIM,
)
from torch_concepts.distributions import Delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_concept(name="c", **kwargs):
    kwargs.setdefault("distribution", dist.Bernoulli)
    kwargs.setdefault("size", 1)
    return ConceptVariable(name, **kwargs)


def _make_embedding(name="e", **kwargs):
    kwargs.setdefault("distribution", Delta)
    kwargs.setdefault("size", 4)
    return EmbeddingVariable(name, **kwargs)


# ===========================================================================
# 1. String-path construction (single variable)
# ===========================================================================

class TestVariableStrPath:
    def test_str_returns_single_instance(self):
        v = ConceptVariable("x", distribution=dist.Bernoulli, size=1)
        assert isinstance(v, Variable)
        assert not isinstance(v, list)

    def test_name_stored(self):
        v = ConceptVariable("my_var", distribution=Delta, size=3)
        assert v.name == "my_var"

    def test_distribution_stored(self):
        v = ConceptVariable("c", distribution=dist.OneHotCategorical, size=3)
        assert v.distribution is dist.OneHotCategorical

    def test_size_from_size_kwarg(self):
        v = ConceptVariable("c", distribution=Delta, size=5)
        assert v.size == 5
        assert v.shape == torch.Size([5])

    def test_size_from_shape_kwarg(self):
        v = ConceptVariable("c", distribution=Delta, shape=(3, 2))
        assert v.shape == torch.Size([3, 2])
        assert v.size == 6

    def test_default_size_is_1(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli)
        assert v.size == 1
        assert v.shape == torch.Size([1])

    def test_dist_kwargs_stored(self):
        dk = {"temperature": 0.5}
        v = ConceptVariable("c", distribution=dist.RelaxedBernoulli, size=1, dist_kwargs=dk)
        assert v.dist_kwargs == {"temperature": 0.5}

    def test_dist_kwargs_defaults_empty(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        assert v.dist_kwargs == {}

    def test_metadata_variable_type(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        assert v.metadata["variable_type"] == "concept"

    def test_embedding_variable_type(self):
        v = EmbeddingVariable("e", distribution=Delta, size=8)
        assert v.metadata["variable_type"] == "embedding"

    def test_shape_and_size_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ConceptVariable("c", distribution=Delta, shape=(2,), size=2)

    def test_size_must_be_positive_int(self):
        with pytest.raises(ValueError):
            ConceptVariable("c", distribution=Delta, size=0)
        with pytest.raises(ValueError):
            ConceptVariable("c", distribution=Delta, size=-1)

    def test_shape_must_be_nonempty(self):
        with pytest.raises(ValueError):
            ConceptVariable("c", distribution=Delta, shape=())

    def test_distribution_required(self):
        with pytest.raises(ValueError, match="distribution.*required"):
            ConceptVariable("c", size=1)


# ===========================================================================
# 2. List-path construction (multiple variables)
# ===========================================================================

class TestVariableListPath:
    def test_list_returns_list(self):
        result = ConceptVariable(["a", "b", "c"], distribution=dist.Bernoulli, size=1)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_list_single_element_returns_list(self):
        result = ConceptVariable(["x"], distribution=Delta, size=2)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ConceptVariable)

    def test_list_names_assigned(self):
        result = ConceptVariable(["a", "b", "c"], distribution=Delta, size=1)
        assert [v.name for v in result] == ["a", "b", "c"]

    def test_list_single_distribution_broadcast(self):
        result = ConceptVariable(["a", "b"], distribution=dist.Bernoulli, size=1)
        assert all(v.distribution is dist.Bernoulli for v in result)

    def test_list_single_size_broadcast(self):
        result = ConceptVariable(["a", "b", "c"], distribution=Delta, size=7)
        assert all(v.size == 7 for v in result)

    def test_list_per_name_distribution(self):
        result = ConceptVariable(
            ["a", "b", "c"],
            distribution=[dist.Bernoulli, dist.OneHotCategorical, Delta],
            size=[1, 3, 2],
        )
        assert result[0].distribution is dist.Bernoulli
        assert result[1].distribution is dist.OneHotCategorical
        assert result[2].distribution is Delta

    def test_list_per_name_size(self):
        result = ConceptVariable(
            ["a", "b", "c"],
            distribution=Delta,
            size=[1, 2, 4],
        )
        assert [v.size for v in result] == [1, 2, 4]

    def test_list_dist_kwargs_deep_copied(self):
        dk = {"extra": [1, 2, 3]}
        result = ConceptVariable(["a", "b"], distribution=dist.Bernoulli, size=1, dist_kwargs=dk)
        result[0].dist_kwargs["extra"].append(99)
        assert result[1].dist_kwargs["extra"] == [1, 2, 3]

    def test_list_distribution_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ConceptVariable(
                ["a", "b", "c"],
                distribution=[dist.Bernoulli, Delta],  # 2, need 3
                size=1,
            )

    def test_list_size_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ConceptVariable(
                ["a", "b"],
                distribution=Delta,
                size=[1, 2, 3],  # 3, need 2
            )

    def test_list_members_kwarg_raises(self):
        with pytest.raises(TypeError, match="members.*only valid with a single"):
            ConceptVariable(
                ["a", "b"],
                distribution=dist.Bernoulli,
                size=1,
                members=["x", "y"],
            )

    def test_list_non_strings_raises(self):
        with pytest.raises(TypeError):
            ConceptVariable([1, 2], distribution=Delta, size=1)


# ===========================================================================
# 3. Plate (members=) construction
# ===========================================================================

class TestVariablePlate:
    def test_plate_returns_single_instance(self):
        v = ConceptVariable("concepts", members=["c1", "c2", "c3"], distribution=dist.Bernoulli)
        assert isinstance(v, Variable)

    def test_plate_name_is_group_name(self):
        v = ConceptVariable("concepts", members=["c1", "c2"], distribution=dist.Bernoulli)
        assert v.name == "concepts"

    def test_plate_members_stored(self):
        v = ConceptVariable("concepts", members=["c1", "c2", "c3"], distribution=dist.Bernoulli)
        assert v.members == ["c1", "c2", "c3"]

    def test_plate_member_size_default_1(self):
        v = ConceptVariable("concepts", members=["c1", "c2"], distribution=dist.Bernoulli)
        assert v.member_size == 1

    def test_plate_member_size_from_size_kwarg(self):
        v = ConceptVariable("concepts", members=["c1", "c2"], distribution=dist.Bernoulli, size=2)
        assert v.member_size == 2
        assert v.size == 4  # 2 members * 2

    def test_plate_is_plate(self):
        v = ConceptVariable("concepts", members=["c1", "c2"], distribution=dist.Bernoulli)
        assert v.is_plate is True

    def test_single_var_is_not_plate(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        assert v.is_plate is False

    def test_plate_total_size(self):
        v = ConceptVariable("concepts", members=["c1", "c2", "c3"], distribution=dist.Bernoulli)
        assert v.size == 3
        assert v.shape == torch.Size([3])

    def test_plate_total_size_multi_member(self):
        v = ConceptVariable("concepts", members=["c1", "c2"], distribution=dist.Bernoulli, size=2)
        assert v.size == 4

    def test_plate_shape_and_members_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ConceptVariable("concepts", members=["c1"], distribution=dist.Bernoulli, shape=(2,))

    def test_plate_duplicate_members_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            ConceptVariable("concepts", members=["c1", "c1"], distribution=dist.Bernoulli)

    def test_plate_empty_members_raises(self):
        with pytest.raises(ValueError):
            ConceptVariable("concepts", members=[], distribution=dist.Bernoulli)

    def test_plate_non_string_members_raises(self):
        with pytest.raises(ValueError):
            ConceptVariable("concepts", members=[1, 2], distribution=dist.Bernoulli)

    def test_plate_mvn_guard_raises(self):
        with pytest.raises(ValueError, match="non-per-element parameter"):
            ConceptVariable("x", members=["c1", "c2"], distribution=dist.MultivariateNormal)

    def test_plate_distribution_required(self):
        with pytest.raises(ValueError, match="distribution.*required"):
            ConceptVariable("x", members=["c1", "c2"])


# ===========================================================================
# 4. Properties: size, shape, is_plate, plate, param_sizes
# ===========================================================================

class TestVariableProperties:
    def test_size_scalar(self):
        v = ConceptVariable("c", distribution=Delta, size=5)
        assert v.size == 5

    def test_size_shape_product(self):
        v = ConceptVariable("c", distribution=Delta, shape=(2, 3))
        assert v.size == 6

    def test_is_plate_false_for_single(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        assert v.is_plate is False

    def test_is_plate_true_for_plate(self):
        v = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli)
        assert v.is_plate is True

    def test_plate_property_returns_self_for_non_plate(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        assert v.plate is v

    def test_param_sizes_bernoulli(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=3)
        ps = v.param_sizes
        assert ps == {"probs": 3, "logits": 3}

    def test_param_sizes_normal(self):
        v = ConceptVariable("c", distribution=dist.Normal, size=2)
        ps = v.param_sizes
        assert ps == {"loc": 2, "scale": 2}

    def test_param_sizes_mvn(self):
        v = ConceptVariable("c", distribution=dist.MultivariateNormal, size=3)
        ps = v.param_sizes
        assert ps["loc"] == 3
        assert ps["scale_tril"] == 3 * 4 // 2  # 6

    def test_param_sizes_unknown_distribution_raises(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        v.distribution = object  # inject unsupported distribution
        with pytest.raises(ValueError, match="no PARAM_DIM entry"):
            _ = v.param_sizes

    def test_members_list_for_single(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        assert v.members == ["c"]

    def test_member_size_for_single(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=5)
        assert v.member_size == 5  # math.prod(shape)


# ===========================================================================
# 5. column_of and member() methods
# ===========================================================================

class TestVariableAddressing:
    def test_column_of_plate_member(self):
        v = ConceptVariable("g", members=["a", "b", "c"], distribution=dist.Bernoulli)
        assert v.column_of("a") == slice(0, 1)
        assert v.column_of("b") == slice(1, 2)
        assert v.column_of("c") == slice(2, 3)

    def test_column_of_multi_size_member(self):
        v = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli, size=3)
        assert v.column_of("a") == slice(0, 3)
        assert v.column_of("b") == slice(3, 6)

    def test_column_of_single_var_is_full_slice(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=5)
        assert v.column_of("c") == slice(0, 5)

    def test_column_of_unknown_raises(self):
        v = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli)
        with pytest.raises(KeyError):
            v.column_of("MISSING")

    def test_member_returns_variable(self):
        plate = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli)
        h = plate.member("a")
        assert isinstance(h, Variable)

    def test_member_name(self):
        plate = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli)
        h = plate.member("a")
        assert h.name == "a"

    def test_member_plate_backref(self):
        plate = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli)
        h = plate.member("a")
        assert h._plate is plate
        assert h.plate is plate

    def test_member_size(self):
        plate = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli, size=2)
        h = plate.member("a")
        assert h.size == 2

    def test_member_unknown_raises(self):
        plate = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli)
        with pytest.raises(KeyError):
            plate.member("MISSING")

    def test_plate_property_of_member_handle(self):
        plate = ConceptVariable("g", members=["a", "b"], distribution=dist.Bernoulli)
        h = plate.member("a")
        assert h.plate is plate

    def test_plate_property_of_ordinary_var(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        assert v.plate is v


# ===========================================================================
# 6. variable_type and subclass identity
# ===========================================================================

class TestVariableSubclasses:
    def test_concept_variable_type(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        assert v.variable_type == "concept"

    def test_embedding_variable_type(self):
        v = EmbeddingVariable("e", distribution=Delta, size=8)
        assert v.variable_type == "embedding"

    def test_list_of_concept_variables_all_concept(self):
        vs = ConceptVariable(["a", "b"], distribution=dist.Bernoulli, size=1)
        assert all(isinstance(v, ConceptVariable) for v in vs)

    def test_list_of_embedding_variables_all_embedding(self):
        vs = EmbeddingVariable(["e1", "e2"], distribution=Delta, size=4)
        assert all(isinstance(v, EmbeddingVariable) for v in vs)


# ===========================================================================
# 7. __repr__
# ===========================================================================

class TestVariableRepr:
    def test_repr_contains_name(self):
        v = ConceptVariable("my_concept", distribution=dist.Bernoulli, size=1)
        assert "my_concept" in repr(v)

    def test_repr_contains_class_name(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        assert "ConceptVariable" in repr(v)

    def test_repr_contains_distribution(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        assert "Bernoulli" in repr(v)

    def test_repr_contains_shape(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        assert "1" in repr(v)

    def test_repr_shows_members_for_plate(self):
        v = ConceptVariable("g", members=["c1", "c2"], distribution=dist.Bernoulli)
        r = repr(v)
        assert "members" in r
        assert "c1" in r
        assert "c2" in r

    def test_repr_no_members_for_single(self):
        v = ConceptVariable("c", distribution=dist.Bernoulli, size=1)
        assert "members" not in repr(v)


# ===========================================================================
# 8. column_of works with tensor slicing
# ===========================================================================

class TestColumnSlicing:
    def test_column_slices_correct_values(self):
        plate = ConceptVariable("g", members=["a", "b", "c"], distribution=dist.Bernoulli)
        B = 4
        t = torch.arange(B * 3, dtype=torch.float).reshape(B, 3)
        for i, name in enumerate(["a", "b", "c"]):
            col = t[..., plate.column_of(name)]
            assert col.shape == (B, 1)
            assert torch.allclose(col, t[:, i:i+1])

    def test_multisize_column_slices(self):
        plate = ConceptVariable("g", members=["a", "b"], distribution=dist.Normal, size=3)
        B = 2
        t = torch.randn(B, 6)
        slice_a = t[..., plate.column_of("a")]
        slice_b = t[..., plate.column_of("b")]
        assert slice_a.shape == (B, 3)
        assert slice_b.shape == (B, 3)
        assert torch.allclose(torch.cat([slice_a, slice_b], dim=-1), t)
