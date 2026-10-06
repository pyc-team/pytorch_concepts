"""InferenceOutput.union_with / rename_variable: merging successive queries."""
import pytest
import torch

from torch_concepts import Annotations
from torch_concepts.nn import InferenceOutput
from torch_concepts.tensor import AnnotatedTensor


def _out(*labels, probabilities=None):
    """An output reporting ``logits`` for ``labels`` and a guide ``loc`` for ``z_<label>``."""
    n = len(labels)
    logits = AnnotatedTensor(torch.randn(4, n), Annotations(labels=list(labels)), axis=-1)
    loc = AnnotatedTensor(torch.randn(4, n), Annotations(
        labels=[f"z_{l}" for l in labels], types=["continuous"] * n), axis=-1)
    return InferenceOutput(params={"logits": logits}, guide_params={"loc": loc},
                           probabilities=probabilities)


def test_union_concatenates_each_quantity():
    a, b = _out("c1"), _out("c2", "c3")
    merged = a.union_with(b)
    assert merged.logits.annotations.labels == ["c1", "c2", "c3"]
    assert torch.equal(merged.logits["c2"].tensor, b.logits["c2"].tensor)
    assert torch.equal(merged.guide_params["loc"]["z_c1"].tensor, a.guide_params["loc"].tensor)


def test_quantities_present_in_one_output_only_are_carried():
    a = _out("c1")
    b = InferenceOutput(loc=AnnotatedTensor(
        torch.randn(4, 1), Annotations(labels=["y"], types=["continuous"]), axis=-1))
    merged = a.union_with(b)
    assert set(merged.params) == {"logits", "loc"}


def test_an_overlapping_variable_is_refused():
    with pytest.raises(ValueError, match="c1"):
        _out("c1", "c2").union_with(_out("c1"))


def test_the_same_query_twice_merges_after_a_rename():
    first, second = _out("c1"), _out("c1")
    second = second.rename_variable("c1", "c1_again").rename_variable("z_c1", "z_c1_again")
    merged = first.union_with(second)
    assert torch.equal(merged.logits["c1_again"].tensor, second.logits.tensor)
    assert torch.equal(merged.logits["c1"].tensor, first.logits.tensor)


def test_renaming_a_plate_keeps_it_addressable():
    logits = AnnotatedTensor(torch.randn(4, 2), Annotations(labels=["a", "b"]), axis=-1)
    logits.register_plate_label("plate", ["a", "b"])
    renamed = InferenceOutput(logits=logits).rename_variable("plate", "plate2")
    assert renamed.logits["plate2"].shape == (4, 2)
    assert "plate" not in renamed.variables


def test_two_probability_estimates_are_refused():
    with pytest.raises(ValueError, match="probabilities"):
        _out("c1", probabilities=torch.rand(4)).union_with(
            _out("c2", probabilities=torch.rand(4)))
