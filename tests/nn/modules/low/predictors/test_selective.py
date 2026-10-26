"""SelectiveLinearConceptToConcept: reads only the concepts it is built with, by name."""
import pytest
import torch

from torch_concepts import Annotations, AnnotatedTensor
from torch_concepts.nn import SelectiveLinearConceptToConcept


def test_parent_layer_reads_its_parents_by_name():
    bottleneck = AnnotatedTensor(torch.randn(4, 3), Annotations(labels=["a", "b", "c"]))
    layer = SelectiveLinearConceptToConcept(in_concepts=Annotations(labels=["c", "a"]), out_concepts=1)
    expected = layer.linear(bottleneck.tensor[:, [2, 0]])
    assert torch.allclose(layer(bottleneck), expected)
    changed = AnnotatedTensor(bottleneck.tensor.clone(), bottleneck.annotations)
    changed["b"] = 100.0  # an undeclared concept has no effect
    assert torch.equal(layer(changed), layer(bottleneck))


def test_selective_layer_realigns_reordered_columns_and_rejects_other_types_or_cardinalities():
    ann = Annotations(labels=["color", "size"], cardinalities=[3, 1], types=["categorical", "binary"])
    layer = SelectiveLinearConceptToConcept(in_concepts=ann, out_concepts=1)
    x = torch.randn(4, 4)
    reordered = AnnotatedTensor(x[:, [3, 0, 1, 2]], Annotations(labels=["size", "color"], cardinalities=[1, 3],
                                                               types=["binary", "categorical"]))
    assert torch.allclose(layer(reordered), layer(AnnotatedTensor(x, ann)))  # labels: realigned
    as_continuous = Annotations(labels=["color", "size"], cardinalities=[3, 1], types=["categorical", "continuous"])
    with pytest.raises(ValueError, match="types"):  # type: same name, now continuous
        layer(AnnotatedTensor(x, as_continuous))
    two = Annotations(labels=["color", "shape"], cardinalities=[3, 2], types=["categorical", "categorical"])
    other_states = Annotations(labels=["color", "shape"], cardinalities=[2, 3], types=["categorical", "categorical"])
    with pytest.raises(ValueError, match="cardinalities"):  # cardinality: same names, types and width, other states
        SelectiveLinearConceptToConcept(in_concepts=two, out_concepts=1)(AnnotatedTensor(torch.randn(4, 5), other_states))
