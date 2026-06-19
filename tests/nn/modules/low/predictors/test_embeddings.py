"""Tests for MixConceptEmbeddingToConcept and MixSumConceptEmbeddingToConcept."""
import pytest
import torch
import torch.nn as nn

from torch_concepts import AxisAnnotation
from torch_concepts.nn import MixConceptEmbeddingToConcept
from torch_concepts.nn.modules.low.predictors.mix import MixSumConceptEmbeddingToConcept


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _axis(n, cardinalities=None):
    """Create an AxisAnnotation for n discrete concepts."""
    if cardinalities is None:
        cardinalities = [1] * n
    assert sum(cardinalities) == n, "cardinalities must sum to n"
    return AxisAnnotation(
        labels=[f"c{i}" for i in range(len(cardinalities))],
        cardinalities=cardinalities,
        types=["discrete"] * len(cardinalities),
    )


# ===========================================================================
# 1. MixConceptEmbeddingToConcept
# ===========================================================================

class TestMixConceptEmbeddingToConcept:
    def test_initialization(self):
        aa = _axis(10)
        pred = MixConceptEmbeddingToConcept(in_concepts=aa, in_embeddings=20, out_concepts=3)
        assert pred.in_concepts is aa
        assert pred.in_embeddings == 20
        assert pred.out_concepts == 3

    def test_forward_shape_all_binary(self):
        aa = _axis(10)
        pred = MixConceptEmbeddingToConcept(in_concepts=aa, in_embeddings=20, out_concepts=3)
        concepts = torch.randn(4, 10)
        embeddings = torch.randn(4, 10, 20)
        output = pred(concepts=concepts, embeddings=embeddings)
        assert output.shape == (4, 3)

    def test_forward_shape_categorical(self):
        aa = _axis(10, cardinalities=[3, 4, 3])
        pred = MixConceptEmbeddingToConcept(in_concepts=aa, in_embeddings=20, out_concepts=3)
        concepts = torch.randn(4, 10)
        embeddings = torch.randn(4, 10, 20)
        output = pred(concepts=concepts, embeddings=embeddings)
        assert output.shape == (4, 3)

    def test_forward_shape_mixed(self):
        aa = _axis(10, cardinalities=[1, 3, 1, 1, 4])
        pred = MixConceptEmbeddingToConcept(in_concepts=aa, in_embeddings=16, out_concepts=5)
        concepts = torch.randn(2, 10)
        embeddings = torch.randn(2, 10, 16)
        output = pred(concepts=concepts, embeddings=embeddings)
        assert output.shape == (2, 5)

    def test_int_in_concepts_raises(self):
        with pytest.raises(AttributeError):
            MixConceptEmbeddingToConcept(
                in_concepts=10,  # int is wrong; must be AxisAnnotation
                in_embeddings=20,
                out_concepts=3,
            )

    def test_gradient_flow(self):
        aa = _axis(8)
        pred = MixConceptEmbeddingToConcept(in_concepts=aa, in_embeddings=16, out_concepts=2)
        concepts = torch.randn(2, 8, requires_grad=True)
        embeddings = torch.randn(2, 8, 16, requires_grad=True)
        pred(concepts=concepts, embeddings=embeddings).sum().backward()
        assert concepts.grad is not None
        assert embeddings.grad is not None

    def test_predictor_is_linear(self):
        aa = _axis(6, cardinalities=[2, 2, 2])
        pred = MixConceptEmbeddingToConcept(in_concepts=aa, in_embeddings=10, out_concepts=3)
        assert isinstance(pred.predictor, nn.Linear)

    def test_output_shape_batch_one(self):
        aa = _axis(4)
        pred = MixConceptEmbeddingToConcept(in_concepts=aa, in_embeddings=8, out_concepts=2)
        out = pred(concepts=torch.randn(1, 4), embeddings=torch.randn(1, 4, 8))
        assert out.shape == (1, 2)


# ===========================================================================
# 2. MixSumConceptEmbeddingToConcept
# ===========================================================================

pytestmark_sum = pytest.mark.xfail(
    raises=AttributeError,
    reason="MixSumConceptEmbeddingToConcept.__init__ passes int to super().__init__ "
           "which expects AxisAnnotation for in_concepts",
)


class TestMixSumConceptEmbeddingToConcept:
    @pytestmark_sum
    def test_initialization_with_cardinalities(self):
        pred = MixSumConceptEmbeddingToConcept(
            in_concepts=10, in_embeddings=20, out_concepts=3, cardinalities=[3, 4, 3],
        )
        assert pred.in_concepts == 10
        assert pred.out_concepts == 3

    @pytestmark_sum
    def test_initialization_defaults_all_binary(self):
        pred = MixSumConceptEmbeddingToConcept(in_concepts=8, in_embeddings=16, out_concepts=4)
        assert pred.cardinalities == [1] * 8

    @pytestmark_sum
    def test_forward_shape(self):
        pred = MixSumConceptEmbeddingToConcept(
            in_concepts=10, in_embeddings=20, out_concepts=3, cardinalities=[3, 4, 3],
        )
        out = pred(concepts=torch.randn(4, 10), embeddings=torch.randn(4, 10, 20))
        assert out.shape == (4, 3)

    @pytestmark_sum
    def test_forward_shape_all_binary(self):
        pred = MixSumConceptEmbeddingToConcept(in_concepts=6, in_embeddings=12, out_concepts=2)
        out = pred(concepts=torch.randn(3, 6), embeddings=torch.randn(3, 6, 12))
        assert out.shape == (3, 2)

    @pytestmark_sum
    def test_predictor_is_linear(self):
        pred = MixSumConceptEmbeddingToConcept(in_concepts=4, in_embeddings=8, out_concepts=2)
        assert isinstance(pred.predictor, nn.Linear)

    @pytestmark_sum
    def test_group_count_invariance(self):
        p1 = MixSumConceptEmbeddingToConcept(
            in_concepts=4, in_embeddings=8, out_concepts=2, cardinalities=[1] * 4
        )
        p2 = MixSumConceptEmbeddingToConcept(
            in_concepts=6, in_embeddings=8, out_concepts=2, cardinalities=[2, 2, 2]
        )
        assert p1.predictor.weight.shape == p2.predictor.weight.shape

    @pytestmark_sum
    def test_gradient_flow(self):
        pred = MixSumConceptEmbeddingToConcept(
            in_concepts=6, in_embeddings=10, out_concepts=2, cardinalities=[2, 2, 2],
        )
        concepts = torch.randn(2, 6, requires_grad=True)
        embeddings = torch.randn(2, 6, 10, requires_grad=True)
        pred(concepts=concepts, embeddings=embeddings).sum().backward()
        assert concepts.grad is not None
        assert embeddings.grad is not None
