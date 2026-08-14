"""Tests for MixConceptEmbeddingToConcept and MixSumConceptEmbeddingToConcept."""
import pytest
import torch
import torch.nn as nn

from torch_concepts import Annotations
from torch_concepts.nn import (BaseConceptLayer, MixConceptEmbeddingToConcept,
                               MixConceptEmbeddings)
from torch_concepts.nn.modules.low.predictors.mix import MixSumConceptEmbeddingToConcept


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _axis(n, cardinalities=None):
    """Create an Annotations for n concepts."""
    if cardinalities is None:
        cardinalities = [1] * n
    assert sum(cardinalities) == n, "cardinalities must sum to n"
    types = ['binary' if c == 1 else 'categorical' for c in cardinalities]
    return Annotations(
        labels=[f"c{i}" for i in range(len(cardinalities))],
        cardinalities=cardinalities,
        types=types,
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
                in_concepts=10,  # int is wrong; must be Annotations
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

class TestMixSumConceptEmbeddingToConcept:
    def test_initialization(self):
        pred = MixSumConceptEmbeddingToConcept(
            in_concepts=_axis(10, cardinalities=[3, 4, 3]), in_embeddings=20, out_concepts=3,
        )
        assert len(pred.in_concepts.labels) == 3  # 3 groups
        assert pred.out_concepts == 3

    def test_forward_shape(self):
        pred = MixSumConceptEmbeddingToConcept(
            in_concepts=_axis(10, cardinalities=[3, 4, 3]), in_embeddings=20, out_concepts=3,
        )
        out = pred(concepts=torch.randn(4, 10), embeddings=torch.randn(4, 10, 20))
        assert out.shape == (4, 3)

    def test_forward_shape_all_binary(self):
        pred = MixSumConceptEmbeddingToConcept(
            in_concepts=_axis(6), in_embeddings=12, out_concepts=2,
        )
        out = pred(concepts=torch.randn(3, 6), embeddings=torch.randn(3, 6, 12))
        assert out.shape == (3, 2)

    def test_predictor_is_linear(self):
        pred = MixSumConceptEmbeddingToConcept(
            in_concepts=_axis(4), in_embeddings=8, out_concepts=2,
        )
        assert isinstance(pred.predictor, nn.Linear)

    def test_group_count_invariance(self):
        p1 = MixSumConceptEmbeddingToConcept(
            in_concepts=_axis(4), in_embeddings=8, out_concepts=2,
        )
        p2 = MixSumConceptEmbeddingToConcept(
            in_concepts=_axis(6, cardinalities=[2, 2, 2]), in_embeddings=8, out_concepts=2,
        )
        assert p1.predictor.weight.shape == p2.predictor.weight.shape

    def test_out_concepts_may_be_annotations(self):
        """Same Annotations contract as the parent, on the output side too."""
        out_ann = _axis(2)
        pred = MixSumConceptEmbeddingToConcept(
            in_concepts=_axis(4), in_embeddings=8, out_concepts=out_ann,
        )
        assert pred.predictor.out_features == out_ann.size

    def test_gradient_flow(self):
        pred = MixSumConceptEmbeddingToConcept(
            in_concepts=_axis(6, cardinalities=[2, 2, 2]), in_embeddings=10, out_concepts=2,
        )
        concepts = torch.randn(2, 6, requires_grad=True)
        embeddings = torch.randn(2, 6, 10, requires_grad=True)
        pred(concepts=concepts, embeddings=embeddings).sum().backward()
        assert concepts.grad is not None
        assert embeddings.grad is not None


# ===========================================================================
# 3. MixConceptEmbeddings — the mixture without the concept head
# ===========================================================================

class TestMixConceptEmbeddings:
    def test_returns_the_mixture_not_a_prediction(self):
        aa = _axis(4, cardinalities=[1, 3])   # 1 binary + 1 three-way categorical
        layer = MixConceptEmbeddings(in_concepts=aa, in_embeddings=8)
        out = layer(concepts=torch.rand(2, 4), embeddings=torch.randn(2, 4, 8))
        assert out.shape == (2, 2, 8)         # (batch, n_groups, in_embeddings)

    def test_is_a_plain_module_not_a_concept_layer(self):
        """It emits embeddings, so it must stay outside the BaseConceptLayer
        taxonomy and compose the concept layer instead of inheriting it."""
        layer = MixConceptEmbeddings(in_concepts=_axis(2), in_embeddings=8)
        assert isinstance(layer, nn.Module)
        assert not isinstance(layer, BaseConceptLayer)

    def test_carries_no_prediction_head(self):
        layer = MixConceptEmbeddings(in_concepts=_axis(2), in_embeddings=8)
        assert not any("predictor" in n for n, _ in layer.named_parameters())

    def test_convex_combination_of_the_state_embeddings(self):
        """A categorical group mixes its own state embeddings by their scores."""
        aa = _axis(2, cardinalities=[2])
        layer = MixConceptEmbeddings(in_concepts=aa, in_embeddings=8)
        e0, e1 = torch.randn(3, 1, 8), torch.randn(3, 1, 8)
        embeddings = torch.cat([e0, e1], dim=1)
        c = torch.tensor([[0.3, 0.7]] * 3)
        assert torch.allclose(
            layer(concepts=c, embeddings=embeddings), 0.3 * e0 + 0.7 * e1, atol=1e-6
        )

    def test_gradients_flow_to_both_inputs(self):
        layer = MixConceptEmbeddings(in_concepts=_axis(2), in_embeddings=8)
        concepts = torch.rand(2, 2, requires_grad=True)
        embeddings = torch.randn(2, 2, 8, requires_grad=True)
        layer(concepts=concepts, embeddings=embeddings).sum().backward()
        assert concepts.grad is not None and embeddings.grad is not None
