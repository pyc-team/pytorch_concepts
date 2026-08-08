"""Tests for MixConceptEmbeddingToConcept and MixSumConceptEmbeddingToConcept."""
import pytest
import torch
import torch.nn as nn

from torch_concepts import Annotations
from torch_concepts.nn import MixConceptEmbeddingToConcept, MixConceptEmbeddingToEmbedding
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
    def test_initialization_with_cardinalities(self):
        pred = MixSumConceptEmbeddingToConcept(
            in_concepts=10, in_embeddings=20, out_concepts=3, cardinalities=[3, 4, 3],
        )
        assert len(pred.in_concepts.labels) == 3  # 3 groups
        assert pred.out_concepts == 3

    def test_initialization_defaults_all_binary(self):
        pred = MixSumConceptEmbeddingToConcept(in_concepts=8, in_embeddings=16, out_concepts=4)
        assert pred.in_concepts.cardinalities == [1] * 8

    def test_forward_shape(self):
        pred = MixSumConceptEmbeddingToConcept(
            in_concepts=10, in_embeddings=20, out_concepts=3, cardinalities=[3, 4, 3],
        )
        out = pred(concepts=torch.randn(4, 10), embeddings=torch.randn(4, 10, 20))
        assert out.shape == (4, 3)

    def test_forward_shape_all_binary(self):
        pred = MixSumConceptEmbeddingToConcept(in_concepts=6, in_embeddings=12, out_concepts=2)
        out = pred(concepts=torch.randn(3, 6), embeddings=torch.randn(3, 6, 12))
        assert out.shape == (3, 2)

    def test_predictor_is_linear(self):
        pred = MixSumConceptEmbeddingToConcept(in_concepts=4, in_embeddings=8, out_concepts=2)
        assert isinstance(pred.predictor, nn.Linear)

    def test_group_count_invariance(self):
        p1 = MixSumConceptEmbeddingToConcept(
            in_concepts=4, in_embeddings=8, out_concepts=2, cardinalities=[1] * 4
        )
        p2 = MixSumConceptEmbeddingToConcept(
            in_concepts=6, in_embeddings=8, out_concepts=2, cardinalities=[2, 2, 2]
        )
        assert p1.predictor.weight.shape == p2.predictor.weight.shape

    def test_gradient_flow(self):
        pred = MixSumConceptEmbeddingToConcept(
            in_concepts=6, in_embeddings=10, out_concepts=2, cardinalities=[2, 2, 2],
        )
        concepts = torch.randn(2, 6, requires_grad=True)
        embeddings = torch.randn(2, 6, 10, requires_grad=True)
        pred(concepts=concepts, embeddings=embeddings).sum().backward()
        assert concepts.grad is not None
        assert embeddings.grad is not None


# ===========================================================================
# 3. Binary concepts with two INDEPENDENT state embeddings
#    (expand_binary_embeddings=False): CEM/CBGM's own layout.
# ===========================================================================

class TestMixConceptEmbeddingBinaryStates:
    def test_default_still_allocates_the_splitter(self):
        """Backward compat: the flag defaults to True, so a caller who never
        heard of it gets the historical single-embedding-per-binary-concept
        layout untouched."""
        aa = _axis(2, cardinalities=[1, 1])
        layer = MixConceptEmbeddingToEmbedding(in_concepts=aa, in_embeddings=8)
        assert layer.expand_binary_embeddings is True
        assert layer.bernoulli_to_categorical_embedding_splitter is not None

    def test_expand_false_allocates_no_splitter_and_zero_parameters(self):
        aa = _axis(2, cardinalities=[1, 1])
        layer = MixConceptEmbeddingToEmbedding(
            in_concepts=aa, in_embeddings=8, expand_binary_embeddings=False,
        )
        assert layer.bernoulli_to_categorical_embedding_splitter is None
        assert sum(p.numel() for p in layer.parameters()) == 0

    def test_all_categorical_allocates_no_splitter_either_way(self):
        """No binary concept in the annotation -> nothing to expand, so the
        splitter is dead weight regardless of the flag."""
        aa = _axis(6, cardinalities=[3, 3])
        layer = MixConceptEmbeddingToEmbedding(in_concepts=aa, in_embeddings=8)
        assert layer.bernoulli_to_categorical_embedding_splitter is None

    def test_row_count_contract_binary_plus_categorical(self):
        # 1 binary (2 rows) + 1 three-way categorical (3 rows) = 5 embedding rows;
        # the concept-SCORE axis stays 1 + 3 = 4 columns either way.
        aa = _axis(4, cardinalities=[1, 3])
        embeddings_pre_expanded = torch.randn(2, 5, 8)
        concepts = torch.rand(2, 4)

        layer_false = MixConceptEmbeddingToEmbedding(
            in_concepts=aa, in_embeddings=8, expand_binary_embeddings=False,
        )
        out_false = layer_false(concepts=concepts, embeddings=embeddings_pre_expanded)
        assert out_false.shape == (2, 2, 8)  # 2 groups: the binary concept + the categorical one

        embeddings_unexpanded = torch.randn(2, 4, 8)  # 1 row per binary concept, old convention
        layer_true = MixConceptEmbeddingToEmbedding(in_concepts=aa, in_embeddings=8)
        out_true = layer_true(concepts=concepts, embeddings=embeddings_unexpanded)
        assert out_true.shape == (2, 2, 8)

    def test_true_convex_combination(self):
        """With the embedding rows supplied pre-paired (w+, w-), the mixture is
        a genuine convex combination of the two — the headline property this
        change exists for."""
        aa = _axis(1, cardinalities=[1])
        layer = MixConceptEmbeddingToEmbedding(
            in_concepts=aa, in_embeddings=8, expand_binary_embeddings=False,
        )
        w_plus = torch.randn(3, 1, 8)
        w_minus = torch.randn(3, 1, 8)
        embeddings = torch.cat([w_plus, w_minus], dim=1)  # (3, 2, 8): [w+, w-]

        c = torch.tensor([[0.3]] * 3)
        out = layer(concepts=c, embeddings=embeddings)          # (3, 1, 8)
        expected = 0.3 * w_plus + 0.7 * w_minus                  # (3, 1, 8)
        assert torch.allclose(out, expected, atol=1e-6)

        # The two endpoints select a state exactly.
        out_on = layer(concepts=torch.ones(3, 1), embeddings=embeddings)
        assert torch.allclose(out_on, w_plus, atol=1e-6)
        out_off = layer(concepts=torch.zeros(3, 1), embeddings=embeddings)
        assert torch.allclose(out_off, w_minus, atol=1e-6)

    def test_buffers_are_non_persistent(self):
        aa = _axis(2, cardinalities=[1, 1])
        layer = MixConceptEmbeddingToEmbedding(in_concepts=aa, in_embeddings=8)
        buffer_names = dict(layer.named_buffers())
        assert "cardinalities_expanded" in buffer_names
        assert "binary_concept_columns" in buffer_names
        assert "cardinalities_expanded" not in layer.state_dict()
        assert "binary_concept_columns" not in layer.state_dict()
