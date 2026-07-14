"""Tests for LinearEmbeddingToConcept."""
import pytest
import torch
import torch.nn as nn

from torch_concepts.nn.modules.low.encoders.linear import LinearEmbeddingToConcept


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestLinearEmbeddingToConceptConstruction:
    def test_stores_in_embeddings(self):
        enc = LinearEmbeddingToConcept(in_embeddings=16, out_concepts=5)
        assert enc.in_embeddings == 16

    def test_stores_out_concepts(self):
        enc = LinearEmbeddingToConcept(in_embeddings=16, out_concepts=5)
        assert enc.out_concepts == 5

    def test_encoder_is_linear(self):
        enc = LinearEmbeddingToConcept(in_embeddings=16, out_concepts=5)
        assert isinstance(enc.encoder, nn.Linear)

    def test_encoder_in_features(self):
        enc = LinearEmbeddingToConcept(in_embeddings=32, out_concepts=8)
        assert enc.encoder.in_features == 32

    def test_encoder_out_features(self):
        enc = LinearEmbeddingToConcept(in_embeddings=32, out_concepts=8)
        assert enc.encoder.out_features == 8

    def test_in_embeddings_shape_is_int(self):
        enc = LinearEmbeddingToConcept(in_embeddings=16, out_concepts=5)
        assert enc.in_embeddings_shape == 16

    def test_in_concepts_is_none(self):
        enc = LinearEmbeddingToConcept(in_embeddings=16, out_concepts=5)
        assert enc.in_concepts is None

    def test_with_bias_false(self):
        enc = LinearEmbeddingToConcept(in_embeddings=8, out_concepts=4, bias=False)
        assert enc.encoder.bias is None


# ===========================================================================
# 2. Forward pass
# ===========================================================================

class TestLinearEmbeddingToConceptForward:
    def test_output_shape(self):
        enc = LinearEmbeddingToConcept(in_embeddings=8, out_concepts=3)
        x = torch.randn(4, 8)
        assert enc(x).shape == (4, 3)

    def test_single_concept(self):
        enc = LinearEmbeddingToConcept(in_embeddings=10, out_concepts=1)
        x = torch.randn(3, 10)
        assert enc(x).shape == (3, 1)

    def test_various_batch_sizes(self):
        enc = LinearEmbeddingToConcept(in_embeddings=32, out_concepts=5)
        for bs in [1, 4, 8, 16]:
            out = enc(torch.randn(bs, 32))
            assert out.shape == (bs, 5)

    def test_no_activation_raw_linear(self):
        enc = LinearEmbeddingToConcept(in_embeddings=4, out_concepts=2)
        x = torch.randn(2, 4)
        out = enc(x)
        # Raw linear output — values can be outside [0,1]
        assert out.shape == (2, 2)

    def test_output_is_float_tensor(self):
        enc = LinearEmbeddingToConcept(in_embeddings=8, out_concepts=3)
        out = enc(torch.randn(2, 8))
        assert out.dtype == torch.float32


# ===========================================================================
# 3. Gradient flow
# ===========================================================================

class TestLinearEmbeddingToConceptGradients:
    def test_gradient_flows_to_input(self):
        enc = LinearEmbeddingToConcept(in_embeddings=8, out_concepts=3)
        x = torch.randn(2, 8, requires_grad=True)
        enc(x).sum().backward()
        assert x.grad is not None

    def test_gradient_flows_to_weights(self):
        enc = LinearEmbeddingToConcept(in_embeddings=8, out_concepts=3)
        enc(torch.randn(2, 8)).sum().backward()
        assert enc.encoder.weight.grad is not None

    def test_gradient_is_nonzero(self):
        enc = LinearEmbeddingToConcept(in_embeddings=8, out_concepts=3)
        x = torch.randn(2, 8, requires_grad=True)
        enc(x).sum().backward()
        assert x.grad.abs().sum() > 0

    def test_no_bias_gradient(self):
        enc = LinearEmbeddingToConcept(in_embeddings=8, out_concepts=3, bias=False)
        enc(torch.randn(2, 8)).sum().backward()
        assert enc.encoder.bias is None


# ===========================================================================
# 4. As part of a pipeline
# ===========================================================================

class TestLinearEmbeddingInPipeline:
    def test_pipeline_with_second_layer(self):
        enc = LinearEmbeddingToConcept(in_embeddings=16, out_concepts=8)
        pred = nn.Linear(8, 3)
        x = torch.randn(4, 16)
        c = enc(x)
        y = pred(c)
        assert y.shape == (4, 3)

    def test_pipeline_gradients_end_to_end(self):
        enc = LinearEmbeddingToConcept(in_embeddings=16, out_concepts=8)
        pred = nn.Linear(8, 3)
        x = torch.randn(4, 16, requires_grad=True)
        y = pred(torch.relu(enc(x)))
        y.sum().backward()
        assert x.grad is not None
        assert enc.encoder.weight.grad is not None

    def test_large_embedding_to_few_concepts(self):
        enc = LinearEmbeddingToConcept(in_embeddings=512, out_concepts=10)
        x = torch.randn(8, 512)
        out = enc(x)
        assert out.shape == (8, 10)
