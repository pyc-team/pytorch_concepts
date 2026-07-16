"""Tests for ConceptWhitening and WhitenedEmbeddingToConcept."""
import pytest
import torch
import torch.nn as nn

from torch_concepts.nn import Sequential
from torch_concepts.nn.modules.low.encoders.whitening import (
    ConceptWhitening,
    WhitenedEmbeddingToConcept,
)


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestConceptWhiteningConstruction:
    def test_stores_dims(self):
        cw = ConceptWhitening(in_features=16)
        assert cw.in_features == 16

    def test_buffers_initialized_to_identity(self):
        cw = ConceptWhitening(in_features=8)
        assert torch.equal(cw.running_wm, torch.eye(8))
        assert torch.equal(cw.running_rot, torch.eye(8))
        assert torch.equal(cw.running_mean, torch.zeros(8))

    def test_no_trainable_parameters(self):
        cw = ConceptWhitening(in_features=8)
        assert len(list(cw.parameters())) == 0

    def test_buffers_in_state_dict(self):
        cw = ConceptWhitening(in_features=8)
        sd = cw.state_dict()
        assert set(sd.keys()) == {
            "running_mean", "running_wm", "running_rot", "sum_G", "counter"
        }

    def test_state_dict_round_trip_restores_alignment(self):
        cw, _, concept_batch = _make_layer_and_data(in_features=8)
        with cw.align(0):
            cw(concept_batch)
        cw.update_rotation_matrix()
        cw.eval()
        expected = cw(concept_batch)

        restored = ConceptWhitening(in_features=8)
        restored.load_state_dict(cw.state_dict())
        restored.eval()
        assert torch.equal(restored(concept_batch), expected)


# ===========================================================================
# 2. Forward pass / whitening property
# ===========================================================================

class TestConceptWhiteningForward:
    def test_output_shape_preserved(self):
        cw = ConceptWhitening(in_features=16)
        x = torch.randn(32, 16)
        assert cw(x).shape == (32, 16)

    def test_leading_dims_preserved(self):
        cw = ConceptWhitening(in_features=8)
        x = torch.randn(4, 5, 8)
        assert cw(x).shape == (4, 5, 8)

    def test_output_is_whitened_in_training(self):
        torch.manual_seed(0)
        # Newton-Schulz whitening is approximate; near-exact for a
        # well-conditioned covariance and T=10 iterations
        cw = ConceptWhitening(
            in_features=8, num_iterations=10
        ).train()
        # correlated, shifted input with singular values in [0.5, 2]
        q1, _ = torch.linalg.qr(torch.randn(8, 8))
        q2, _ = torch.linalg.qr(torch.randn(8, 8))
        mix = q1 @ torch.diag(torch.linspace(0.5, 2.0, 8)) @ q2
        x = torch.randn(2048, 8) @ mix + 3.0
        z = cw(x)
        assert torch.allclose(z.mean(0), torch.zeros(8), atol=1e-2)
        cov = z.t() @ z / z.size(0)
        assert torch.allclose(cov, torch.eye(8), atol=0.05)

    def test_eval_uses_running_stats(self):
        torch.manual_seed(0)
        cw = ConceptWhitening(in_features=8).train()
        x = torch.randn(512, 8) * 2.0 + 1.0
        for _ in range(200):
            cw(x)
        cw.eval()
        z = cw(x)
        assert torch.allclose(z.mean(0), torch.zeros(8), atol=0.1)
        cov = z.t() @ z / z.size(0)
        assert torch.allclose(cov, torch.eye(8), atol=0.15)

    def test_eval_forward_is_deterministic(self):
        cw = ConceptWhitening(in_features=8).eval()
        x = torch.randn(16, 8)
        assert torch.equal(cw(x), cw(x))

    def test_unbatched_1d_input(self):
        cw = ConceptWhitening(in_features=8).eval()
        x = torch.randn(8)
        assert cw(x).shape == (8,)

    def test_single_feature_dimension(self):
        cw = ConceptWhitening(in_features=1).train()
        x = torch.randn(32, 1) * 3.0 + 5.0
        z = cw(x)
        assert z.shape == (32, 1)
        assert torch.isfinite(z).all()


# ===========================================================================
# 3. Gradient flow
# ===========================================================================

class TestConceptWhiteningGradients:
    def test_gradient_flows_to_input_in_training(self):
        cw = ConceptWhitening(in_features=8).train()
        x = torch.randn(64, 8, requires_grad=True)
        cw(x).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flows_to_input_in_eval(self):
        cw = ConceptWhitening(in_features=8).eval()
        x = torch.randn(4, 8, requires_grad=True)
        cw(x).sum().backward()
        assert x.grad is not None


# ===========================================================================
# 4. Concept alignment (rotation)
# ===========================================================================

def _make_layer_and_data(cls=ConceptWhitening, **kwargs):
    torch.manual_seed(0)
    d = 8
    layer = cls(**kwargs).train()
    # warm up whitening statistics
    base = torch.randn(4096, d)
    for _ in range(50):
        layer(base)
    # concept samples: shifted along a fixed latent direction
    direction = torch.zeros(d)
    direction[5] = 1.0
    concept_batch = torch.randn(512, d) + 4.0 * direction
    return layer, base, concept_batch


class TestConceptWhiteningAlignment:
    def _make(self):
        return _make_layer_and_data(in_features=8)

    def test_align_context_sets_and_resets_mode(self):
        cw = ConceptWhitening(in_features=8)
        assert cw.mode == -1
        with cw.align(1):
            assert cw.mode == 1
        assert cw.mode == -1

    def test_align_switches_to_eval_and_restores(self):
        cw = ConceptWhitening(in_features=8).train()
        with cw.align(0):
            assert not cw.training
        assert cw.training

    def test_align_does_not_update_running_stats(self):
        cw, _, concept_batch = self._make()
        mean_before = cw.running_mean.clone()
        with cw.align(0):
            cw(concept_batch)
        assert torch.equal(cw.running_mean, mean_before)

    def test_align_any_axis_is_valid(self):
        cw = ConceptWhitening(in_features=8)
        with cw.align(7):
            assert cw.mode == 7

    def test_align_invalid_axis_raises(self):
        cw = ConceptWhitening(in_features=8)
        with pytest.raises(ValueError):
            cw.align(8)

    def test_align_negative_axis_raises(self):
        cw = ConceptWhitening(in_features=8)
        with pytest.raises(ValueError):
            cw.align(-1)

    def test_align_accumulates_gradient(self):
        cw, _, concept_batch = self._make()
        with cw.align(0):
            cw(concept_batch)
        assert cw.sum_G[:, 0].abs().sum() > 0
        assert cw.counter[0] > 1e-3

    def test_rotation_stays_orthogonal_after_update(self):
        cw, _, concept_batch = self._make()
        for _ in range(5):
            with cw.align(0):
                cw(concept_batch)
            cw.update_rotation_matrix()
        RtR = cw.running_rot.t() @ cw.running_rot
        assert torch.allclose(RtR, torch.eye(8), atol=1e-4)

    def test_update_increases_concept_activation(self):
        cw, _, concept_batch = self._make()
        cw.eval()
        before = cw(concept_batch)[:, 0].mean()
        for _ in range(20):
            with cw.align(0):
                cw(concept_batch)
            cw.update_rotation_matrix()
        after = cw(concept_batch)[:, 0].mean()
        assert after > before

    def test_update_resets_counter(self):
        cw, _, concept_batch = self._make()
        with cw.align(0):
            cw(concept_batch)
        cw.update_rotation_matrix()
        assert torch.allclose(cw.counter, torch.full((8,), 1e-3))

    def test_no_accumulation_without_align(self):
        cw, base, _ = self._make()
        cw(base)
        assert cw.sum_G.abs().sum() == 0

    def test_update_without_alignment_is_a_no_op(self):
        # calling update_rotation_matrix before any align() must not move R
        # away from identity or produce NaNs (guards the counter=1e-3 default
        # against a division blow-up)
        cw = ConceptWhitening(in_features=8)
        cw.update_rotation_matrix()
        assert torch.equal(cw.running_rot, torch.eye(8))

    def test_multiple_concepts_aligned_independently(self):
        # the real training pattern: align several concepts on their own
        # batches, then update once, repeated over rounds
        torch.manual_seed(0)
        d = 8
        cw = ConceptWhitening(in_features=d).train()
        base = torch.randn(4096, d)
        for _ in range(50):
            cw(base)

        dir0, dir1 = torch.zeros(d), torch.zeros(d)
        dir0[3], dir1[6] = 1.0, 1.0
        batch0 = torch.randn(512, d) + 4.0 * dir0
        batch1 = torch.randn(512, d) + 4.0 * dir1

        cw.eval()
        before0 = cw(batch0)[:, 0].mean()
        before1 = cw(batch1)[:, 1].mean()

        for _ in range(30):
            with cw.align(0):
                cw(batch0)
            with cw.align(1):
                cw(batch1)
            cw.update_rotation_matrix()

        after0 = cw(batch0)[:, 0].mean()
        after1 = cw(batch1)[:, 1].mean()
        assert after0 > before0
        assert after1 > before1
        RtR = cw.running_rot.t() @ cw.running_rot
        assert torch.allclose(RtR, torch.eye(d), atol=1e-4)


# ===========================================================================
# 5. WhitenedEmbeddingToConcept encoder
# ===========================================================================

class TestWhitenedEmbeddingToConcept:
    def test_is_base_concept_layer(self):
        from torch_concepts.nn import BaseConceptLayer
        enc = WhitenedEmbeddingToConcept(in_embeddings=16, out_concepts=4)
        assert isinstance(enc, BaseConceptLayer)

    def test_encoder_is_concept_whitening(self):
        enc = WhitenedEmbeddingToConcept(in_embeddings=16, out_concepts=4)
        assert isinstance(enc.encoder, ConceptWhitening)

    def test_more_concepts_than_embeddings_raises(self):
        with pytest.raises(ValueError):
            WhitenedEmbeddingToConcept(in_embeddings=4, out_concepts=5)

    def test_out_concepts_equal_in_embeddings_is_allowed(self):
        # boundary: full-width "bottleneck" (k == d) is valid, only k > d
        # should raise
        enc = WhitenedEmbeddingToConcept(in_embeddings=4, out_concepts=4)
        x = torch.randn(3, 4)
        assert enc(x).shape == (3, 4)

    def test_align_beyond_concepts_raises(self):
        # axis 2 exists in the wrapped layer, but is not a concept axis
        enc = WhitenedEmbeddingToConcept(in_embeddings=8, out_concepts=2)
        with pytest.raises(ValueError):
            enc.align(2)

    def test_output_shape_is_concepts(self):
        enc = WhitenedEmbeddingToConcept(in_embeddings=16, out_concepts=4)
        x = torch.randn(8, 16)
        assert enc(x).shape == (8, 4)

    def test_kwargs_forwarded(self):
        enc = WhitenedEmbeddingToConcept(
            in_embeddings=16,
            out_concepts=4,
            num_iterations=7,
            eps=1e-3,
            momentum=0.2,
        )
        assert enc.encoder.num_iterations == 7
        assert enc.encoder.eps == 1e-3
        assert enc.encoder.momentum == 0.2

    def test_matches_full_output_slice(self):
        enc = WhitenedEmbeddingToConcept(in_embeddings=16, out_concepts=4).eval()
        x = torch.randn(8, 16)
        assert torch.equal(enc(x), enc.encoder(x)[:, :4])

    def test_alignment_delegation(self):
        enc, _, concept_batch = _make_layer_and_data(
            WhitenedEmbeddingToConcept, in_embeddings=8, out_concepts=2
        )
        enc.eval()
        before = enc(concept_batch)[:, 0].mean()
        for _ in range(20):
            with enc.align(0):
                enc(concept_batch)
            enc.update_rotation_matrix()
        after = enc(concept_batch)[:, 0].mean()
        assert after > before

    def test_in_sequential_pipeline(self):
        backbone = nn.Linear(32, 16)
        enc = WhitenedEmbeddingToConcept(in_embeddings=16, out_concepts=4)
        head = nn.Linear(4, 3)
        model = Sequential(backbone, enc, head)
        x = torch.randn(8, 32)
        assert model(x).shape == (8, 3)

    def test_sequential_gradients_reach_backbone(self):
        backbone = nn.Linear(32, 16)
        enc = WhitenedEmbeddingToConcept(in_embeddings=16, out_concepts=4)
        head = nn.Linear(4, 3)
        model = Sequential(backbone, enc, head)
        model(torch.randn(8, 32)).sum().backward()
        assert backbone.weight.grad is not None

    def test_concept_whitening_as_normalization_in_sequential(self):
        # paper-faithful use: full-width CW inside the chain, no bottleneck
        model = Sequential(
            nn.Linear(32, 16),
            ConceptWhitening(in_features=16),
            nn.Linear(16, 3),
        )
        x = torch.randn(8, 32)
        assert model(x).shape == (8, 3)
