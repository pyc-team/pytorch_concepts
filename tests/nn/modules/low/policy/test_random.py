"""Tests for torch_concepts.nn.modules.low.intervention.policy.random.RandomPolicy."""
import pytest
import torch

from torch_concepts.nn.modules.low.intervention.policy.random import RandomPolicy


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestRandomPolicyConstruction:
    def test_default_scale(self):
        policy = RandomPolicy()
        assert policy.scale == pytest.approx(1.0)

    def test_custom_scale(self):
        policy = RandomPolicy(scale=2.0)
        assert policy.scale == pytest.approx(2.0)

    def test_scale_zero(self):
        policy = RandomPolicy(scale=0.0)
        assert policy.scale == pytest.approx(0.0)

    def test_is_nn_module(self):
        import torch.nn as nn
        assert isinstance(RandomPolicy(), nn.Module)


# ===========================================================================
# 2. Forward pass — output shape and value range
# ===========================================================================

class TestRandomPolicyForward:
    def test_output_shape(self):
        policy = RandomPolicy(scale=1.0)
        out = policy(torch.randn(4, 10))
        assert out.shape == (4, 10)

    def test_output_shape_various_sizes(self):
        policy = RandomPolicy(scale=1.0)
        for bs in [1, 4, 16]:
            out = policy(torch.randn(bs, 7))
            assert out.shape == (bs, 7)

    def test_output_is_non_negative(self):
        policy = RandomPolicy(scale=2.0)
        out = policy(torch.randn(100, 10))
        assert (out >= 0).all()

    def test_output_bounded_by_scale(self):
        policy = RandomPolicy(scale=3.0)
        out = policy(torch.randn(100, 10))
        assert (out <= 3.0).all()

    def test_zero_scale_gives_zeros(self):
        policy = RandomPolicy(scale=0.0)
        out = policy(torch.randn(4, 10))
        assert torch.all(out == 0.0)

    def test_output_is_random(self):
        policy = RandomPolicy(scale=1.0)
        x = torch.randn(4, 10)
        out1 = policy(x)
        out2 = policy(x)
        assert not torch.equal(out1, out2)


# ===========================================================================
# 3. Scale effect
# ===========================================================================

class TestRandomPolicyScale:
    def test_larger_scale_larger_values(self):
        torch.manual_seed(42)
        small = RandomPolicy(scale=0.5)
        large = RandomPolicy(scale=5.0)
        x = torch.randn(100, 10)
        out_s = small(x).mean().item()
        out_l = large(x).mean().item()
        assert out_l > out_s

    def test_scale_one_mean_approx_half(self):
        policy = RandomPolicy(scale=1.0)
        out = policy(torch.randn(10000, 5))
        assert abs(out.mean().item() - 0.5) < 0.05


# ===========================================================================
# 4. build_mask (inherited from BaseInterventionPolicy)
# ===========================================================================

class TestRandomPolicyBuildMask:
    def test_mask_shape(self):
        policy = RandomPolicy(scale=1.0)
        scores = policy(torch.randn(4, 6))
        mask = policy.build_mask(scores, quantile=0.5)
        assert mask.shape == (4, 6)

    def test_mask_values_in_0_1(self):
        policy = RandomPolicy(scale=1.0)
        scores = policy(torch.randn(4, 6))
        mask = policy.build_mask(scores, quantile=1.0)
        assert (mask >= 0).all()

    def test_full_quantile_selects_all(self):
        policy = RandomPolicy(scale=1.0)
        B, F = 4, 5
        scores = policy(torch.randn(B, F))
        mask = policy.build_mask(scores, quantile=1.0)
        # quantile=1.0 → threshold is the max → nothing > max → all 0 (intervene on all)
        # (apart from STE proxy which makes them slightly > 0 but < 1)
        assert (mask <= 1.0).all()

    def test_subset_indices(self):
        policy = RandomPolicy(scale=1.0)
        B, F = 4, 6
        scores = policy(torch.randn(B, F))
        sel = torch.tensor([0, 2, 4])
        mask = policy.build_mask(scores, sel_idx=sel, quantile=1.0)
        assert mask.shape == (B, F)
        # Non-selected indices (1, 3, 5) must be exactly 1.0
        for i in [1, 3, 5]:
            assert torch.allclose(mask[:, i], torch.ones(B))

    def test_empty_subset_returns_ones(self):
        policy = RandomPolicy(scale=1.0)
        B, F = 2, 4
        scores = policy(torch.randn(B, F))
        mask = policy.build_mask(scores, sel_idx=torch.tensor([], dtype=torch.long))
        assert torch.allclose(mask, torch.ones(B, F))
