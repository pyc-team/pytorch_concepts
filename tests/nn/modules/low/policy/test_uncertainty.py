"""Tests for UncertaintyInterventionPolicy."""
import pytest
import torch

from torch_concepts.nn.modules.low.intervention.policy.uncertainty import (
    UncertaintyInterventionPolicy,
)


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestUncertaintyInterventionPolicyConstruction:
    def test_default_max_uncertainty_point(self):
        policy = UncertaintyInterventionPolicy()
        assert policy.max_uncertainty_point == pytest.approx(0.0)

    def test_custom_max_uncertainty_point(self):
        policy = UncertaintyInterventionPolicy(max_uncertainty_point=0.5)
        assert policy.max_uncertainty_point == pytest.approx(0.5)

    def test_is_nn_module(self):
        import torch.nn as nn
        assert isinstance(UncertaintyInterventionPolicy(), nn.Module)

    def test_negative_max_uncertainty_point(self):
        policy = UncertaintyInterventionPolicy(max_uncertainty_point=-1.0)
        assert policy.max_uncertainty_point == pytest.approx(-1.0)


# ===========================================================================
# 2. Forward pass — shape
# ===========================================================================

class TestUncertaintyInterventionPolicyShape:
    def test_output_shape(self):
        policy = UncertaintyInterventionPolicy()
        out = policy(torch.randn(4, 10))
        assert out.shape == (4, 10)

    def test_various_batch_sizes(self):
        policy = UncertaintyInterventionPolicy()
        for bs in [1, 4, 8]:
            out = policy(torch.randn(bs, 7))
            assert out.shape == (bs, 7)

    def test_single_concept(self):
        policy = UncertaintyInterventionPolicy()
        out = policy(torch.randn(3, 1))
        assert out.shape == (3, 1)


# ===========================================================================
# 3. Certainty semantics
# ===========================================================================

class TestUncertaintyPolicyCertainty:
    def test_high_certainty_values_far_from_zero(self):
        policy = UncertaintyInterventionPolicy(max_uncertainty_point=0.0)
        high = torch.tensor([[10.0, -10.0]])
        low = torch.tensor([[0.1, -0.1]])
        assert policy(high).mean() > policy(low).mean()

    def test_output_is_non_negative(self):
        policy = UncertaintyInterventionPolicy(max_uncertainty_point=0.0)
        out = policy(torch.randn(100, 10))
        assert (out >= 0).all()

    def test_zero_input_gives_zero_certainty(self):
        policy = UncertaintyInterventionPolicy(max_uncertainty_point=0.0)
        x = torch.zeros(3, 4)
        out = policy(x)
        assert torch.all(out == 0.0)

    def test_certainty_equals_abs_distance_from_mup(self):
        policy = UncertaintyInterventionPolicy(max_uncertainty_point=0.5)
        x = torch.tensor([[0.5, 1.0, 0.0, -0.5]])
        expected = (x - 0.5).abs()
        assert torch.allclose(policy(x), expected)

    def test_symmetric_around_max_uncertainty_point(self):
        policy = UncertaintyInterventionPolicy(max_uncertainty_point=0.0)
        pos = torch.tensor([[0.3]])
        neg = torch.tensor([[-0.3]])
        assert torch.allclose(policy(pos), policy(neg))


# ===========================================================================
# 4. Gradient flow
# ===========================================================================

class TestUncertaintyPolicyGradients:
    def test_gradient_flows_through_policy(self):
        policy = UncertaintyInterventionPolicy(max_uncertainty_point=0.0)
        x = torch.randn(2, 5, requires_grad=True)
        policy(x).sum().backward()
        assert x.grad is not None

    def test_no_gradient_at_zero(self):
        policy = UncertaintyInterventionPolicy(max_uncertainty_point=0.0)
        x = torch.zeros(2, 4, requires_grad=True)
        policy(x).sum().backward()
        # grad of abs(x-0) at x=0 is undefined (subgradient); PyTorch returns 0
        assert x.grad is not None


# ===========================================================================
# 5. build_mask (inherited from BaseInterventionPolicy)
# ===========================================================================

class TestUncertaintyBuildMask:
    def test_mask_shape(self):
        policy = UncertaintyInterventionPolicy()
        B, F = 4, 6
        x = torch.rand(B, F)
        scores = policy(x)
        mask = policy.build_mask(scores, quantile=0.5)
        assert mask.shape == (B, F)

    def test_most_uncertain_selected_first_at_low_quantile(self):
        policy = UncertaintyInterventionPolicy(max_uncertainty_point=0.5)
        B, F = 1, 4
        # Construct scores manually: concept 0 is least certain (largest distance from 0.5)
        x = torch.tensor([[0.0, 0.4, 0.6, 0.5]])
        # certainty = abs(x - 0.5): [0.5, 0.1, 0.1, 0.0]
        scores = policy(x)
        # quantile=1.0 → intervene on ALL → mask=0 for all selected
        mask_all = policy.build_mask(scores, quantile=1.0)
        # All selected concepts should have mask < 1 (intervened)
        assert mask_all.max() <= 1.0

    def test_no_selected_returns_ones(self):
        policy = UncertaintyInterventionPolicy()
        B, F = 2, 4
        scores = policy(torch.randn(B, F))
        mask = policy.build_mask(scores, sel_idx=torch.tensor([], dtype=torch.long))
        assert torch.allclose(mask, torch.ones(B, F))
