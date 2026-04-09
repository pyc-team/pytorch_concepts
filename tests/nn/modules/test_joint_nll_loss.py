"""
Tests for JointNLLLoss.

Tests cover:
- Correct NLL indexing into multi-dimensional log-joint
- BCE fallback for 2D (flat logit) input
- Gradient flow through log-joint path
- Gradient flow through BCE fallback path
- Known-value computation check
- Single-concept edge case
"""
import pytest
import torch
from torch import nn

from torch_concepts.nn.modules.loss import JointNLLLoss


class TestJointNLLLossLogJoint:
    """Tests for the multi-dim log-joint path (ndim > 2)."""

    def test_known_value(self):
        """Hand-computed NLL for a (2, 2, 2) log-joint."""
        # Two binary concepts, batch=2
        joint = torch.tensor([
            [[0.1, 0.2], [0.3, 0.4]],  # sample 0
            [[0.05, 0.15], [0.35, 0.45]],  # sample 1
        ])
        log_joint = torch.log(joint)
        target = torch.tensor([[0, 1], [1, 0]])  # sample0: (0,1), sample1: (1,0)

        loss_fn = JointNLLLoss()
        loss = loss_fn(input=log_joint, target=target)

        # sample 0: joint[0, 0, 1] = 0.2 → -log(0.2)
        # sample 1: joint[1, 1, 0] = 0.35 → -log(0.35)
        expected = -(torch.log(torch.tensor(0.2)) + torch.log(torch.tensor(0.35))) / 2
        torch.testing.assert_close(loss, expected)

    def test_output_is_scalar(self):
        log_joint = torch.randn(16, 2, 3)
        target = torch.randint(0, 2, (16, 1)).long()
        target = torch.cat([target, torch.randint(0, 3, (16, 1)).long()], dim=1)
        loss = JointNLLLoss()(input=log_joint, target=target)
        assert loss.ndim == 0

    def test_gradient_flows(self):
        log_joint = torch.randn(8, 2, 2, requires_grad=True)
        target = torch.randint(0, 2, (8, 2))
        loss = JointNLLLoss()(input=log_joint, target=target)
        loss.backward()
        assert log_joint.grad is not None
        assert log_joint.grad.shape == log_joint.shape

    def test_single_concept(self):
        """Edge case: one concept with cardinality 3 → (batch, 3) log-joint
        is 2D so falls through to BCE. Use 3D (batch, 1, 3) for log-joint."""
        # Reshape to (batch, 1, 3) so it's treated as log-joint
        log_joint = torch.log_softmax(torch.randn(4, 1, 3), dim=-1)
        target = torch.randint(0, 1, (4, 1)).long()  # index into dim of size 1
        # This should still work for the log-joint path
        loss = JointNLLLoss()(input=log_joint, target=target)
        assert loss.ndim == 0

    def test_many_concepts(self):
        """Batch with 4 binary concepts → shape (batch, 2, 2, 2, 2)."""
        log_joint = torch.log_softmax(torch.randn(8, 2, 2, 2, 2).view(8, -1),
                                      dim=-1).view(8, 2, 2, 2, 2)
        target = torch.randint(0, 2, (8, 4))
        loss = JointNLLLoss()(input=log_joint, target=target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestJointNLLLossBCEFallback:
    """Tests for the 2D BCE fallback path."""

    def test_matches_bce(self):
        logits = torch.randn(16, 4)
        target = torch.rand(16, 4)
        loss_joint = JointNLLLoss()(input=logits, target=target)
        loss_bce = nn.BCEWithLogitsLoss()(logits, target)
        torch.testing.assert_close(loss_joint, loss_bce)

    def test_gradient_flows_bce(self):
        logits = torch.randn(8, 3, requires_grad=True)
        target = torch.rand(8, 3)
        loss = JointNLLLoss()(input=logits, target=target)
        loss.backward()
        assert logits.grad is not None
