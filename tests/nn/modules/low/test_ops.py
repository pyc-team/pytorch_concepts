"""Tests for torch_concepts.nn.modules.low.ops (SumOp, ResidualCorrectionOp)."""

import pytest
import torch

from torch_concepts.nn.modules.low.ops import SumOp, ResidualCorrectionOp


# =============================================================================
# SumOp
# =============================================================================

class TestSumOp:
    def test_invalid_n_terms_raises(self):
        with pytest.raises(ValueError, match="n_terms must be >= 1"):
            SumOp(input_size=4, n_terms=0)

    def test_sum_of_three_terms(self):
        layer = SumOp(input_size=4, n_terms=3)
        out = layer(torch.ones(2, 12))
        assert out.shape == (2, 4)
        assert torch.equal(out, torch.full((2, 4), 3.0))

    def test_single_term_is_identity(self):
        layer = SumOp(input_size=5, n_terms=1)
        x = torch.randn(3, 5)
        out = layer(x)
        assert torch.equal(out, x)

    def test_leading_dims_preserved(self):
        layer = SumOp(input_size=2, n_terms=2)
        out = layer(torch.ones(7, 3, 4))   # (*, n_terms*input_size)
        assert out.shape == (7, 3, 2)
        assert torch.equal(out, torch.full((7, 3, 2), 2.0))

    def test_wrong_last_dim_raises(self):
        layer = SumOp(input_size=4, n_terms=3)
        with pytest.raises(ValueError, match="got last dim"):
            layer(torch.ones(2, 11))

    def test_actual_sum_values(self):
        # Two chunks [1,2] and [10,20] -> [11, 22]
        layer = SumOp(input_size=2, n_terms=2)
        x = torch.tensor([[1.0, 2.0, 10.0, 20.0]])
        assert torch.equal(layer(x), torch.tensor([[11.0, 22.0]]))


# =============================================================================
# ResidualCorrectionOp — construction validation
# =============================================================================

class TestResidualCorrectionInit:
    def test_invalid_n_terms(self):
        with pytest.raises(ValueError, match="n_terms must be >= 1"):
            ResidualCorrectionOp(input_size=4, n_terms=0)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="residual_mode must be one of"):
            ResidualCorrectionOp(input_size=4, n_terms=2, residual_mode="bogus")

    def test_stop_grad_index_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            ResidualCorrectionOp(input_size=4, n_terms=2, stop_grad_parts=(2,))

    def test_stop_grad_parts_stored_as_tuple(self):
        op = ResidualCorrectionOp(input_size=4, n_terms=2, stop_grad_parts=[0, 1])
        assert op.stop_grad_parts == (0, 1)

    def test_default_stop_grad_parts_empty(self):
        op = ResidualCorrectionOp(input_size=4, n_terms=2)
        assert op.stop_grad_parts == ()


# =============================================================================
# ResidualCorrectionOp — compute values
# =============================================================================

class TestResidualCorrectionValues:
    def test_block_parts_reconstructs_target(self):
        op = ResidualCorrectionOp(input_size=4, n_terms=2, residual_mode="block_parts")
        target = torch.randn(3, 4)
        p0, p1 = torch.randn(3, 4), torch.randn(3, 4)
        eps = op.compute(target, p0, p1)
        torch.testing.assert_close(p0 + p1 + eps, target)

    def test_keep_parts_reconstructs_target(self):
        op = ResidualCorrectionOp(input_size=4, n_terms=2, residual_mode="keep_parts")
        target = torch.randn(3, 4)
        p0, p1 = torch.randn(3, 4), torch.randn(3, 4)
        eps = op.compute(target, p0, p1)
        torch.testing.assert_close(p0 + p1 + eps, target)

    def test_off_mode_epsilon_is_zero(self):
        op = ResidualCorrectionOp(input_size=4, n_terms=2, residual_mode="off")
        target = torch.randn(3, 4)
        p0, p1 = torch.randn(3, 4), torch.randn(3, 4)
        eps = op.compute(target, p0, p1)
        assert torch.equal(eps, torch.zeros_like(target))

    def test_off_mode_hbar_is_sum_of_parts(self):
        op = ResidualCorrectionOp(input_size=4, n_terms=2, residual_mode="off")
        target = torch.randn(3, 4)
        p0, p1 = torch.randn(3, 4), torch.randn(3, 4)
        eps = op.compute(target, p0, p1)
        torch.testing.assert_close(p0 + p1 + eps, p0 + p1)

    def test_stop_grad_adds_zero_to_value(self):
        # stop-grad contribution p.detach()-p is numerically zero.
        op = ResidualCorrectionOp(input_size=4, n_terms=2, residual_mode="off",
                                  stop_grad_parts=(1,))
        target = torch.randn(2, 4)
        p0, p1 = torch.randn(2, 4), torch.randn(2, 4)
        eps = op.compute(target, p0, p1)
        torch.testing.assert_close(eps, torch.zeros_like(target), atol=1e-6, rtol=0)

    def test_wrong_number_of_parts_raises(self):
        op = ResidualCorrectionOp(input_size=4, n_terms=2)
        with pytest.raises(ValueError, match="Expected 2 parts"):
            op.compute(torch.randn(2, 4), torch.randn(2, 4))


# =============================================================================
# ResidualCorrectionOp — gradient-flow behaviour table (n_terms=2)
# =============================================================================

def _part_grads(op, target, p0, p1):
    """Return (grad_p0, grad_p1) of sum(h_bar) where h_bar = p0+p1+epsilon."""
    eps = op.compute(target, p0, p1)
    h_bar = p0 + p1 + eps
    h_bar.sum().backward()
    return p0.grad.abs().max().item(), p1.grad.abs().max().item()


class TestResidualCorrectionGradients:
    def _parts(self):
        return (torch.randn(2, 3),
                torch.randn(2, 3, requires_grad=True),
                torch.randn(2, 3, requires_grad=True))

    def test_block_parts_blocks_both(self):
        op = ResidualCorrectionOp(input_size=3, n_terms=2, residual_mode="block_parts")
        t, p0, p1 = self._parts()
        g0, g1 = _part_grads(op, t, p0, p1)
        assert g0 == pytest.approx(0.0) and g1 == pytest.approx(0.0)

    def test_keep_parts_keeps_both(self):
        op = ResidualCorrectionOp(input_size=3, n_terms=2, residual_mode="keep_parts")
        t, p0, p1 = self._parts()
        g0, g1 = _part_grads(op, t, p0, p1)
        assert g0 == pytest.approx(1.0) and g1 == pytest.approx(1.0)

    def test_keep_parts_with_stop_grad_on_one(self):
        op = ResidualCorrectionOp(input_size=3, n_terms=2, residual_mode="keep_parts",
                                  stop_grad_parts=(1,))
        t, p0, p1 = self._parts()
        g0, g1 = _part_grads(op, t, p0, p1)
        assert g0 == pytest.approx(1.0) and g1 == pytest.approx(0.0)

    def test_off_keeps_both(self):
        op = ResidualCorrectionOp(input_size=3, n_terms=2, residual_mode="off")
        t, p0, p1 = self._parts()
        g0, g1 = _part_grads(op, t, p0, p1)
        assert g0 == pytest.approx(1.0) and g1 == pytest.approx(1.0)

    def test_off_with_stop_grad_blocks_one(self):
        op = ResidualCorrectionOp(input_size=3, n_terms=2, residual_mode="off",
                                  stop_grad_parts=(1,))
        t, p0, p1 = self._parts()
        g0, g1 = _part_grads(op, t, p0, p1)
        assert g0 == pytest.approx(1.0) and g1 == pytest.approx(0.0)


# =============================================================================
# ResidualCorrectionOp — forward (concatenated input)
# =============================================================================

class TestResidualCorrectionForward:
    def test_forward_matches_compute(self):
        op = ResidualCorrectionOp(input_size=4, n_terms=2, residual_mode="block_parts")
        target = torch.randn(2, 4)
        p0, p1 = torch.randn(2, 4), torch.randn(2, 4)
        packed = torch.cat([target, p0, p1], dim=-1)   # (target, *parts)
        torch.testing.assert_close(op(packed), op.compute(target, p0, p1))

    def test_forward_wrong_last_dim_raises(self):
        op = ResidualCorrectionOp(input_size=4, n_terms=2)
        with pytest.raises(ValueError, match="got last dim"):
            op(torch.randn(2, 4 * 2))   # missing the target chunk

    def test_forward_n_terms_one_resnet_residual(self):
        # ResNet-style: y = part + (target - part) == target
        op = ResidualCorrectionOp(input_size=4, n_terms=1, residual_mode="block_parts")
        target = torch.randn(2, 4)
        part = torch.randn(2, 4, requires_grad=True)
        eps = op(torch.cat([target, part], dim=-1))
        torch.testing.assert_close(part + eps, target)
