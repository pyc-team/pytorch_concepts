"""Tests for TrilActivation and GlobalScale (torch_concepts.nn.modules.low.scales).

A scale head is a ``Sequential`` of a raw layer and the activation that makes
its output a valid distribution parameter. For a ``Normal``'s per-element
``scale`` that activation is stock ``nn.Softplus``; ``TrilActivation`` supplies
the one case torch has no equivalent for — a ``MultivariateNormal``'s
matrix-valued ``scale_tril``. ``GlobalScale`` is a raw *layer* for the same
slot: one learnable value, input-independent, for a homoscedastic likelihood.
"""
import pytest
import torch
import torch.nn.functional as F

from torch_concepts.nn.modules.low.scales import TrilActivation, GlobalScale
from torch_concepts.nn.modules.low.sequential import Sequential


class TestTrilActivation:
    def test_builds_a_lower_triangular_factor_with_positive_diagonal(self):
        head = Sequential(torch.nn.Linear(4, 6), TrilActivation(size=3))
        tril = head(torch.randn(8, 4))
        assert tril.shape == (8, 3, 3)
        assert bool((tril.diagonal(dim1=-2, dim2=-1) > 0).all())
        assert bool((tril.triu(diagonal=1) == 0).all())

    def test_factor_is_a_valid_covariance_root(self):
        head = Sequential(torch.nn.Linear(4, 6), TrilActivation(size=3))
        tril = head(torch.randn(8, 4))
        # L L^T is positive definite, so it has a Cholesky decomposition.
        torch.linalg.cholesky(tril @ tril.transpose(-1, -2))

    def test_parametrizes_a_multivariate_normal(self):
        head = Sequential(torch.nn.Linear(4, 6), TrilActivation(size=3))
        d = torch.distributions.MultivariateNormal(
            loc=torch.zeros(8, 3), scale_tril=head(torch.randn(8, 4))
        )
        sample = d.rsample()
        assert sample.shape == (8, 3)
        assert d.log_prob(sample).shape == (8,)

    def test_floor_keeps_the_diagonal_off_zero(self):
        linear = torch.nn.Linear(4, 6)
        with torch.no_grad():  # drive softplus to ~0 so only the floor is left
            linear.weight.zero_()
            linear.bias.fill_(-1e4)
        tril = Sequential(linear, TrilActivation(size=3, floor=1e-6))(torch.randn(8, 4))
        diag = tril.diagonal(dim1=-2, dim2=-1)
        assert bool((diag > 0).all())
        assert torch.allclose(diag, torch.full_like(diag, 1e-6))

    def test_rejects_a_head_of_the_wrong_width(self):
        head = Sequential(torch.nn.Linear(4, 3), TrilActivation(size=3))  # needs 6
        with pytest.raises(ValueError, match="needs 6"):
            head(torch.randn(8, 4))

    def test_gradients_reach_the_head(self):
        head = Sequential(torch.nn.Linear(4, 6), TrilActivation(size=3))
        head(torch.randn(8, 4)).sum().backward()
        assert head[0].weight.grad is not None


class TestSoftplusScaleHead:
    """The per-element case needs no library class: a Sequential of the raw head
    and stock nn.Softplus is the whole scale head."""

    def test_output_is_positive_and_shaped_like_the_head(self):
        head = Sequential(torch.nn.Linear(4, 3), torch.nn.Softplus())
        out = head(torch.randn(8, 4))
        assert out.shape == (8, 3)
        assert bool((out > 0).all())

    def test_forwards_keyword_arguments_to_the_head(self):
        """A multi-input head keeps its calling convention (Sequential passes
        every argument to its first module), so a CPD can drive it unchanged."""
        class PyCLayer(torch.nn.Module):
            def forward(self, concepts, embeddings):
                return concepts + embeddings

        head = Sequential(PyCLayer(), torch.nn.Softplus())
        out = head(concepts=torch.zeros(2, 3), embeddings=torch.zeros(2, 3))
        assert out.shape == (2, 3)


class TestGlobalScale:
    def test_exactly_one_trainable_parameter(self):
        scale = GlobalScale(size=784)
        assert sum(p.numel() for p in scale.parameters()) == 1

    def test_softplus_of_raw_matches_init(self):
        for init in (0.1, 1.0, 3.0):
            scale = GlobalScale(size=16, init=init)
            assert F.softplus(scale.raw).item() == pytest.approx(init, abs=1e-6)

    def test_output_shape_and_uniformity(self):
        scale = GlobalScale(size=784)
        out = scale(torch.randn(5, 4, 16))  # only shape[0] is read
        assert out.shape == (5, 784)
        assert bool((out == out[0, 0]).all())

    def test_ignores_the_input_content(self):
        scale = GlobalScale(size=8)
        a = scale(torch.zeros(3, 2))
        b = scale(torch.randn(3, 2) * 100)
        assert torch.equal(a, b)

    def test_gradient_reaches_the_single_parameter_through_expand(self):
        scale = GlobalScale(size=8)
        out = scale(torch.randn(3, 2))
        out.sum().backward()
        assert scale.raw.grad is not None
        assert scale.raw.grad.item() != 0.0

    def test_composes_as_a_raw_scale_head(self):
        """The class's whole purpose: drop-in as the raw layer under Softplus,
        exactly like a per-element head, but parameter-free w.r.t. the input."""
        head = Sequential(GlobalScale(size=4, init=0.5), torch.nn.Softplus())
        out = head(torch.randn(6, 3))
        assert out.shape == (6, 4)
        assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-6)
