"""Tests for ConvDecoder (torch_concepts.nn.modules.low.conv).

A PGM keeps an image variable on one flat event axis, so a decoder to it is a
head with as many outputs as pixels. ConvDecoder spends those parameters on
transposed convolutions instead of one dense layer, while keeping the flat-in /
flat-out contract a CPD parametrization needs.
"""
import pytest
import torch
import torch.nn as nn

from torch_concepts.nn import ConvDecoder


class TestShapes:
    @pytest.mark.parametrize(
        "out_shape, hidden_channels, base_size",
        [
            ((3, 28, 28), (64, 32), 7),      # ColorMNIST
            ((3, 64, 64), (128, 64, 32), 8), # CelebA at 64px
            ((3, 32, 32), (64, 32, 16), 4),
            ((1, 16, 16), (32, 16), 4),      # single channel
        ],
    )
    def test_base_size_is_derived_from_the_target(
        self, out_shape, hidden_channels, base_size
    ):
        decoder = ConvDecoder(48, out_shape, hidden_channels=hidden_channels)
        assert decoder.base_size == base_size
        out = decoder(torch.randn(4, 48))
        assert out.shape == (4, out_shape[0] * out_shape[1] * out_shape[2])
        assert decoder.out_features == out.shape[-1]

    def test_leading_dimensions_are_preserved(self):
        decoder = ConvDecoder(8, (3, 32, 32), hidden_channels=(32, 16, 8))
        assert decoder(torch.randn(2, 5, 8)).shape == (2, 5, 3072)

    def test_explicit_base_size_is_honoured(self):
        decoder = ConvDecoder(8, (3, 32, 32), hidden_channels=(16, 8), base_size=8)
        assert decoder(torch.randn(2, 8)).shape == (2, 3072)


class TestErrors:
    def test_a_depth_the_target_cannot_reach_is_rejected(self):
        with pytest.raises(ValueError, match="not reachable"):
            ConvDecoder(8, (3, 28, 28), hidden_channels=(64, 32, 16))

    def test_a_non_square_target_is_rejected(self):
        with pytest.raises(ValueError, match="not square"):
            ConvDecoder(8, (3, 218, 178))

    def test_a_non_image_shape_is_rejected(self):
        with pytest.raises(ValueError, match=r"\(channels, height, width\)"):
            ConvDecoder(8, (32, 32))

    def test_an_inconsistent_explicit_base_size_is_rejected(self):
        with pytest.raises(ValueError, match="upsampling stages reach"):
            ConvDecoder(8, (3, 32, 32), hidden_channels=(16, 8), base_size=4)


class TestBehaviour:
    def test_it_is_far_smaller_than_the_dense_equivalent(self):
        # The reason it exists: a flat head to 64x64x3 is an order of magnitude
        # more parameters than the whole conv stack.
        conv = ConvDecoder(96, (3, 64, 64), hidden_channels=(128, 64, 32))
        dense = nn.Linear(96, 3 * 64 * 64)
        assert sum(p.numel() for p in conv.parameters()) < sum(
            p.numel() for p in dense.parameters()
        )

    def test_the_output_is_raw_so_an_activation_can_be_composed(self):
        # A CPD applies no activation, so the decoder must NOT squash on its own
        # — DefaultActivation is composed on top.
        decoder = ConvDecoder(8, (3, 16, 16), hidden_channels=(16, 8))
        out = decoder(torch.randn(64, 8) * 5)
        assert out.min() < 0, "output should be unconstrained"

    def test_gradients_flow(self):
        decoder = ConvDecoder(8, (3, 16, 16), hidden_channels=(16, 8))
        decoder(torch.randn(4, 8)).sum().backward()
        assert decoder.project.weight.grad is not None
