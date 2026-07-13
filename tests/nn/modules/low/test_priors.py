"""Tests for prior modules (torch_concepts.nn.modules.low.priors).

Covers that ``LearnablePrior`` accepts both an int length and a tuple shape.
"""
import torch

from torch_concepts.nn.modules.low.priors import LearnablePrior


class TestLearnablePrior:
    def test_int_size_gives_1d_vector(self):
        prior = LearnablePrior(3)
        out = prior()
        assert out.shape == (3,)
        assert prior.param.requires_grad

    def test_tuple_size_gives_shaped_parameter(self):
        prior = LearnablePrior((2, 4))
        out = prior()
        assert out.shape == (2, 4)
        assert prior.param.requires_grad
