"""Tests for UniformPolicy."""
import pytest
import torch

from torch_concepts.nn.modules.low.intervention.policy.uniform import UniformPolicy


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestUniformPolicyConstruction:
    def test_construction_no_args(self):
        policy = UniformPolicy()
        assert isinstance(policy, UniformPolicy)

    def test_is_nn_module(self):
        import torch.nn as nn
        assert isinstance(UniformPolicy(), nn.Module)


# ===========================================================================
# 2. Forward pass
# ===========================================================================

class TestUniformPolicyForward:
    def test_returns_zeros(self):
        policy = UniformPolicy()
        out = policy(torch.randn(4, 10))
        assert torch.all(out == 0.0)

    def test_output_shape(self):
        policy = UniformPolicy()
        out = policy(torch.randn(4, 10))
        assert out.shape == (4, 10)

    def test_uniform_values_all_equal_per_row(self):
        policy = UniformPolicy()
        out = policy(torch.randn(4, 10))
        for i in range(out.shape[0]):
            assert torch.allclose(out[i], out[i, 0].expand_as(out[i]))

    def test_different_inputs_same_output(self):
        policy = UniformPolicy()
        out1 = policy(torch.randn(2, 5))
        out2 = policy(torch.randn(2, 5))
        assert torch.allclose(out1, out2)

    def test_output_independent_of_input_values(self):
        policy = UniformPolicy()
        x_small = torch.full((3, 5), 0.01)
        x_large = torch.full((3, 5), 100.0)
        assert torch.allclose(policy(x_small), policy(x_large))

    def test_various_concept_sizes(self):
        policy = UniformPolicy()
        for F in [1, 5, 20, 100]:
            out = policy(torch.randn(4, F))
            assert out.shape == (4, F)
            assert torch.all(out == 0.0)
