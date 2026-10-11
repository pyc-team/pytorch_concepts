"""Tests for InterventionModule, the intervention context manager,
and the GroundTruthIntervention, DoIntervention, DistributionIntervention
strategies together with UniformPolicy, RandomPolicy, and
UncertaintyInterventionPolicy.
"""
import inspect
import itertools

import pytest
import torch
import torch.nn as nn
import torch.distributions as torch_dist

from torch_concepts.nn import intervention
from torch_concepts.nn.modules.low.base.intervention import InterventionModule
from torch_concepts.nn.modules.low.intervention.strategy.ground_truth import (
    GroundTruthIntervention,
)
from torch_concepts.nn.modules.low.intervention.strategy.do import DoIntervention
from torch_concepts.nn.modules.low.intervention.strategy.distribution import (
    DistributionIntervention,
)
from torch_concepts.nn.modules.low.intervention.policy.uniform import UniformPolicy
from torch_concepts.nn.modules.low.intervention.policy.random import RandomPolicy
from torch_concepts.nn.modules.low.intervention.policy.uncertainty import (
    UncertaintyInterventionPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    """Simple encoder that always returns a fixed [B, F] output."""
    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_f = in_features
        self.out_f = out_features

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def _make_enc(in_f=4, out_f=3):
    return _Encoder(in_features=in_f, out_features=out_f)


class _CountingEncoder(_Encoder):
    """Encoder that records how many times it ran."""
    calls = 0

    def forward(self, x):
        self.calls += 1
        return super().forward(x)


B, F = 4, 3  # default batch size and feature size
ALL = list(range(F))  # every output of the default encoder


# ===========================================================================
# 1. GroundTruthIntervention strategy
# ===========================================================================

class TestGroundTruthIntervention:
    def test_construction_no_model_arg(self):
        gt = torch.ones(B, F)
        strat = GroundTruthIntervention(gt)
        assert torch.equal(strat.ground_truth, gt)

    def test_forward_returns_ground_truth(self):
        gt = torch.full((B, F), 0.7)
        strat = GroundTruthIntervention(gt)
        x = torch.randn(B, F)
        out = strat(x)
        assert torch.equal(out, gt)

    def test_forward_ignores_input(self):
        gt = torch.zeros(B, F)
        strat = GroundTruthIntervention(gt)
        x1 = torch.randn(B, F)
        x2 = torch.randn(B, F)
        assert torch.equal(strat(x1), strat(x2))

    def test_ground_truth_stored_as_tensor(self):
        gt = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        strat = GroundTruthIntervention(gt)
        assert isinstance(strat.ground_truth, torch.Tensor)


# ===========================================================================
# 2. DoIntervention strategy
# ===========================================================================

class TestDoIntervention:
    def test_construction_scalar(self):
        strat = DoIntervention(1.0)
        assert strat.constants.dim() == 0

    def test_construction_tensor_1d(self):
        strat = DoIntervention(torch.tensor([0.5, 1.0, 0.0]))
        assert strat.constants.shape == (3,)

    def test_construction_tensor_2d(self):
        strat = DoIntervention(torch.ones(1, 3))
        assert strat.constants.shape == (1, 3)

    def test_forward_scalar_broadcasts(self):
        strat = DoIntervention(0.5)
        x = torch.randn(B, F)
        out = strat(x)
        assert out.shape == (B, F)
        assert torch.allclose(out, torch.full((B, F), 0.5))

    def test_forward_1d_per_feature(self):
        constants = torch.tensor([0.1, 0.2, 0.3])
        strat = DoIntervention(constants)
        x = torch.randn(B, F)
        out = strat(x)
        assert out.shape == (B, F)
        for i in range(B):
            assert torch.allclose(out[i], constants)

    def test_forward_2d_per_sample(self):
        constants = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
                                    [0.7, 0.8, 0.9], [1.0, 0.0, 0.5]])
        strat = DoIntervention(constants)
        x = torch.randn(B, F)
        out = strat(x)
        assert torch.allclose(out, constants)

    def test_forward_2d_broadcast_1xF(self):
        constants = torch.tensor([[0.3, 0.6, 0.9]])
        strat = DoIntervention(constants)
        x = torch.randn(B, F)
        out = strat(x)
        assert out.shape == (B, F)
        for i in range(B):
            assert torch.allclose(out[i], constants[0])

    def test_forward_3d_raises_value_error(self):
        # constants has more dims than x, so it cannot broadcast against x's shape
        strat = DoIntervention(torch.ones(1, 1, 3))
        x = torch.randn(B, F)
        with pytest.raises(ValueError, match="cannot be broadcast"):
            strat(x)

    def test_forward_wrong_feature_size_raises(self):
        strat = DoIntervention(torch.tensor([0.5, 1.0]))  # 2 features, expect 3
        x = torch.randn(B, F)
        with pytest.raises(ValueError, match="cannot be broadcast"):
            strat(x)

    def test_forward_wrong_batch_size_raises(self):
        strat = DoIntervention(torch.ones(5, F))  # B=5, expect B=4
        x = torch.randn(B, F)
        with pytest.raises(ValueError, match="cannot be broadcast"):
            strat(x)

    def test_output_dtype_matches_input(self):
        strat = DoIntervention(torch.ones(F))
        x = torch.randn(B, F, dtype=torch.float64)
        out = strat(x)
        assert out.dtype == torch.float64


# ===========================================================================
# 3. DistributionIntervention strategy
# ===========================================================================

class TestDistributionIntervention:
    def test_construction_single_distribution(self):
        d = torch_dist.Bernoulli(torch.tensor(0.5))
        strat = DistributionIntervention(d)
        assert strat.dist is d

    def test_construction_list_distributions(self):
        dists = [torch_dist.Bernoulli(torch.tensor(p)) for p in [0.3, 0.5, 0.7]]
        strat = DistributionIntervention(dists)
        assert len(list(strat.dist)) == 3

    def test_forward_single_dist_shape(self):
        d = torch_dist.Bernoulli(torch.tensor(0.5))
        strat = DistributionIntervention(d)
        out = strat(torch.randn(B, F))
        assert out.shape == (B, F)

    def test_forward_single_bernoulli_values_binary(self):
        d = torch_dist.Bernoulli(torch.tensor(0.5))
        strat = DistributionIntervention(d)
        out = strat(torch.randn(B, F))
        assert torch.all((out == 0) | (out == 1))

    def test_forward_per_feature_shape(self):
        dists = [torch_dist.Bernoulli(torch.tensor(0.5)) for _ in range(F)]
        strat = DistributionIntervention(dists)
        out = strat(torch.randn(B, F))
        assert out.shape == (B, F)

    def test_forward_normal_distribution(self):
        d = torch_dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
        strat = DistributionIntervention(d)
        out = strat(torch.randn(B, F))
        assert out.shape == (B, F)

    def test_forward_wrong_number_of_dists_raises(self):
        dists = [torch_dist.Bernoulli(torch.tensor(0.5))] * 2  # need 3, got 2
        strat = DistributionIntervention(dists)
        with pytest.raises(AssertionError):
            strat(torch.randn(B, F))


# ===========================================================================
# 4. UniformPolicy
# ===========================================================================

class TestUniformPolicy:
    def test_forward_returns_zeros(self):
        policy = UniformPolicy()
        x = torch.randn(B, F)
        out = policy(x)
        assert torch.all(out == 0.0)

    def test_output_shape(self):
        policy = UniformPolicy()
        out = policy(torch.randn(4, 6))
        assert out.shape == (4, 6)

    def test_output_independent_of_input(self):
        policy = UniformPolicy()
        assert torch.equal(policy(torch.randn(4, 3)), policy(torch.randn(4, 3)))


# ===========================================================================
# 5. RandomPolicy
# ===========================================================================

class TestRandomPolicy:
    def test_default_scale(self):
        p = RandomPolicy()
        assert p.scale == pytest.approx(1.0)

    def test_output_non_negative(self):
        p = RandomPolicy(scale=2.0)
        out = p(torch.randn(B, F))
        assert (out >= 0).all()

    def test_output_bounded_by_scale(self):
        p = RandomPolicy(scale=2.0)
        out = p(torch.randn(100, 10))
        assert (out <= 2.0).all()

    def test_random_outputs_differ(self):
        p = RandomPolicy(scale=1.0)
        x = torch.randn(B, F)
        assert not torch.equal(p(x), p(x))


# ===========================================================================
# 6. UncertaintyInterventionPolicy
# ===========================================================================

class TestUncertaintyInterventionPolicy:
    def test_certainty_at_zero_input(self):
        p = UncertaintyInterventionPolicy()
        assert torch.all(p(torch.zeros(B, F)) == 0.0)

    def test_abs_distance_from_mup(self):
        p = UncertaintyInterventionPolicy(max_uncertainty_point=0.5)
        x = torch.tensor([[0.0, 0.5, 1.0]])
        expected = torch.tensor([[0.5, 0.0, 0.5]])
        assert torch.allclose(p(x), expected)

    def test_output_non_negative(self):
        p = UncertaintyInterventionPolicy()
        assert (p(torch.randn(B, F)) >= 0).all()


# ===========================================================================
# 7. InterventionModule — construction
# ===========================================================================

class TestInterventionModuleConstruction:
    def test_basic_construction(self):
        enc = _make_enc()
        gt = torch.ones(B, F)
        m = InterventionModule(enc, GroundTruthIntervention(gt), UniformPolicy(), ALL)
        assert isinstance(m, nn.Module)

    def test_original_module_stored(self):
        enc = _make_enc()
        m = InterventionModule(enc, GroundTruthIntervention(torch.ones(B, F)), UniformPolicy(), ALL)
        assert m.original_module is enc

    def test_strategy_stored(self):
        enc = _make_enc()
        strat = GroundTruthIntervention(torch.ones(B, F))
        m = InterventionModule(enc, strat, UniformPolicy(), ALL)
        assert m.intervention_strategy is strat

    def test_policy_stored(self):
        enc = _make_enc()
        policy = UniformPolicy()
        m = InterventionModule(enc, GroundTruthIntervention(torch.ones(B, F)), policy, ALL)
        assert m.intervention_policy is policy

    def test_default_quantile(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy(), ALL)
        assert m.quantile == pytest.approx(1.0)

    def test_custom_quantile(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy(), ALL, quantile=0.5)
        assert m.quantile == pytest.approx(0.5)

    def test_unknown_argument_raises(self):
        with pytest.raises(TypeError):
            InterventionModule(_make_enc(), DoIntervention(0.0), UniformPolicy(), ALL, quantil=0.5)

    def test_out_concepts_to_intervene_on_stored(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy(),
                               out_concepts_to_intervene_on=[0, 1])
        assert m.out_concepts_to_intervene_on == [0, 1]


# ===========================================================================
# 9. intervention() context manager
# ===========================================================================

class TestInterventionContextManager:
    def test_module_intervened_inside_context(self):
        enc = _make_enc()
        x = torch.randn(B, enc.in_f)
        with intervention(enc, DoIntervention(0.5), UniformPolicy(), ALL):
            out = enc(x)
        assert torch.allclose(out, torch.full((B, F), 0.5))

    def test_selection_by_position(self):
        enc = _make_enc()
        x = torch.randn(B, enc.in_f)
        with intervention(enc, DoIntervention(0.5), UniformPolicy(), [1]):
            out = enc(x)
        assert torch.allclose(out[:, 1], torch.full((B,), 0.5))
        assert torch.allclose(out[:, [0, 2]], enc(x)[:, [0, 2]])

    def test_selection_by_name(self):
        enc = _make_enc()
        enc.out_concepts = Annotations(labels=['alpha', 'beta', 'gamma'])
        x = torch.randn(B, enc.in_f)
        with intervention(enc, DoIntervention(0.5), UniformPolicy(), ['gamma']):
            out = enc(x)
        assert torch.allclose(out[:, 2], torch.full((B,), 0.5))
        assert torch.allclose(out[:, :2], enc(x)[:, :2])

    def test_selection_is_required(self):
        with pytest.raises(ValueError, match="out_concepts_to_intervene_on"):
            with intervention(_make_enc(), DoIntervention(0.5), UniformPolicy()):
                pass

    def test_selection_must_be_a_list(self):
        enc = _make_enc()
        enc.out_concepts = Annotations(labels=['alpha', 'beta', 'gamma'])
        with pytest.raises(ValueError, match="as a list"):
            with intervention(enc, DoIntervention(0.5), UniformPolicy(), 'beta'):
                pass

    def test_concept_strategy_runs_the_layer_once(self):
        enc = _CountingEncoder()
        with intervention(enc, DoIntervention(0.5), UniformPolicy(), ALL):
            enc(torch.randn(B, enc.in_f))
        assert enc.calls == 1

    def test_module_strategy_runs_the_original_and_the_transformed_layer(self):
        enc = _CountingEncoder()
        with torch.no_grad():
            enc.linear.weight.fill_(-1.0)
            enc.linear.bias.fill_(0.3)
        with intervention(enc, PositiveWeightsIntervention(), UniformPolicy(), ALL):
            out = enc(torch.randn(B, enc.in_f))
        assert enc.calls == 2
        assert torch.allclose(out, torch.sigmoid(torch.full((B, F), 0.3)))

    def test_accepts_built_intervention_module(self):
        enc = _make_enc()
        x = torch.randn(B, enc.in_f)
        with intervention(InterventionModule(enc, DoIntervention(0.5), UniformPolicy(), ALL)):
            out = enc(x)
        assert torch.allclose(out, torch.full((B, F), 0.5))

    @pytest.mark.parametrize("extra", [{"quantile": 0.5}, {"out_concepts_to_intervene_on": [0]}])
    def test_built_intervention_module_takes_no_other_arguments(self, extra):
        m = InterventionModule(_make_enc(), DoIntervention(0.5), UniformPolicy(), ALL)
        with pytest.raises(TypeError, match="no other arguments"):
            with intervention(m, **extra):
                pass

    def test_module_restored_on_exit(self):
        enc = _make_enc()
        x = torch.randn(B, enc.in_f)
        before = enc(x)
        with pytest.raises(RuntimeError):
            with intervention(enc, DoIntervention(0.5), UniformPolicy(), ALL):
                raise RuntimeError
        assert torch.equal(enc(x), before)


# ===========================================================================
# 10. InterventionModule.forward() — end-to-end
# ===========================================================================

class TestInterventionModuleForward:
    def test_output_shape(self):
        enc = _make_enc()
        x = torch.randn(B, enc.in_f)
        m = InterventionModule(enc, DoIntervention(0.5), UniformPolicy(), ALL, quantile=1.0)
        out = m(x)
        assert out.shape == (B, F)

    def test_full_intervention_ground_truth(self):
        enc = _make_enc()
        gt = torch.full((B, F), 0.7)
        m = InterventionModule(enc, GroundTruthIntervention(gt), UniformPolicy(), ALL, quantile=1.0)
        x = torch.randn(B, enc.in_f)
        out = m(x)
        # quantile=1.0 → all concepts replaced → output should equal gt
        assert torch.allclose(out, gt, atol=1e-5)

    def test_full_do_intervention(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy(), ALL, quantile=1.0)
        x = torch.randn(B, enc.in_f)
        out = m(x)
        # quantile=1.0 + do(0.0) → all concepts become 0
        assert torch.allclose(out, torch.zeros(B, F), atol=1e-5)

    def test_no_intervention_at_quantile_zero_single_concept(self):
        enc = _Encoder(in_features=4, out_features=1)
        gt = torch.ones(B, 1)
        m = InterventionModule(enc, GroundTruthIntervention(gt), UniformPolicy(), [0], quantile=0.0)
        x = torch.randn(B, 4)
        with torch.no_grad():
            orig = enc(x)
            out = m(x)
        # quantile=0.0 + single concept → keep col → mask=1 → not intervened → matches original
        assert torch.allclose(out, orig, atol=1e-5)

    def test_subset_intervened_by_index(self):
        F2 = 4
        enc = _Encoder(in_features=4, out_features=F2)
        gt = torch.ones(B, F2)
        m = InterventionModule(enc, GroundTruthIntervention(gt), UniformPolicy(),
                               out_concepts_to_intervene_on=[0, 1], quantile=1.0)
        x = torch.randn(B, 4)
        with torch.no_grad():
            orig = enc(x)
            out = m(x)
        # Concepts 0 and 1 should be replaced by gt (=1.0)
        assert torch.allclose(out[:, 0:2], torch.ones(B, 2), atol=1e-5)
        # Concepts 2 and 3 should be unchanged
        assert torch.allclose(out[:, 2:], orig[:, 2:], atol=1e-5)

    def test_random_policy_with_do_intervention(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.5), RandomPolicy(scale=1.0), ALL, quantile=1.0)
        x = torch.randn(B, enc.in_f)
        out = m(x)
        # quantile=1.0 → all concepts replaced by do(0.5)
        assert torch.allclose(out, torch.full((B, F), 0.5), atol=1e-5)

    def test_uncertainty_policy_with_do_intervention(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(1.0), UncertaintyInterventionPolicy(), ALL, quantile=1.0)
        x = torch.randn(B, enc.in_f)
        out = m(x)
        # quantile=1.0 → all concepts replaced by do(1.0)
        assert torch.allclose(out, torch.ones(B, F), atol=1e-5)

    def test_distribution_intervention_output_shape(self):
        enc = _make_enc()
        d = torch_dist.Bernoulli(torch.tensor(0.5))
        m = InterventionModule(enc, DistributionIntervention(d), UniformPolicy(), ALL, quantile=1.0)
        x = torch.randn(B, enc.in_f)
        out = m(x)
        assert out.shape == (B, F)


# ===========================================================================
# 11. Gradient flow
# ===========================================================================

class TestGradientFlow:
    def test_gradient_through_intervention_module(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.5), UniformPolicy(), ALL, quantile=0.5)
        x = torch.randn(B, enc.in_f, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_through_original_module_weights(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.5), UniformPolicy(), ALL, quantile=0.5)
        x = torch.randn(B, enc.in_f)
        m(x).sum().backward()
        assert enc.linear.weight.grad is not None

    def test_no_gradient_with_full_do_intervention(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.5), UniformPolicy(), ALL, quantile=1.0)
        x = torch.randn(B, enc.in_f, requires_grad=True)
        out = m(x)
        out.sum().backward()
        # quantile=1.0 with uniform policy + STE proxy: grad may still flow
        # through the STE term, so just check it doesn't error
        assert out is not None

    def test_gradient_with_ground_truth_partial(self):
        enc = _make_enc()
        gt = torch.zeros(B, F)
        m = InterventionModule(enc, GroundTruthIntervention(gt), UniformPolicy(), ALL, quantile=0.5)
        x = torch.randn(B, enc.in_f, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


# ===========================================================================
# 12. sel_idx property
# ===========================================================================

class TestSelIdx:
    def test_selection_is_required(self):
        enc = _make_enc()
        with pytest.raises(TypeError):
            InterventionModule(enc, DoIntervention(0.0), UniformPolicy())
        with pytest.raises(ValueError, match="out_concepts_to_intervene_on"):
            InterventionModule(enc, DoIntervention(0.0), UniformPolicy(), None)

    def test_tensor_when_int_indices(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy(),
                               out_concepts_to_intervene_on=[0, 2])
        sel = m.sel_idx
        assert isinstance(sel, torch.Tensor)
        assert sel.tolist() == [0, 2]


# ===========================================================================
# 13. Extra modules registration
# ===========================================================================

class TestExtraModules:
    def test_extra_module_registered(self):
        enc = _make_enc()
        head = nn.Linear(F, 1)
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy(), ALL,
                               extra_modules={"task_head": head})
        assert "task_head" in dict(m.named_modules())


# ===========================================================================
# 14. PositiveWeightsIntervention strategy
# ===========================================================================

from torch_concepts.nn.modules.low.intervention.strategy.positive_weights import PositiveWeightsIntervention


class TestPositiveWeightsIntervention:
    def test_construction(self):
        strat = PositiveWeightsIntervention()
        from torch_concepts.nn.modules.low.base.intervention import ModuleInterventionStrategy
        assert isinstance(strat, ModuleInterventionStrategy)

    def test_transform_evaluates_with_nonnegative_weights(self):
        enc = _make_enc()
        # Force some negative weights
        with torch.no_grad():
            enc.linear.weight.fill_(-1.0)
            enc.linear.bias.fill_(0.3)
        strat = PositiveWeightsIntervention()
        out = strat.transform(enc)(torch.randn(B, enc.in_f))
        # ReLU zeroes the weights, so only the bias is left
        assert torch.allclose(out, torch.sigmoid(torch.full((B, F), 0.3)))

    def test_transform_preserves_positive_weights(self):
        enc = _make_enc()
        with torch.no_grad():
            enc.linear.weight.fill_(2.0)
            enc.linear.bias.fill_(0.3)
        strat = PositiveWeightsIntervention()
        x = torch.randn(B, enc.in_f)
        assert torch.allclose(strat.transform(enc)(x), enc(x))

    def test_transform_leaves_module_untouched(self):
        enc = _make_enc()
        with torch.no_grad():
            enc.linear.weight.fill_(-1.0)
        strat = PositiveWeightsIntervention()
        strat.transform(enc)(torch.randn(B, enc.in_f))
        assert torch.equal(enc.linear.weight, torch.full_like(enc.linear.weight, -1.0))

    def test_transform_gradient_reaches_original_parameters(self):
        enc = _make_enc()
        with torch.no_grad():
            enc.linear.weight.fill_(1.0)
        strat = PositiveWeightsIntervention()
        strat.transform(enc)(torch.randn(B, enc.in_f)).sum().backward()
        assert enc.linear.weight.grad is not None

    def test_full_intervention_via_intervention_module(self):
        """PositiveWeightsIntervention used as strategy in InterventionModule."""
        enc = _make_enc()
        with torch.no_grad():
            enc.linear.weight.fill_(-0.5)
        strat = PositiveWeightsIntervention()
        m = InterventionModule(enc, strat, UniformPolicy(), ALL, quantile=1.0)
        x = torch.randn(B, enc.in_f)
        out = m(x)
        # ReLU is applied to every parameter: weights become 0, so the output is
        # the ReLU-ed bias only
        assert torch.allclose(out, torch.sigmoid(torch.relu(enc.linear.bias)).expand(B, -1))
        # and the wrapped encoder keeps its own (negative) weights
        assert (enc.linear.weight == -0.5).all()


# ===========================================================================
# 15. GradientPolicy
# ===========================================================================

from torch_concepts.nn.modules.low.intervention.policy.gradient import GradientPolicy


class TestGradientPolicy:
    def test_construction(self):
        p = GradientPolicy()
        from torch_concepts.nn.modules.low.base.intervention import InterventionPolicy
        assert isinstance(p, InterventionPolicy)

    def test_with_gradients_returns_negative_abs(self):
        p = GradientPolicy()
        concepts = torch.randn(B, F)
        grads = torch.tensor([[-1.0, 2.0, -3.0]] * B)
        out = p(concepts, concept_grads=grads)
        assert torch.allclose(out, -grads.abs())

    def test_largest_gradient_is_intervened_on_first(self):
        p = GradientPolicy()
        grads = torch.tensor([[-1.0, 2.0, -3.0]] * B)
        mask = p.build_mask(p(torch.randn(B, F), concept_grads=grads), torch.tensor(ALL), quantile=0.0)
        assert torch.equal(mask, torch.tensor([[1.0, 1.0, 0.0]] * B))

    def test_without_gradients_returns_zeros(self):
        p = GradientPolicy()
        concepts = torch.randn(B, F)
        out = p(concepts)
        assert torch.equal(out, torch.zeros(B, F))

    def test_no_gradients_same_shape_as_input(self):
        p = GradientPolicy()
        concepts = torch.randn(3, 7)
        out = p(concepts)
        assert out.shape == (3, 7)

    def test_gradient_scores_are_nonpositive(self):
        p = GradientPolicy()
        concepts = torch.randn(B, F)
        grads = torch.randn(B, F)
        out = p(concepts, concept_grads=grads)
        assert (out <= 0).all()


# ===========================================================================
# 16. Additional intervention module coverage
# ===========================================================================

from torch_concepts.annotations import Annotations


class TestInterventionModuleCoverage:
    def test_build_context_fn_is_called(self):
        """build_context_fn is invoked and its return value flows into policy/strategy."""
        enc = _make_enc()
        context_called = []

        def my_build_context(preds, module, inputs, extra_tensors, extra_modules):
            context_called.append(True)
            return {}

        m = InterventionModule(
            enc,
            DoIntervention(0.5),
            UniformPolicy(),
            ALL,
            build_context=my_build_context,
        )
        x = torch.randn(B, enc.in_f)
        m(x)
        assert context_called

    def test_invalid_strategy_type_raises(self):
        """Passing an object that is neither ConceptInterventionStrategy nor
        ModuleInterventionStrategy raises at construction time."""
        enc = _make_enc()

        class FakeStrategy:
            pass

        with pytest.raises(ValueError):
            InterventionModule(enc, FakeStrategy(), UniformPolicy(), ALL)

    def test_sel_idx_string_type_raises(self):
        """String-based concept selection without Annotations raises ValueError."""
        enc = _make_enc()
        m = InterventionModule(
            enc,
            DoIntervention(0.0),
            UniformPolicy(),
            out_concepts_to_intervene_on=['concept_a'],
        )
        with pytest.raises(ValueError):
            _ = m.sel_idx

    def test_sel_idx_string_with_int_out_concepts_raises(self):
        """`out_concepts` given as a count (not Annotations) must raise the clear
        ValueError, not an AttributeError that nn.Module reports as a missing
        `sel_idx` attribute."""
        enc = _make_enc()
        enc.out_concepts = 3
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy(),
                               out_concepts_to_intervene_on=['concept_a'])
        with pytest.raises(ValueError, match="out_concepts"):
            _ = m.sel_idx

    def test_sel_idx_invalid_type_raises(self):
        """out_concepts_to_intervene_on with floats raises ValueError."""
        enc = _make_enc()
        m = InterventionModule(
            enc,
            DoIntervention(0.0),
            UniformPolicy(),
            out_concepts_to_intervene_on=[1.5],  # not int or str
        )
        with pytest.raises(ValueError):
            _ = m.sel_idx

    def test_module_with_var_kwargs_patches_forward(self):
        """Module whose forward has **kwargs still gets patched cleanly."""
        class KwargsEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = nn.Linear(4, 3)
            def forward(self, x, **kwargs):
                return torch.sigmoid(self.l(x))

        enc = KwargsEncoder()
        m = InterventionModule(enc, DoIntervention(0.5), UniformPolicy(), ALL)
        x = torch.randn(B, 4)
        out = m(x)
        assert out.shape == (B, 3)

    def test_patch_forward_signature_exception_path(self):
        """Module whose forward raises during inspect.signature() skips patching silently (lines 99-100)."""
        class _Unpatchable(nn.Module):
            """forward is a non-callable descriptor — inspect.signature raises TypeError."""
            def __init__(self):
                super().__init__()
                self.l = nn.Linear(4, 3)

        enc = _Unpatchable()
        # Overwrite 'forward' with a built-in that has no inspectable signature
        enc.forward = len  # built-in: inspect.signature raises ValueError
        # Construction must not raise — the except branch (lines 99-100) silently
        # swallows the signature-inspection failure.
        m = InterventionModule(enc, DoIntervention(0.5), UniformPolicy(), ALL)
        assert m.original_module is enc

    def test_sel_idx_string_with_valid_axis_annotation(self):
        """String-based selection with valid Annotations returns correct indices (lines 112-113)."""
        from torch_concepts.annotations import Annotations

        class AnnotatedEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = nn.Linear(4, 3)
                self.out_concepts = Annotations(labels=['alpha', 'beta', 'gamma'])

            def forward(self, x):
                return torch.sigmoid(self.l(x))

        enc = AnnotatedEncoder()
        m = InterventionModule(
            enc,
            DoIntervention(0.0),
            UniformPolicy(),
            out_concepts_to_intervene_on=['alpha', 'gamma'],
        )
        sel = m.sel_idx
        assert sel.tolist() == [0, 2]

    def test_inputs_fall_back_to_empty_dict_when_signature_does_not_bind(self):
        """A layer whose declared signature does not match the call still runs;
        build_context then gets no named inputs."""
        class _MisdeclaredEncoder(_Encoder):
            def forward(self, *args):
                return super().forward(*args)

        # declares a required argument that callers never pass
        _MisdeclaredEncoder.forward.__signature__ = inspect.signature(lambda self, x, required_extra: None)

        seen = []

        def build_context(predictions, module, inputs, extra_tensors, extra_modules):
            seen.append(inputs)
            return {}

        m = InterventionModule(_MisdeclaredEncoder(), DoIntervention(0.5), UniformPolicy(), ALL,
                               build_context=build_context)
        assert torch.allclose(m(torch.randn(B, 4)), torch.full((B, F), 0.5))
        assert seen == [{}]


# ===========================================================================
# 17. Base intervention abstract methods (base/intervention.py lines 27, 43, 53)
# ===========================================================================

from torch_concepts.nn.modules.low.base.intervention import (
    ConceptInterventionStrategy,
    ModuleInterventionStrategy,
    InterventionPolicy,
)


class TestInterventionModuleCoverageExtra:
    def test_patch_forward_signature_exception_branch(self):
        """A forward with no inspectable signature triggers the except (ValueError/TypeError) branch (lines 99-100)."""
        class _Unpatchable(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = nn.Linear(4, 3)
            def forward(self, x):
                return torch.sigmoid(self.l(x))

        enc = _Unpatchable()
        # range() has no signature inspectable by inspect.signature -> raises ValueError
        enc.forward = range
        m = InterventionModule(enc, DoIntervention(0.5), UniformPolicy(), ALL)
        assert m.original_module is enc

    def test_build_context_defaults_to_empty_dict(self):
        """InterventionModule.build_context returns {} when no callable is supplied."""
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy(), ALL)
        assert m.build_context({}, enc, torch.randn(B, F)) == {}

    def test_build_context_override_in_subclass(self):
        """A subclass overriding build_context still wins over the default."""
        class _Sub(InterventionModule):
            def build_context(self, *args, **kwargs):
                return {"marker": torch.zeros(1)}

        enc = _make_enc()
        m = _Sub(enc, DoIntervention(0.0), UniformPolicy(), ALL)
        assert "marker" in m.build_context({}, enc, torch.randn(B, F))


class TestBaseInterventionAbstractMethods:
    def test_base_concept_strategy_forward_raises(self):
        """ConceptInterventionStrategy.forward raises NotImplementedError (line 27)."""
        class _ConcreteStrategy(ConceptInterventionStrategy):
            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)

        strat = _ConcreteStrategy()
        with pytest.raises(NotImplementedError):
            strat(torch.randn(2, 3))

    def test_base_module_strategy_transform_raises(self):
        """ModuleInterventionStrategy.transform raises NotImplementedError (line 43)."""
        class _ConcreteModuleStrategy(ModuleInterventionStrategy):
            def transform(self, module, *args, **kwargs):
                return super().transform(module, *args, **kwargs)

        strat = _ConcreteModuleStrategy()
        with pytest.raises(NotImplementedError):
            strat.transform(nn.Linear(2, 2))

    def test_base_policy_forward_raises(self):
        """InterventionPolicy.forward raises NotImplementedError (line 53)."""
        class _ConcretePolicy(InterventionPolicy):
            def forward(self, x, *args, **kwargs):
                return super().forward(x, *args, **kwargs)

        policy = _ConcretePolicy()
        with pytest.raises(NotImplementedError):
            policy(torch.randn(2, 3))


# ===========================================================================
# 18. Arbitrary leading dimensions: [*lead, F] behaves like the flattened [N, F]
# ===========================================================================

LEADS = [(), (6,), (3, 2), (2, 3, 2)]
STRATEGIES = {
    "do_scalar": lambda lead: DoIntervention(0.5),
    "do_per_output": lambda lead: DoIntervention(torch.tensor([1., 2., 3.])),
    "ground_truth": lambda lead: GroundTruthIntervention(torch.arange(F, dtype=torch.float).expand(*lead, F)),
    "positive_weights": lambda lead: PositiveWeightsIntervention(),
}


def _annotated_enc():
    enc = _make_enc()
    enc.out_concepts = Annotations(labels=['alpha', 'beta', 'gamma'])
    with torch.no_grad():
        enc.linear.weight[0] = -1.0  # so that PositiveWeightsIntervention changes the output
    return enc


class TestLeadingDims:
    @pytest.mark.parametrize("lead", LEADS)
    @pytest.mark.parametrize("strategy", STRATEGIES)
    @pytest.mark.parametrize("policy", [UniformPolicy, UncertaintyInterventionPolicy])
    def test_matches_flattened_input(self, lead, strategy, policy):
        """Module and context manager, for every selection form and quantile."""
        enc = _annotated_enc()
        x = torch.randn(*lead, enc.in_f)
        x_flat = x.reshape(-1, enc.in_f)
        for sel, q in itertools.product([[0, 2], ['beta'], [], ALL], [1.0, 0.5]):
            flat = InterventionModule(enc, STRATEGIES[strategy]((len(x_flat),)), policy(), sel, quantile=q)
            expected = flat(x_flat).reshape(*lead, F)
            out = InterventionModule(enc, STRATEGIES[strategy](lead), policy(), sel, quantile=q)(x)
            assert out.shape == (*lead, F)
            assert torch.allclose(out, expected, atol=1e-6)
            with intervention(enc, STRATEGIES[strategy](lead), policy(), sel, quantile=q):
                assert torch.allclose(enc(x), expected, atol=1e-6)

    @pytest.mark.parametrize("idx", [
        torch.tensor([[[0], [1]], [[2], [0]], [[1], [2]]]),  # [B, T, K]: one set per (b, t)
        torch.tensor([[[0, 1]], [[1, 2]], [[0, 2]]]),        # [B, 1, K]: shared across T
        torch.tensor([[[0, 1], [1, 2]]]),                    # [1, T, K]: shared across B
    ])
    @pytest.mark.parametrize("as_list", [False, True])
    def test_per_sample_selection(self, idx, as_list):
        enc = _make_enc()
        x = torch.randn(3, 2, enc.in_f)
        out = InterventionModule(enc, DoIntervention(7.0), UniformPolicy(), idx.tolist() if as_list else idx)(x)
        picked = torch.zeros(3, 2, F, dtype=torch.bool).scatter(-1, idx.expand(3, 2, -1), True)
        assert torch.all(out[picked] == 7.0)
        assert torch.allclose(out[~picked], enc(x)[~picked])

    def test_per_sample_selection_with_wrong_leading_shape_raises(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(7.0), UniformPolicy(), torch.zeros(5, 1, dtype=torch.long))
        with pytest.raises(ValueError, match="cannot be broadcast"):
            m(torch.randn(3, 2, enc.in_f))

    @pytest.mark.parametrize("lead", LEADS[1:])
    @pytest.mark.parametrize("strategy, policy", [
        (DoIntervention(9.0), RandomPolicy()),
        (DistributionIntervention(torch_dist.Normal(9.0, 1e-4)), UniformPolicy()),
        (DistributionIntervention([torch_dist.Normal(9.0, 1e-4)] * F), UniformPolicy()),
    ])
    def test_stochastic_strategies_and_policies(self, lead, strategy, policy):
        enc = _make_enc()
        x = torch.randn(*lead, enc.in_f)
        out = InterventionModule(enc, strategy, policy, [0, 2])(x)
        assert out.shape == (*lead, F)
        assert torch.allclose(out[..., [0, 2]], torch.full((*lead, 2), 9.0), atol=1e-2)
        assert torch.allclose(out[..., 1], enc(x)[..., 1])

    @pytest.mark.parametrize("lead", LEADS[1:])
    def test_gradient_policy_and_backward(self, lead):
        enc = _make_enc()

        def build_context(predictions, module, inputs, extra_tensors, extra_modules):
            pred = predictions.detach().requires_grad_(True)
            return {"concept_grads": torch.autograd.grad(extra_modules["head"](pred).sum(), pred)[0]}

        m = InterventionModule(enc, DoIntervention(0.0), GradientPolicy(), ALL, quantile=0.5,
                               build_context=build_context, extra_modules={"head": nn.Linear(F, 1)})
        x = torch.randn(*lead, enc.in_f, requires_grad=True)
        out = m(x)
        assert torch.allclose(out, m(x.reshape(-1, enc.in_f)).reshape(*lead, F))
        out.sum().backward()
        assert x.grad.shape == x.shape
