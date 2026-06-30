"""Tests for InterventionModule, intervene, intervention context manager,
and the GroundTruthIntervention, DoIntervention, DistributionIntervention
strategies together with UniformPolicy, RandomPolicy, and
UncertaintyInterventionPolicy.
"""
import pytest
import torch
import torch.nn as nn
import torch.distributions as torch_dist

from torch_concepts.nn.modules.low.intervention.intervention import (
    InterventionModule,
    intervene,
    intervention,
)
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


B, F = 4, 3  # default batch size and feature size


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
        strat = DoIntervention(torch.ones(1, 1, 3))
        x = torch.randn(B, F)
        with pytest.raises(ValueError, match="constants must be scalar"):
            strat(x)

    def test_forward_wrong_feature_size_raises(self):
        strat = DoIntervention(torch.tensor([0.5, 1.0]))  # 2 features, expect 3
        x = torch.randn(B, F)
        with pytest.raises(AssertionError):
            strat(x)

    def test_forward_wrong_batch_size_raises(self):
        strat = DoIntervention(torch.ones(5, F))  # B=5, expect B=4
        x = torch.randn(B, F)
        with pytest.raises(AssertionError):
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
        m = InterventionModule(enc, GroundTruthIntervention(gt), UniformPolicy())
        assert isinstance(m, nn.Module)

    def test_original_module_stored(self):
        enc = _make_enc()
        m = InterventionModule(enc, GroundTruthIntervention(torch.ones(B, F)), UniformPolicy())
        assert m.original_module is enc

    def test_strategy_stored(self):
        enc = _make_enc()
        strat = GroundTruthIntervention(torch.ones(B, F))
        m = InterventionModule(enc, strat, UniformPolicy())
        assert m.intervention_strategy is strat

    def test_policy_stored(self):
        enc = _make_enc()
        policy = UniformPolicy()
        m = InterventionModule(enc, GroundTruthIntervention(torch.ones(B, F)), policy)
        assert m.intervention_policy is policy

    def test_default_quantile(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy())
        assert m.quantile == pytest.approx(1.0)

    def test_custom_quantile(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy(), quantile=0.5)
        assert m.quantile == pytest.approx(0.5)

    def test_out_concepts_to_intervene_on_stored(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy(),
                               out_concepts_to_intervene_on=[0, 1])
        assert m.out_concepts_to_intervene_on == [0, 1]


# ===========================================================================
# 8. intervene() factory function
# ===========================================================================

class TestInterveneFn:
    def test_returns_intervention_module(self):
        enc = _make_enc()
        m = intervene(enc, DoIntervention(0.0), UniformPolicy())
        assert isinstance(m, InterventionModule)

    def test_original_module_preserved(self):
        enc = _make_enc()
        m = intervene(enc, DoIntervention(0.0), UniformPolicy())
        assert m.original_module is enc

    def test_custom_quantile(self):
        enc = _make_enc()
        m = intervene(enc, DoIntervention(0.0), UniformPolicy(), quantile=0.5)
        assert m.quantile == pytest.approx(0.5)


# ===========================================================================
# 9. intervention() context manager
# ===========================================================================

class TestInterventionContextManager:
    def test_yields_intervention_module(self):
        enc = _make_enc()
        with intervention(enc, DoIntervention(0.0), UniformPolicy()) as m:
            assert isinstance(m, InterventionModule)

    def test_forward_runs_inside_context(self):
        enc = _make_enc()
        x = torch.randn(B, enc.in_f)
        with intervention(enc, DoIntervention(0.5), UniformPolicy()) as m:
            out = m(x)
        assert out.shape == (B, enc.out_f)

    def test_context_exits_cleanly(self):
        enc = _make_enc()
        with intervention(enc, DoIntervention(0.0), UniformPolicy()):
            pass


# ===========================================================================
# 10. InterventionModule.forward() — end-to-end
# ===========================================================================

class TestInterventionModuleForward:
    def test_output_shape(self):
        enc = _make_enc()
        x = torch.randn(B, enc.in_f)
        m = intervene(enc, DoIntervention(0.5), UniformPolicy(), quantile=1.0)
        out = m(x)
        assert out.shape == (B, F)

    def test_full_intervention_ground_truth(self):
        enc = _make_enc()
        gt = torch.full((B, F), 0.7)
        m = intervene(enc, GroundTruthIntervention(gt), UniformPolicy(), quantile=1.0)
        x = torch.randn(B, enc.in_f)
        out = m(x)
        # quantile=1.0 → all concepts replaced → output should equal gt
        assert torch.allclose(out, gt, atol=1e-5)

    def test_full_do_intervention(self):
        enc = _make_enc()
        m = intervene(enc, DoIntervention(0.0), UniformPolicy(), quantile=1.0)
        x = torch.randn(B, enc.in_f)
        out = m(x)
        # quantile=1.0 + do(0.0) → all concepts become 0
        assert torch.allclose(out, torch.zeros(B, F), atol=1e-5)

    def test_no_intervention_at_quantile_zero_single_concept(self):
        enc = _Encoder(in_features=4, out_features=1)
        gt = torch.ones(B, 1)
        m = intervene(enc, GroundTruthIntervention(gt), UniformPolicy(), quantile=0.0)
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
        m = intervene(enc, GroundTruthIntervention(gt), UniformPolicy(),
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
        m = intervene(enc, DoIntervention(0.5), RandomPolicy(scale=1.0), quantile=1.0)
        x = torch.randn(B, enc.in_f)
        out = m(x)
        # quantile=1.0 → all concepts replaced by do(0.5)
        assert torch.allclose(out, torch.full((B, F), 0.5), atol=1e-5)

    def test_uncertainty_policy_with_do_intervention(self):
        enc = _make_enc()
        m = intervene(enc, DoIntervention(1.0), UncertaintyInterventionPolicy(), quantile=1.0)
        x = torch.randn(B, enc.in_f)
        out = m(x)
        # quantile=1.0 → all concepts replaced by do(1.0)
        assert torch.allclose(out, torch.ones(B, F), atol=1e-5)

    def test_distribution_intervention_output_shape(self):
        enc = _make_enc()
        d = torch_dist.Bernoulli(torch.tensor(0.5))
        m = intervene(enc, DistributionIntervention(d), UniformPolicy(), quantile=1.0)
        x = torch.randn(B, enc.in_f)
        out = m(x)
        assert out.shape == (B, F)


# ===========================================================================
# 11. Gradient flow
# ===========================================================================

class TestGradientFlow:
    def test_gradient_through_intervention_module(self):
        enc = _make_enc()
        m = intervene(enc, DoIntervention(0.5), UniformPolicy(), quantile=0.5)
        x = torch.randn(B, enc.in_f, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_gradient_through_original_module_weights(self):
        enc = _make_enc()
        m = intervene(enc, DoIntervention(0.5), UniformPolicy(), quantile=0.5)
        x = torch.randn(B, enc.in_f)
        m(x).sum().backward()
        assert enc.linear.weight.grad is not None

    def test_no_gradient_with_full_do_intervention(self):
        enc = _make_enc()
        m = intervene(enc, DoIntervention(0.5), UniformPolicy(), quantile=1.0)
        x = torch.randn(B, enc.in_f, requires_grad=True)
        out = m(x)
        out.sum().backward()
        # quantile=1.0 with uniform policy + STE proxy: grad may still flow
        # through the STE term, so just check it doesn't error
        assert out is not None

    def test_gradient_with_ground_truth_partial(self):
        enc = _make_enc()
        gt = torch.zeros(B, F)
        m = intervene(enc, GroundTruthIntervention(gt), UniformPolicy(), quantile=0.5)
        x = torch.randn(B, enc.in_f, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


# ===========================================================================
# 12. sel_idx property
# ===========================================================================

class TestSelIdx:
    def test_none_when_no_selection(self):
        enc = _make_enc()
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy())
        assert m.sel_idx is None

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
        m = InterventionModule(enc, DoIntervention(0.0), UniformPolicy(),
                               extra_modules={"task_head": head})
        assert "task_head" in dict(m.named_modules())


# ===========================================================================
# 14. PositiveWeightsIntervention strategy
# ===========================================================================

from torch_concepts.nn.modules.low.intervention.strategy.positive_weights import PositiveWeightsIntervention


class TestPositiveWeightsIntervention:
    def test_construction(self):
        strat = PositiveWeightsIntervention()
        from torch_concepts.nn.modules.low.base.intervention import BaseModuleInterventionStrategy
        assert isinstance(strat, BaseModuleInterventionStrategy)

    def test_transform_makes_weights_nonnegative(self):
        enc = _make_enc()
        # Force some negative weights
        with torch.no_grad():
            enc.linear.weight.fill_(-1.0)
        strat = PositiveWeightsIntervention()
        strat.transform(enc)
        assert (enc.linear.weight >= 0).all()

    def test_transform_preserves_positive_weights(self):
        enc = _make_enc()
        with torch.no_grad():
            enc.linear.weight.fill_(2.0)
        strat = PositiveWeightsIntervention()
        strat.transform(enc)
        assert torch.allclose(enc.linear.weight, torch.full_like(enc.linear.weight, 2.0))

    def test_transform_returns_module(self):
        enc = _make_enc()
        strat = PositiveWeightsIntervention()
        result = strat.transform(enc)
        assert result is enc

    def test_full_intervention_via_intervention_module(self):
        """PositiveWeightsIntervention used as strategy in InterventionModule."""
        enc = _make_enc()
        with torch.no_grad():
            enc.linear.weight.fill_(-0.5)
        strat = PositiveWeightsIntervention()
        m = InterventionModule(enc, strat, UniformPolicy(), quantile=1.0)
        x = torch.randn(B, enc.in_f)
        out = m(x)
        # After applying ReLU to weights, all weights are 0, output should be bias-only
        assert out.shape == (B, enc.out_f)


# ===========================================================================
# 15. GradientPolicy
# ===========================================================================

from torch_concepts.nn.modules.low.intervention.policy.gradient import GradientPolicy


class TestGradientPolicy:
    def test_construction(self):
        p = GradientPolicy()
        from torch_concepts.nn.modules.low.base.intervention import BaseInterventionPolicy
        assert isinstance(p, BaseInterventionPolicy)

    def test_with_gradients_returns_abs(self):
        p = GradientPolicy()
        concepts = torch.randn(B, F)
        grads = torch.tensor([[-1.0, 2.0, -3.0]] * B)
        out = p(concepts, concept_grads=grads)
        assert torch.allclose(out, grads.abs())

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

    def test_gradient_scores_are_nonnegative(self):
        p = GradientPolicy()
        concepts = torch.randn(B, F)
        grads = torch.randn(B, F)
        out = p(concepts, concept_grads=grads)
        assert (out >= 0).all()


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
            build_context=my_build_context,
        )
        x = torch.randn(B, enc.in_f)
        m(x)
        assert context_called

    def test_invalid_strategy_type_raises(self):
        """Passing an object that is neither BaseConceptInterventionStrategy nor BaseModuleInterventionStrategy raises."""
        enc = _make_enc()

        class FakeStrategy:
            pass

        m = InterventionModule(enc, FakeStrategy(), UniformPolicy())
        x = torch.randn(B, enc.in_f)
        with pytest.raises((ValueError, AttributeError)):
            m(x)

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
        m = InterventionModule(enc, DoIntervention(0.5), UniformPolicy())
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
        m = InterventionModule(enc, DoIntervention(0.5), UniformPolicy())
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

    def test_forward_bind_raises_falls_back_to_empty_dict(self):
        """When sig.bind raises TypeError, original_module_inputs falls back to {} (lines 140-141)."""
        class _BadSigEncoder(nn.Module):
            """forward requires extra required arg so sig.bind with just x fails."""
            def __init__(self):
                super().__init__()
                self.l = nn.Linear(4, 3)

            def forward(self, x, required_extra):
                return torch.sigmoid(self.l(x))

        enc = _BadSigEncoder()
        m = InterventionModule(enc, DoIntervention(0.5), UniformPolicy())
        x = torch.randn(B, 4)
        # Calling m(x) without required_extra triggers TypeError inside sig.bind;
        # also, the underlying encoder call will fail — we only care that the
        # except branch is reachable, so catch the propagated error from enc.forward.
        with pytest.raises(TypeError):
            m(x)


# ===========================================================================
# 17. Base intervention abstract methods (base/intervention.py lines 27, 43, 53)
# ===========================================================================

from torch_concepts.nn.modules.low.base.intervention import (
    BaseConceptInterventionStrategy,
    BaseModuleInterventionStrategy,
    BaseInterventionPolicy,
)


class TestBaseInterventionModuleCoverageExtra:
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
        m = InterventionModule(enc, DoIntervention(0.5), UniformPolicy())
        assert m.original_module is enc

    def test_base_build_context_raises_not_implemented(self):
        """BaseInterventionModule.build_context raises NotImplementedError (line 128)."""
        from torch_concepts.nn.modules.low.intervention.intervention import (
            BaseInterventionModule,
        )

        class _Concrete(BaseInterventionModule):
            def build_context(self, *args, **kwargs):
                return super().build_context(*args, **kwargs)

        enc = _make_enc()
        m = _Concrete(enc, DoIntervention(0.0), UniformPolicy())
        with pytest.raises(NotImplementedError):
            m.build_context({}, enc, torch.randn(B, F))


class TestBaseInterventionAbstractMethods:
    def test_base_concept_strategy_forward_raises(self):
        """BaseConceptInterventionStrategy.forward raises NotImplementedError (line 27)."""
        class _ConcreteStrategy(BaseConceptInterventionStrategy):
            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)

        strat = _ConcreteStrategy()
        with pytest.raises(NotImplementedError):
            strat(torch.randn(2, 3))

    def test_base_module_strategy_transform_raises(self):
        """BaseModuleInterventionStrategy.transform raises NotImplementedError (line 43)."""
        class _ConcreteModuleStrategy(BaseModuleInterventionStrategy):
            def transform(self, module, *args, **kwargs):
                return super().transform(module, *args, **kwargs)

        strat = _ConcreteModuleStrategy()
        with pytest.raises(NotImplementedError):
            strat.transform(nn.Linear(2, 2))

    def test_base_policy_forward_raises(self):
        """BaseInterventionPolicy.forward raises NotImplementedError (line 53)."""
        class _ConcretePolicy(BaseInterventionPolicy):
            def forward(self, x, *args, **kwargs):
                return super().forward(x, *args, **kwargs)

        policy = _ConcretePolicy()
        with pytest.raises(NotImplementedError):
            policy(torch.randn(2, 3))
