"""Tests for torch_concepts.data.scalers.module.ScalerModule.

Covers:
- Fitting from annotations (continuous concepts only)
- Round-tripping transform/inverse
- Label-based alignment: subsets and reorderings
- Device/dtype moves reaching the fitted statistics
- Error handling for unknown labels and bad shapes
"""

import pytest
import torch

from torch_concepts.annotations import Annotations
from torch_concepts.data.scalers import ScalerModule, StandardScaler


@pytest.fixture
def annotations():
    """Mixed-type axis: a categorical concept sits *before* the continuous ones,
    so a continuous concept's logit index differs from its concept index."""
    return Annotations(
        labels=['flag', 'colour', 'small', 'large'],
        cardinalities=[1, 3, 1, 1],
        types=['binary', 'categorical', 'continuous', 'continuous'],
    )


@pytest.fixture
def concepts():
    """Concept-space ground truth; the two continuous columns differ by ~100x."""
    return torch.tensor([
        [0., 0., 1., 100.],
        [1., 2., 3., 300.],
        [0., 1., 5., 500.],
        [1., 0., 7., 700.],
    ])


@pytest.fixture
def fitted(annotations, concepts):
    return ScalerModule.fit(
        annotations=annotations, concepts=concepts, concept_scaler=StandardScaler()
    )


class TestFit:
    """Fitting picks up exactly the continuous concepts."""

    def test_only_continuous_concepts_are_scaled(self, fitted):
        assert fitted.concept_names == ['small', 'large']
        assert fitted.has_concepts
        assert not fitted.has_input

    def test_statistics_come_from_the_right_column(self, fitted, concepts):
        # 'large' holds 100..700, not the 1..7 of 'small'.
        assert fitted.concept_scalers['large'].mean.item() == pytest.approx(400.0)
        assert fitted.concept_scalers['small'].mean.item() == pytest.approx(4.0)

    def test_no_scaler_prototype_yields_empty_module(self, annotations, concepts):
        empty = ScalerModule.fit(annotations=annotations, concepts=concepts)
        assert not empty.has_concepts
        assert not empty.has_input

    def test_no_continuous_concepts_yields_empty_module(self):
        binary_only = Annotations(labels=['a', 'b'], cardinalities=[1, 1],
                                  types=['binary', 'binary'])
        module = ScalerModule.fit(
            annotations=binary_only, concepts=torch.rand(8, 2),
            concept_scaler=StandardScaler(),
        )
        assert not module.has_concepts

    def test_wide_continuous_concept_is_rejected(self):
        """Alignment is one column per label, so a wider continuous concept must
        raise rather than silently misalign."""
        wide = Annotations(labels=['v'], cardinalities=[3], types=['continuous'])
        with pytest.raises(ValueError, match="cardinality"):
            ScalerModule.fit(annotations=wide, concepts=torch.rand(8, 1),
                             concept_scaler=StandardScaler())

    def test_input_scaler_is_fitted(self, annotations, concepts):
        x = torch.randn(16, 5) * 10 + 3
        module = ScalerModule.fit(
            annotations=annotations, concepts=concepts, input_data=x,
            input_scaler=StandardScaler(),
        )
        assert module.has_input
        scaled = module.transform_input(x)
        assert torch.allclose(scaled.mean(0), torch.zeros(5), atol=1e-5)
        assert torch.allclose(module.inverse_input(scaled), x, atol=1e-4)


class TestRoundTrip:
    """transform → inverse recovers the original values."""

    def test_round_trip(self, fitted, concepts):
        labels = ['small', 'large']
        original = concepts[:, [2, 3]]
        scaled = fitted.transform_concepts(original, labels)
        assert torch.allclose(fitted.inverse_concepts(scaled, labels), original, atol=1e-4)

    def test_scaled_values_are_standardised(self, fitted, concepts):
        scaled = fitted.transform_concepts(concepts[:, [2, 3]], ['small', 'large'])
        assert torch.allclose(scaled.mean(0), torch.zeros(2), atol=1e-5)
        # Both columns had the same *relative* spread, so standardising collapses
        # the 100x difference between them.
        assert torch.allclose(scaled[:, 0], scaled[:, 1], atol=1e-5)

    def test_leading_dims_are_preserved(self, fitted):
        x = torch.randn(2, 5, 2)
        scaled = fitted.transform_concepts(x, ['small', 'large'])
        assert scaled.shape == x.shape
        assert torch.allclose(fitted.inverse_concepts(scaled, ['small', 'large']), x, atol=1e-4)


class TestLabelAlignment:
    """Columns are matched to scalers by label, not by position."""

    def test_reordering_labels_reorders_the_statistics(self, fitted, concepts):
        forward = fitted.transform_concepts(concepts[:, [2, 3]], ['small', 'large'])
        reversed_ = fitted.transform_concepts(concepts[:, [3, 2]], ['large', 'small'])
        assert torch.allclose(reversed_, forward[:, [1, 0]], atol=1e-6)

    def test_subset_of_labels(self, fitted, concepts):
        """A task-only model reports a subset of the concepts."""
        full = fitted.transform_concepts(concepts[:, [2, 3]], ['small', 'large'])
        subset = fitted.transform_concepts(concepts[:, 3:4], ['large'])
        assert torch.allclose(subset, full[:, 1:2], atol=1e-6)

    def test_unknown_label_raises(self, fitted, concepts):
        with pytest.raises(KeyError, match="no fitted scaler"):
            fitted.transform_concepts(concepts[:, 2:3], ['flag'])

    def test_column_count_must_match_labels(self, fitted, concepts):
        with pytest.raises(ValueError, match="one column per label"):
            fitted.transform_concepts(concepts[:, [2, 3]], ['small'])


class TestFusedFastPath:
    """The vectorised path must be numerically identical to the per-column loop,
    and must step aside whenever the scalers cannot be fused."""

    @staticmethod
    def _loop_result(module, x, labels, inverse=False):
        """Same call with the fast path disabled."""
        original = module._fused_for
        module._fused_for = lambda _labels: None
        try:
            return module._apply_concepts(x, labels, inverse=inverse)
        finally:
            module._fused_for = original

    def test_fused_path_is_used(self, fitted):
        assert fitted._fused_for(('small', 'large')) is not None

    def test_fused_matches_loop(self, fitted, concepts):
        x = concepts[:, [2, 3]]
        for labels in [('small', 'large'), ('large', 'small'), ('large',)]:
            cols = [{'small': 2, 'large': 3}[n] for n in labels]
            data = concepts[:, cols]
            assert torch.allclose(
                fitted.transform_concepts(data, labels),
                self._loop_result(fitted, data, labels),
                atol=1e-6,
            ), labels

    def test_fused_matches_loop_for_inverse(self, fitted, concepts):
        scaled = fitted.transform_concepts(concepts[:, [2, 3]], ['small', 'large'])
        assert torch.allclose(
            fitted.inverse_concepts(scaled, ['small', 'large']),
            self._loop_result(fitted, scaled, ['small', 'large'], inverse=True),
            atol=1e-6,
        )

    def test_cache_is_per_label_tuple(self, fitted, concepts):
        fitted.transform_concepts(concepts[:, [2, 3]], ['small', 'large'])
        fitted.transform_concepts(concepts[:, 3:4], ['large'])
        assert ('small', 'large') in fitted._fused_cache
        assert ('large',) in fitted._fused_cache

    def test_cache_is_invalidated_when_statistics_change(self, fitted, concepts):
        """The fused scaler holds tensors *derived* from the buffers, so replacing
        a buffer must drop it — otherwise it keeps scaling with the stale
        statistics. ``_sync`` is the point every buffer replacement funnels
        through (see ``_apply``), so the contract is pinned there directly: a
        dtype cast alone would not catch this, being value-preserving."""
        labels = ['small', 'large']
        x = concepts[:, [2, 3]]
        before = fitted.transform_concepts(x, labels)  # populates the cache

        fitted.stat__concept_small__mean = fitted.stat__concept_small__mean + 100.0
        fitted._sync()

        after = fitted.transform_concepts(x, labels)
        assert not torch.allclose(before[:, 0], after[:, 0])
        assert torch.allclose(after, self._loop_result(fitted, x, labels), atol=1e-6)

    def test_dtype_move_keeps_results_correct(self, fitted, concepts):
        labels = ['small', 'large']
        fitted.transform_concepts(concepts[:, [2, 3]], labels)  # populates the cache
        moved = fitted.to(torch.float64)

        x = concepts[:, [2, 3]].double()
        out = moved.transform_concepts(x, labels)
        assert out.dtype == torch.float64
        assert torch.allclose(out, self._loop_result(moved, x, labels), atol=1e-9)

    def test_non_scalar_statistics_fall_back_to_the_loop(self, annotations, concepts):
        """A scaler holding a per-concept *vector* cannot be concatenated into a
        broadcastable row, so the loop must take over rather than misalign."""
        class VectorStatScaler(StandardScaler):
            def fit(self, x):
                super().fit(x)
                self.extra = torch.zeros(3)  # not a per-column scalar
                return self

        module = ScalerModule.fit(
            annotations=annotations, concepts=concepts,
            concept_scaler=VectorStatScaler(),
        )
        assert module._fused_for(('small', 'large')) is None
        # ...and the result is still correct.
        out = module.transform_concepts(concepts[:, [2, 3]], ['small', 'large'])
        assert torch.allclose(out.mean(0), torch.zeros(2), atol=1e-5)

    def test_differing_configuration_falls_back(self, annotations, concepts):
        """Non-tensor state must agree across concepts to be represented by one
        fused scaler."""
        module = ScalerModule.fit(
            annotations=annotations, concepts=concepts, concept_scaler=StandardScaler(),
        )
        module.concept_scalers['large'].axis = 1  # diverge from 'small'
        module._fused_cache.clear()
        assert module._fused_for(('small', 'large')) is None


class TestModuleIntegration:
    """The statistics behave like real buffers."""

    def test_statistics_are_registered_as_buffers(self, fitted):
        keys = set(fitted.state_dict())
        assert 'stat__concept_small__mean' in keys
        assert 'stat__concept_large__std' in keys

    def test_dtype_move_reaches_the_scalers(self, fitted, concepts):
        """Guards the _apply/_sync wiring: without it the scalers would keep
        pointing at the pre-move tensors."""
        moved = fitted.to(torch.float64)
        assert moved.concept_scalers['small'].mean.dtype == torch.float64
        out = moved.transform_concepts(concepts[:, [2, 3]].double(), ['small', 'large'])
        assert out.dtype == torch.float64

    def test_state_dict_round_trip(self, annotations, concepts, fitted):
        fresh = ScalerModule.fit(
            annotations=annotations, concepts=torch.zeros_like(concepts),
            concept_scaler=StandardScaler(),
        )
        fresh.load_state_dict(fitted.state_dict())
        fresh._sync()
        assert fresh.concept_scalers['large'].mean.item() == pytest.approx(400.0)

    def test_repr(self, fitted):
        assert 'small' in repr(fitted)
        assert 'ScalerModule' in repr(fitted)
