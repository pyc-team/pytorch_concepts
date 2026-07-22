"""Tests for input/concept scaling in BaseLearner.

The contract under test: with a ``ScalerModule`` attached, everything the model
touches (evidence, teacher-forced query values, loss target) is in **scaled**
space, while metrics are updated in the **original** data scale. Without one,
every scaling step is an identity.
"""

import pytest
import torch
import torch.nn as nn
from torchmetrics.regression import MeanSquaredError

from torch_concepts.annotations import Annotations
from torch_concepts.data.scalers import ScalerModule, StandardScaler
from torch_concepts.nn.modules.high.models.blackbox import BlackBoxTaskOnly
from torch_concepts.nn.modules.high.models.cbm import ConceptBottleneckModel
from torch_concepts.nn.modules.loss import ConceptLoss
from torch_concepts.nn.modules.metrics import ConceptMetrics
from torch_concepts.nn.modules.outputs import ModelOutput
from torch_concepts.tensor import AnnotatedTensor


N_SAMPLES = 64
INPUT_SIZE = 8


@pytest.fixture
def annotations():
    return Annotations(
        labels=['c1', 'c2', 'task'], cardinalities=[1, 1, 1],
        types=['continuous', 'continuous', 'continuous'],
    )


@pytest.fixture
def batch():
    """Concepts on wildly different scales — the case scaling exists for."""
    torch.manual_seed(0)
    concepts = torch.stack([
        torch.randn(N_SAMPLES) * 3 + 10,
        torch.randn(N_SAMPLES) * 100 + 500,
        torch.randn(N_SAMPLES) * 0.01,
    ], dim=1)
    return {
        'inputs': {'x': torch.randn(N_SAMPLES, INPUT_SIZE)},
        'concepts': {'c': concepts},
    }


@pytest.fixture
def scalers(annotations, batch):
    return ScalerModule.fit(
        annotations=annotations, concepts=batch['concepts']['c'],
        concept_scaler=StandardScaler(),
    )


def build_model(annotations, scalers, cls=ConceptBottleneckModel, **kwargs):
    """Fresh model with identical initial weights for a given seed."""
    torch.manual_seed(0)
    return cls(
        input_size=INPUT_SIZE,
        annotations=annotations,
        task_names=['task'],
        lightning=True,
        loss=ConceptLoss(continuous=nn.MSELoss()),
        metrics=ConceptMetrics(
            annotations=annotations, continuous={'mse': MeanSquaredError()}
        ),
        optim_class=torch.optim.Adam,
        scalers=scalers,
        **kwargs,
    )


class TestNoScalerIsIdentity:
    """Without scalers the learner must behave exactly as before."""

    def test_helpers_pass_through(self, annotations, batch):
        model = build_model(annotations, None)
        c = batch['concepts']['c']
        assert model.scale_concepts(c) is c
        assert model.scale_inputs(batch['inputs']) is batch['inputs']

    def test_unscale_output_returns_same_object(self, annotations, batch):
        model = build_model(annotations, None)
        out = model.forward(query=model.default_query(batch['concepts']['c']),
                            evidence={'input': batch['inputs']['x']})
        assert model.unscale_output(out) is out

    def test_loss_and_metric_agree(self, annotations, batch):
        """With MSE on both sides and no scaling, the loss and the summary metric
        are the same quantity — a baseline the scaled run is compared against."""
        model = build_model(annotations, None)
        loss = model.shared_step(batch, 'test')
        metric = model.test_metrics.compute()['test/SUMMARY-continuous_mse']
        assert float(metric) == pytest.approx(float(loss), rel=1e-4)


def _metric_for_prediction(model, c, loc_scaled):
    """Feed *loc_scaled* (a scaled-space prediction) through the metrics path and
    return the resulting summary MSE, exactly as ``shared_step`` would."""
    out = ModelOutput()
    out.loc = AnnotatedTensor(
        loc_scaled, model.concept_annotations.to_concept_space(), axis=-1
    )
    model.update_and_log_metrics(
        model.unscale_output(out), model.prepare_target(c), 'test', N_SAMPLES
    )
    return float(model.test_metrics.compute()['test/SUMMARY-continuous_mse'])


class TestScaledSpaceSeparation:
    """Loss in scaled units, metrics in original units."""

    def test_perfect_scaled_prediction_gives_zero_metric(self, annotations, batch, scalers):
        """The round-trip that makes the metric trustworthy: a model whose output
        exactly equals the scaled target must score 0 error in original units."""
        model = build_model(annotations, scalers)
        c = batch['concepts']['c']
        mse = _metric_for_prediction(model, c, model.scale_concepts(c).tensor.clone())
        assert mse == pytest.approx(0.0, abs=1e-6)

    def test_known_scaled_error_maps_to_known_original_error(self, annotations, batch, scalers):
        """Pin the inverse transform exactly: an error of ``delta`` in scaled space
        is an error of ``delta * std`` in original space, per concept. This is what
        distinguishes a correct inversion from no inversion at all."""
        model = build_model(annotations, scalers)
        c = batch['concepts']['c']
        delta = 0.5

        mse = _metric_for_prediction(
            model, c, model.scale_concepts(c).tensor + delta
        )

        stds = torch.tensor([
            scalers.concept_scalers[name].std.item() for name in ['c1', 'c2', 'task']
        ])
        expected = ((delta * stds) ** 2).mean()
        assert mse == pytest.approx(float(expected), rel=1e-4)

    def test_shared_step_metric_matches_original_scale_recomputation(
        self, annotations, batch, scalers
    ):
        """End-to-end pin on ``shared_step``: the logged metric must equal the MSE
        between the inverse-transformed predictions and the **untouched** batch
        concepts. Catches a metrics call that is handed the scaled target."""
        model = build_model(annotations, scalers)
        model.shared_step(batch, 'test')
        mse = float(model.test_metrics.compute()['test/SUMMARY-continuous_mse'])

        c = batch['concepts']['c']
        c_scaled = model.scale_concepts(c)
        out = model.forward(
            query=model.default_query(c_scaled),
            evidence=model.default_evidence(model.scale_inputs(batch['inputs'])),
        )
        loc = out.loc
        labels = list(loc.annotation.labels)
        preds = scalers.inverse_concepts(loc.tensor, labels)
        target = model.prepare_target(c)[labels].tensor
        assert mse == pytest.approx(float(((preds - target) ** 2).mean()), rel=1e-4)

    def test_loss_is_in_scaled_units_metric_is_not(self, annotations, batch, scalers):
        """The two are the same MSE quantity without scaling (see
        ``test_loss_and_metric_agree``); with scaling they must diverge, the loss
        following the scaled concepts and the metric the original ones."""
        model = build_model(annotations, scalers)
        loss = float(model.shared_step(batch, 'test'))
        mse = float(model.test_metrics.compute()['test/SUMMARY-continuous_mse'])

        assert loss == pytest.approx(
            _scaled_space_mse(model, batch), rel=1e-4
        )
        assert mse != pytest.approx(loss, rel=0.1)


def _scaled_space_mse(model, batch):
    """MSE the loss should see: predictions and target both in scaled space."""
    c_scaled = model.scale_concepts(batch['concepts']['c'])
    out = model.forward(
        query=model.default_query(c_scaled),
        evidence=model.default_evidence(model.scale_inputs(batch['inputs'])),
    )
    loc = out.loc
    target = c_scaled[list(loc.annotation.labels)]
    return float(((loc.tensor - target.tensor) ** 2).mean())


class TestUnscaleOutput:
    def test_uncovered_columns_pass_through(self, annotations, batch, scalers):
        """A query may also ask for a non-concept variable — 'latent' is a Delta
        and reports under 'value' — whose columns have no scaler."""
        from torch_concepts.annotations import Annotations as Ann

        model = build_model(annotations, scalers)
        mixed_ann = Ann(
            labels=['c1', 'latent'], cardinalities=[1, 4],
            types=['continuous', 'continuous'],
        )
        data = torch.randn(N_SAMPLES, 5)
        out = ModelOutput()
        out.value = AnnotatedTensor(data, mixed_ann, axis=-1)

        restored = model.unscale_output(out).value.tensor
        # 'latent' columns untouched, 'c1' column inverted.
        assert torch.equal(restored[:, 1:], data[:, 1:])
        expected = scalers.inverse_concepts(data[:, 0:1], ['c1'])
        assert torch.allclose(restored[:, 0:1], expected, atol=1e-5)

    def test_original_output_is_not_mutated(self, annotations, batch, scalers):
        model = build_model(annotations, scalers)
        data = torch.randn(N_SAMPLES, 3)
        out = ModelOutput()
        out.loc = AnnotatedTensor(
            data.clone(), model.concept_annotations.to_concept_space(), axis=-1
        )
        model.unscale_output(out)
        assert torch.equal(out.loc.tensor, data)


class TestScaleConcepts:
    """scale_concepts touches continuous columns only."""

    def test_continuous_columns_are_standardised(self, annotations, batch, scalers):
        model = build_model(annotations, scalers)
        scaled = model.scale_concepts(batch['concepts']['c'])
        assert torch.allclose(scaled.tensor.mean(0), torch.zeros(3), atol=1e-5)

    def test_annotation_is_preserved(self, annotations, batch, scalers):
        model = build_model(annotations, scalers)
        scaled = model.scale_concepts(batch['concepts']['c'])
        assert isinstance(scaled, AnnotatedTensor)
        assert list(scaled.annotation.labels) == ['c1', 'c2', 'task']

    def test_discrete_columns_pass_through(self, batch):
        """Binary and categorical columns are class labels; scaling them would be
        meaningless, so they must come out bit-identical."""
        mixed = Annotations(
            labels=['flag', 'colour', 'value'], cardinalities=[1, 3, 1],
            types=['binary', 'categorical', 'continuous'],
        )
        concepts = torch.stack([
            torch.randint(0, 2, (N_SAMPLES,)).float(),
            torch.randint(0, 3, (N_SAMPLES,)).float(),
            torch.randn(N_SAMPLES) * 50 + 200,
        ], dim=1)
        scalers = ScalerModule.fit(annotations=mixed, concepts=concepts,
                                   concept_scaler=StandardScaler())
        model = ConceptBottleneckModel(
            input_size=INPUT_SIZE, annotations=mixed, task_names=['value'],
            lightning=True, scalers=scalers,
        )
        scaled = model.scale_concepts(concepts)
        assert torch.equal(scaled.tensor[:, :2], concepts[:, :2])
        assert not torch.equal(scaled.tensor[:, 2], concepts[:, 2])

    def test_none_concepts(self, annotations, scalers):
        model = build_model(annotations, scalers)
        assert model.scale_concepts(None) is None


class TestScaleInputs:
    def test_input_is_scaled(self, annotations, batch):
        x = batch['inputs']['x'] * 20 + 5
        scalers = ScalerModule.fit(
            annotations=annotations, input_data=x, input_scaler=StandardScaler(),
        )
        model = build_model(annotations, scalers)
        scaled = model.scale_inputs({'x': x})
        assert torch.allclose(scaled['x'].mean(0), torch.zeros(INPUT_SIZE), atol=1e-5)

    def test_other_input_keys_are_kept(self, annotations, batch):
        scalers = ScalerModule.fit(
            annotations=annotations, input_data=batch['inputs']['x'],
            input_scaler=StandardScaler(),
        )
        model = build_model(annotations, scalers)
        scaled = model.scale_inputs({'x': batch['inputs']['x'], 'extra': 'kept'})
        assert scaled['extra'] == 'kept'


class TestTaskOnlyModel:
    """BlackBoxTaskOnly slices the target but queries with full-axis offsets."""

    def test_shared_step_runs(self, annotations, batch, scalers):
        model = self._model(annotations, scalers)
        loss = model.shared_step(batch, 'test')
        assert torch.isfinite(loss)
        assert 'test/SUMMARY-continuous_mse' in model.test_metrics.compute()

    def test_metric_is_inverted_on_the_task_subset(self, annotations, batch, scalers):
        """The target is sliced to 'task' only, so the inverse transform has to
        pick that concept's statistics by name out of the full set."""
        model = self._model(annotations, scalers)
        c = batch['concepts']['c']
        delta = 0.5

        out = ModelOutput()
        out.loc = AnnotatedTensor(
            model.scale_concepts(c).tensor[:, 2:3] + delta,
            model.task_annotations.to_concept_space(),
            axis=-1,
        )
        model.update_and_log_metrics(
            model.unscale_output(out), model.prepare_target(c), 'test', N_SAMPLES
        )
        mse = float(model.test_metrics.compute()['test/SUMMARY-continuous_mse'])
        expected = (delta * scalers.concept_scalers['task'].std.item()) ** 2
        assert mse == pytest.approx(expected, rel=1e-4)

    @staticmethod
    def _model(annotations, scalers):
        torch.manual_seed(0)
        return BlackBoxTaskOnly(
            input_size=INPUT_SIZE,
            annotations=annotations,
            task_names=['task'],
            lightning=True,
            loss=ConceptLoss(continuous=nn.MSELoss()),
            metrics=ConceptMetrics(
                annotations=annotations, continuous={'mse': MeanSquaredError()}
            ),
            scalers=scalers,
        )

    def test_query_uses_full_concept_axis(self, annotations, batch, scalers):
        """Regression guard: build_query indexes the *full* concept tensor with
        task offsets, so it must not be handed the task-sliced target."""
        torch.manual_seed(0)
        model = BlackBoxTaskOnly(
            input_size=INPUT_SIZE, annotations=annotations,
            task_names=['task'], lightning=True, scalers=scalers,
        )
        c_scaled = model.scale_concepts(batch['concepts']['c'])
        assert c_scaled.shape[-1] == 3
        query = model.default_query(c_scaled)
        assert torch.allclose(query['task'].squeeze(-1), c_scaled.tensor[:, 2])
