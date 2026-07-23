"""Tests for concept/input scaling in BaseLearner.

The contract under test: scalers arrive **via the batch** (``batch['scalers']``,
shipped by ``ConceptDataset.collate``), not through a model constructor
argument. With a scaler present and ``scale_concepts`` on, everything the
model touches (evidence, teacher-forced query values, loss target) is in
**scaled** space, while metrics are updated in the **original** data scale.
Without one — or with ``scale_concepts=False`` — every scaling step is an
identity.

``unscale_output`` assumes a queried quantity covers exactly the concepts the
scaler was fit on (the standard case: ``default_query`` asks for every
concept). ``BlackBoxTaskOnly`` queries a strict subset and explicitly refuses
that combination instead of silently producing wrong values.
"""

import pytest
import torch
import torch.nn as nn
from torchmetrics.regression import MeanSquaredError

from torch_concepts.annotations import Annotations
from torch_concepts.data.scalers import StandardScaler
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
def concepts(annotations):
    """Concepts on wildly different scales — the case scaling exists for."""
    torch.manual_seed(0)
    data = torch.stack([
        torch.randn(N_SAMPLES) * 3 + 10,
        torch.randn(N_SAMPLES) * 100 + 500,
        torch.randn(N_SAMPLES) * 0.01,
    ], dim=1)
    return AnnotatedTensor(data, annotations, axis=1)


@pytest.fixture
def batch(concepts):
    return {
        'inputs': {'x': torch.randn(N_SAMPLES, INPUT_SIZE)},
        'concepts': {'c': concepts},
    }


@pytest.fixture
def scaler(concepts):
    """Fitted exactly as `ConceptDataModule.setup()` fits a 'concepts' scaler:
    on the continuous-only slice, so `.mean`/`.std` stay label-addressable."""
    s = StandardScaler(axis=0)
    s.fit(concepts.continuous())
    return s


def build_model(cls=ConceptBottleneckModel, annotations=None, **kwargs):
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
        **kwargs,
    )


class TestNoScalerIsIdentity:
    """Without a scaler shipped, every scaling step must be a no-op."""

    def test_helpers_pass_through(self, annotations, batch):
        model = build_model(annotations=annotations)
        c = batch['concepts']['c']
        assert model.maybe_scale_concepts({'c': c}, {})['c'] is c
        assert model.maybe_scale_inputs(batch['inputs'], {}) is batch['inputs']

    def test_unscale_output_returns_same_object(self, annotations, batch):
        model = build_model(annotations=annotations)
        out = model.forward(
            query=model.fully_observed_query(batch['concepts']['c']),
            evidence={'input': batch['inputs']['x']},
        )
        assert model.unscale_output(out, {}) is out

    def test_loss_and_metric_agree(self, annotations, batch):
        """With MSE on both sides and no scaling, the loss and the summary metric
        are the same quantity — a baseline the scaled run is compared against."""
        model = build_model(annotations=annotations)
        full_batch = {**batch, 'scalers': {}}
        loss = model.shared_step(full_batch, 'test')
        metric = model.test_metrics.compute()['test/SUMMARY-continuous_mse']
        assert float(metric) == pytest.approx(float(loss), rel=1e-4)


class TestScaledSpaceSeparation:
    """Loss in scaled units, metrics in original units."""

    def test_perfect_scaled_prediction_gives_zero_metric(self, annotations, batch, scaler):
        """The round-trip that makes the metric trustworthy: a model whose output
        exactly equals the scaled target must score 0 error in original units."""
        model = build_model(annotations=annotations)
        c = batch['concepts']['c']
        transforms = {'concepts': scaler}

        c_scaled = model.maybe_scale_concepts({'c': c}, transforms)['c']
        out = ModelOutput()
        out.loc = AnnotatedTensor(c_scaled.tensor.clone(), annotations.to_concept_space(), axis=-1)
        model.update_and_log_metrics(
            model.unscale_output(out, transforms), model.prepare_target(c), 'test', N_SAMPLES
        )
        mse = float(model.test_metrics.compute()['test/SUMMARY-continuous_mse'])
        assert mse == pytest.approx(0.0, abs=1e-6)

    def test_known_scaled_error_maps_to_known_original_error(self, annotations, batch, scaler):
        """Pin the inverse transform exactly: an error of `delta` in scaled space
        is an error of `delta * std` in original space, per concept. This is what
        distinguishes a correct inversion from no inversion at all."""
        model = build_model(annotations=annotations)
        c = batch['concepts']['c']
        transforms = {'concepts': scaler}
        delta = 0.5

        c_scaled = model.maybe_scale_concepts({'c': c}, transforms)['c']
        out = ModelOutput()
        out.loc = AnnotatedTensor(
            c_scaled.tensor + delta, annotations.to_concept_space(), axis=-1
        )
        model.update_and_log_metrics(
            model.unscale_output(out, transforms), model.prepare_target(c), 'test', N_SAMPLES
        )
        mse = float(model.test_metrics.compute()['test/SUMMARY-continuous_mse'])

        stds = scaler.std[['c1', 'c2', 'task']].tensor
        expected = ((delta * stds) ** 2).mean()
        assert mse == pytest.approx(float(expected), rel=1e-4)

    def test_shared_step_metric_matches_original_scale_recomputation(
        self, annotations, batch, scaler
    ):
        """End-to-end pin on `shared_step`: the logged metric must equal the MSE
        between the inverse-transformed predictions and the **untouched** batch
        concepts. Catches a metrics call that is handed the scaled target."""
        model = build_model(annotations=annotations)
        full_batch = {**batch, 'scalers': {'concepts': scaler}}
        model.shared_step(full_batch, 'test')
        mse = float(model.test_metrics.compute()['test/SUMMARY-continuous_mse'])

        c = batch['concepts']['c']
        transforms = {'concepts': scaler}
        c_scaled = model.maybe_scale_concepts({'c': c}, transforms)['c']
        out = model.forward(
            query=model.fully_observed_query(c_scaled),
            evidence=model.default_evidence(model.maybe_scale_inputs(batch['inputs'], transforms)),
        )
        loc = out.loc
        preds = scaler.inverse_transform(loc.tensor)
        target = model.prepare_target(c).tensor
        assert mse == pytest.approx(float(((preds - target) ** 2).mean()), rel=1e-4)

    def test_loss_is_in_scaled_units_metric_is_not(self, annotations, batch, scaler):
        """The two are the same MSE quantity without scaling (see
        `test_loss_and_metric_agree`); with scaling they must diverge, the loss
        following the scaled concepts and the metric the original ones."""
        model = build_model(annotations=annotations)
        full_batch = {**batch, 'scalers': {'concepts': scaler}}
        loss = float(model.shared_step(full_batch, 'test'))
        mse = float(model.test_metrics.compute()['test/SUMMARY-continuous_mse'])
        assert mse != pytest.approx(loss, rel=0.1)


class TestUnscaleOutput:
    def test_original_tensor_is_not_mutated_in_place(self, annotations, batch, scaler):
        """`unscale_output` must build a new tensor (reassigning `out.loc`), not
        mutate the scaled-space tensor in place — that tensor may still be
        referenced elsewhere (e.g. by the loss computed earlier in `shared_step`)."""
        model = build_model(annotations=annotations)
        data = torch.randn(N_SAMPLES, 3)
        scaled_space_tensor = data.clone()
        out = ModelOutput()
        out.loc = AnnotatedTensor(scaled_space_tensor, annotations.to_concept_space(), axis=-1)

        model.unscale_output(out, {'concepts': scaler})

        assert torch.equal(scaled_space_tensor, data)  # untouched in place
        assert out.loc.tensor is not scaled_space_tensor  # reassigned, not mutated
        assert not torch.equal(out.loc.tensor, data)  # actually inverse-transformed

    def test_scale_off_is_a_noop_even_with_scaler_shipped(self, annotations, batch, scaler):
        model = build_model(annotations=annotations, scale_concepts=False)
        data = torch.randn(N_SAMPLES, 3)
        out = ModelOutput()
        out.loc = AnnotatedTensor(data.clone(), annotations.to_concept_space(), axis=-1)
        restored = model.unscale_output(out, {'concepts': scaler})
        assert restored is out
        assert torch.equal(restored.loc.tensor, data)


class TestScaleConcepts:
    """maybe_scale_concepts touches continuous columns only."""

    def test_continuous_columns_are_standardised(self, annotations, batch, scaler):
        model = build_model(annotations=annotations)
        scaled = model.maybe_scale_concepts(batch['concepts'], {'concepts': scaler})['c']
        assert torch.allclose(scaled.tensor.mean(0), torch.zeros(3), atol=1e-5)

    def test_annotation_is_preserved(self, annotations, batch, scaler):
        model = build_model(annotations=annotations)
        scaled = model.maybe_scale_concepts(batch['concepts'], {'concepts': scaler})['c']
        assert isinstance(scaled, AnnotatedTensor)
        assert list(scaled.annotation.labels) == ['c1', 'c2', 'task']

    def test_discrete_columns_pass_through(self):
        """Binary and categorical columns are class labels; scaling them would be
        meaningless, so they must come out bit-identical."""
        mixed = Annotations(
            labels=['flag', 'colour', 'value'], cardinalities=[1, 3, 1],
            types=['binary', 'categorical', 'continuous'],
        )
        data = torch.stack([
            torch.randint(0, 2, (N_SAMPLES,)).float(),
            torch.randint(0, 3, (N_SAMPLES,)).float(),
            torch.randn(N_SAMPLES) * 50 + 200,
        ], dim=1)
        c = AnnotatedTensor(data, mixed.to_concept_space(), axis=1)
        scaler = StandardScaler(axis=0)
        scaler.fit(c.continuous())
        # No loss/metrics here: this test only exercises maybe_scale_concepts,
        # which needs nothing but `self.scale_concepts`.
        model = ConceptBottleneckModel(
            input_size=INPUT_SIZE, annotations=mixed, task_names=['value'], lightning=True,
        )

        scaled = model.maybe_scale_concepts({'c': c}, {'concepts': scaler})['c']
        assert torch.equal(scaled.tensor[:, :2], data[:, :2])
        assert not torch.equal(scaled.tensor[:, 2], data[:, 2])

    def test_none_concepts(self, annotations, scaler):
        model = build_model(annotations=annotations)
        assert model.maybe_scale_concepts({'c': None}, {'concepts': scaler})['c'] is None

    def test_multiple_concept_keys_raise(self, annotations, scaler):
        model = build_model(annotations=annotations)
        with pytest.raises(NotImplementedError, match="multiple keys"):
            model.maybe_scale_concepts({'c': None, 'extra': None}, {'concepts': scaler})


class TestScaleInputs:
    def test_input_is_scaled(self, annotations):
        x = torch.randn(N_SAMPLES, INPUT_SIZE) * 20 + 5
        scaler = StandardScaler(axis=0)
        scaler.fit(x)
        model = build_model(annotations=annotations)
        scaled = model.maybe_scale_inputs({'x': x}, {'input': scaler})
        assert torch.allclose(scaled['x'].mean(0), torch.zeros(INPUT_SIZE), atol=1e-5)

    def test_other_input_keys_are_kept(self, annotations):
        x = torch.randn(N_SAMPLES, INPUT_SIZE)
        scaler = StandardScaler(axis=0)
        scaler.fit(x)
        model = build_model(annotations=annotations)
        scaled = model.maybe_scale_inputs({'x': x, 'extra': 'kept'}, {'input': scaler})
        assert scaled['extra'] == 'kept'


class TestTaskOnlyModel:
    """BlackBoxTaskOnly queries a strict subset of the fitted concepts, which
    `unscale_output`'s full-coverage assumption cannot handle — it must refuse
    rather than silently produce wrong values (see BaseLearner.unscale_output)."""

    def test_shared_step_runs_without_scaler(self, annotations, batch):
        model = self._model(annotations)
        full_batch = {**batch, 'scalers': {}}
        loss = model.shared_step(full_batch, 'test')
        assert torch.isfinite(loss)
        assert 'test/SUMMARY-continuous_mse' in model.test_metrics.compute()

    def test_shared_step_raises_with_scaler_and_scaling_on(self, annotations, batch, scaler):
        model = self._model(annotations)
        full_batch = {**batch, 'scalers': {'concepts': scaler}}
        with pytest.raises(NotImplementedError, match="does not support concept scaling"):
            model.shared_step(full_batch, 'test')

    def test_shared_step_runs_with_scaler_when_scale_concepts_false(self, annotations, batch, scaler):
        model = self._model(annotations, scale_concepts=False)
        full_batch = {**batch, 'scalers': {'concepts': scaler}}
        loss = model.shared_step(full_batch, 'test')
        assert torch.isfinite(loss)

    @staticmethod
    def _model(annotations, **kwargs):
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
            **kwargs,
        )
