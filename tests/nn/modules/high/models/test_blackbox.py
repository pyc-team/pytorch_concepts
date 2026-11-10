"""Tests for :class:`~torch_concepts.nn.BlackBoxModel`.

The black box is a Bayesian network ``input → latent → outputs``. It inherits
``forward``, inference and the output contract from the concept models, so these
tests cover only what is specific to it: which labels it predicts, how it reports
them, and the learner hooks that score a task subset.
"""
import pytest
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy

from torch_concepts.annotations import Annotations
from torch_concepts.nn import MLP, BlackBoxModel, ConceptLoss, ConceptMetrics
from torch_concepts.nn.modules.high.base.learner import BaseLearner
from torch_concepts.tensor import AnnotatedTensor


LABELS = ['c1', 'c2', 'task']


@pytest.fixture
def ann():
    # binary, categorical(3), binary
    return Annotations(labels=LABELS, cardinalities=[1, 3, 1])


def _typed(types):
    return Annotations(labels=[f'c{i}' for i in range(len(types))],
                       cardinalities=[1] * len(types), types=types)


def _batch(ann, n=4):
    c = torch.stack([torch.randint(0, 2, (n,)), torch.randint(0, 3, (n,)),
                     torch.randint(0, 2, (n,))], dim=1).float()
    return {'inputs': {'x': torch.randn(n, 8)},
            'concepts': {'c': AnnotatedTensor(c, ann.to_concept_space(), axis=-1)}}


# -- construction -------------------------------------------------------------

def test_task_names_default_to_every_label(ann):
    assert BlackBoxModel(input_size=8, annotations=ann).task_names == LABELS


def test_task_names_accepts_a_string(ann):
    assert BlackBoxModel(input_size=8, annotations=ann, task_names='task').task_names == ['task']


def test_pgm_holds_only_the_predicted_labels(ann):
    model = BlackBoxModel(input_size=8, annotations=ann, task_names='task')
    members = [m for v in model.pgm.variables.values()
               if v.variable_type == 'concept' for m in v.members]
    assert members == ['task']


def test_backbone_and_latent_size(ann):
    model = BlackBoxModel(input_size=8, annotations=ann, backbone=MLP(8, 16), latent_size=16)
    assert model.latent_size == 16
    out = model(query=LABELS, input=torch.randn(2, 8))
    assert out.logits.shape == (2, 5)
    assert isinstance(BlackBoxModel(input_size=8, annotations=ann).backbone, nn.Identity)


def test_accepts_the_runner_graph_and_inference_args(ann):
    """conceptarium passes `graph` and the inference engines to every model."""
    from torch_concepts.nn import DeterministicInference
    model = BlackBoxModel(input_size=8, annotations=ann, graph=None,
                     inference=DeterministicInference, train_inference=DeterministicInference)
    assert model.eval_inference is not None and model.train_inference is not None


def test_unknown_argument_raises(ann):
    with pytest.raises(TypeError):
        BlackBoxModel(input_size=8, annotations=ann, not_a_real_argument=1)


def test_unknown_task_raises(ann):
    with pytest.raises((KeyError, ValueError)):
        BlackBoxModel(input_size=8, annotations=ann, task_names='nope')


# -- forward: same call and output contract as every other model ----------------

def test_query_subset_matches_full_output(ann):
    model = BlackBoxModel(input_size=8, annotations=ann).eval()
    x = torch.randn(3, 8)
    full = model(query=LABELS, input=x).logits
    assert full.shape == (3, 5)
    sub = model(query=['c2'], input=x).logits
    assert torch.allclose(sub.tensor, full['c2'].tensor)


def test_task_only_output(ann):
    out = BlackBoxModel(input_size=8, annotations=ann, task_names='task')(
        query=['task'], input=torch.randn(2, 8))
    assert list(out.logits.annotations.labels) == ['task']


def test_continuous_reports_loc_and_scale():
    out = BlackBoxModel(input_size=8, annotations=_typed(['binary', 'continuous']))(
        query=['c0', 'c1'], input=torch.randn(4, 8))
    assert list(out.logits.annotations.labels) == ['c0']
    assert list(out.loc.annotations.labels) == ['c1']
    assert out.scale.shape == (4, 1) and bool((out.scale > 0).all())


def test_continuous_loss_trains_the_head():
    ann = _typed(['continuous', 'continuous'])
    model = BlackBoxModel(input_size=8, annotations=ann)
    out = model(query=['c0', 'c1'], input=torch.randn(4, 8))
    loss = ConceptLoss(continuous=nn.MSELoss())(
        out, AnnotatedTensor(torch.randn(4, 2), ann.to_concept_space(), axis=-1))
    loss.backward()
    assert loss > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


# -- learner hooks --------------------------------------------------------------

def test_prepare_target_reads_the_tasks_by_name(ann):
    model = BlackBoxModel(input_size=8, annotations=ann, task_names='task')
    shuffled = Annotations(labels=list(reversed(LABELS)), cardinalities=[1, 3, 1])
    c = torch.arange(3).float().unsqueeze(0)
    target = model.prepare_target({'inputs': {}, 'concepts': {'c': AnnotatedTensor(
        c, shuffled.to_concept_space(), axis=-1)}})['c']
    assert float(target.tensor) == float(shuffled.labels.index('task'))


def test_lightning_flag(ann):
    assert not isinstance(BlackBoxModel(input_size=8, annotations=ann), BaseLearner)
    assert isinstance(BlackBoxModel(input_size=8, annotations=ann, lightning=True), BaseLearner)


@pytest.mark.parametrize('task_names', [None, 'task'])
def test_shared_step_with_per_concept_metrics(ann, task_names):
    """Per-concept metrics built on the full annotations must only score the
    labels the head predicts (setup_metrics), and the loss must see a target
    aligned to them (prepare_target)."""
    model = BlackBoxModel(
        input_size=8, annotations=ann, task_names=task_names, lightning=True,
        loss=ConceptLoss(binary=nn.BCEWithLogitsLoss(), categorical=nn.CrossEntropyLoss()),
        metrics=ConceptMetrics(annotations=ann, binary={'acc': BinaryAccuracy()},
                               categorical={'acc': MulticlassAccuracy(num_classes=3)},
                               per_concept=True),
        optim_class=torch.optim.AdamW, optim_kwargs={'lr': 1e-3},
    )
    for step in ('train', 'val'):
        assert torch.isfinite(model.shared_step(_batch(ann), step))
    keys = [k.split('/')[1] for k in model.val_metrics.compute()]
    scored = {k.split('_')[0] for k in keys if not k.startswith('SUMMARY')}
    assert scored == set(model.task_names)
    model.on_train_batch_end(outputs=None, batch=None, batch_idx=0)
    assert isinstance(model.configure_optimizers()['optimizer'], torch.optim.AdamW)
