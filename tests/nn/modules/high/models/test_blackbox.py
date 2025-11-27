"""
Comprehensive tests for BlackBox model in torch_concepts.nn.modules.high.models.blackbox
"""
import pytest
import torch
import torch.nn as nn
from torch_concepts.nn.modules.high.models.blackbox import BlackBox
from torch_concepts.annotations import AxisAnnotation, Annotations

class DummyBackbone(nn.Module):
    def __init__(self, out_features=8):
        super().__init__()
        self.out_features = out_features
    def forward(self, x):
        return torch.ones(x.shape[0], self.out_features)

class DummyLatentEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=4):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
    def forward(self, x):
        return self.linear(x)

def test_blackbox_init():
    ann = Annotations({
        1: AxisAnnotation(labels=['output'])
    })
    model = BlackBox(
        input_size=8,
        annotations=ann,
        backbone=DummyBackbone(),
        latent_encoder=DummyLatentEncoder,
        latent_encoder_kwargs={'hidden_size': 4}
    )
    assert isinstance(model.backbone, DummyBackbone)
    assert isinstance(model.latent_encoder, DummyLatentEncoder)
    assert model.latent_encoder.linear.in_features == 8
    assert model.latent_encoder.linear.out_features == 4

def test_blackbox_forward_shape():
    ann = Annotations({
        1: AxisAnnotation(labels=['output'])
    })
    model = BlackBox(
        input_size=8,
        annotations=ann,
        backbone=DummyBackbone(),
        latent_encoder=DummyLatentEncoder,
        latent_encoder_kwargs={'hidden_size': 4}
    )
    x = torch.randn(2, 8)
    out = model(x)
    assert out.shape == (2, 4)

def test_blackbox_filter_output_for_loss_and_metric():
    ann = Annotations({
        1: AxisAnnotation(labels=['output'])
    })
    model = BlackBox(
        input_size=8,
        annotations=ann,
        backbone=DummyBackbone(),
        latent_encoder=DummyLatentEncoder,
        latent_encoder_kwargs={'hidden_size': 4}
    )
    x = torch.randn(2, 8)
    out = model(x)
    target = torch.randint(0, 2, out.shape)
    loss_out = model.filter_output_for_loss(out, target)
    metric_out = model.filter_output_for_metrics(out, target)
    assert 'input' in loss_out and 'target' in loss_out
    assert 'preds' in metric_out and 'target' in metric_out
    assert torch.allclose(loss_out['input'], out)
    assert torch.allclose(loss_out['target'], target)
    assert torch.allclose(metric_out['preds'], out)
    assert torch.allclose(metric_out['target'], target)

def test_blackbox_repr():
    ann = Annotations({
        1: AxisAnnotation(labels=['output'])
    })
    model = BlackBox(
        input_size=8,
        annotations=ann,
        backbone=DummyBackbone(),
        latent_encoder=DummyLatentEncoder,
        latent_encoder_kwargs={'hidden_size': 4}
    )
    rep = repr(model)
    assert 'DummyBackbone' in rep
    assert 'BlackBox' in rep
