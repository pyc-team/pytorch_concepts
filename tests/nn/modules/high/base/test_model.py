"""
Comprehensive tests for BaseModel class in torch_concepts.nn.modules.high.base.model
"""
import pytest
import torch
import torch.nn as nn
from torch_concepts.nn.modules.high.base.model import BaseModel
from torch_concepts.annotations import Annotations, AxisAnnotation
from torch_concepts.nn.modules.utils import GroupConfig

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

class DummyModel(BaseModel):
    def filter_output_for_loss(self, forward_out, target=None):
        return forward_out
    def filter_output_for_metrics(self, forward_out, target=None):
        return forward_out
    def forward(self, x):
        x = self.maybe_apply_backbone(x)
        x = self.latent_encoder(x)
        return x

def make_annotations():
    return Annotations({
        1: AxisAnnotation(
            metadata={
                'binary_concept': {'type': 'discrete'},
                'cat_concept': {'type': 'discrete'},
            },
            cardinalities=[1, 3],
            labels=['binary_concept', 'cat_concept']
        )
    })

def make_distributions():
    return GroupConfig(
        binary=torch.distributions.Bernoulli,
        categorical=torch.distributions.Categorical
    )

def test_init_with_backbone_and_latent_encoder():
    ann = make_annotations()
    dist = make_distributions()
    model = DummyModel(
        input_size=8,
        annotations=ann,
        variable_distributions=dist,
        backbone=DummyBackbone(),
        latent_encoder=DummyLatentEncoder,
        latent_encoder_kwargs={'hidden_size': 4}
    )
    assert isinstance(model.backbone, DummyBackbone)
    assert isinstance(model.latent_encoder, DummyLatentEncoder)
    assert model.latent_encoder.linear.in_features == 8
    assert model.latent_encoder.linear.out_features == 4
    assert hasattr(model, 'concept_annotations')
    assert model.concept_names == ['binary_concept', 'cat_concept']

def test_init_with_identity_encoder():
    ann = make_annotations()
    dist = make_distributions()
    model = DummyModel(
        input_size=8,
        annotations=ann,
        variable_distributions=dist,
        backbone=None,
        latent_encoder=None,
        latent_encoder_kwargs=None
    )
    assert model.backbone is None
    assert isinstance(model.latent_encoder, nn.Identity)
    assert model.latent_size == 8

def test_forward_pass():
    ann = make_annotations()
    dist = make_distributions()
    model = DummyModel(
        input_size=8,
        annotations=ann,
        variable_distributions=dist,
        backbone=DummyBackbone(),
        latent_encoder=DummyLatentEncoder,
        latent_encoder_kwargs={'hidden_size': 4}
    )
    x = torch.randn(2, 8)
    out = model(x)
    assert out.shape == (2, 4)

def test_repr():
    ann = make_annotations()
    dist = make_distributions()
    model = DummyModel(
        input_size=8,
        annotations=ann,
        variable_distributions=dist,
        backbone=DummyBackbone(),
        latent_encoder=DummyLatentEncoder,
        latent_encoder_kwargs={'hidden_size': 4}
    )
    rep = repr(model)
    assert 'DummyBackbone' in rep
    assert 'DummyLatentEncoder' in rep

def test_maybe_apply_backbone_none():
    ann = make_annotations()
    dist = make_distributions()
    model = DummyModel(
        input_size=8,
        annotations=ann,
        variable_distributions=dist,
        backbone=None,
        latent_encoder=None,
        latent_encoder_kwargs=None
    )
    x = torch.randn(2, 8)
    out = model.maybe_apply_backbone(x)
    assert torch.allclose(out, x)

def test_maybe_apply_backbone_callable():
    ann = make_annotations()
    dist = make_distributions()
    model = DummyModel(
        input_size=8,
        annotations=ann,
        variable_distributions=dist,
        backbone=DummyBackbone(),
        latent_encoder=None,
        latent_encoder_kwargs=None
    )
    x = torch.randn(2, 8)
    out = model.maybe_apply_backbone(x)
    assert out.shape == (2, 8)

def test_concept_annotations_distribution():
    ann = make_annotations()
    dist = make_distributions()
    model = DummyModel(
        input_size=8,
        annotations=ann,
        variable_distributions=dist,
        backbone=None,
        latent_encoder=None,
        latent_encoder_kwargs=None
    )
    meta = model.concept_annotations.metadata
    assert 'distribution' in meta['binary_concept']
    assert meta['binary_concept']['distribution'] == torch.distributions.Bernoulli
    assert meta['cat_concept']['distribution'] == torch.distributions.Categorical

def test_filter_output_for_loss_and_metric():
    ann = make_annotations()
    dist = make_distributions()
    model = DummyModel(
        input_size=8,
        annotations=ann,
        variable_distributions=dist,
        backbone=None,
        latent_encoder=None,
        latent_encoder_kwargs=None
    )
    x = torch.randn(2, 8)
    out = model(x)
    loss_out = model.filter_output_for_loss(out)
    metric_out = model.filter_output_for_metrics(out)
    assert torch.allclose(loss_out, out)
    assert torch.allclose(metric_out, out)
