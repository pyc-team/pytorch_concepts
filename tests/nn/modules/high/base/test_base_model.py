"""
Comprehensive tests for BaseModel abstract class.

Tests cover:
- Initialization with various configurations
- Backbone integration
- Latent encoder setup
- Annotation and distribution handling
- Properties and methods
- Forward pass functionality
"""
import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch_concepts.nn.modules.high.base.model import BaseModel
from torch_concepts.annotations import AxisAnnotation, Annotations
from torch_concepts.nn.modules.utils import GroupConfig


# Test Fixtures
class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""
    
    def forward(self, x, query=None):
        features = self.maybe_apply_backbone(x)
        latent = self.latent_encoder(features)
        return latent
    
    def filter_output_for_loss(self, forward_out, target=None):
        if target is None:
            return forward_out
        return {'input': forward_out, 'target': target}
    
    def filter_output_for_metrics(self, forward_out, target=None):
        if target is None:
            return forward_out
        return {'preds': forward_out, 'target': target}


class DummyBackbone(nn.Module):
    """Simple backbone for testing."""
    def __init__(self, in_features=100, out_features=20):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.out_features = out_features
    
    def forward(self, x):
        return self.linear(x)


class DummyLatentEncoder(nn.Module):
    """Simple encoder for testing."""
    def __init__(self, input_size, hidden_size=16):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.hidden_size = hidden_size
    
    def forward(self, x):
        return self.linear(x)


# Fixtures
@pytest.fixture
def annotations_with_distributions():
    """Annotations with distributions in metadata."""
    return Annotations({
        1: AxisAnnotation(
            labels=['c1', 'c2', 'task'],
            cardinalities=[1, 1, 1],
            metadata={
                'c1': {'type': 'discrete', 'distribution': Bernoulli},
                'c2': {'type': 'discrete', 'distribution': Bernoulli},
                'task': {'type': 'discrete', 'distribution': Bernoulli}
            }
        )
    })


@pytest.fixture
def annotations_without_distributions():
    """Annotations without distributions but with type metadata."""
    return Annotations({
        1: AxisAnnotation(
            labels=['c1', 'c2', 'task'],
            cardinalities=[1, 1, 1],
            metadata={
                'c1': {'type': 'discrete'},
                'c2': {'type': 'discrete'},
                'task': {'type': 'discrete'}
            }
        )
    })


@pytest.fixture
def mixed_annotations():
    """Annotations with mixed concept types."""
    return Annotations({
        1: AxisAnnotation(
            labels=['binary_c', 'cat_c'],
            cardinalities=[1, 3],
            metadata={
                'binary_c': {'type': 'discrete'},
                'cat_c': {'type': 'discrete'}
            }
        )
    })


@pytest.fixture
def variable_distributions_dict():
    """Variable distributions as dict."""
    return {
        'c1': Bernoulli,
        'c2': Bernoulli,
        'task': Bernoulli
    }


@pytest.fixture
def variable_distributions_groupconfig():
    """Variable distributions as GroupConfig."""
    return GroupConfig(
        binary=Bernoulli,
        categorical=Categorical
    )


# Initialization Tests
class TestBaseModelInitialization:
    """Test BaseModel initialization with various configurations."""
    
    def test_init_with_distributions_in_annotations(self, annotations_with_distributions):
        """Test initialization when distributions are in annotations."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        assert model.concept_names == ['c1', 'c2', 'task']
        assert model.concept_annotations.has_metadata('distribution')
        assert model.latent_size == 10  # No encoder, uses input_size
    
    def test_init_with_variable_distributions_dict(
        self, annotations_without_distributions, variable_distributions_dict
    ):
        """Test initialization with variable_distributions as dict."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_without_distributions,
            variable_distributions=variable_distributions_dict
        )
        
        assert model.concept_names == ['c1', 'c2', 'task']
        assert model.concept_annotations.has_metadata('distribution')
        meta = model.concept_annotations.metadata
        assert meta['c1']['distribution'] == Bernoulli
        assert meta['c2']['distribution'] == Bernoulli
        assert meta['task']['distribution'] == Bernoulli
    
    def test_init_with_variable_distributions_groupconfig(
        self, mixed_annotations, variable_distributions_groupconfig
    ):
        """Test initialization with variable_distributions as GroupConfig."""
        model = ConcreteModel(
            input_size=10,
            annotations=mixed_annotations,
            variable_distributions=variable_distributions_groupconfig
        )
        
        assert model.concept_names == ['binary_c', 'cat_c']
        assert model.concept_annotations.has_metadata('distribution')
        meta = model.concept_annotations.metadata
        assert meta['binary_c']['distribution'] == Bernoulli
        assert meta['cat_c']['distribution'] == Categorical
    
    def test_init_without_distributions_raises_error(self, annotations_without_distributions):
        """Test that missing distributions raises assertion error."""
        with pytest.raises(AssertionError, match="variable_distributions must be provided"):
            ConcreteModel(
                input_size=10,
                annotations=annotations_without_distributions
            )
    
    def test_init_with_latent_encoder_class(self, annotations_with_distributions):
        """Test initialization with latent encoder class and kwargs."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 64}
        )
        
        assert isinstance(model.latent_encoder, DummyLatentEncoder)
        assert model.latent_size == 64
        assert model.latent_encoder.linear.in_features == 10
        assert model.latent_encoder.linear.out_features == 64
    
    def test_init_with_latent_encoder_kwargs_only(self, annotations_with_distributions):
        """Test initialization with only latent encoder kwargs (uses MLP)."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            latent_encoder_kwargs={'hidden_size': 64, 'n_layers': 2}
        )
        
        assert model.latent_size == 64
        assert isinstance(model.latent_encoder, nn.Module)
        assert not isinstance(model.latent_encoder, nn.Identity)
    
    def test_init_without_latent_encoder_uses_identity(self, annotations_with_distributions):
        """Test that no encoder config results in Identity."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        assert isinstance(model.latent_encoder, nn.Identity)
        assert model.latent_size == 10


# Backbone Tests
class TestBaseModelBackbone:
    """Test backbone integration."""
    
    def test_model_with_backbone(self, annotations_with_distributions):
        """Test model with custom backbone."""
        backbone = DummyBackbone(in_features=100, out_features=20)
        model = ConcreteModel(
            input_size=20,
            annotations=annotations_with_distributions,
            backbone=backbone
        )
        
        assert model.backbone is not None
        assert model.backbone == backbone
        assert isinstance(model.backbone, DummyBackbone)
    
    def test_model_without_backbone(self, annotations_with_distributions):
        """Test model without backbone (pre-computed features)."""
        model = ConcreteModel(
            input_size=20,
            annotations=annotations_with_distributions,
            backbone=None
        )
        
        assert model.backbone is None
    
    def test_maybe_apply_backbone_with_backbone(self, annotations_with_distributions):
        """Test maybe_apply_backbone when backbone exists."""
        backbone = DummyBackbone(in_features=100, out_features=20)
        model = ConcreteModel(
            input_size=20,
            annotations=annotations_with_distributions,
            backbone=backbone
        )
        
        x = torch.randn(8, 100)
        features = model.maybe_apply_backbone(x)
        
        assert features.shape == (8, 20)
    
    def test_maybe_apply_backbone_without_backbone(self, annotations_with_distributions):
        """Test maybe_apply_backbone when no backbone."""
        model = ConcreteModel(
            input_size=20,
            annotations=annotations_with_distributions,
            backbone=None
        )
        
        x = torch.randn(8, 20)
        features = model.maybe_apply_backbone(x)
        
        # Should return input unchanged
        assert torch.equal(features, x)
    
    def test_maybe_apply_backbone_returns_tensor(self, annotations_with_distributions):
        """Test maybe_apply_backbone always returns a tensor."""
        backbone = DummyBackbone()
        model = ConcreteModel(
            input_size=20,
            annotations=annotations_with_distributions,
            backbone=backbone
        )
        
        x = torch.randn(4, 100)
        out = model.maybe_apply_backbone(x)
        
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == 4  # Batch dimension preserved


# Forward Pass Tests
class TestBaseModelForward:
    """Test forward pass functionality."""
    
    def test_forward_basic(self, annotations_with_distributions):
        """Test basic forward pass."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            latent_encoder_kwargs={'hidden_size': 16}
        )
        
        x = torch.randn(4, 10)
        out = model(x)
        
        assert out.shape == (4, 16)
        assert isinstance(out, torch.Tensor)
    
    def test_forward_with_backbone(self, annotations_with_distributions):
        """Test forward pass with backbone."""
        backbone = DummyBackbone(in_features=50, out_features=10)
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            backbone=backbone
        )
        
        x = torch.randn(4, 50)
        out = model(x)
        
        assert out.shape == (4, 10)
    
    def test_forward_with_backbone_and_encoder(self, annotations_with_distributions):
        """Test forward pass with both backbone and encoder."""
        backbone = DummyBackbone(in_features=100, out_features=20)
        model = ConcreteModel(
            input_size=20,
            annotations=annotations_with_distributions,
            backbone=backbone,
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 32}
        )
        
        x = torch.randn(8, 100)
        out = model(x)
        
        assert out.shape == (8, 32)
    
    def test_forward_preserves_batch_size(self, annotations_with_distributions):
        """Test forward pass preserves batch dimension."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            latent_encoder_kwargs={'hidden_size': 16}
        )
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 10)
            out = model(x)
            assert out.shape[0] == batch_size


# Filter Methods Tests
class TestBaseModelFilterMethods:
    """Test filter_output methods."""
    
    def test_filter_output_for_loss_with_target(self, annotations_with_distributions):
        """Test filter_output_for_loss returns correct format with target."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        forward_out = torch.randn(4, 3)
        target = torch.randint(0, 2, (4, 3)).float()
        
        filtered = model.filter_output_for_loss(forward_out, target)
        
        assert isinstance(filtered, dict)
        assert 'input' in filtered
        assert 'target' in filtered
        assert torch.equal(filtered['input'], forward_out)
        assert torch.equal(filtered['target'], target)
    
    def test_filter_output_for_loss_without_target(self, annotations_with_distributions):
        """Test filter_output_for_loss without target."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        forward_out = torch.randn(4, 3)
        filtered = model.filter_output_for_loss(forward_out)
        
        assert torch.equal(filtered, forward_out)
    
    def test_filter_output_for_metrics_with_target(self, annotations_with_distributions):
        """Test filter_output_for_metrics returns correct format with target."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        forward_out = torch.randn(4, 3)
        target = torch.randint(0, 2, (4, 3)).float()
        
        filtered = model.filter_output_for_metrics(forward_out, target)
        
        assert isinstance(filtered, dict)
        assert 'preds' in filtered
        assert 'target' in filtered
        assert torch.equal(filtered['preds'], forward_out)
        assert torch.equal(filtered['target'], target)
    
    def test_filter_output_for_metrics_without_target(self, annotations_with_distributions):
        """Test filter_output_for_metrics without target."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        forward_out = torch.randn(4, 3)
        filtered = model.filter_output_for_metrics(forward_out)
        
        assert torch.equal(filtered, forward_out)


# Properties Tests
class TestBaseModelProperties:
    """Test model properties and attributes."""
    
    def test_backbone_property(self, annotations_with_distributions):
        """Test backbone property."""
        backbone = DummyBackbone()
        model = ConcreteModel(
            input_size=20,
            annotations=annotations_with_distributions,
            backbone=backbone
        )
        
        assert model.backbone == backbone
        assert isinstance(model.backbone, nn.Module)
    
    def test_latent_encoder_property(self, annotations_with_distributions):
        """Test latent_encoder property."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            latent_encoder_kwargs={'hidden_size': 32}
        )
        
        assert isinstance(model.latent_encoder, nn.Module)
        assert hasattr(model.latent_encoder, 'forward')
    
    def test_concept_names_property(self, annotations_with_distributions):
        """Test concept_names attribute."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        assert model.concept_names == ['c1', 'c2', 'task']
        assert isinstance(model.concept_names, list)
    
    def test_concept_annotations_property(self, annotations_with_distributions):
        """Test concept_annotations attribute."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        assert hasattr(model, 'concept_annotations')
        assert isinstance(model.concept_annotations, AxisAnnotation)
        assert model.concept_annotations.has_metadata('distribution')
    
    def test_latent_size_property_with_encoder(self, annotations_with_distributions):
        """Test latent_size attribute with encoder."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            latent_encoder_kwargs={'hidden_size': 64}
        )
        
        assert model.latent_size == 64
    
    def test_latent_size_property_without_encoder(self, annotations_with_distributions):
        """Test latent_size attribute without encoder."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        assert model.latent_size == 10


# Representation Tests
class TestBaseModelRepr:
    """Test model string representation."""
    
    def test_repr_with_backbone(self, annotations_with_distributions):
        """Test __repr__ with backbone."""
        backbone = DummyBackbone()
        model = ConcreteModel(
            input_size=20,
            annotations=annotations_with_distributions,
            backbone=backbone
        )
        
        repr_str = repr(model)
        assert 'ConcreteModel' in repr_str
        assert 'DummyBackbone' in repr_str
    
    def test_repr_without_backbone(self, annotations_with_distributions):
        """Test __repr__ without backbone."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        repr_str = repr(model)
        assert 'ConcreteModel' in repr_str
        assert 'backbone=None' in repr_str
    
    def test_repr_with_encoder(self, annotations_with_distributions):
        """Test __repr__ with latent encoder."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 32}
        )
        
        repr_str = repr(model)
        assert 'DummyLatentEncoder' in repr_str
    
    def test_repr_contains_key_info(self, annotations_with_distributions):
        """Test __repr__ contains essential information."""
        backbone = DummyBackbone()
        model = ConcreteModel(
            input_size=20,
            annotations=annotations_with_distributions,
            backbone=backbone,
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 32}
        )
        
        repr_str = repr(model)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0


# Integration Tests
class TestBaseModelIntegration:
    """Test model integration scenarios."""
    
    def test_full_pipeline_with_all_components(self, annotations_with_distributions):
        """Test complete pipeline with backbone and encoder."""
        backbone = DummyBackbone(in_features=100, out_features=20)
        model = ConcreteModel(
            input_size=20,
            annotations=annotations_with_distributions,
            backbone=backbone,
            latent_encoder=DummyLatentEncoder,
            latent_encoder_kwargs={'hidden_size': 32}
        )
        
        # Forward pass
        x = torch.randn(8, 100)
        out = model(x)
        assert out.shape == (8, 32)
        
        # Filter for loss
        target = torch.randint(0, 2, (8, 3)).float()
        loss_input = model.filter_output_for_loss(out, target)
        assert isinstance(loss_input, dict)
        assert 'input' in loss_input and 'target' in loss_input
        
        # Filter for metrics
        metrics_input = model.filter_output_for_metrics(out, target)
        assert isinstance(metrics_input, dict)
        assert 'preds' in metrics_input and 'target' in metrics_input
    
    def test_minimal_model_pipeline(self, annotations_with_distributions):
        """Test minimal model with no backbone or encoder."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 10)
        
        # Check identity passthrough
        assert torch.equal(out, x)
    
    def test_gradient_flow(self, annotations_with_distributions):
        """Test gradients flow through the model."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            latent_encoder_kwargs={'hidden_size': 16}
        )
        
        x = torch.randn(4, 10, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
