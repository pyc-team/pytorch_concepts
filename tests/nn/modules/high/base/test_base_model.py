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
from torch.distributions import Bernoulli, OneHotCategorical, RelaxedBernoulli
from torch_concepts.nn.modules.high.base.model import BaseModel
from torch_concepts.nn.modules.low.dense_layers import MLP
from torch_concepts.annotations import AxisAnnotation, Annotations
from torch_concepts.nn.modules.utils import GroupConfig
from torch_concepts.utils import add_distribution_to_annotations, add_activation_to_annotations


# Test Fixtures
class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""

    def forward(self, x, query=None):
        return self.backbone(x)


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
                'c1': {'type': 'discrete'},
                'c2': {'type': 'discrete'},
                'task': {'type': 'discrete'}
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





# Initialization Tests
class TestBaseModelInitialization:
    """Test BaseModel initialization with various configurations."""
    
    def test_init_defaults(self, annotations_with_distributions):
        """Test initialization fills default distributions and activations."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        assert model.concept_names == ['c1', 'c2', 'task']
        ann = model.concept_annotations
        assert ann.distributions is not None
        assert ann.concept('c1').distribution == Bernoulli

    def test_init_with_variable_distributions_dict(
        self, annotations_without_distributions
    ):
        """Test initialization with variable_distributions dict passed to constructor."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_without_distributions,
            variable_distributions={
                'c1': RelaxedBernoulli,
                'c2': RelaxedBernoulli,
                'task': Bernoulli,
            },
        )
        
        assert model.concept_names == ['c1', 'c2', 'task']
        ann = model.concept_annotations
        assert ann.distributions is not None
        assert ann.concept('c1').distribution == RelaxedBernoulli
        assert model.latent_size == 10  # No encoder, uses input_size
        assert ann.concept('c2').distribution == RelaxedBernoulli
        assert ann.concept('task').distribution == Bernoulli
    
    def test_init_with_variable_distributions_groupconfig(
        self, mixed_annotations
    ):
        """Test initialization with variable_distributions as GroupConfig passed to constructor."""
        model = ConcreteModel(
            input_size=10,
            annotations=mixed_annotations,
            variable_distributions=GroupConfig(
                binary=RelaxedBernoulli,
                categorical=OneHotCategorical,
            ),
        )
        
        assert model.concept_names == ['binary_c', 'cat_c']
        ann = model.concept_annotations
        assert ann.distributions is not None
        assert ann.concept('binary_c').distribution == RelaxedBernoulli
        assert ann.concept('cat_c').distribution == OneHotCategorical
    
    def test_init_without_distributions_uses_defaults(self, annotations_without_distributions):
        """Test that missing distributions are filled with defaults (Bernoulli for binary discrete)."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_without_distributions
        )
        ann = model.concept_annotations
        assert ann.distributions is not None
        assert ann.concept('c1').distribution == Bernoulli
        assert ann.concept('c2').distribution == Bernoulli
        assert ann.concept('task').distribution == Bernoulli
    
    def test_init_with_backbone_class(self, annotations_with_distributions):
        """Test initialization with a custom backbone instance."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            backbone=DummyLatentEncoder(10, hidden_size=64),
            latent_size=64,
        )

        assert isinstance(model.backbone, DummyLatentEncoder)
        assert model.latent_size == 64
        assert model.backbone.linear.in_features == 10
        assert model.backbone.linear.out_features == 64

    def test_init_with_mlp_backbone(self, annotations_with_distributions):
        """Test initialization with an MLP backbone."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            backbone=MLP(input_size=10, hidden_size=64, n_layers=2),
            latent_size=64,
        )

        assert model.latent_size == 64
        assert isinstance(model.backbone, nn.Module)
        assert not isinstance(model.backbone, nn.Identity)

    def test_init_without_backbone_uses_identity(self, annotations_with_distributions):
        """Test that no backbone results in Identity."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )

        assert isinstance(model.backbone, nn.Identity)
        assert model.latent_size == 10


# Backbone Tests
class TestBaseModelBackbone:
    """Test backbone integration."""
    
    def test_model_with_backbone(self, annotations_with_distributions):
        """Test model with custom backbone."""
        backbone = DummyBackbone(in_features=100, out_features=20)
        model = ConcreteModel(
            input_size=100,
            annotations=annotations_with_distributions,
            backbone=backbone,
            latent_size=20,
        )

        assert model.backbone is not None
        assert model.backbone == backbone
        assert isinstance(model.backbone, DummyBackbone)

    def test_model_without_backbone(self, annotations_with_distributions):
        """Test model without backbone (backbone defaults to Identity)."""
        model = ConcreteModel(
            input_size=20,
            annotations=annotations_with_distributions,
            backbone=None
        )

        assert isinstance(model.backbone, nn.Identity)

    def test_backbone_applied_with_backbone(self, annotations_with_distributions):
        """Test that the backbone transforms the input."""
        backbone = DummyBackbone(in_features=100, out_features=20)
        model = ConcreteModel(
            input_size=100,
            annotations=annotations_with_distributions,
            backbone=backbone,
            latent_size=20,
        )

        x = torch.randn(8, 100)
        features = model.backbone(x)

        assert features.shape == (8, 20)

    def test_backbone_identity_without_backbone(self, annotations_with_distributions):
        """Test that the default Identity backbone returns input unchanged."""
        model = ConcreteModel(
            input_size=20,
            annotations=annotations_with_distributions,
            backbone=None
        )

        x = torch.randn(8, 20)
        features = model.backbone(x)

        # Should return input unchanged
        assert torch.equal(features, x)

    def test_backbone_returns_tensor(self, annotations_with_distributions):
        """Test the backbone always returns a tensor."""
        backbone = DummyBackbone()
        model = ConcreteModel(
            input_size=100,
            annotations=annotations_with_distributions,
            backbone=backbone,
            latent_size=20,
        )

        x = torch.randn(4, 100)
        out = model.backbone(x)

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
            backbone=MLP(input_size=10, hidden_size=16),
            latent_size=16,
        )

        x = torch.randn(4, 10)
        out = model(x)

        assert out.shape == (4, 16)
        assert isinstance(out, torch.Tensor)

    def test_forward_with_backbone(self, annotations_with_distributions):
        """Test forward pass with backbone."""
        backbone = DummyBackbone(in_features=50, out_features=10)
        model = ConcreteModel(
            input_size=50,
            annotations=annotations_with_distributions,
            backbone=backbone,
            latent_size=10,
        )

        x = torch.randn(4, 50)
        out = model(x)

        assert out.shape == (4, 10)

    def test_forward_with_backbone_maps_to_latent(self, annotations_with_distributions):
        """Test forward pass with a backbone mapping raw input to the latent."""
        backbone = DummyBackbone(in_features=100, out_features=32)
        model = ConcreteModel(
            input_size=100,
            annotations=annotations_with_distributions,
            backbone=backbone,
            latent_size=32,
        )

        x = torch.randn(8, 100)
        out = model(x)

        assert out.shape == (8, 32)

    def test_forward_preserves_batch_size(self, annotations_with_distributions):
        """Test forward pass preserves batch dimension."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            backbone=MLP(input_size=10, hidden_size=16),
            latent_size=16,
        )

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 10)
            out = model(x)
            assert out.shape[0] == batch_size


# Prepare Target Tests
class TestBaseModelPrepareTarget:
    """Test prepare_target method."""
    
    def test_prepare_target_returns_identity(self, annotations_with_distributions):
        """Test prepare_target returns target unchanged for base models."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )
        
        target = torch.randint(0, 2, (4, 3)).float()
        prepared = model.prepare_target(target)
        assert torch.equal(prepared, target)


# Properties Tests
class TestBaseModelProperties:
    """Test model properties and attributes."""
    
    def test_backbone_property(self, annotations_with_distributions):
        """Test backbone property."""
        backbone = DummyBackbone()
        model = ConcreteModel(
            input_size=100,
            annotations=annotations_with_distributions,
            backbone=backbone,
            latent_size=20,
        )

        assert model.backbone == backbone
        assert isinstance(model.backbone, nn.Module)

    def test_backbone_property_with_module(self, annotations_with_distributions):
        """Test backbone property is a callable nn.Module."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            backbone=MLP(input_size=10, hidden_size=32),
            latent_size=32,
        )

        assert isinstance(model.backbone, nn.Module)
        assert hasattr(model.backbone, 'forward')
    
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
        assert model.concept_annotations.distributions is not None
    
    def test_latent_size_property_with_backbone(self, annotations_with_distributions):
        """Test latent_size attribute with backbone."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            backbone=MLP(input_size=10, hidden_size=64),
            latent_size=64,
        )

        assert model.latent_size == 64

    def test_latent_size_property_without_backbone(self, annotations_with_distributions):
        """Test latent_size attribute without backbone."""
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
            input_size=100,
            annotations=annotations_with_distributions,
            backbone=backbone,
            latent_size=20,
        )

        repr_str = repr(model)
        assert 'ConcreteModel' in repr_str
        assert 'backbone=DummyBackbone' in repr_str
        assert 'latent_encoder=' not in repr_str

    def test_repr_without_backbone(self, annotations_with_distributions):
        """Test __repr__ without backbone (defaults to Identity)."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions
        )

        repr_str = repr(model)
        assert 'ConcreteModel' in repr_str
        assert 'backbone=Identity' in repr_str
        assert 'latent_encoder=' not in repr_str

    def test_repr_with_mlp_backbone(self, annotations_with_distributions):
        """Test __repr__ with an MLP backbone."""
        model = ConcreteModel(
            input_size=10,
            annotations=annotations_with_distributions,
            backbone=MLP(input_size=10, hidden_size=32),
            latent_size=32,
        )

        repr_str = repr(model)
        assert 'backbone=MLP' in repr_str

    def test_repr_contains_key_info(self, annotations_with_distributions):
        """Test __repr__ contains essential information."""
        backbone = DummyBackbone()
        model = ConcreteModel(
            input_size=100,
            annotations=annotations_with_distributions,
            backbone=backbone,
            latent_size=32,
        )

        repr_str = repr(model)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0


# Integration Tests
class TestBaseModelIntegration:
    """Test model integration scenarios."""
    
    def test_full_pipeline_with_all_components(self, annotations_with_distributions):
        """Test complete pipeline with a backbone mapping input to latent."""
        backbone = DummyBackbone(in_features=100, out_features=32)
        model = ConcreteModel(
            input_size=100,
            annotations=annotations_with_distributions,
            backbone=backbone,
            latent_size=32,
        )

        # Forward pass
        x = torch.randn(8, 100)
        out = model(x)
        assert out.shape == (8, 32)

        # prepare_target returns identity for base models
        target = torch.randint(0, 2, (8, 3)).float()
        prepared = model.prepare_target(target)
        assert torch.equal(prepared, target)
    
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
            backbone=MLP(input_size=10, hidden_size=16),
            latent_size=16,
        )

        x = torch.randn(4, 10, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
