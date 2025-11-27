"""
Comprehensive tests for BaseModel abstract class and its core functionality.

Tests cover:
- Initialization with various configurations
- Backbone integration
- Latent encoder setup
- Annotation and distribution handling
- Properties and methods
"""
import unittest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical
from torch_concepts.nn.modules.high.base.model import BaseModel
from torch_concepts.annotations import AxisAnnotation, Annotations
from torch_concepts.nn.modules.utils import GroupConfig


class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""
    
    def forward(self, x, query=None):
        features = self.maybe_apply_backbone(x)
        latent = self.latent_encoder(features)
        return latent
    
    def filter_output_for_loss(self, forward_out, target):
        return {'input': forward_out, 'target': target}
    
    def filter_output_for_metrics(self, forward_out, target):
        return {'preds': forward_out, 'target': target}


class TestBaseModelInitialization(unittest.TestCase):
    """Test BaseModel initialization with various configurations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Annotations with distributions in metadata
        self.ann_with_dist = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'task'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'task': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        # Annotations without distributions (but with type metadata)
        self.ann_no_dist = Annotations({
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
        
        self.variable_distributions = {
            'c1': Bernoulli,
            'c2': Bernoulli,
            'task': Bernoulli
        }
    
    def test_init_with_distributions_in_annotations(self):
        """Test initialization when distributions are in annotations."""
        model = ConcreteModel(
            input_size=10,
            annotations=self.ann_with_dist
        )
        
        self.assertEqual(model.concept_names, ['c1', 'c2', 'task'])
        self.assertTrue(model.concept_annotations.has_metadata('distribution'))
        self.assertEqual(model.latent_size, 10)  # No encoder, uses input_size
    
    def test_init_with_variable_distributions(self):
        """Test initialization with variable_distributions parameter."""
        model = ConcreteModel(
            input_size=10,
            annotations=self.ann_no_dist,
            variable_distributions=self.variable_distributions
        )
        
        self.assertEqual(model.concept_names, ['c1', 'c2', 'task'])
        self.assertTrue(model.concept_annotations.has_metadata('distribution'))
    
    def test_init_without_distributions_raises_error(self):
        """Test that missing distributions raises assertion error."""
        with self.assertRaises(AssertionError) as context:
            ConcreteModel(
                input_size=10,
                annotations=self.ann_no_dist
            )
        self.assertIn("variable_distributions must be provided", str(context.exception))
    
    def test_init_with_latent_encoder_kwargs(self):
        """Test initialization with latent encoder configuration."""
        model = ConcreteModel(
            input_size=10,
            annotations=self.ann_with_dist,
            latent_encoder_kwargs={'hidden_size': 64, 'n_layers': 2}
        )
        
        self.assertEqual(model.latent_size, 64)
        self.assertIsInstance(model.latent_encoder, nn.Module)
    
    def test_init_without_latent_encoder_uses_identity(self):
        """Test that no encoder config results in Identity."""
        model = ConcreteModel(
            input_size=10,
            annotations=self.ann_with_dist
        )
        
        self.assertIsInstance(model.latent_encoder, nn.Identity)
        self.assertEqual(model.latent_size, 10)


class TestBaseModelBackbone(unittest.TestCase):
    """Test backbone integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        # Simple backbone
        self.backbone = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20)
        )
    
    def test_model_with_backbone(self):
        """Test model with custom backbone."""
        model = ConcreteModel(
            input_size=20,  # Backbone output size
            annotations=self.ann,
            backbone=self.backbone
        )
        
        self.assertIsNotNone(model.backbone)
        self.assertEqual(model.backbone, self.backbone)
    
    def test_model_without_backbone(self):
        """Test model without backbone (pre-computed features)."""
        model = ConcreteModel(
            input_size=20,
            annotations=self.ann,
            backbone=None
        )
        
        self.assertIsNone(model.backbone)
    
    def test_maybe_apply_backbone_with_backbone(self):
        """Test maybe_apply_backbone when backbone exists."""
        model = ConcreteModel(
            input_size=20,
            annotations=self.ann,
            backbone=self.backbone
        )
        
        x = torch.randn(8, 100)
        features = model.maybe_apply_backbone(x)
        
        self.assertEqual(features.shape, (8, 20))
    
    def test_maybe_apply_backbone_without_backbone(self):
        """Test maybe_apply_backbone when no backbone."""
        model = ConcreteModel(
            input_size=20,
            annotations=self.ann,
            backbone=None
        )
        
        x = torch.randn(8, 20)
        features = model.maybe_apply_backbone(x)
        
        # Should return input unchanged
        self.assertTrue(torch.equal(features, x))


class TestBaseModelForward(unittest.TestCase):
    """Test forward pass functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2', 'c3'],
                cardinalities=[1, 1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli},
                    'c3': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        model = ConcreteModel(
            input_size=10,
            annotations=self.ann,
            latent_encoder_kwargs={'hidden_size': 16}
        )
        
        x = torch.randn(4, 10)
        out = model(x)
        
        self.assertEqual(out.shape, (4, 16))
    
    def test_forward_with_backbone(self):
        """Test forward pass with backbone."""
        backbone = nn.Linear(50, 10)
        model = ConcreteModel(
            input_size=10,
            annotations=self.ann,
            backbone=backbone
        )
        
        x = torch.randn(4, 50)
        out = model(x)
        
        self.assertEqual(out.shape, (4, 10))


class TestBaseModelFilterMethods(unittest.TestCase):
    """Test filter_output methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
        
        self.model = ConcreteModel(
            input_size=10,
            annotations=self.ann
        )
    
    def test_filter_output_for_loss(self):
        """Test filter_output_for_loss returns correct format."""
        forward_out = torch.randn(4, 2)
        target = torch.randint(0, 2, (4, 2)).float()
        
        filtered = self.model.filter_output_for_loss(forward_out, target)
        
        self.assertIsInstance(filtered, dict)
        self.assertIn('input', filtered)
        self.assertIn('target', filtered)
        self.assertTrue(torch.equal(filtered['input'], forward_out))
        self.assertTrue(torch.equal(filtered['target'], target))
    
    def test_filter_output_for_metrics(self):
        """Test filter_output_for_metrics returns correct format."""
        forward_out = torch.randn(4, 2)
        target = torch.randint(0, 2, (4, 2)).float()
        
        filtered = self.model.filter_output_for_metrics(forward_out, target)
        
        self.assertIsInstance(filtered, dict)
        self.assertIn('preds', filtered)
        self.assertIn('target', filtered)
        self.assertTrue(torch.equal(filtered['preds'], forward_out))
        self.assertTrue(torch.equal(filtered['target'], target))


class TestBaseModelProperties(unittest.TestCase):
    """Test model properties."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ann = Annotations({
            1: AxisAnnotation(
                labels=['c1', 'c2'],
                cardinalities=[1, 1],
                metadata={
                    'c1': {'type': 'binary', 'distribution': Bernoulli},
                    'c2': {'type': 'binary', 'distribution': Bernoulli}
                }
            )
        })
    
    def test_backbone_property(self):
        """Test backbone property."""
        backbone = nn.Linear(10, 5)
        model = ConcreteModel(
            input_size=5,
            annotations=self.ann,
            backbone=backbone
        )
        
        self.assertEqual(model.backbone, backbone)
    
    def test_latent_encoder_property(self):
        """Test latent_encoder property."""
        model = ConcreteModel(
            input_size=10,
            annotations=self.ann,
            latent_encoder_kwargs={'hidden_size': 32}
        )
        
        self.assertIsInstance(model.latent_encoder, nn.Module)
    
    def test_concept_names_property(self):
        """Test concept_names attribute."""
        model = ConcreteModel(
            input_size=10,
            annotations=self.ann
        )
        
        self.assertEqual(model.concept_names, ['c1', 'c2'])
    
    def test_latent_size_property(self):
        """Test latent_size attribute."""
        model = ConcreteModel(
            input_size=10,
            annotations=self.ann,
            latent_encoder_kwargs={'hidden_size': 64}
        )
        
        self.assertEqual(model.latent_size, 64)


class TestBaseModelRepr(unittest.TestCase):
    """Test model string representation."""
    
    def test_repr_with_backbone(self):
        """Test __repr__ with backbone."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1'],
                cardinalities=[1],
                metadata={'c1': {'type': 'binary', 'distribution': Bernoulli}}
            )
        })
        
        backbone = nn.Linear(10, 5)
        model = ConcreteModel(
            input_size=5,
            annotations=ann,
            backbone=backbone
        )
        
        repr_str = repr(model)
        self.assertIn('ConcreteModel', repr_str)
        self.assertIn('backbone=Linear', repr_str)
    
    def test_repr_without_backbone(self):
        """Test __repr__ without backbone."""
        ann = Annotations({
            1: AxisAnnotation(
                labels=['c1'],
                cardinalities=[1],
                metadata={'c1': {'type': 'binary', 'distribution': Bernoulli}}
            )
        })
        
        model = ConcreteModel(
            input_size=10,
            annotations=ann
        )
        
        repr_str = repr(model)
        self.assertIn('ConcreteModel', repr_str)
        self.assertIn('backbone=None', repr_str)


if __name__ == '__main__':
    unittest.main()
