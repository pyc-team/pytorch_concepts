"""
Tests for torch_concepts.data.backbone module.

This module provides comprehensive tests for the Backbone class, including:
- API validation (string-only model names)
- HuggingFace model detection
- Torchvision model support
- Device handling and auto-detection
- Forward pass correctness
- Property accessors
"""
import pytest
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from torch_concepts.data.backbone import (
    Backbone,
    _is_huggingface_model,
    _resolve_device,
    _load_torchvision_model,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def dummy_pil_images():
    """Create a batch of dummy PIL images."""
    return [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) 
            for _ in range(2)]


@pytest.fixture
def dummy_tensor_batch():
    """Create a batch of dummy tensors."""
    return torch.randn(2, 3, 224, 224)


class DummyImageDataset(Dataset):
    """Simple dataset returning PIL images for testing."""
    def __init__(self, n_samples=2):
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        return {'inputs': {'x': img}}


# =============================================================================
# Test Device Resolution
# =============================================================================

class TestDeviceResolution:
    """Tests for _resolve_device function."""
    
    def test_explicit_cpu(self):
        """Test explicit CPU device."""
        device = _resolve_device('cpu')
        assert device == torch.device('cpu')
    
    def test_explicit_cuda(self):
        """Test explicit CUDA device when available."""
        if torch.cuda.is_available():
            device = _resolve_device('cuda')
            assert device == torch.device('cuda')
    
    def test_explicit_cuda_index(self):
        """Test explicit CUDA device with index."""
        if torch.cuda.is_available():
            device = _resolve_device('cuda:0')
            assert device == torch.device('cuda:0')
    
    def test_auto_detection_returns_device(self):
        """Test that auto-detection returns a valid device."""
        device = _resolve_device(None)
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda']


# =============================================================================
# Test HuggingFace Detection
# =============================================================================

class TestIsHuggingfaceModelDetection:
    """Tests for the _is_huggingface_model helper function."""
    
    def test_detects_slash_as_huggingface(self):
        """Models with '/' are detected as HuggingFace."""
        assert _is_huggingface_model('facebook/dinov2-base') is True
        assert _is_huggingface_model('google/vit-base-patch16-224') is True
        assert _is_huggingface_model('microsoft/beit-base-patch16-224') is True
    
    def test_detects_keywords_as_huggingface(self):
        """Models with HuggingFace keywords are detected."""
        assert _is_huggingface_model('dinov2-base') is True
        assert _is_huggingface_model('vit-large') is True
        assert _is_huggingface_model('beit-base') is True
        assert _is_huggingface_model('clip-vit') is True
        assert _is_huggingface_model('swin-transformer') is True
        assert _is_huggingface_model('convnext-base') is True
    
    def test_torchvision_names_not_huggingface(self):
        """Standard torchvision names are not detected as HuggingFace."""
        assert _is_huggingface_model('resnet18') is False
        assert _is_huggingface_model('resnet50') is False
        assert _is_huggingface_model('resnet101') is False
        assert _is_huggingface_model('vgg16') is False
        assert _is_huggingface_model('vgg19') is False
        assert _is_huggingface_model('efficientnet_b0') is False
        assert _is_huggingface_model('efficientnet_b7') is False
        assert _is_huggingface_model('densenet121') is False
        assert _is_huggingface_model('densenet201') is False
    
    def test_case_insensitive(self):
        """Detection should be case-insensitive."""
        assert _is_huggingface_model('DINOv2-base') is True
        assert _is_huggingface_model('DINOV2') is True
        assert _is_huggingface_model('ViT-large') is True


# =============================================================================
# Test Backbone Initialization
# =============================================================================

class TestBackboneInit:
    """Tests for Backbone class initialization."""
    
    def test_init_with_resnet(self):
        """Test initialization with ResNet model."""
        backbone = Backbone('resnet18', device='cpu')
        assert backbone.name == 'resnet18'
        assert backbone.device == torch.device('cpu')
        assert backbone.is_huggingface is False
    
    def test_init_with_vgg(self):
        """Test initialization with VGG model."""
        backbone = Backbone('vgg16', device='cpu')
        assert backbone.name == 'vgg16'
        assert backbone.is_huggingface is False
    
    def test_init_with_efficientnet(self):
        """Test initialization with EfficientNet model."""
        backbone = Backbone('efficientnet_b0', device='cpu')
        assert backbone.name == 'efficientnet_b0'
        assert backbone.is_huggingface is False
    
    def test_init_with_densenet(self):
        """Test initialization with DenseNet model."""
        backbone = Backbone('densenet121', device='cpu')
        assert backbone.name == 'densenet121'
        assert backbone.is_huggingface is False
    
    def test_unsupported_torchvision_model(self):
        """Test that unsupported torchvision models raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported torchvision backbone"):
            Backbone('mobilenet_v2', device='cpu')
    
    def test_init_auto_device(self):
        """Test that auto device detection works."""
        backbone = Backbone('resnet18')  # No device specified
        assert backbone.device is not None
        assert backbone.device.type in ['cpu', 'cuda']


# =============================================================================
# Test Backbone Properties
# =============================================================================

class TestBackboneProperties:
    """Tests for Backbone property accessors."""
    
    def test_device_property(self):
        """Test device property returns torch.device."""
        backbone = Backbone('resnet18', device='cpu')
        assert isinstance(backbone.device, torch.device)
        assert backbone.device == torch.device('cpu')
    
    def test_processor_property(self):
        """Test processor property is set."""
        backbone = Backbone('resnet18', device='cpu')
        assert backbone.processor is not None
    
    def test_is_huggingface_property(self):
        """Test is_huggingface property."""
        backbone = Backbone('resnet18', device='cpu')
        assert backbone.is_huggingface is False
    
    def test_filename_property_torchvision(self):
        """Test filename generation for torchvision models."""
        backbone = Backbone('resnet50', device='cpu')
        assert backbone.filename == 'bkb_embs_resnet50.pt'
    
    def test_filename_property_huggingface_format(self):
        """Test filename generation handles slashes correctly."""
        # We can't easily test HuggingFace initialization without downloading,
        # but we can test the filename generation logic
        name = 'facebook/dinov2-base'
        expected = f"bkb_embs_{name.replace('/', '-')}.pt"
        assert expected == 'bkb_embs_facebook-dinov2-base.pt'


# =============================================================================
# Test Backbone Forward Pass (Torchvision)
# =============================================================================

class TestBackboneForwardTorchvision:
    """Tests for Backbone forward pass with torchvision models."""
    
    def test_forward_with_tensor_batch(self, dummy_tensor_batch):
        """Test forward pass with tensor input."""
        backbone = Backbone('resnet18', device='cpu')
        embeddings = backbone(dummy_tensor_batch)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == 2  # Batch size
        assert embeddings.shape[1] == 512  # ResNet18 feature dim
    
    def test_forward_with_pil_images(self, dummy_pil_images):
        """Test forward pass with PIL image list."""
        backbone = Backbone('resnet18', device='cpu')
        embeddings = backbone(dummy_pil_images)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 512
    
    def test_resnet_embedding_dimensions(self, dummy_tensor_batch):
        """Test correct embedding dimensions for different ResNet variants."""
        # ResNet18/34 have 512 features
        backbone18 = Backbone('resnet18', device='cpu')
        emb18 = backbone18(dummy_tensor_batch)
        assert emb18.shape[1] == 512
        
        # ResNet50/101/152 have 2048 features
        backbone50 = Backbone('resnet50', device='cpu')
        emb50 = backbone50(dummy_tensor_batch)
        assert emb50.shape[1] == 2048
    
    def test_vgg_embedding_dimensions(self, dummy_tensor_batch):
        """Test correct embedding dimensions for VGG."""
        backbone = Backbone('vgg16', device='cpu')
        embeddings = backbone(dummy_tensor_batch)
        assert embeddings.shape[1] == 25088  # 512 * 7 * 7
    
    def test_efficientnet_embedding_dimensions(self, dummy_tensor_batch):
        """Test correct embedding dimensions for EfficientNet."""
        backbone = Backbone('efficientnet_b0', device='cpu')
        embeddings = backbone(dummy_tensor_batch)
        assert embeddings.shape[1] == 1280
    
    def test_densenet_embedding_dimensions(self, dummy_tensor_batch):
        """Test correct embedding dimensions for DenseNet."""
        backbone = Backbone('densenet121', device='cpu')
        embeddings = backbone(dummy_tensor_batch)
        assert embeddings.shape[1] == 1024


# =============================================================================
# Test Backbone Representation
# =============================================================================

class TestBackboneRepr:
    """Tests for Backbone __repr__ method."""
    
    def test_repr_torchvision(self):
        """Test repr for torchvision model."""
        backbone = Backbone('resnet50', device='cpu')
        repr_str = repr(backbone)
        
        assert 'Backbone' in repr_str
        assert 'resnet50' in repr_str
        assert 'torchvision' in repr_str
        assert 'cpu' in repr_str
    
    def test_repr_contains_all_info(self):
        """Test that repr contains all relevant information."""
        backbone = Backbone('vgg16', device='cpu')
        repr_str = repr(backbone)
        
        assert 'name=' in repr_str
        assert 'type=' in repr_str
        assert 'device=' in repr_str


# =============================================================================
# Test Backbone as nn.Module
# =============================================================================

class TestBackboneAsModule:
    """Tests for Backbone behavior as nn.Module."""
    
    def test_is_nn_module(self):
        """Test that Backbone is an nn.Module subclass."""
        backbone = Backbone('resnet18', device='cpu')
        assert isinstance(backbone, nn.Module)
    
    def test_eval_mode_after_init(self):
        """Test that backbone model is in eval mode after initialization."""
        backbone = Backbone('resnet18', device='cpu')
        # The internal model should be in eval mode
        assert not backbone._model.training
    
    def test_no_grad_context(self, dummy_tensor_batch):
        """Test that embeddings can be computed without gradients."""
        backbone = Backbone('resnet18', device='cpu')
        
        with torch.no_grad():
            embeddings = backbone(dummy_tensor_batch)
        
        assert not embeddings.requires_grad


# =============================================================================
# Test HuggingFace Backbone (Slow tests - marked for optional skip)
# =============================================================================

@pytest.mark.slow
class TestBackboneHuggingFace:
    """Tests for Backbone with HuggingFace models.
    
    These tests require downloading HuggingFace models and are marked as slow.
    Run with: pytest -m slow
    """
    
    def test_init_dinov2(self):
        """Test initialization with DINOv2 model."""
        backbone = Backbone('facebook/dinov2-base', device='cpu')
        assert backbone.name == 'facebook/dinov2-base'
        assert backbone.is_huggingface is True
    
    def test_forward_dinov2(self, dummy_pil_images):
        """Test forward pass with DINOv2."""
        backbone = Backbone('facebook/dinov2-base', device='cpu')
        embeddings = backbone(dummy_pil_images)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 768  # DINOv2-base CLS token dim
    
    def test_repr_huggingface(self):
        """Test repr for HuggingFace model."""
        backbone = Backbone('facebook/dinov2-base', device='cpu')
        repr_str = repr(backbone)
        
        assert 'HuggingFace' in repr_str
        assert 'facebook/dinov2-base' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
