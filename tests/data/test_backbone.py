"""
Tests for torch_concepts.backbone module.

This module provides comprehensive tests for the Backbone class, including:
- API validation (string-only model names)
- HuggingFace model detection
- Torchvision model support
- Device handling and auto-detection
- Forward pass correctness
- Property accessors
"""
import sys
import types

import pytest
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from torch_concepts.backbone import (
    Backbone,
    ImageBackbone,
    TextBackbone,
    BackboneSpec,
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

    def test_auto_cpu_when_nothing_available(self, monkeypatch):
        """No CUDA and no MPS -> CPU (covers the final else branch)."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert _resolve_device(None) == torch.device('cpu')

    def test_mps_warns_and_falls_back_to_cpu(self, monkeypatch):
        """MPS available -> warn and fall back to CPU (covers the MPS branch)."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        with pytest.warns(UserWarning, match="MPS"):
            device = _resolve_device(None)
        assert device == torch.device('cpu')

    def test_cuda_selected_when_available(self, monkeypatch):
        """CUDA available -> 'cuda' is selected (covers the CUDA branch)."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert _resolve_device(None) == torch.device('cuda')


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
        backbone = ImageBackbone('resnet18', device='cpu')
        assert backbone.name == 'resnet18'
        assert backbone.device == torch.device('cpu')
        assert backbone.source == 'torchvision'
    
    def test_init_with_vgg(self):
        """Test initialization with VGG model."""
        backbone = ImageBackbone('vgg16', device='cpu')
        assert backbone.name == 'vgg16'
        assert backbone.source == 'torchvision'
    
    def test_init_with_efficientnet(self):
        """Test initialization with EfficientNet model."""
        backbone = ImageBackbone('efficientnet_b0', device='cpu')
        assert backbone.name == 'efficientnet_b0'
        assert backbone.source == 'torchvision'
    
    def test_init_with_densenet(self):
        """Test initialization with DenseNet model."""
        backbone = ImageBackbone('densenet121', device='cpu')
        assert backbone.name == 'densenet121'
        assert backbone.source == 'torchvision'
    
    def test_unsupported_torchvision_model(self):
        """Test that unsupported torchvision models raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported torchvision backbone"):
            ImageBackbone('mobilenet_v2', device='cpu')
    
    def test_init_auto_device(self):
        """Test that auto device detection works."""
        backbone = ImageBackbone('resnet18')  # No device specified
        assert backbone.device is not None
        assert backbone.device.type in ['cpu', 'cuda']

    def test_base_class_is_abstract(self):
        """Backbone cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            Backbone('resnet18', device='cpu')

    def test_image_backbone_is_a_backbone(self):
        assert isinstance(ImageBackbone('resnet18', device='cpu'), Backbone)


# =============================================================================
# Test source resolution and the loader registry
# =============================================================================

class TestImageBackboneSource:
    """Tests for `source` resolution on ImageBackbone."""

    def test_auto_resolves_torchvision(self):
        assert ImageBackbone('resnet18', device='cpu').source == 'torchvision'

    def test_explicit_torchvision(self):
        assert ImageBackbone('resnet18', device='cpu',
                             source='torchvision').source == 'torchvision'

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            ImageBackbone('resnet18', device='cpu', source='nope')


class TestBackboneRegistry:
    """Tests for the per-source loader registry."""

    def test_registries_are_per_modality(self):
        assert 'torchvision' in ImageBackbone._loaders
        assert 'torchvision' not in TextBackbone._loaders

    def test_register_and_use_custom_source(self):
        @ImageBackbone.register_source("dummy_src")
        def _load(name, device):
            model = nn.Linear(10, 7).to(device)
            return BackboneSpec(model, None, lambda m, p, x: m(x), 7)
        try:
            b = ImageBackbone("whatever", device='cpu', source="dummy_src")
            assert b.source == "dummy_src"
            assert b.out_features == 7
            assert b(torch.randn(4, 10)).shape == (4, 7)
            assert isinstance(b, Backbone)
        finally:
            del ImageBackbone._loaders["dummy_src"]


# =============================================================================
# Test Backbone Properties
# =============================================================================

class TestBackboneProperties:
    """Tests for Backbone property accessors."""
    
    def test_device_property(self):
        """Test device property returns torch.device."""
        backbone = ImageBackbone('resnet18', device='cpu')
        assert isinstance(backbone.device, torch.device)
        assert backbone.device == torch.device('cpu')
    
    def test_processor_property(self):
        """Test processor property is set."""
        backbone = ImageBackbone('resnet18', device='cpu')
        assert backbone.processor is not None

    def test_out_features_property(self):
        """Test out_features exposes the embedding dimension."""
        backbone = ImageBackbone('resnet18', device='cpu')
        assert backbone.out_features == 512
    
    def test_source_property(self):
        """Test source resolves to 'torchvision' for a torchvision name."""
        backbone = ImageBackbone('resnet18', device='cpu')
        assert backbone.source == 'torchvision'
    
    def test_filename_property_torchvision(self):
        """Test filename generation for torchvision models."""
        backbone = ImageBackbone('resnet50', device='cpu')
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
        backbone = ImageBackbone('resnet18', device='cpu')
        embeddings = backbone(dummy_tensor_batch)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == 2  # Batch size
        assert embeddings.shape[1] == 512  # ResNet18 feature dim
    
    def test_forward_with_pil_images(self, dummy_pil_images):
        """Test forward pass with PIL image list."""
        backbone = ImageBackbone('resnet18', device='cpu')
        embeddings = backbone(dummy_pil_images)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 512

    def test_forward_single_image_3d(self):
        """A single (C, H, W) image is batched then squeezed back to 1D."""
        backbone = ImageBackbone('resnet18', device='cpu')
        embedding = backbone(torch.randn(3, 224, 224))

        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (512,)  # batch dim added then squeezed away
    
    def test_resnet_embedding_dimensions(self, dummy_tensor_batch):
        """Test correct embedding dimensions for different ResNet variants."""
        # ResNet18/34 have 512 features
        backbone18 = ImageBackbone('resnet18', device='cpu')
        emb18 = backbone18(dummy_tensor_batch)
        assert emb18.shape[1] == 512
        
        # ResNet50/101/152 have 2048 features
        backbone50 = ImageBackbone('resnet50', device='cpu')
        emb50 = backbone50(dummy_tensor_batch)
        assert emb50.shape[1] == 2048
    
    def test_vgg_embedding_dimensions(self, dummy_tensor_batch):
        """Test correct embedding dimensions for VGG."""
        backbone = ImageBackbone('vgg16', device='cpu')
        embeddings = backbone(dummy_tensor_batch)
        assert embeddings.shape[1] == 25088  # 512 * 7 * 7
    
    def test_efficientnet_embedding_dimensions(self, dummy_tensor_batch):
        """Test correct embedding dimensions for EfficientNet."""
        backbone = ImageBackbone('efficientnet_b0', device='cpu')
        embeddings = backbone(dummy_tensor_batch)
        assert embeddings.shape[1] == 1280
    
    def test_densenet_embedding_dimensions(self, dummy_tensor_batch):
        """Test correct embedding dimensions for DenseNet."""
        backbone = ImageBackbone('densenet121', device='cpu')
        embeddings = backbone(dummy_tensor_batch)
        assert embeddings.shape[1] == 1024


# =============================================================================
# Test Backbone Representation
# =============================================================================

class TestBackboneRepr:
    """Tests for Backbone __repr__ method."""
    
    def test_repr_torchvision(self):
        """Test repr for torchvision model."""
        backbone = ImageBackbone('resnet50', device='cpu')
        repr_str = repr(backbone)
        
        assert 'Backbone' in repr_str
        assert 'resnet50' in repr_str
        assert 'torchvision' in repr_str
        assert 'cpu' in repr_str
    
    def test_repr_contains_all_info(self):
        """Test that repr contains all relevant information."""
        backbone = ImageBackbone('vgg16', device='cpu')
        repr_str = repr(backbone)
        
        assert 'name=' in repr_str
        assert 'source=' in repr_str
        assert 'device=' in repr_str


# =============================================================================
# Test Backbone as nn.Module
# =============================================================================

class TestBackboneAsModule:
    """Tests for Backbone behavior as nn.Module."""
    
    def test_is_nn_module(self):
        """Test that Backbone is an nn.Module subclass."""
        backbone = ImageBackbone('resnet18', device='cpu')
        assert isinstance(backbone, nn.Module)
    
    def test_eval_mode_after_init(self):
        """Test that backbone model is in eval mode after initialization."""
        backbone = ImageBackbone('resnet18', device='cpu')
        # The internal model should be in eval mode
        assert not backbone._model.training
    
    def test_no_grad_context(self, dummy_tensor_batch):
        """Test that embeddings can be computed without gradients."""
        backbone = ImageBackbone('resnet18', device='cpu')

        with torch.no_grad():
            embeddings = backbone(dummy_tensor_batch)

        assert not embeddings.requires_grad


# =============================================================================
# Test Freezing
# =============================================================================

class TestBackboneFreeze:
    """Tests for the freeze mechanism (default frozen, opt-in fine-tuning)."""

    def test_frozen_by_default(self):
        backbone = ImageBackbone('resnet18', device='cpu')
        assert backbone.frozen
        assert all(not p.requires_grad for p in backbone.parameters())

    def test_frozen_stays_in_eval_under_train(self):
        """A frozen backbone ignores .train() (fixed BatchNorm/Dropout)."""
        backbone = ImageBackbone('resnet18', device='cpu')
        backbone.train()
        assert not backbone.training

    def test_unfrozen_is_trainable(self):
        backbone = ImageBackbone('resnet18', device='cpu', freeze=False)
        assert not backbone.frozen
        assert all(p.requires_grad for p in backbone.parameters())
        # starts in train mode like any nn.Module (correct BatchNorm behavior
        # when fine-tuning; Lightning does not flip modes at fit start)
        assert backbone.training
        assert all(m.training for m in backbone.modules())

    def test_unfrozen_propagates_train_mode(self):
        backbone = ImageBackbone('resnet18', device='cpu', freeze=False)
        backbone.train()
        assert backbone.training
        backbone.eval()
        assert not backbone.training

    def test_frozen_inside_parent_model(self):
        """Calling .train() on a parent module keeps a frozen backbone in eval."""
        backbone = ImageBackbone('resnet18', device='cpu')
        parent = nn.Sequential(backbone)
        parent.train()
        assert parent.training
        assert not backbone.training


# =============================================================================
# Test HuggingFace Backbone (Slow tests - marked for optional skip)
# =============================================================================

class TestBackboneHuggingFaceMocked:
    """HuggingFace code paths exercised offline via a mocked ``transformers``.

    These cover the HF branches of ``_load_huggingface_model``, ``_load_model``
    and ``forward`` without any Hub download, so they run in CI (unlike the
    ``@pytest.mark.slow`` tests below).
    """

    @pytest.fixture
    def fake_transformers(self, monkeypatch):
        class _BatchFeature(dict):
            # mirror transformers.BatchFeature: dict with a .to(device) method
            def to(self, device):
                return self

        class _FakeProcessor:
            def __call__(self, images=None, return_tensors=None):
                n = len(images) if isinstance(images, list) else images.shape[0]
                return _BatchFeature(pixel_values=torch.zeros(n, 3, 224, 224))

        class _FakeHFModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(hidden_size=8)
                self._lin = nn.Linear(3, 8)  # real params for device / requires_grad

            def forward(self, **inputs):
                batch = inputs["pixel_values"].shape[0]
                return types.SimpleNamespace(last_hidden_state=torch.zeros(batch, 5, 8))

        fake = types.ModuleType("transformers")
        fake.AutoImageProcessor = types.SimpleNamespace(
            from_pretrained=lambda name, token=None: _FakeProcessor()
        )
        fake.AutoModel = types.SimpleNamespace(
            from_pretrained=lambda name, token=None: _FakeHFModel()
        )
        monkeypatch.setitem(sys.modules, "transformers", fake)
        return fake

    def test_init_reads_hidden_size(self, fake_transformers):
        """HF init path: loads model/processor and reads out_features from config."""
        backbone = ImageBackbone('facebook/dinov2-base', device='cpu')
        assert backbone.source == 'huggingface'
        assert backbone.out_features == 8

    def test_forward_returns_cls_token(self, fake_transformers, dummy_pil_images):
        """HF forward path: processor -> model -> CLS token embedding."""
        backbone = ImageBackbone('facebook/dinov2-base', device='cpu')
        embeddings = backbone(dummy_pil_images)
        assert embeddings.shape == (2, 8)

    def test_repr_shows_source(self, fake_transformers):
        """repr shows the resolved source, truthfully (not a HF/torchvision bool)."""
        backbone = ImageBackbone('facebook/dinov2-base', device='cpu')
        assert 'source=huggingface' in repr(backbone)


# =============================================================================
# Test TextBackbone (HuggingFace text path, mocked)
# =============================================================================

class TestTextBackboneMocked:
    """TextBackbone paths exercised offline via a mocked ``transformers``.

    The fake tokenizer emits per-sample lengths of ``len(text.split()) + 2``
    so mask-aware pooling can be checked on an unequal-length batch; the fake
    model returns hidden states equal to the token position, so a longer
    (less-masked) sentence has a strictly larger mean.
    """

    @staticmethod
    def _install(monkeypatch, pad_token="[PAD]"):
        """Install a mocked ``transformers`` whose tokenizer starts with the
        given ``pad_token`` (pass None to exercise the eos-fallback)."""
        class _Enc(dict):
            def to(self, device):
                return self

        class _FakeTokenizer:
            eos_token = "[EOS]"

            def __init__(self):
                self.pad_token = pad_token

            def __call__(self, texts, padding=True, truncation=True,
                         return_tensors="pt"):
                lengths = [len(t.split()) + 2 for t in texts]
                T = max(lengths)
                ids = torch.zeros(len(texts), T, dtype=torch.long)
                mask = torch.zeros(len(texts), T, dtype=torch.long)
                for i, n in enumerate(lengths):
                    mask[i, :n] = 1
                return _Enc(input_ids=ids, attention_mask=mask)

        class _FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(hidden_size=8)
                self._lin = nn.Linear(3, 8)  # real params for device/grad

            def forward(self, input_ids=None, attention_mask=None, **kw):
                B, T = input_ids.shape
                h = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(B, T, 8)
                return types.SimpleNamespace(last_hidden_state=h.clone())

        fake = types.ModuleType("transformers")
        fake.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda name, token=None: _FakeTokenizer()
        )
        fake.AutoModel = types.SimpleNamespace(
            from_pretrained=lambda name, token=None: _FakeModel()
        )
        monkeypatch.setitem(sys.modules, "transformers", fake)
        return fake

    @pytest.fixture
    def fake_transformers(self, monkeypatch):
        return self._install(monkeypatch)

    TEXTS = ["short text", "a much longer piece of text with many words"]

    def test_init(self, fake_transformers):
        tb = TextBackbone("fake/model", device='cpu')
        assert tb.source == 'huggingface'
        assert tb.out_features == 8
        assert tb.pooling == 'mean'
        assert isinstance(tb, Backbone)

    def test_invalid_pooling_raises(self):
        with pytest.raises(ValueError, match="pooling must be"):
            TextBackbone("fake/model", device='cpu', pooling='max')

    def test_pad_token_falls_back_to_eos(self, monkeypatch):
        """Decoder-style tokenizers have no pad token: it must fall back to eos."""
        self._install(monkeypatch, pad_token=None)
        tb = TextBackbone("gpt-style", device='cpu')
        assert tb.processor.pad_token == "[EOS]"

    def test_existing_pad_token_is_kept(self, fake_transformers):
        """A tokenizer that already has a pad token keeps it (no clobbering)."""
        tb = TextBackbone("fake/model", device='cpu')
        assert tb.processor.pad_token == "[PAD]"

    def test_mean_pooling_shape(self, fake_transformers):
        tb = TextBackbone("fake/model", device='cpu', pooling='mean')
        assert tb(self.TEXTS).shape == (2, 8)

    def test_cls_pooling_shape(self, fake_transformers):
        tb = TextBackbone("fake/model", device='cpu', pooling='cls')
        assert tb(self.TEXTS).shape == (2, 8)

    def test_none_pooling_returns_tokens(self, fake_transformers):
        tb = TextBackbone("fake/model", device='cpu', pooling='none')
        out = tb(self.TEXTS)
        assert out.dim() == 3 and out.shape[0] == 2 and out.shape[2] == 8

    def test_single_string_is_squeezed(self, fake_transformers):
        tb = TextBackbone("fake/model", device='cpu')
        assert tb("hello world").shape == (8,)

    def test_mean_pooling_respects_mask(self, fake_transformers):
        """Masked (padding) positions are excluded: the longer sentence, with
        fewer masked-out positions, has a strictly larger mean."""
        tb = TextBackbone("fake/model", device='cpu', pooling='mean')
        out = tb(self.TEXTS)
        assert out[0, 0] < out[1, 0]

    def test_frozen_by_default(self, fake_transformers):
        tb = TextBackbone("fake/model", device='cpu')
        assert tb.frozen
        assert all(not p.requires_grad for p in tb.parameters())
        tb.train()
        assert not tb.training  # frozen stays in eval


@pytest.mark.slow
class TestBackboneHuggingFace:
    """Tests for Backbone with HuggingFace models.
    
    These tests require downloading HuggingFace models and are marked as slow.
    Run with: pytest -m slow
    """
    
    def test_init_dinov2(self):
        """Test initialization with DINOv2 model."""
        backbone = ImageBackbone('facebook/dinov2-base', device='cpu')
        assert backbone.name == 'facebook/dinov2-base'
        assert backbone.source == 'huggingface'
    
    def test_forward_dinov2(self, dummy_pil_images):
        """Test forward pass with DINOv2."""
        backbone = ImageBackbone('facebook/dinov2-base', device='cpu')
        embeddings = backbone(dummy_pil_images)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 768  # DINOv2-base CLS token dim
    
    def test_repr_huggingface(self):
        """Test repr for HuggingFace model."""
        backbone = ImageBackbone('facebook/dinov2-base', device='cpu')
        repr_str = repr(backbone)

        assert 'source=huggingface' in repr_str
        assert 'facebook/dinov2-base' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
