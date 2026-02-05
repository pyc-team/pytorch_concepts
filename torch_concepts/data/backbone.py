"""
Backbone utilities for feature extraction.

This module provides the :class:`Backbone` class for extracting embeddings
from pre-trained models. It supports both HuggingFace models (DINOv2, ViT, etc.)
and torchvision models (ResNet, VGG, EfficientNet, DenseNet).

The backbone can be used as a regular :class:`torch.nn.Module` and is designed
to work seamlessly with the :class:`ConceptDataModule` for preprocessing
image datasets into feature embeddings.

Example
-------
>>> from torch_concepts.data.backbone import Backbone
>>> backbone = Backbone('resnet50', device='cuda')
>>> embeddings = backbone(batch_images)  # (B, 2048)

Notes
-----
HuggingFace models are detected by presence of '/' in the name or by
keywords like 'dinov2', 'vit-', 'beit', 'clip', 'swin', 'convnext'.
"""
import torch
import torch.nn as nn
import logging
import warnings
from typing import Union, List, Tuple, Optional

logger = logging.getLogger(__name__)


def _resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve device with auto-detection if None.

    Auto-detection priority: CUDA > CPU. MPS is not supported due to
    compatibility issues with torchvision transforms and HuggingFace models.

    Parameters
    ----------
    device : str, optional
        Device string ('cpu', 'cuda', 'cuda:0', etc.). If None, auto-detects.

    Returns
    -------
    torch.device
        Resolved device object.

    Warnings
    --------
    If MPS is available but selected, a warning is raised and CPU is used instead.
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            warnings.warn(
                "MPS may not work with torchvision preprocessing transforms "
                "and HuggingFace models. Falling back to CPU.",
                stacklevel=2
            )
            device = 'cpu'
        else:
            device = 'cpu'
    return torch.device(device)


def _is_huggingface_model(name: str) -> bool:
    """Check if backbone string refers to a HuggingFace model.

    Detection is based on:
    1. Presence of '/' in the name (e.g., 'facebook/dinov2-base')
    2. Presence of known HuggingFace keywords (dinov2, vit-, beit, etc.)

    Parameters
    ----------
    name : str
        Model name to check.

    Returns
    -------
    bool
        True if the name refers to a HuggingFace model.
    """
    hf_keywords = ['dinov2', 'dino-', 'vit-', 'beit', 'clip', 'swin', 'convnext']
    name_lower = name.lower()
    if '/' in name:
        return True
    return any(kw in name_lower for kw in hf_keywords)


def _load_huggingface_model(
    name: str, 
    device: torch.device
) -> Tuple[nn.Module, object]:
    """Load a HuggingFace model and processor.

    Parameters
    ----------
    name : str
        HuggingFace model identifier (e.g., 'facebook/dinov2-base').
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    tuple
        (model, processor) where model is the HuggingFace model in eval mode
        and processor is the AutoImageProcessor for preprocessing.
    """
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    return model, processor


def _load_torchvision_model(
    name: str, 
    device: torch.device
) -> Tuple[nn.Module, object]:
    """Load a torchvision model and its preprocessing transforms.

    Supported model families:
    - ResNet (resnet18, resnet34, resnet50, resnet101, resnet152)
    - VGG (vgg11, vgg13, vgg16, vgg19)
    - EfficientNet (efficientnet_b0 through efficientnet_b7)
    - DenseNet (densenet121, densenet161, densenet169, densenet201)

    Parameters
    ----------
    name : str
        Torchvision model name (e.g., 'resnet50', 'vgg16').
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    tuple
        (model, preprocess) where model is a feature extractor (without
        classification head) and preprocess is the transforms pipeline.

    Raises
    ------
    ValueError
        If the model name is not supported.
    """
    from torchvision.models import get_model, get_model_weights

    weights = get_model_weights(name).DEFAULT
    full_model = get_model(name, weights=weights)

    name_lower = name.lower()
    if 'resnet' in name_lower:
        model = nn.Sequential(*list(full_model.children())[:-1], nn.Flatten())
    elif 'vgg' in name_lower:
        model = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten())
    elif 'efficientnet' in name_lower:
        model = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten())
    elif 'densenet' in name_lower:
        model = nn.Sequential(full_model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    else:
        raise ValueError(f"Unsupported torchvision backbone: {name}")

    model = model.to(device).eval()
    preprocess = weights.transforms()
    return model, preprocess


class Backbone(nn.Module):
    """Wrapper class for backbone models used for feature extraction.

    Supports both HuggingFace models (DINOv2, ViT, CLIP, etc.) and torchvision
    models (ResNet, VGG, EfficientNet, DenseNet). The backbone extracts
    embeddings from images and can be used as a regular :class:`torch.nn.Module`.

    The class automatically handles:
    - Model loading and initialization in eval mode
    - Preprocessing transforms appropriate for each model
    - Device management (auto-detection of CUDA/CPU)
    - Image format conversion (PIL to tensor for torchvision)

    Parameters
    ----------
    name : str
        Model name for feature extraction. Can be:

        - **HuggingFace model**: 'facebook/dinov2-base', 'google/vit-base-patch16-224'
        - **torchvision model**: 'resnet18', 'resnet50', 'vgg16', 'efficientnet_b0'

    device : str, optional
        Device to use ('cpu', 'cuda', 'cuda:0', etc.).
        If None, auto-detects available hardware (CUDA > CPU).
        Default is None.

    Attributes
    ----------
    name : str
        The model name used for initialization.
    device : torch.device
        The device the model is loaded on.
    processor : object
        The preprocessing transform/processor (varies by model type).
    is_huggingface : bool
        Whether this is a HuggingFace model.
    filename : str
        Safe filename for caching embeddings (e.g., 'bkb_embs_resnet50.pt').

    Examples
    --------
    Using with torchvision model:

    >>> from torch_concepts.data.backbone import Backbone
    >>> import torch
    >>> backbone = Backbone('resnet50', device='cpu')
    >>> images = torch.randn(4, 3, 224, 224)  # batch of 4 images
    >>> embeddings = backbone(images)
    >>> embeddings.shape
    torch.Size([4, 2048])

    Using with HuggingFace model:

    >>> backbone = Backbone('facebook/dinov2-base', device='cuda')
    >>> from PIL import Image
    >>> images = [Image.new('RGB', (224, 224)) for _ in range(4)]
    >>> embeddings = backbone(images)
    >>> embeddings.shape
    torch.Size([4, 768])

    See Also
    --------
    ConceptDataModule : DataModule that integrates with Backbone for preprocessing.
    _is_huggingface_model : Helper function for model type detection.
    """

    def __init__(self, name: str, device: Optional[str] = None):
        super().__init__()
        self.name = name
        self._device = _resolve_device(device)
        self._is_huggingface = _is_huggingface_model(name)
        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the backbone model and processor based on model type.

        For HuggingFace models, loads via transformers library.
        For torchvision models, loads pretrained weights and removes
        classification head to create a feature extractor.
        """
        if self._is_huggingface:
            self._model, self._processor = _load_huggingface_model(self.name, self._device)
        else:
            self._model, self._processor = _load_torchvision_model(self.name, self._device)
            # Cache ToTensor transform for PIL image conversion
            from torchvision import transforms
            self._to_tensor = transforms.ToTensor()

    @property
    def device(self) -> torch.device:
        """The device this backbone is on.

        Returns
        -------
        torch.device
            The device (e.g., cpu, cuda:0).
        """
        return self._device

    @property
    def processor(self):
        """The preprocessing transform/processor for this backbone.

        For HuggingFace models, this is an AutoImageProcessor.
        For torchvision models, this is a transforms.Compose pipeline.

        Returns
        -------
        object
            The preprocessor appropriate for the model type.
        """
        return self._processor

    @property
    def is_huggingface(self) -> bool:
        """Whether this is a HuggingFace model.

        Returns
        -------
        bool
            True if the backbone is a HuggingFace model.
        """
        return self._is_huggingface

    @property
    def filename(self) -> str:
        """Generate a safe filename for caching embeddings.

        Replaces '/' with '-' to ensure filesystem compatibility.

        Returns
        -------
        str
            Filename like 'bkb_embs_resnet50.pt' or 'bkb_embs_facebook-dinov2-base.pt'.
        """
        return f"bkb_embs_{self.name.replace('/', '-')}.pt"

    def forward(self, x: Union[torch.Tensor, List]) -> torch.Tensor:
        """Forward pass through the backbone to extract embeddings.

        Parameters
        ----------
        x : torch.Tensor or list
            Input data. Format depends on model type:

            - **torchvision**: Tensor of shape (B, C, H, W) or list of PIL Images
            - **HuggingFace**: List of PIL Images or preprocessed tensors

        Returns
        -------
        torch.Tensor
            Embeddings of shape (B, embedding_dim), where embedding_dim
            depends on the model (e.g., 2048 for ResNet50, 768 for DINOv2-base).

        Notes
        -----
        For HuggingFace models, the CLS token embedding is returned.
        For torchvision models, the output of the average pooling layer is used.
        """
        if self._is_huggingface:
            inputs = self._processor(images=x, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            return outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            if isinstance(x, list):
                x = torch.stack([self._to_tensor(img) for img in x])
            x = x.to(self._device)
            x = self._processor(x)
            return self._model(x)

    def __repr__(self) -> str:
        """Return string representation of the Backbone.

        Returns
        -------
        str
            Formatted string with model name, type, and device.
        """
        model_type = "HuggingFace" if self._is_huggingface else "torchvision"
        return f"Backbone(name='{self.name}', type={model_type}, device={self._device})"

