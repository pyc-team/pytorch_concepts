"""
Backbone utilities for feature extraction.
"""
import torch
import torch.nn as nn
import logging
import warnings
from typing import Union, List, Tuple, Optional

from .utils import resolve_hf_token

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
    If MPS is available and selected, a warning is raised and CPU is used instead.
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
    # FIXME: This is a heuristic; consider using a more robust method 
    # (e.g., checking against a list of known HF models).
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

    Supported model families:
    - DINOv2 (facebook/dinov2-base, facebook/dinov2-large)

    Parameters
    ----------
    name : str
        HuggingFace model identifier (e.g., 'facebook/dinov2-base').
    device : torch.device
        Device to load the model onto.

    Returns
    -------
    tuple
        (model, processor) where 'model' is the HuggingFace model
        and 'processor' is the AutoImageProcessor for preprocessing.
    """
    from transformers import AutoImageProcessor, AutoModel
    token = resolve_hf_token()
    processor = AutoImageProcessor.from_pretrained(name, token=token)
    model = AutoModel.from_pretrained(name, token=token).to(device)
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
        (model, preprocess) where 'model' is a feature extractor (without
        classification head) and 'preprocess' is the transforms pipeline.

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

    model = model.to(device)
    preprocess = weights.transforms()
    return model, preprocess


class Backbone(nn.Module):
    """
    This module provides the :class:`Backbone` class for extracting embeddings
    from pre-trained models. It supports both HuggingFace models (DINOv2, ViT, etc.)
    and torchvision models (ResNet, VGG, EfficientNet, DenseNet).

    The backbone is a regular :class:`torch.nn.Module`. 
    Pass it to :meth:`ConceptDataset.precompute_embeddings` /
    :meth:`ConceptDataModule.precompute_embeddings` to precompute and cache embeddings,
    or pass it to a high-level model as its ``backbone`` (frozen or unfrozen).

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
    freeze : bool, default True
        If True, parameters are excluded from gradients. 
        Pass ``freeze=False`` to fine-tune.

    Attributes
    ----------
    name : str
        The model name used for initialization.
    frozen : bool
        Whether the backbone is frozen.
    device : torch.device
        The device the model currently lives on.
    processor : object
        The preprocessing transform/processor (varies by model type).
    is_huggingface : bool
        Whether this is a HuggingFace model.
    filename : str
        Safe filename for caching embeddings (e.g., 'bkb_embs_resnet50.pt').

    Examples
    --------
    >>> from torch_concepts import Backbone
    >>> import torch
    >>>
    >>> backbone = Backbone('resnet50', device='cpu')
    >>>
    >>> images = torch.randn(4, 3, 224, 224)  # batch of 4 images
    >>> embeddings = backbone(images)
    >>> embeddings.shape
    torch.Size([4, 2048])
    """

    def __init__(self, name: str, device: Optional[str] = None, freeze: bool = True):
        super().__init__()
        self.name = name
        self.frozen = freeze
        self._device = _resolve_device(device)
        self._is_huggingface = _is_huggingface_model(name)
        self._model = None
        self._processor = None
        self._out_features = None
        self._load_model()
        if freeze:
            self.requires_grad_(False)

    def _load_model(self) -> None:
        """Load the backbone model and processor based on model type.

        For HuggingFace models, loads via transformers library.
        For torchvision models, loads pretrained weights and removes
        classification head to create a feature extractor.

        Also computes the output feature dimension via a dummy forward pass.
        """
        if self._is_huggingface:
            self._model, self._processor = _load_huggingface_model(self.name, self._device)
            # Get output size from model config
            self._out_features = self._model.config.hidden_size
        else:
            self._model, self._processor = _load_torchvision_model(self.name, self._device)
            # Cache ToTensor transform for PIL image conversion
            from torchvision import transforms
            self._to_tensor = transforms.ToTensor()
            # Compute output size with dummy forward pass
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 224, 224, device=self._device)
                dummy_output = self._model(dummy_input)
                self._out_features = dummy_output.shape[-1]

    @property
    def device(self) -> torch.device:
        """The device this backbone currently lives on.

        Read from the parameters so it stays correct when a parent model
        (e.g. Lightning) moves the module after construction.

        Returns
        -------
        torch.device
            The device (e.g., cpu, cuda:0).
        """
        return next(self._model.parameters()).device

    @property
    def out_features(self) -> int:
        """The output embedding dimension of the backbone.

        Returns
        -------
        int
            The size of the output embedding (e.g., 2048 for ResNet50,
            768 for DINOv2-base).
        """
        return self._out_features

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

            - **torchvision**: Tensor of shape (B, C, H, W), (C, H, W), or list of PIL Images
            - **HuggingFace**: Tensor of shape (B, C, H, W), (C, H, W), or list of PIL Images

        Returns
        -------
        torch.Tensor
            Embeddings of shape (B, embedding_dim) or (embedding_dim,) for single images,
            where embedding_dim depends on the model (e.g., 2048 for ResNet50, 768 for DINOv2-base).

        Notes
        -----
        For HuggingFace models, the CLS token embedding is returned.
        For torchvision models, the output of the average pooling layer is used.
        Single images (3D tensors) are automatically batched and the result is squeezed.
        """
        if self._is_huggingface:
            inputs = self._processor(images=x, return_tensors="pt")
            outputs = self._model(**inputs)
            return outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            if isinstance(x, list):
                # list of PIL Images case
                x = torch.stack([self._to_tensor(img) for img in x])
            # Handle single image (3D tensor) by adding batch dimension
            squeeze_output = False
            if x.dim() == 3:
                # single image case
                x = x.unsqueeze(0)
                squeeze_output = True
            x = self._processor(x)
            out = self._model(x)
            if squeeze_output:
                out = out.squeeze(0)
            return out

    def __repr__(self) -> str:
        """Return string representation of the Backbone.

        Returns
        -------
        str
            Formatted string with model name, type, and device.
        """
        model_type = "HuggingFace" if self._is_huggingface else "torchvision"
        return f"Backbone(name='{self.name}', type={model_type}, device={self.device}, frozen={self.frozen})"

