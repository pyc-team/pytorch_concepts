"""
Backbone utilities for feature extraction.

- :class:`Backbone` — abstract base holding all modality-independent logic:
  device resolution, freeze semantics, cache filename, common properties,
  and the per-source loader registry. Not instantiable directly.
- :class:`ImageBackbone` — image embeddings. Built-in sources:
  ``'torchvision'`` (ResNet, VGG, EfficientNet, DenseNet) and
  ``'huggingface'`` (DINOv2, ViT, ...).
- :class:`TextBackbone` — text embeddings. Built-in source:
  ``'huggingface'`` (``AutoTokenizer`` + ``AutoModel``), with mean / CLS /
  no pooling.

Extensibility:

- **per-name** (resnet18 vs resnet50): a string, zero code;
- **per-family** (resnet vs vgg): one entry in ``_TORCHVISION_FAMILIES``;
- **per-source** (torchvision vs huggingface vs your own): a registered
  loader — additive, no library changes::

      @ImageBackbone.register_source("mylab")
      def _load_mylab(name, device):
          model = ...                       # nn.Module on `device`
          def embed(model, processor, x):   # x -> (B, out_features)
              return model(x)
          return BackboneSpec(model, None, embed, out_features=512)

      backbone = ImageBackbone("my-model-v2", source="mylab")

- **per-modality** (image vs text): a :class:`Backbone` subclass; freezing,
  device handling, caching and ``precompute_embeddings`` compatibility are
  inherited.
"""
import torch
import torch.nn as nn
import logging
import warnings
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

from .utils import resolve_hf_token

logger = logging.getLogger(__name__)


class BackboneSpec(NamedTuple):
    """What a source loader returns.
    
    Attributes
    ----------
    model : nn.Module
        Loaded model on the target device.
    processor : object
        Transform/tokenizer for preprocessing (may be None).
    embed : Callable
        Function ``embed(model, processor, x)`` to turn raw inputs into embeddings.
    out_features : int
        Embedding dimension.
    """
    model: nn.Module
    processor: object
    embed: Callable
    out_features: int


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
    """Heuristic used by ``source='auto'``: a '/' or a known keyword in the
    name means HuggingFace. Pass an explicit ``source`` to bypass it (e.g.
    for new HuggingFace families whose names carry no known keyword).

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
    """Load a HuggingFace model and its AutoImageProcessor..

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


# Family -> feature-extractor builder (drop the classification head).
# Supporting a new torchvision family is one entry here.
_TORCHVISION_FAMILIES = {
    'resnet': lambda m: nn.Sequential(*list(m.children())[:-1], nn.Flatten()),
    'vgg': lambda m: nn.Sequential(m.features, m.avgpool, nn.Flatten()),
    'efficientnet': lambda m: nn.Sequential(m.features, m.avgpool, nn.Flatten()),
    'densenet': lambda m: nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()),
}


def _load_torchvision_model(
    name: str,
    device: torch.device
) -> Tuple[nn.Module, object]:
    """Load a torchvision model as a headless feature extractor
    together with its preprocessing transforms.

    Supported model families are in _TORCHVISION_FAMILIES:

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
        If the model family is not in ``_TORCHVISION_FAMILIES``.
    """
    from torchvision.models import get_model, get_model_weights

    weights = get_model_weights(name).DEFAULT
    full_model = get_model(name, weights=weights)

    name_lower = name.lower()
    for family, build in _TORCHVISION_FAMILIES.items():
        if family in name_lower:
            model = build(full_model)
            break
    else:
        raise ValueError(f"Unsupported torchvision backbone: {name}")

    model = model.to(device)
    preprocess = weights.transforms()
    return model, preprocess


class Backbone(nn.Module):
    """
    Abstract base class for feature-extraction backbones.

    Holds all modality-independent logic; subclasses only choose sources
    and, if needed, override :meth:`forward`. Not instantiable directly —
    use :class:`ImageBackbone`, :class:`TextBackbone`, or your own
    subclass. To support a new *source* of models within an existing
    modality, register a loader instead of subclassing (see the module
    docstring).

    Parameters
    ----------
    name : str
        Model identifier. Interpretation is source-specific.
        For example:
        - **HuggingFace model**: 'facebook/dinov2-base', 'google/vit-base-patch16-224'
        - **torchvision model**: 'resnet18', 'resnet50', 'vgg16', 'efficientnet_b0'

    device : str, optional
        Device to use ('cpu', 'cuda', 'cuda:0', etc.).
        If None, auto-detects available hardware (CUDA > CPU).
        Default is None.
    freeze : bool, default True
        If True, parameters take no gradients and the module stays in eval
        mode. Pass ``freeze=False`` to fine-tune.
    """

    _loaders: dict = {}  # per-source loaders; each subclass has its own dict

    def __init__(self, name: str, device: Optional[str] = None, freeze: bool = True):
        if type(self) is Backbone:
            raise TypeError(
                "Backbone is abstract; instantiate ImageBackbone, "
                "TextBackbone, or a custom subclass instead."
            )
        super().__init__()
        self.name = name
        self.frozen = freeze
        self._device = _resolve_device(device)
        self._load_model()
        if freeze:
            self.requires_grad_(False)
            self.eval()  # fixed feature extractor: lock BatchNorm/Dropout

    @classmethod
    def register_source(cls, source: str) -> Callable:
        """Register a loader for ``source`` on this backbone class.

        The decorated function must have signature
        ``loader(name: str, device: torch.device) -> BackboneSpec``.
        Registration is additive: user code can add sources without
        modifying the library.
        """
        def decorator(fn: Callable) -> Callable:
            cls._loaders[source] = fn
            return fn
        return decorator

    def _resolve_source(self) -> str:
        """The source key to load from; subclasses may override (e.g. for
        an ``'auto'`` heuristic)."""
        return self._source

    def _load_model(self) -> None:
        """Resolve the source, run its loader, and store the results."""
        source = self._resolve_source()
        if source not in self._loaders:
            raise ValueError(
                f"Unknown source {source!r} for {type(self).__name__}; "
                f"registered sources: {sorted(self._loaders)}. Register new "
                f"ones with @{type(self).__name__}.register_source(...)."
            )
        spec = BackboneSpec(*self._loaders[source](self.name, self._device))
        self._model = spec.model
        self._processor = spec.processor
        self._embed = spec.embed
        self._out_features = spec.out_features
        self.source = source  # resolved source key ('torchvision', 'huggingface', ...)

    def forward(self, x):
        """Extract embeddings from ``x`` using the source's embed function."""
        return self._embed(self._model, self._processor, x)

    def train(self, mode: bool = True):
        """Set train mode, but keep a frozen backbone in eval.

        A frozen backbone is a fixed feature extractor, so its BatchNorm
        running statistics and Dropout must stay fixed even when a parent
        module (or Lightning at the start of each epoch) switches to train
        mode. Unfrozen backbones behave like a normal ``nn.Module``.
        """
        return super().train(mode and not self.frozen)

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
        For text, this can be a tokenizer.

        Returns
        -------
        object
            The preprocessor appropriate for the model type.
        """
        return self._processor

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

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name='{self.name}', source={self.source}, "
                f"device={self.device}, frozen={self.frozen})")


class ImageBackbone(Backbone):
    """
    Image feature-extraction backbone.

    Accepts a tensor of shape ``(B, C, H, W)`` or ``(C, H, W)``, or a list
    of PIL Images; returns embeddings of shape ``(B, out_features)`` (a
    single 3D image is squeezed back to ``(out_features,)``). HuggingFace
    models return the CLS token; torchvision models the pooled features.

    Parameters
    ----------
    name : str
        Model name: HuggingFace ('facebook/dinov2-base', ...) or
        torchvision ('resnet50', 'vgg16', ...).
    device : str, optional
        Device string; auto-detected if None.
    freeze : bool, default True
        Freeze parameters (see :class:`Backbone`).
    source : str, default 'auto'
        Which registered source to load from (``'torchvision'``,
        ``'huggingface'``, or a custom key). ``'auto'`` infers from the
        name; pass it explicitly for HuggingFace models the heuristic does
        not recognize.

    Examples
    --------
    >>> from torch_concepts import ImageBackbone
    >>> import torch
    >>>
    >>> backbone = ImageBackbone('resnet50')
    >>> backbone(torch.randn(4, 3, 224, 224)).shape
    torch.Size([4, 2048])
    """

    _loaders: dict = {}

    def __init__(
        self,
        name: str,
        device: Optional[str] = None,
        freeze: bool = True,
        source: str = "auto",
    ):
        self._source = source
        super().__init__(name=name, device=device, freeze=freeze)

    def _resolve_source(self) -> str:
        if self._source == "auto":
            return "huggingface" if _is_huggingface_model(self.name) else "torchvision"
        return self._source


@ImageBackbone.register_source("huggingface")
def _load_hf_image_source(name: str, device: torch.device) -> BackboneSpec:
    """Built-in HuggingFace vision source: processor -> model -> CLS token."""
    model, processor = _load_huggingface_model(name, device)

    def embed(model, processor, x):
        inputs = processor(images=x, return_tensors="pt")
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

    return BackboneSpec(model, processor, embed, model.config.hidden_size)


@ImageBackbone.register_source("torchvision")
def _load_torchvision_source(name: str, device: torch.device) -> BackboneSpec:
    """Built-in torchvision source: transforms -> headless model."""
    model, preprocess = _load_torchvision_model(name, device)
    from torchvision import transforms
    to_tensor = transforms.ToTensor()  # for PIL-image lists

    def embed(model, processor, x):
        if isinstance(x, list):  # list of PIL Images
            x = torch.stack([to_tensor(img) for img in x])
        if squeeze_single := x.dim() == 3:  # single image: batch it
            x = x.unsqueeze(0)
        out = model(processor(x))
        return out.squeeze(0) if squeeze_single else out

    with torch.no_grad():  # output size via dummy forward pass
        out_features = model(torch.zeros(1, 3, 224, 224, device=device)).shape[-1]

    return BackboneSpec(model, preprocess, embed, out_features)


class TextBackbone(Backbone):
    """
    Text embedding backbone.

    Accepts a string or a list of strings. Pooling of the per-token hidden
    states is applied by this class, uniformly across sources:

    - ``'mean'`` (default): attention-mask-weighted mean — the standard
      recipe for sentence embeddings. Output ``(B, out_features)``.
    - ``'cls'``: first-token ([CLS]) embedding. Output ``(B, out_features)``.
    - ``'none'``: unpooled ``(B, T, out_features)`` — the hook for
      token-level pipelines. Not usable with ``precompute_embeddings``
      caching (T varies per batch).

    A single input string is squeezed to ``(out_features,)``. A custom text
    source's ``embed`` must return ``(hidden_states, attention_mask)``.

    Parameters
    ----------
    name : str
        HuggingFace model identifier (e.g., 'bert-base-uncased',
        'sentence-transformers/all-MiniLM-L6-v2').
    device : str, optional
        Device string; auto-detected if None.
    freeze : bool, default True
        Freeze parameters (see :class:`Backbone`).
    pooling : str, default 'mean'
        How to pool token hidden states: 'mean', 'cls' or 'none'.
    source : str, default 'huggingface'
        Which registered source to load from.

    Examples
    --------
    >>> from torch_concepts import TextBackbone
    >>>
    >>> backbone = TextBackbone('sentence-transformers/all-MiniLM-L6-v2',
    ...                         device='cpu')
    >>> backbone(["a smiling person", "a person wearing a hat"]).shape
    torch.Size([2, 384])
    """

    _loaders: dict = {}

    def __init__(
        self,
        name: str,
        device: Optional[str] = None,
        freeze: bool = True,
        pooling: str = "mean",
        source: str = "huggingface",
    ):
        if pooling not in ("mean", "cls", "none"):
            raise ValueError(
                f"pooling must be 'mean', 'cls' or 'none', got {pooling!r}."
            )
        self.pooling = pooling
        self._source = source
        super().__init__(name=name, device=device, freeze=freeze)

    def forward(self, x: Union[str, List[str]]) -> torch.Tensor:
        """Embed a string or a batch of strings (see class docstring for
        output shapes per pooling mode)."""
        if squeeze_single := isinstance(x, str):
            x = [x]

        hidden, mask = self._embed(self._model, self._processor, x)  # (B, T, H)

        if self.pooling == "mean":
            mask = mask.unsqueeze(-1)  # (B, T, 1)
            out = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        elif self.pooling == "cls":
            out = hidden[:, 0, :]
        else:  # 'none'
            out = hidden

        return out.squeeze(0) if squeeze_single else out


@TextBackbone.register_source("huggingface")
def _load_hf_text_source(name: str, device: torch.device) -> BackboneSpec:
    """Built-in HuggingFace text source: tokenizer -> model hidden states.

    The embed function returns ``(last_hidden_state, attention_mask)``;
    pooling is applied by :class:`TextBackbone`.
    """
    from transformers import AutoTokenizer, AutoModel
    token = resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(name, token=token)
    model = AutoModel.from_pretrained(name, token=token).to(device)
    if tokenizer.pad_token is None:  # decoder-style tokenizers (GPT-family)
        tokenizer.pad_token = tokenizer.eos_token

    def embed(model, tokenizer, texts):
        inputs = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(next(model.parameters()).device)
        return model(**inputs).last_hidden_state, inputs["attention_mask"]

    return BackboneSpec(model, tokenizer, embed, model.config.hidden_size)
