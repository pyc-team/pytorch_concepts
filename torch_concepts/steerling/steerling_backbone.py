"""Steerling transformer backbone for text feature extraction.

``SteerlingBackbone`` wraps the official Steerling causal-diffusion
transformer and exposes the PyC-friendly mapping ``input_ids -> hidden``.
It is responsible only for token-level hidden states; concept scoring and
concept mixing are handled by the Steerling encoder and mixer layers.

By default the backbone builds the Steerling-8B architecture from the Hub
config, loads backbone weights, freezes them, and exposes the matching
tokenizer.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from .steerling_configs import (
    DEFAULT_MODEL_ID,
    config_to_dict,
    resolve_steerling_configs,
)
from .steerling_utils import (
    load_steerling_backbone_weights, get_steerling_tokenizer
)

logger = logging.getLogger(__name__)
DEFAULT_VOCAB_SIZE = 100281


def _import_steerling_transformer():
    """Lazily import Steerling model/config classes."""
    try:
        from steerling.models.causal_diffusion import CausalDiffusionLM
        from steerling.configs.causal_diffusion import CausalDiffusionConfig
        return CausalDiffusionLM, CausalDiffusionConfig
    except ImportError as exc:
        raise ImportError(
            "SteerlingBackbone requires the `steerling` package. "
            "Install it with: pip install steerling  (requires Python >= 3.13)"
        ) from exc

def _import_steerling_generator():
    """Lazily import the official Steerling inference loader."""
    try:
        from steerling import SteerlingGenerator
        return SteerlingGenerator
    except ImportError as exc:
        raise ImportError(
            "SteerlingBackbone requires the `steerling` package. "
            "Install it with: pip install steerling  (requires Python >= 3.13)"
        ) from exc


class SteerlingBackbone(nn.Module):
    """Steerling text backbone.

    The layer wraps ``steerling.models.causal_diffusion.CausalDiffusionLM`` and
    returns final hidden states with shape ``(batch, sequence, n_embd)``.

    Args:
        config: Optional Steerling model config as a mapping, dataclass, or
            Pydantic-style object. If omitted, the Steerling-8B Hub config is
            used.
        vocab_size: Optional vocabulary size override. Defaults to the config
            value, then to the local Steerling tokenizer vocabulary size.
        load_weights: If ``True``, load Steerling-8B backbone weights from the
            Hugging Face Hub.
        freeze: If ``True``, set all backbone parameters to
            ``requires_grad=False`` after loading.
        device: Device for the transformer weights. CUDA requests fall back to
            CPU when CUDA is unavailable.

    Attributes:
        out_features: Hidden size ``n_embd`` emitted by :meth:`forward`.
        vocab_size: Vocabulary size used to instantiate the transformer.
        tokenizer: Steerling tokenizer matching the configured model id.
        model_id: Hub model id used for weights/tokenizer, or ``None`` when
            weights were not loaded.

    Example:
        >>> backbone = SteerlingBackbone(load_weights=True, freeze=True)
        >>> input_ids = torch.tensor([[1, 2, 3]])
        >>> hidden = backbone(input_ids)
        >>> tuple(hidden.shape)
        (1, 3, backbone.out_features)
    """

    def __init__(
        self,
        config: Any = None,
        vocab_size: int = None,
        load_weights: bool = True,
        freeze: bool = True,
        device: str = "cuda",
    ):
        super().__init__()

        # Resolve device before any downloads or model instantiation.
        # Falls back to CPU gracefully if CUDA is requested but unavailable.
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU.")
            device = "cpu"

        # Lazily import the Steerling transformer classes
        CausalDiffusionLM, CausalDiffusionConfig = _import_steerling_transformer()

        if config is None and load_weights:
            SteerlingGenerator = _import_steerling_generator()
            generator = SteerlingGenerator.from_pretrained(
                DEFAULT_MODEL_ID,
                device=device,
            )
            full_model = generator.model
            self.transformer = getattr(full_model, "transformer", full_model).to(device)

            self._out_features = int(generator.model_config.n_embd)
            self._vocab_size = int(
                vocab_size if vocab_size is not None
                else getattr(generator.tokenizer, "vocab_size", DEFAULT_VOCAB_SIZE)
            )
            self._model_id = DEFAULT_MODEL_ID
            self._tokenizer = generator.tokenizer
            self._tokenizer_model_id = DEFAULT_MODEL_ID

            if freeze:
                self.transformer.requires_grad_(False)
            
            return

        # Fallback path: build the plain CausalDiffusionLM backbone directly.
        # Used when config is provided, or when load_weights=False.
        if config is None:
            model_cfg_dict, _, _ = resolve_steerling_configs(
                config_source="pyc",
                model_id=DEFAULT_MODEL_ID,
            )
        else:
            model_cfg_dict = config_to_dict(config)

        config = CausalDiffusionConfig(**model_cfg_dict)

        self._vocab_size = int(
            vocab_size
            if vocab_size is not None
            else model_cfg_dict.get("vocab_size", DEFAULT_VOCAB_SIZE)
        )

        self.transformer = CausalDiffusionLM(
            config,
            vocab_size=self._vocab_size,
        ).to(device)

        self._out_features = int(config.n_embd)

        model_id: str | None = DEFAULT_MODEL_ID if load_weights else None
        self._model_id = model_id

        if load_weights:
            state_dict = load_steerling_backbone_weights(
                DEFAULT_MODEL_ID,
                device=device,
            )
            missing, unexpected = self.transformer.load_state_dict(
                state_dict,
                strict=False,
            )
            logger.info(
                "Loaded pretrained Steerling backbone weights. "
                "Missing keys: %s; unexpected keys: %s",
                missing,
                unexpected,
            )

        if freeze:
            self.transformer.requires_grad_(False)

        self._tokenizer = None
        self._tokenizer_model_id = DEFAULT_MODEL_ID

    @property
    def out_features(self) -> int:
        """Hidden dimension ``n_embd`` returned by :meth:`forward`."""
        return self._out_features

    @property
    def vocab_size(self) -> int:
        """Vocabulary size used by the Steerling transformer."""
        return self._vocab_size

    @property
    def tokenizer(self):
        """Tokenizer corresponding to ``model_id`` or the default Steerling model."""
        if self._tokenizer is None:
            self._tokenizer = get_steerling_tokenizer(self._tokenizer_model_id)
        return self._tokenizer

    @property
    def model_id(self) -> Optional[str]:
        """Hub model id used for weights/tokenizer, or ``None`` without weights."""
        return self._model_id

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        input_embeds: torch.Tensor | None = None,
        return_hidden: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices [B, T] (may contain mask tokens)
            input_embeds: Pre-computed embeddings [B, T, D]. If provided, input_ids is ignored.
            return_hidden: If True, return hidden states before lm_head.

        Returns:
            logits [B, T, vocab_size] or hidden_states [B, T, n_embd]
        """
        return self.transformer(input_ids, return_hidden=return_hidden, input_embeds=input_embeds)

    def __repr__(self) -> str:
        params_b = sum(p.numel() for p in self.parameters()) / 1e9
        return (
            f"SteerlingBackbone(out_features={self._out_features}, "
            f"params={params_b:.1f}B)"
        )
