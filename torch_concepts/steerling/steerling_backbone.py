"""Steerling transformer backbone for text feature extraction.

``CausalDiffusionTextBackbone`` wraps the official Steerling causal-diffusion
transformer and exposes the PyC-friendly mapping ``input_ids -> hidden``.
It is responsible only for token-level hidden states; concept scoring and
concept mixing are handled by the Steerling encoder and mixer layers.

By default the backbone builds the Steerling-8B architecture from the Hub
config and exposes the matching tokenizer. Weight loading and freezing are
handled by the owning model.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from .steerling_configs import (
    DEFAULT_MODEL_ID,
    SteerlingConfigSource,
    config_to_dict,
    resolve_steerling_configs,
)
from .steerling_utils import get_steerling_tokenizer

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
            "CausalDiffusionTextBackbone requires the `steerling` package. "
            "Install it with: pip install steerling  (requires Python >= 3.13)"
        ) from exc


def _causal_diffusion_config_fields(CausalDiffusionConfig: type) -> set[str]:
    """Return fields accepted by the upstream pydantic config class."""
    if hasattr(CausalDiffusionConfig, "model_fields"):
        return set(CausalDiffusionConfig.model_fields)
    if hasattr(CausalDiffusionConfig, "__fields__"):
        return set(CausalDiffusionConfig.__fields__)
    return set()


def _to_causal_diffusion_config_kwargs(
    CausalDiffusionConfig: type,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Strip HF wrapper metadata before constructing ``CausalDiffusionConfig``."""
    config = dict(config)
    if config.get("model_type") == "steerling":
        config["model_type"] = "causal_diffusion"

    fields = _causal_diffusion_config_fields(CausalDiffusionConfig)
    if not fields:
        return config
    return {key: value for key, value in config.items() if key in fields}


class CausalDiffusionTextBackbone(nn.Module):
    """Steerling text backbone.

    The layer wraps ``steerling.models.causal_diffusion.CausalDiffusionLM`` and
    returns final hidden states with shape ``(batch, sequence, n_embd)``.

    Args:
        config: Optional Steerling model config as a mapping, dataclass, or
            Pydantic-style object. If omitted, the Steerling-8B Hub config is
            used.
        model_id: Hugging Face model id or local path used for default config,
            and tokenizer.
        config_source: Source used to resolve the default config when
            ``config`` is omitted.
        vocab_size: Optional vocabulary size override. Defaults to the config
            value, then to the local Steerling default vocabulary size.

    Attributes:
        out_features: Hidden size ``n_embd`` emitted by :meth:`forward`.
        vocab_size: Vocabulary size used to instantiate the transformer.
        tokenizer: Steerling tokenizer matching the configured model id.
        model_id: Hub model id used for default config and tokenizer.

    Example:
        >>> backbone = CausalDiffusionTextBackbone()
        >>> input_ids = torch.tensor([[1, 2, 3]])
        >>> hidden = backbone(input_ids)
        >>> hidden.shape[-1] == backbone.out_features
        True
    """

    def __init__(
        self,
        config: Any = None,
        model_id: str = DEFAULT_MODEL_ID,
        config_source: SteerlingConfigSource = "hub",
        vocab_size: int = None,
    ):
        super().__init__()

        # Lazily import the Steerling transformer classes
        CausalDiffusionLM, CausalDiffusionConfig = _import_steerling_transformer()

        if config is None:
            model_cfg_dict, _, _ = resolve_steerling_configs(
                config_source=config_source,
                model_id=model_id,
            )
        else:
            model_cfg_dict = config_to_dict(config)

        self._vocab_size = int(
            vocab_size
            if vocab_size is not None
            else model_cfg_dict.get("vocab_size", DEFAULT_VOCAB_SIZE)
        )

        config_kwargs = _to_causal_diffusion_config_kwargs(
            CausalDiffusionConfig,
            model_cfg_dict,
        )
        config = CausalDiffusionConfig(**config_kwargs)

        self.transformer = CausalDiffusionLM(
            config,
            vocab_size=self._vocab_size,
        )

        self._out_features = int(config.n_embd)

        self._model_id = model_id
        self._tokenizer = None
        self._tokenizer_model_id = model_id

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
    def model_id(self) -> str:
        """Hub model id used for default config and tokenizer."""
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
            input_ids: Token indices [B, T] (may contain mask tokens).
            input_embeds: Pre-computed embeddings [B, T, D]. If provided,
                ``input_ids`` is ignored.
            return_hidden: If True (the default), return hidden states before
                the language-model head; otherwise return token logits.

        Returns:
            With the default ``return_hidden=True``, hidden states with shape
            ``(B, T, n_embd)``. With ``return_hidden=False``, token logits with
            shape ``(B, T, vocab_size)``.
        """
        return self.transformer(input_ids, return_hidden=return_hidden, input_embeds=input_embeds)

    def __repr__(self) -> str:
        params_b = sum(p.numel() for p in self.parameters()) / 1e9
        return (
            f"CausalDiffusionTextBackbone(out_features={self._out_features}, "
            f"params={params_b:.1f}B)"
        )
