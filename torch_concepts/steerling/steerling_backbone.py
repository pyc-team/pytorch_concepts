"""
Steerling transformer backbone for text feature extraction.

Provides a :class:`SteerlingBackbone` that follows the same pattern as
:class:`~torch_concepts.data.backbone.Backbone` (for images) but wraps
the Steerling causal-diffusion transformer for text.

When ``pretrained=True`` the full transformer weights are downloaded
from HuggingFace Hub (~16 GB for Steerling-8B).

Example::

    from torch_concepts.steerling import SteerlingBackbone

    backbone = SteerlingBackbone(pretrained=True, freeze=True, device="cuda")
    tokenizer = backbone.tokenizer

    tokens = tokenizer.encode("The key to understanding AI is")
    input_ids = torch.tensor([tokens], device="cuda")
    hidden = backbone(input_ids)   # (1, T, 4096)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
DEFAULT_VOCAB_SIZE = 100277


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


class SteerlingBackbone(nn.Module):
    """Steerling transformer backbone for text feature extraction.

    Wraps ``steerling.models.causal_diffusion.CausalDiffusionLM`` and
    exposes it as a PyC-style backbone: ``input_ids → hidden_states``.

    The backbone returns the **final hidden states** after all
    transformer blocks and layer-norm, with shape ``(B, T, n_embd)``.

    Parameters
    ----------
    pretrained : bool or str
        If ``True``, download Steerling-8B weights from HuggingFace
        Hub.  A string is interpreted as a custom model id / path.
        ``False`` = random init (requires explicit config kwargs).
    freeze : bool
        If ``True``, freeze all parameters after loading.
    device : str, optional
        Device to load onto (``"cpu"``, ``"cuda"``, etc.).
        Defaults to ``"cpu"``.
    dtype : torch.dtype, optional
        Parameter dtype.  Defaults to ``torch.bfloat16`` for pretrained,
        ``None`` (= float32) for random init.

    Attributes
    ----------
    out_features : int
        Hidden dimension of the transformer (``n_embd``).
    tokenizer
        A ``SteerlingTokenizer`` instance for encoding text.
    """

    def __init__(
        self,
        pretrained: bool | str = True,
        freeze: bool = True,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # Resolve device before any downloads or model instantiation.
        # Falls back to CPU gracefully if CUDA is requested but unavailable.
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU.")
            device = "cpu"

        CausalDiffusionLM, CausalDiffusionConfig = _import_steerling_transformer()
        from torch_concepts.steerling.steerling_utils import (
            DEFAULT_MODEL_ID,
            load_steerling_config,
            load_steerling_backbone_weights,
        )

        model_id: str | None = (
            DEFAULT_MODEL_ID if pretrained is True
            else pretrained if isinstance(pretrained, str) and pretrained
            else None
        )

        self._model_id = model_id
        model_cfg_dict: dict[str, object] = {}

        if model_id is not None:
            model_cfg_dict, _ = load_steerling_config(model_id)
            config_values = {
                k: v for k, v in model_cfg_dict.items()
                if k in CausalDiffusionConfig.model_fields
            }
            config_values.pop("model_type", None)
            config = CausalDiffusionConfig(**config_values)
        else:
            config = CausalDiffusionConfig()

        vocab_size = int(model_cfg_dict.get("vocab_size", DEFAULT_VOCAB_SIZE))
        self.transformer = CausalDiffusionLM(config, vocab_size=vocab_size)
        self._out_features = config.n_embd

        if model_id is not None:
            logger.info("Loading backbone weights from %s...", model_id)
            state_dict = load_steerling_backbone_weights(model_id, device=device)
            # strict=False: CausalDiffusionLM may declare concept-head keys
            # that are intentionally absent from the backbone-only state dict.
            self.transformer.load_state_dict(state_dict, strict=False)
            if dtype is None:
                dtype = torch.bfloat16

        if dtype is not None:
            self.transformer = self.transformer.to(dtype=dtype)

        self.transformer = self.transformer.to(device)
        self.transformer.eval()

        if freeze:
            self.transformer.requires_grad_(False)

        self._tokenizer: Optional[object] = None

    @property
    def out_features(self) -> int:
        """Hidden dimension of the backbone (``n_embd``)."""
        return self._out_features

    @property
    def tokenizer(self):
        """The Steerling ``AutoTokenizer`` for encoding text."""
        if self._tokenizer is None:
            from torch_concepts.steerling.steerling_utils import (
                get_steerling_tokenizer, DEFAULT_MODEL_ID,
            )
            self._tokenizer = get_steerling_tokenizer(
                self._model_id or DEFAULT_MODEL_ID
            )
        return self._tokenizer

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from the transformer.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token ids, shape ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Hidden states ``(B, T, n_embd)`` after all layers + LN.
        """
        return self.transformer(input_ids, return_hidden=True)

    def __repr__(self) -> str:
        params_b = sum(p.numel() for p in self.parameters()) / 1e9
        return (
            f"SteerlingBackbone(out_features={self._out_features}, "
            f"params={params_b:.1f}B)"
        )
