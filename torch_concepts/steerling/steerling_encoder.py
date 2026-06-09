"""Steerling latent-to-concept encoder.

``SteerlingLatentToConcept`` wraps the official Steerling ``ConceptHead`` and
adapts it to the PyC ``BaseEncoder`` interface. It maps transformer hidden
states to dense concept logits. The concept embeddings exposed by this class
are used by downstream mixing layers to reconstruct Steerling concept features.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from torch_concepts.nn.modules.low.base.layer import BaseEncoder
from torch_concepts.steerling.steerling_configs import (
    DEFAULT_MODEL_ID,
    config_to_dict,
    normalize_concept_config,
    resolve_steerling_configs,
)

logger = logging.getLogger(__name__)

_CONCEPT_CONFIG_ALIASES = {
    False: {
        "topk_known_features": "topk_features",
        "topk_known": "topk",
        "use_attention_known": "use_attention",
    },
    True: {
        "factorize_unknown": "factorize",
        "n_unknown_concepts": "n_concepts",
        "unknown_topk": "topk",
        "use_attention_unknown": "use_attention",
    },
}

def _concept_head_init_keys(ConceptHead: type[nn.Module]) -> set[str]:
    """Return kwargs accepted by upstream ``ConceptHead.__init__``."""
    signature = inspect.signature(ConceptHead.__init__)
    return {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def _filter_concept_head_config(
    config: Optional[Dict[str, Any]],
    allowed_keys: set[str],
    is_unknown: bool,
) -> dict[str, Any]:
    """Keep only external config entries accepted by ``ConceptHead``."""
    if not config:
        return {}
    filtered = {}
    ignored_keys = []
    aliases = _CONCEPT_CONFIG_ALIASES[is_unknown]
    for key, value in config.items():
        target_key = aliases.get(key)
        if target_key is not None:
            if target_key in allowed_keys:
                filtered[target_key] = value
            else:
                ignored_keys.append(key)
        elif key in allowed_keys:
            filtered[key] = value
        else:
            ignored_keys.append(key)
    if ignored_keys:
        logger.info(f"Ignoring unsupported ({'unknown' if is_unknown else 'known'}) \
                    ConceptHead config keys: {ignored_keys}")
    return filtered


def _import_concept_head():
    try:
        import steerling.models.interpretable.concept_head as _ch
        from steerling.models.interpretable.concept_head import ConceptHead
        # Allow dense logits for all heads (including >50k unknown concepts).
        _ch.LARGE_CONCEPT_THRESHOLD = float("inf")
        return ConceptHead
    except ImportError as exc:
        raise ImportError(
            "SteerlingLatentToConcept requires the `steerling` package. "
            "Install it with: pip install steerling  (requires Python >= 3.13)"
        ) from exc


class SteerlingLatentToConcept(BaseEncoder):
    """PyC wrapper around Steerling's latent-to-concept head.

    The encoder accepts hidden states with shape ``(..., in_latent)`` and
    returns raw dense concept logits with shape ``(..., out_concepts)``.
    :meth:`forward` always scores densely from the head's projection/embedding
    weights and never applies Steerling's top-k sparsification; the wrapper does
    not itself sanitize top-k config keys (the owning model is responsible).

    If ``config`` is omitted, the Steerling-8B Hub concept config is used.
    ``ConceptConfig`` names that differ from ``ConceptHead`` names are mapped
    for the selected head. For example, known heads use
    ``use_attention_known -> use_attention`` and unknown heads use
    ``factorize_unknown -> factorize``. Constructor arguments such as
    ``embedding_size`` and ``factorize`` override config-derived values when
    they are not ``None``.

    Official Steerling-8B uses projection mode by default
    (``use_attention_known=False`` and ``use_attention_unknown=False``). The
    attention path remains supported for configs that enable it.

    Args:
        in_latent: Hidden-state dimension ``n_embd``.
        out_concepts: Number of concept logits to emit.
        embedding_size: Per-concept embedding dimension. Defaults to config
            ``concept_dim`` and then to ``in_latent``.
        is_unknown: Whether this instance represents the unknown/discovered
            concept head. Controls side-specific config aliasing.
        use_attention: Optional override for attention-style concept scoring.
            ``None`` means use the resolved config value.
        factorize: Optional override for low-rank factorized embeddings.
            ``None`` means use the resolved config value.
        factorize_rank: Optional low-rank dimension for factorized heads.
            ``None`` means use the resolved config value or ``256``.
        config: Optional concept config mapping/object. If omitted, the Hub
            default concept config is resolved.
    Attributes:
        head: Wrapped Steerling ``ConceptHead`` module.
        embedding_size: Concept embedding dimension.
        use_attention: Whether logits are computed by the attention-style
            embedding scorer.
        factorize: Whether embeddings/predictor weights use low-rank factors.

    Example:
        >>> enc = SteerlingLatentToConcept(4096, 33732, is_unknown=False)  # doctest: +SKIP
        >>> hidden = torch.randn(2, 8, 4096)  # doctest: +SKIP
        >>> logits = enc(hidden)  # doctest: +SKIP
        >>> logits.shape  # doctest: +SKIP
        torch.Size([2, 8, 33732])
    """

    def __init__(
        self,
        in_latent: int,
        out_concepts: int,
        embedding_size: int | None = None,
        is_unknown: bool = False,
        use_attention: bool | None = None,
        factorize: bool | None = None,
        factorize_rank: int | None = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(in_latent=in_latent, out_concepts=out_concepts)

        ConceptHead = _import_concept_head()

        self.is_unknown = is_unknown
        if config is None:
            _, config = resolve_steerling_configs(
                config_source="hub",
                model_id=DEFAULT_MODEL_ID,
            )
        else:
            config = normalize_concept_config(config_to_dict(config))

        # Build kwargs for the ConceptHead constructor, prioritizing
        # explicit wrapper args over config entries.
        head_kwargs = self._build_concept_head_kwargs(
            ConceptHead=ConceptHead,
            config=config,
            in_latent=in_latent,
            out_concepts=out_concepts,
            embedding_size=embedding_size,
            is_unknown=is_unknown,
            use_attention=use_attention,
            factorize=factorize,
            factorize_rank=factorize_rank,
        )
        self.embedding_size = int(head_kwargs["concept_dim"])

        # Initialize the ConceptHead with the resolved kwargs
        self.head: nn.Module = ConceptHead(**head_kwargs)
        self.use_attention = bool(
            getattr(self.head, "use_attention", head_kwargs["use_attention"])
        )
        self.factorize = bool(getattr(self.head, "factorize", head_kwargs["factorize"]))

    @staticmethod
    def _build_concept_head_kwargs(
        *,
        ConceptHead: type[nn.Module],
        config: Optional[Dict[str, Any]],
        in_latent: int,
        out_concepts: int,
        embedding_size: int | None,
        is_unknown: bool,
        use_attention: bool | None,
        factorize: bool | None,
        factorize_rank: int | None,
    ) -> dict[str, Any]:
        """Build upstream ``ConceptHead`` kwargs from config plus explicit args.

        Precedence is intentionally simple:

        1. Copy every external config key accepted by ``ConceptHead``.
        2. Override the few fields exposed by the PyC wrapper interface.

        The encoder builds faithfully from the config it is given. It does not
        sanitize top-k: keeping the layer's behavior a pure function of its
        config is the caller's contract. The owning model is responsible for
        passing a config that is coherent with the rest of the graph (e.g.
        ``SteerlingLowLevelModel`` disables top-k before constructing heads).
        """
        allowed_keys = _concept_head_init_keys(ConceptHead)
        head_kwargs = _filter_concept_head_config(config, allowed_keys, is_unknown)

        concept_dim = (
            embedding_size
            if embedding_size is not None
            else int(head_kwargs.get("concept_dim", in_latent))
        )
        if use_attention is None:
            use_attention = bool(head_kwargs.get("use_attention", False))
        if factorize is None:
            factorize = bool(head_kwargs.get("factorize", False))
        if factorize_rank is None:
            factorize_rank = int(head_kwargs.get("factorize_rank", 256))

        head_kwargs.update(
            n_concepts=out_concepts,
            concept_dim=concept_dim,
            n_embd=in_latent,
            is_unknown=is_unknown,
            use_attention=use_attention,
            factorize=factorize,
            factorize_rank=factorize_rank,
        )
        return head_kwargs

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embeddings(self, factorized: bool | None = None):
        """Return concept embeddings for downstream mixing.

        Args:
            factorized: Controls the returned layout. ``False`` always returns
                dense embeddings. ``True`` or ``None`` returns packed
                factorized components when this head is factorized; otherwise
                it returns dense embeddings.

        Returns:
            Dense layout: ``(out_concepts, embedding_size)``.
            Packed factorized layout: ``(out_concepts + embedding_size, rank)``,
            where the first ``out_concepts`` rows are coefficients and the
            remaining rows are basis vectors.
        """
        if factorized is not False and self.factorize:
            return self.get_embedding_components()
        return self.head._get_embedding_weight()[:self.out_concepts]

    def get_embedding_components(self) -> torch.Tensor:
        """Return the layout expected by a factorized Steerling mixer.

        Factorized heads return a packed tensor containing coefficient rows
        followed by basis rows. Non-factorized heads fall back to the dense
        embedding matrix so callers can use this method without branching.
        """
        if not self.factorize:
            return self.head._get_embedding_weight()[:self.out_concepts]

        coef = self.head.embedding_coef.weight[:self.out_concepts]
        basis = self.head.embedding_basis.weight
        return torch.cat([coef, basis], dim=0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dense raw concept logits.

        Args:
            latent: Hidden states with shape ``(batch, in_latent)`` or
                ``(batch, sequence, in_latent)``.

        Returns:
            Dense logits with shape ``(batch, out_concepts)`` for 2-D input or
            ``(batch, sequence, out_concepts)`` for 3-D input.
        """
        squeeze = latent.dim() == 2
        if squeeze:
            latent = latent.unsqueeze(1)

        if self.use_attention:
            embeddings = self.get_embeddings(factorized=False)
            query = self.head.concept_query_projection(latent)
            logits = self.head.blocked_logits(
                query,
                embeddings,
                block_size=int(getattr(self.head, "block_size", 8192)),
            )
        elif self.factorize:
            # TODO: use the factorized score path
            # (`q_compressed = latent @ basis.T` → `q_compressed @ coef.T`)
            # instead of materializing the full (C, D) predictor weight, to
            # recover the memory win of factorization at inference time.
            weight = self.head._get_predictor_weight()[:self.out_concepts]
            logits = (latent @ weight.T).float().clamp(-15, 15)
        else:
            logits = (
                self.head.concept_predictor(latent)[..., :self.out_concepts]
                .float()
                .clamp(-15, 15)
            )

        # Restore the caller's dtype.  `.float()` above is for safe clamping
        # in fp32.
        logits = logits.to(latent.dtype)

        return logits.squeeze(1) if squeeze else logits
