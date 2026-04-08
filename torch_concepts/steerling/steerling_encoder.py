"""
Steerling-style concept encoder.

Wraps the official ``steerling.ConceptHead`` behind the standard PyC
:class:`BaseEncoder` interface.

When ``pretrained=True`` the encoder loads the known-concept-head
weights from Steerling-8B on HuggingFace Hub (only ~553 MB, not the
full 16 GB model).

References:
    Guide Labs, "Scaling Interpretable Language Models to 8 Billion
    Parameters", 2025.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from torch_concepts.nn.modules.low.base.layer import BaseEncoder

logger = logging.getLogger(__name__)


def _import_concept_head():
    try:
        import steerling.models.interpretable.concept_head as _ch
        from steerling.models.interpretable.concept_head import (
            ConceptHead,
            ConceptHeadOutput,
        )
        # Allow dense logits for all heads (including >50k unknown concepts).
        # Users should be mindful of memory when using return_logits=True on
        # large heads.
        _ch.LARGE_CONCEPT_THRESHOLD = float("inf")
        return ConceptHead, ConceptHeadOutput
    except ImportError as exc:
        raise ImportError(
            "SteerlingLatentToConcept requires the `steerling` package. "
            "Install it with: pip install steerling  (requires Python >= 3.13)"
        ) from exc


class SteerlingLatentToConcept(BaseEncoder):
    """PyC encoder that wraps the official steerling ``ConceptHead``.

    **forward()** returns raw concept logits ``(*, out_concepts)``
    before sigmoid.

    Args:
        in_latent: Hidden dimension (``n_embd``).
        out_concepts: Number of concepts.
        embedding_size: Per-concept embedding dimension.
            Defaults to ``in_latent``.
        topk: Top-k sparsity.  ``None`` = dense.
        use_attention: Sigmoid-attention scoring instead of linear.
        is_unknown: Unknown-concept mode (no GT / teacher forcing).
        factorize: Low-rank factorised embeddings.
        factorize_rank: Rank *r* when ``factorize=True``.
        pretrained: If ``True``, load Steerling-8B known-head weights
            from HuggingFace Hub (dimensions are inferred from the
            checkpoint config).  A string is interpreted as a custom
            model id / path.  ``False`` or ``None`` = random init.
        freeze: Freeze all parameters after construction.
        concept_head_kwargs: Extra kwargs forwarded to ``ConceptHead``.

    Example::

        # Random init
        enc = SteerlingLatentToConcept(in_latent=128, out_concepts=500)

        # Pretrained Steerling-8B known head, frozen
        enc = SteerlingLatentToConcept(pretrained=True, freeze=True)
    """

    def __init__(
        self,
        in_latent: int | None = None,
        out_concepts: int | None = None,
        embedding_size: int | None = None,
        topk: int = 16,
        use_attention: bool = False,
        is_unknown: bool = False,
        factorize: bool = False,
        factorize_rank: int = 256,
        apply_topk_to_unknown: bool = False,
        pretrained: bool | str = False,
        freeze: bool = False,
        concept_head_kwargs: Optional[Dict[str, Any]] = None,
    ):
        ConceptHead, _ = _import_concept_head()

        # Resolve pretrained -> model_id (or None)
        if pretrained is True:
            from torch_concepts.steerling.steerling_utils import DEFAULT_MODEL_ID
            model_id: str | None = DEFAULT_MODEL_ID
        elif isinstance(pretrained, str) and pretrained:
            model_id = pretrained
        else:
            model_id = None

        # When pretrained, fill missing args from the checkpoint config
        if model_id is not None:
            from torch_concepts.steerling.steerling_utils import load_steerling_config

            model_cfg, concept_cfg = load_steerling_config(model_id)
            # concept keys live at the top level (model_cfg), concept_cfg
            # may be empty depending on the checkpoint format.
            cfg = {**model_cfg, **concept_cfg}
            in_latent = in_latent or cfg.get("n_embd", 4096)
            embedding_size = embedding_size or cfg.get("concept_dim", in_latent)

            if is_unknown:
                out_concepts = out_concepts or cfg.get("n_unknown_concepts", 101196)
                topk = cfg.get("unknown_topk", topk)
                use_attention = cfg.get("use_attention_unknown", use_attention)
                factorize = cfg.get("factorize_unknown", factorize)
                factorize_rank = cfg.get("factorize_rank", factorize_rank)
                apply_topk_to_unknown = cfg.get("apply_topk_to_unknown", True)
            else:
                out_concepts = out_concepts or cfg.get("n_concepts", 33732)
                topk = cfg.get("topk_known", topk)
                use_attention = cfg.get("use_attention_known", use_attention)

        if in_latent is None or out_concepts is None:
            raise ValueError(
                "in_latent and out_concepts are required when pretrained "
                "is False"
            )

        super().__init__(in_latent=in_latent, out_concepts=out_concepts)

        self.embedding_size = embedding_size if embedding_size is not None else in_latent

        head_kwargs = dict(
            n_concepts=out_concepts,
            concept_dim=self.embedding_size,
            n_embd=in_latent,
            is_unknown=is_unknown,
            use_attention=use_attention,
            topk=topk,
            factorize=factorize,
            factorize_rank=factorize_rank,
            enforce_gt_for_known=False,
            apply_topk_to_unknown=apply_topk_to_unknown,
        )
        if concept_head_kwargs:
            head_kwargs.update(concept_head_kwargs)

        self.head: nn.Module = ConceptHead(**head_kwargs)

        # Load pretrained weights
        if model_id is not None:
            if is_unknown:
                from torch_concepts.steerling.steerling_utils import load_steerling_unknown_head_weights
                state_dict = load_steerling_unknown_head_weights(model_id)
            else:
                from torch_concepts.steerling.steerling_utils import load_steerling_known_head_weights
                state_dict = load_steerling_known_head_weights(model_id)
            self.head.load_state_dict(state_dict, strict=False)

        # Freeze
        if freeze:
            for p in self.head.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embeddings(self) -> torch.Tensor:
        """Return the concept embedding weight matrix ``(out_concepts, embedding_size)``.

        For standard (non-factorized) heads this returns
        ``concept_embedding.weight.data`` directly.  For factorized heads
        (typically the unknown head) the full embedding matrix is
        reconstructed as ``embedding_coef @ embedding_basis.weight.T``.

        The result is truncated to ``self.out_concepts`` rows because
        the underlying ``Embedding`` layers may have extra padding rows.
        """
        head = self.head
        if getattr(head, "concept_embedding", None) is not None:
            return head.concept_embedding.weight.data[:self.out_concepts]
        # Factorized: coef (N, r) @ basis (D, r).T -> (N, D)
        full = head.embedding_coef.weight.data @ head.embedding_basis.weight.data.T
        return full[:self.out_concepts]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return concept output tensor ``(*, out_concepts)`` or ``(*, D)``.

        For known heads (non-factorized), returns raw concept logits
        ``(*, out_concepts)`` before sigmoid.

        For unknown heads (factorized/large), dense logits are not
        available.  Returns the ``predicted`` feature tensor
        ``(*, D)`` — i.e. the top-k weighted concept embeddings that
        reconstruct the hidden state contribution.

        Accepts 2-D ``(batch, D)`` or 3-D ``(batch, seq, D)`` input.
        """
        squeezed = False
        if latent.dim() == 2:
            latent = latent.unsqueeze(1)
            squeezed = True

        is_large = getattr(self.head, "_is_large", False)

        if is_large:
            # Unknown / large head: use sparse path, return features
            out = self.head(latent, use_teacher_forcing=False, return_logits=False)
            result = out.predicted
        else:
            # Known / small head: return dense logits
            out = self.head(latent, use_teacher_forcing=False, return_logits=True)
            result = out.logits

        if squeezed:
            result = result.squeeze(1)
        return result

    # ------------------------------------------------------------------
    # Full output
    # ------------------------------------------------------------------

    def forward_full(self, latent: torch.Tensor, **kwargs):
        """Return the full ``ConceptHeadOutput`` dataclass."""
        squeezed = False
        if latent.dim() == 2:
            latent = latent.unsqueeze(1)
            squeezed = True

        kwargs.setdefault("use_teacher_forcing", False)
        out = self.head(latent, **kwargs)

        if squeezed:
            out.features = out.features.squeeze(1)
            out.predicted = out.predicted.squeeze(1)
            if out.logits is not None:
                out.logits = out.logits.squeeze(1)
            if out.weights is not None:
                out.weights = out.weights.squeeze(1)
            if out.topk_indices is not None:
                out.topk_indices = out.topk_indices.squeeze(1)
            if out.topk_logits is not None:
                out.topk_logits = out.topk_logits.squeeze(1)
        return out


class SteerlingConceptExogenousToLatent(nn.Module):
    r"""Reconstruct a concept-based hidden state from concept activations
    and their embeddings.

    Implements the Steerling reconstruction:

    .. math::

        \bar{h} = \sum_{i=1}^{n} k_i\, K_i \;+\; \sum_{j=1}^{m} u_j\, U_j

    where :math:`k_i` / :math:`u_j` are concept activations (after sigmoid)
    and :math:`K_i` / :math:`U_j` are the corresponding learned concept
    embeddings.

    Embedding matrices are stored as non-learnable buffers at construction
    time (they come from the pretrained ``SteerlingLatentToConcept`` heads).
    The ``forward`` signature uses only **concepts** so that the PyC
    inference engine routes it through the ``{'concepts'}`` path
    (``torch.cat(parent_concepts, dim=-1)``).

    Args:
        known_embeddings: ``(n_known, D)`` embedding matrix for known
            (supervised) concepts.
        unknown_embeddings: ``(n_unknown, D)`` embedding matrix for
            unknown (unsupervised) concepts.

    Example::

        recon = SteerlingConceptExogenousToLatent(
            known_embeddings=sup_head.get_embeddings(),    # (n_known, D)
            unknown_embeddings=unsup_head.get_embeddings(),  # (n_unknown, D)
        )

        # Via the PGM inference engine:
        h_bar_cpd = ParametricCPD(
            "h_bar",
            parents=sup_concepts_names + unsup_concepts_names,
            parametrization=recon,
        )

        # Manually:
        h_bar = recon(
            concepts=torch.cat([k_act, u_act], dim=-1),   # (B, N)
        )  # → (B, D)
    """

    def __init__(self, known_embeddings: torch.Tensor, unknown_embeddings: torch.Tensor):
        super().__init__()
        self.register_buffer("embeddings", torch.cat([known_embeddings, unknown_embeddings], dim=0))

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        r"""Reconstruct the concept-based hidden state.

        Args:
            concepts: ``(*, N)`` concatenated concept activations
                (post-sigmoid), where ``N`` matches ``self.embeddings``
                first dimension.

        Returns:
            ``(*, D)`` reconstructed hidden state
            :math:`\bar{h} = \text{concepts} \times \text{embeddings}`.
        """
        return concepts @ self.embeddings
