"""
Steerling decoders — LM head wrapper and concept-name utilities."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SteerlingLMHead(nn.Module):
    """Pretrained Steerling LM head for use as a PGM decoder.

    Maps a hidden state ``(*, D)`` to next-token logits ``(*, vocab)``
    via the weight-tied token-embedding projection.

    Follows the same ``pretrained`` / ``freeze`` pattern as
    :class:`SteerlingLatentToConcept`.

    Args:
        pretrained: If ``True``, load Steerling-8B LM-head weights
            from HuggingFace Hub (~4 GB shard, extracts one tensor).
            A string is interpreted as a custom model id / path.
            ``False`` = random init (requires ``vocab_size`` and
            ``n_embd``).
        freeze: Freeze parameters after construction.
        vocab_size: Token vocabulary size (inferred from config when
            pretrained).
        n_embd: Hidden dimension (inferred from config when pretrained).

    Example::

        decoder = SteerlingLMHead(pretrained=True, freeze=True)
        token_pred_cpd = ParametricCPD(
            "token_pred",
            parents=["concept_latent"],
            parametrization=decoder,
        )
    """

    def __init__(
        self,
        pretrained: bool | str = True,
        freeze: bool = True,
        vocab_size: int | None = None,
        n_embd: int | None = None,
        device: str = "cuda",
    ):
        super().__init__()

        # Resolve device before any downloads or model instantiation.
        # Falls back to CPU gracefully if CUDA is requested but unavailable.
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU.")
            device = "cpu"

        # Resolve model id
        if pretrained is True:
            from torch_concepts.steerling.steerling_utils import DEFAULT_MODEL_ID
            model_id: str | None = DEFAULT_MODEL_ID
        elif isinstance(pretrained, str) and pretrained:
            model_id = pretrained
        else:
            model_id = None

        # Fill dims from config when pretrained
        if model_id is not None:
            from torch_concepts.steerling.steerling_utils import load_steerling_config
            model_cfg, _ = load_steerling_config(model_id)
            vocab_size = vocab_size or model_cfg.get("vocab_size", 100281)
            n_embd = n_embd or model_cfg.get("n_embd", 4096)

        if vocab_size is None or n_embd is None:
            raise ValueError(
                "vocab_size and n_embd are required when pretrained is False"
            )

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Load pretrained weights (tok_emb.weight, weight-tied)
        if model_id is not None:
            from torch_concepts.steerling.steerling_utils import load_steerling_lm_head_weights
            state_dict = load_steerling_lm_head_weights(model_id, device=device)
            self.lm_head.load_state_dict(state_dict)
            logger.info("Loaded LM head weights (%d vocab, %d embd)", vocab_size, n_embd)

        self.lm_head = self.lm_head.to(device)

        if freeze:
            for p in self.lm_head.parameters():
                p.requires_grad = False

    @property
    def vocab_size(self) -> int:
        return self.lm_head.out_features

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Map ``(*, D)`` hidden states to ``(*, vocab)`` logits (float32)."""
        target_device = self.lm_head.weight.device
        target_dtype = self.lm_head.weight.dtype
        return self.lm_head(latent.to(device=target_device, dtype=target_dtype)).float()


def prepare_generation_sequence(
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> tuple[torch.Tensor, torch.BoolTensor, torch.BoolTensor]:
    """Build the ``[prompt | MASK × N]`` input for causal-diffusion generation.

    Parameters
    ----------
    tokenizer : SteerlingTokenizer
        Tokenizer with ``mask_token_id``.
    prompt : str
        The text prompt.
    max_new_tokens : int
        Number of masked positions to append.

    Returns
    -------
    input_ids : torch.Tensor
        Shape ``(1, prompt_len + max_new_tokens)``.
    is_finalized : torch.BoolTensor
        Shape ``(prompt_len + max_new_tokens,)``; ``True`` for prompt positions.
    gen_region : torch.BoolTensor
        Shape ``(prompt_len + max_new_tokens,)``; ``True`` for generation positions.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    total_len = prompt_len + max_new_tokens

    input_ids = torch.full((1, total_len), tokenizer.mask_token_id, dtype=torch.long)
    input_ids[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long)

    is_finalized = torch.zeros(total_len, dtype=torch.bool)
    is_finalized[:prompt_len] = True
    gen_region = ~is_finalized.clone()

    return input_ids, is_finalized, gen_region


@torch.no_grad()
def prepare_steerling_evidence(
    backbone,
    prompt: str,
    n_new_tokens: int = 0,
) -> dict:
    """Tokenize, optionally append MASK tokens, and compute hidden states.

    Parameters
    ----------
    backbone : SteerlingBackbone
        Pretrained backbone (used for tokenizer and forward pass).
    prompt : str
        The text prompt.
    n_new_tokens : int
        Number of MASK tokens to append after the prompt for generation.
        Use 0 for pure concept-querying (no generation).

    Returns
    -------
    dict
        ``{"input_ids": (1, T), "hidden": (1, T, D)}``
        where ``T = prompt_len + n_new_tokens`` and ``D = n_embd``.
        Tensors are float32.
    """
    tokenizer = backbone.tokenizer

    if n_new_tokens > 0:
        input_ids, _, _ = prepare_generation_sequence(
            tokenizer, prompt, n_new_tokens)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    device = next(backbone.parameters()).device
    input_ids = input_ids.to(device)
    hidden = backbone(input_ids).float()
    
    return {"input_ids": input_ids, "hidden": hidden}


def print_concepts(
    logits: torch.Tensor,
    topk: int = 10,
) -> "pandas.DataFrame":  # noqa: F821
    """Map concept logits to human-readable concept names.

    Parameters
    ----------
    logits : torch.Tensor
        Concept logits, shape ``(T, n_concepts)`` for a single
        sequence or ``(n_concepts,)`` for a single position.
    topk : int
        Number of top concepts to return per token position.

    Returns
    -------
    pandas.DataFrame
        Columns: ``position``, ``concept_idx``, ``concept_name``,
        ``logit``.
    """
    import pandas as pd
    from torch_concepts.steerling.steerling_utils import load_steerling_concepts

    labels = load_steerling_concepts()

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    rows = []
    for pos in range(logits.shape[0]):
        tk = torch.topk(logits[pos], k=topk)
        for idx, val in zip(tk.indices.tolist(), tk.values.tolist()):
            rows.append({
                "position": pos,
                "concept_idx": idx,
                "concept_name": labels.loc[idx, "concept_name"]
                if idx in labels.index
                else f"<unknown:{idx}>",
                "logit": round(val, 4),
            })

    return pd.DataFrame(rows)
