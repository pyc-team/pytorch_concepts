"""
Steerling hub utilities — download, cache and access pretrained components.

This module handles HuggingFace Hub interaction for Steerling tokenizers,
weights, and concept metadata. Config loading lives in
``steerling_configs.py``.

Public helpers:

* :func:`load_steerling_known_head_weights` – fetch the single shard
  that contains the known-concept-head tensors and return a state dict.
* :func:`load_steerling_backbone_weights` – fetch all shards for the
  transformer backbone and return a state dict.
* :func:`get_steerling_tokenizer` – return a ``SteerlingTokenizer``.
* :func:`load_steerling_concepts` – download concept label CSV.
* :func:`load_steerling_concept_names` – concept_idx → concept_name mapping.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

import torch

from ..utils import resolve_hf_token
from .steerling_configs import DEFAULT_MODEL_ID, load_steerling_hub_config

logger = logging.getLogger(__name__)

# Seed the huggingface_hub global session token once at import time.
# This suppresses "unauthenticated" warnings from any internal HF Hub
# calls (e.g. within the steerling package) that do not go through our
# resolve_hf_token() helpers.
def _login_hf_hub() -> None:
    """Silently log in to the HF Hub if a token is available."""
    token = resolve_hf_token()
    if token is None:
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    except Exception:
        # huggingface_hub is optional; token is still passed per-call.
        pass

_login_hf_hub()


# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------

def get_steerling_tokenizer(model_name_or_path: str = DEFAULT_MODEL_ID):
    """Return the Steerling tokenizer via ``AutoTokenizer``.

    Uses ``trust_remote_code=True`` to pull the custom tokenizer class
    from the HuggingFace Hub.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "get_steerling_tokenizer requires `transformers`. "
            "Install with: pip install transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        token=resolve_hf_token(),
    )

# In-process cache for the safetensors weight-map index.
# huggingface_hub already caches the file on disk; this avoids
# re-parsing the JSON on every function call within the same process.
_weight_map_cache: Dict[str, Dict[str, str]] = {}


def _download_index(model_name_or_path: str) -> Dict[str, str]:
    """Download ``model.safetensors.index.json`` and return its weight_map.

    The result is cached in memory so repeated calls within the same
    process skip the JSON parse entirely.
    """
    if model_name_or_path in _weight_map_cache:
        return _weight_map_cache[model_name_or_path]

    from huggingface_hub import hf_hub_download

    index_path = hf_hub_download(
        model_name_or_path,
        "model.safetensors.index.json",
        token=resolve_hf_token(),
    )
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    _weight_map_cache[model_name_or_path] = weight_map
    return weight_map


def load_steerling_known_head_weights(
    model_name_or_path: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Download only the known-head shard and return a state dict.

    Keys have the ``known_head.`` prefix stripped so they can be loaded
    directly into a ``ConceptHead`` via ``load_state_dict``.

    For Steerling-8B this downloads ~1 GB (shard 4/4) and extracts
    ~553 MB of tensors.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    weight_map = _download_index(model_name_or_path)

    # Find which shard(s) contain known_head tensors
    shards = {
        v for k, v in weight_map.items() if k.startswith("known_head.")
    }

    state_dict: Dict[str, torch.Tensor] = {}
    for shard_file in shards:
        shard_path = hf_hub_download(
            model_name_or_path,
            shard_file,
            token=resolve_hf_token(),
        )
        with safe_open(shard_path, framework="pt", device=device) as f:
            for key in f.keys():
                if key.startswith("known_head."):
                    state_dict[key.removeprefix("known_head.")] = f.get_tensor(key)

    logger.info("Loaded known-head weights from %s", model_name_or_path)
    return state_dict


def load_steerling_unknown_head_weights(
    model_name_or_path: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Download only the unknown-head shard and return a state dict.

    Keys have the ``unknown_head.`` prefix stripped so they can be loaded
    directly into a ``ConceptHead`` via ``load_state_dict``.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    weight_map = _download_index(model_name_or_path)

    shards = {
        v for k, v in weight_map.items() if k.startswith("unknown_head.")
    }

    state_dict: Dict[str, torch.Tensor] = {}
    for shard_file in shards:
        shard_path = hf_hub_download(
            model_name_or_path,
            shard_file,
            token=resolve_hf_token(),
        )
        with safe_open(shard_path, framework="pt", device=device) as f:
            for key in f.keys():
                if key.startswith("unknown_head."):
                    state_dict[key.removeprefix("unknown_head.")] = f.get_tensor(key)

    logger.info("Loaded unknown-head weights from %s", model_name_or_path)
    return state_dict


def load_steerling_lm_head_weights(
    model_name_or_path: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Download the LM-head/tied-token-embedding shard and return a state dict.

    Steerling checkpoints may either store an explicit ``lm_head.weight`` or
    rely on the standard tied-weight layout where the output projection uses
    ``transformer.tok_emb.weight``.  The returned dict is keyed as
    ``{"weight": tensor}``, ready for ``nn.Linear.load_state_dict``.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    weight_map = _download_index(model_name_or_path)
    weight_key = next(
        (
            key
            for key in (
                "lm_head.weight",
                "transformer.lm_head.weight",
                "transformer.tok_emb.weight",
            )
            if key in weight_map
        ),
        None,
    )
    if weight_key is None:
        raise KeyError(
            "Could not find LM-head weights in the Steerling checkpoint index. "
            "Expected one of 'lm_head.weight', 'transformer.lm_head.weight', "
            "or 'transformer.tok_emb.weight'."
        )

    shard_file = weight_map[weight_key]
    shard_path = hf_hub_download(
        model_name_or_path,
        shard_file,
        token=resolve_hf_token(),
    )

    with safe_open(shard_path, framework="pt", device=device) as f:
        weight = f.get_tensor(weight_key)

    logger.info("Loaded lm_head weights from %s:%s", model_name_or_path, weight_key)
    return {"weight": weight}


def load_steerling_backbone_weights(
    model_name_or_path: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Download the transformer backbone weights and return a state dict.

    Keys have the ``transformer.`` prefix stripped so they can be
    loaded directly into a ``CausalDiffusionLM``.

    .. warning::
       For Steerling-8B this downloads the full ~16 GB checkpoint
       (all shards are needed because transformer weights are spread
       across shards 1–3).
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    weight_map = _download_index(model_name_or_path)

    # Find shards that contain transformer.* keys
    shards = {
        v for k, v in weight_map.items() if k.startswith("transformer.")
    }

    state_dict: Dict[str, torch.Tensor] = {}
    logger.info(
        "Loading backbone weights from %s (%d shards)...",
        model_name_or_path, len(shards),
    )
    for shard_file in sorted(shards):
        logger.info("  Loading shard %s...", shard_file)
        shard_path = hf_hub_download(
            model_name_or_path,
            shard_file,
            token=resolve_hf_token(),
        )
        with safe_open(shard_path, framework="pt", device=device) as f:
            for key in f.keys():
                if key.startswith("transformer."):
                    state_dict[key.removeprefix("transformer.")] = f.get_tensor(key)

    logger.info("Backbone weights loaded from %s", model_name_or_path)
    return state_dict


# ------------------------------------------------------------------
# Concept labels
# ------------------------------------------------------------------

KNOWN_CONCEPTS_URL = (
    "https://raw.githubusercontent.com/guidelabs/steerling/"
    "main/assets/concepts/known_concepts.csv"
)

_concept_labels_cache: Dict[str, "pandas.DataFrame"] = {}  # noqa: F821


def load_steerling_concepts(
    url: str = KNOWN_CONCEPTS_URL,
) -> "pandas.DataFrame":  # noqa: F821
    """Download (and cache) the known-concept label mapping.

    The CSV is fetched from the ``guidelabs/steerling`` GitHub repo on
    first call and cached in-process for subsequent calls.  The file is
    also written to ``~/.cache/steerling/known_concepts.csv`` so it
    persists across runs.

    Returns
    -------
    pandas.DataFrame
        Columns: ``concept_idx``, ``concept_name``,
        ``concept_description``, ``top_100_tokens``.
        Indexed by ``concept_idx``.
    """
    import pandas as pd

    if url in _concept_labels_cache:
        return _concept_labels_cache[url]

    # Local disk cache
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "steerling"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "known_concepts.csv"

    if cache_path.exists():
        logger.info("Loading concept labels from cache: %s", cache_path)
        df = pd.read_csv(cache_path, index_col="concept_idx")
    else:
        logger.info("Downloading concept labels from %s", url)
        df = pd.read_csv(url, index_col="concept_idx")
        df.to_csv(cache_path)
        logger.info("Cached concept labels to %s", cache_path)

    _concept_labels_cache[url] = df
    return df


def load_steerling_concept_names(
    url: str = KNOWN_CONCEPTS_URL,
) -> list:
    """Return an ordered list of concept names for all known concepts.

    The list is ordered by ``concept_idx`` so that
    ``names[i]`` corresponds to concept index ``i``.  This is the
    format expected by :class:`~torch_concepts.distributions.ConceptVariable`.

    Use :func:`load_steerling_concept_map` when you need index → name
    random-access lookups instead.
    """
    df = load_steerling_concepts(url)
    # sort by concept_idx (the DataFrame index) to guarantee order
    return list(df.sort_index()["concept_name"])


def load_steerling_concept_map(
    url: str = KNOWN_CONCEPTS_URL,
) -> Dict[int, str]:
    """Return a ``{concept_idx: concept_name}`` dict for all known concepts.

    Use this for index-based lookups (e.g. mapping top-k indices back to
    names).  For ordered variable naming use
    :func:`load_steerling_concept_names` instead.
    """
    df = load_steerling_concepts(url)
    return dict(zip(df.index, df["concept_name"]))


def prepare_generation_sequence(
    tokenizer,
    prompt: str,
    n_new_tokens: int,
) -> tuple[torch.Tensor, torch.BoolTensor, torch.BoolTensor]:
    """Build the ``[prompt | MASK × N]`` input for causal-diffusion generation.

    Parameters
    ----------
    tokenizer : SteerlingTokenizer
        Tokenizer with ``mask_token_id``.
    prompt : str
        The text prompt.
    n_new_tokens : int
        Number of masked positions to append.

    Returns
    -------
    input_ids : torch.Tensor
        Shape ``(1, prompt_len + n_new_tokens)``.
    prompt_mask : torch.BoolTensor
        Shape ``(prompt_len + n_new_tokens,)``; ``True`` for prompt positions.
    gen_mask : torch.BoolTensor
        Shape ``(prompt_len + n_new_tokens,)``; ``True`` for generation positions.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    total_len = prompt_len + n_new_tokens

    input_ids = torch.full((1, total_len), tokenizer.mask_token_id, dtype=torch.long)
    input_ids[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long)

    prompt_mask = torch.zeros(total_len, dtype=torch.bool)
    prompt_mask[:prompt_len] = True
    gen_mask = ~prompt_mask.clone()

    return input_ids, prompt_mask, gen_mask


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
