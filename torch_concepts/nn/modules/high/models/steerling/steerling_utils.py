"""
Steerling utilities — Hub downloads, weight loading, and concept labels.

Public API:
    ``load_steerling_weights``      — download checkpoint shards by key prefix.
    ``get_steerling_tokenizer``     — return the Steerling AutoTokenizer.
    ``load_steerling_concept_names``— ordered list of known-concept names.
    ``top_concepts``                — map concept logits to a named DataFrame.

Config loading lives in ``steerling_configs.py``.
"""

from __future__ import annotations

import json
import logging
from steerling import ConceptCatalog
from typing import Dict

import pandas as pd
import torch

from .steerling_configs import DEFAULT_MODEL_ID

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------

def get_steerling_tokenizer(model_id: str = DEFAULT_MODEL_ID):
    """Return the Steerling tokenizer via ``AutoTokenizer``.

    Args:
        model_id: HuggingFace Hub model id or local path.

    Returns:
        Steerling tokenizer instance (custom class via ``trust_remote_code``).
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "get_steerling_tokenizer requires `transformers`. "
            "Install with: pip install transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )


# ------------------------------------------------------------------
# Weights loading
# ------------------------------------------------------------------

# In-process cache for the safetensors weight-map index.
# huggingface_hub already caches the file on disk; this avoids
# re-parsing the JSON on every function call within the same process.
_weight_map_cache: Dict[str, Dict[str, str]] = {}


def _download_weight_map(model_id: str) -> Dict[str, str]:
    """Download ``model.safetensors.index.json`` and return its weight map.

    Cached in-process so repeated calls within the same process skip the
    JSON parse entirely (huggingface_hub should handle the disk cache).
    """
    if model_id in _weight_map_cache:
        return _weight_map_cache[model_id]

    from huggingface_hub import hf_hub_download

    index_path = hf_hub_download(
        model_id,
        "model.safetensors.index.json",
    )
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    _weight_map_cache[model_id] = weight_map
    return weight_map


def load_steerling_weights(
    model_id: str,
    prefix: str,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Download the checkpoint shards for ``prefix`` and return a state dict.

    Only the shards that actually contain keys under ``prefix`` are fetched,
    so a single concept head pulls one shard rather than the full checkpoint.
    ``prefix`` is stripped from every returned key so the dict loads directly
    into the matching module via ``load_state_dict``.

    Args:
        model_id: HuggingFace Hub model id or local path.
        prefix: Key prefix, e.g. ``"known_head"``, ``"unknown_head"``, or
            ``"transformer"``. A trailing dot is added automatically.
        device: PyTorch device string passed to safetensors.

    Returns:
        State dict with ``prefix`` stripped from every key.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    prefix = prefix.rstrip(".") + "."
    weight_map = _download_weight_map(model_id)
    shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})

    logger.info(
        "Loading %r weights from %s (%d shard%s)...",
        prefix, model_id, len(shards), "" if len(shards) == 1 else "s",
    )
    state_dict: Dict[str, torch.Tensor] = {}
    for shard_file in shards:
        logger.info("  Loading shard %s...", shard_file)
        shard_path = hf_hub_download(model_id, shard_file)
        with safe_open(shard_path, framework="pt", device=device) as f:
            for key in f.keys():
                if key.startswith(prefix):
                    state_dict[key.removeprefix(prefix)] = f.get_tensor(key)

    return state_dict


def _load_lm_head_weights(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Download the LM-head / tied-token-embedding shard and return a state dict.

    Tries ``lm_head.weight``, ``transformer.lm_head.weight``, and
    ``transformer.tok_emb.weight`` in order, covering both the explicit and
    weight-tied checkpoint layouts.

    Args:
        model_id: HuggingFace Hub model id or local path.
        device: PyTorch device string passed to safetensors.

    Returns:
        ``{"weight": tensor}`` ready for ``nn.Linear.load_state_dict``.

    Raises:
        KeyError: If none of the expected LM-head keys are found in the index.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    weight_map = _download_weight_map(model_id)
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
    shard_path = hf_hub_download(model_id, shard_file)

    with safe_open(shard_path, framework="pt", device=device) as f:
        weight = f.get_tensor(weight_key)

    logger.info("Loaded lm_head weights from %s:%s", model_id, weight_key)
    return {"weight": weight}


# ------------------------------------------------------------------
# Concept labels
# ------------------------------------------------------------------

_known_concept_names_cache: list | None = None


def load_steerling_concept_names() -> list:
    """Return an ordered list of unique known-concept labels.

    The catalog's known-concept names are auto-generated labels (not hand-picked
    unique identifiers) and are not guaranteed unique — distinct concepts can
    share a name, disambiguated only by ``concept_id`` (see
    ``steerling.ConceptCatalog``; verified against the real catalog: ~1,900 of
    the 33,732 known-concept names collide with at least one other concept).
    Each label is therefore
    ``"<concept_name>_<concept_id>"``, giving
    :class:`~torch_concepts.distributions.ConceptVariable` a unique identifier
    per concept. The list is ordered by ``concept_id`` so that ``names[i]``
    corresponds to concept index ``i`` (verified to match the known-head's
    logit/embedding row order — see ``steerling.SteerlingGenerator.concept_top_tokens``,
    which indexes the same embedding table directly by ``concept_id``).

    Returns:
        Ordered list of ``"<name>_<id>"`` strings, cached in-process. The
        underlying catalog is loaded via ``steerling.ConceptCatalog`` (cached
        on disk by ``huggingface_hub``).
    """
    global _known_concept_names_cache
    if _known_concept_names_cache is not None:
        return _known_concept_names_cache

    logger.info("Loading known concept labels via steerling.ConceptCatalog")
    known = ConceptCatalog.load().to_df()
    known = known[known["head"] == "known"].sort_index()

    names = [
        f"{name}_{concept_id}"
        for name, concept_id in zip(known["concept_name"], known["concept_id"])
    ]
    _known_concept_names_cache = names
    return names


def top_concepts(
    logits: torch.Tensor,
    topk: int = 10,
) -> pd.DataFrame:
    """Map concept logits to the top-``k`` human-readable concept names.

    A lightweight post-processing helper — no forward passes, just ranks the
    logits and looks up names. Pass a single position
    (e.g. ``out["known_concepts"][0, -1]``) or a whole sequence.

    Args:
        logits: Concept logits, shape ``(n_concepts,)`` for a single position
            or ``(T, n_concepts)`` for a sequence.
        topk: Number of top concepts to return per position. Clamped to the
            number of available concepts.

    Returns:
        ``pandas.DataFrame`` with one row per (position, concept) and columns
        ``position``, ``concept_idx``, ``concept_name``, ``probability``
        (sigmoid of the logit), and ``logit``. Rows are ordered by descending
        logit within each position.

    Raises:
        ValueError: If ``logits`` is not 1-D or 2-D.
    """

    names = load_steerling_concept_names()

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() != 2:
        raise ValueError(
            "Expected logits with shape (n_concepts,) or (T, n_concepts); "
            f"got shape {tuple(logits.shape)}."
        )

    k = min(topk, logits.shape[-1])

    rows = []
    for pos in range(logits.shape[0]):
        values, indices = torch.topk(logits[pos].float(), k=k)
        probs = torch.sigmoid(values)
        for idx, logit, prob in zip(
            indices.tolist(), values.tolist(), probs.tolist()
        ):
            rows.append({
                "position": pos,
                "concept_idx": idx,
                "concept_name": (
                    names[idx] if 0 <= idx < len(names) else f"<unknown:{idx}>"
                ),
                "probability": round(prob, 6),
                "logit": round(logit, 4),
            })

    return pd.DataFrame(
        rows,
        columns=["position", "concept_idx", "concept_name", "probability", "logit"],
    )
