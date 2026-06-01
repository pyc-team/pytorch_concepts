"""Configuration presets for the local Steerling integration.

The low-level wrapper can either follow the official Steerling package
defaults or keep the PyC defaults that existed before config resolution was
made explicit.  All helpers return plain dictionaries so importing this module
does not require the optional ``steerling`` package.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Literal, Mapping



SteerlingConfigSource = Literal["pyc", "steerling", "hub"]
DEFAULT_MODEL_ID = "guidelabs/steerling-8b"

# TODO: at some point, consider moving these to hydra YAML

PYTORCH_CONCEPTS_MODEL_DEFAULTS: dict[str, Any] = {
    # Aligned with upstream `steerling.configs.causal_diffusion.CausalDiffusionConfig`
    # defaults so `config_source="pyc"` ≈ `config_source="steerling"`.  Note that
    # the Hub config.json for Steerling-8B overrides `interpretable=True`; under
    # the default `config_source="hub"` that override wins, so the trained model
    # is built as interpretable.  The HF custom-code `SteerlingConfig` wrapper
    # also defaults `interpretable=True`, but that's a model-specific wrapper,
    # not the underlying source class.
    "model_type": "causal_diffusion",
    "interpretable": False,
    "n_layers": 32,
    "n_head": 32,
    "n_embd": 4096,
    "block_size": 4096,
    "n_kv_heads": 4,
    "diff_block_size": 64,
    "use_rms_norm": True,
    "norm_eps": 1e-5,
    "norm_order": "post",
    "use_qk_norm": True,
    "use_rope": True,
    "rope_base": 500000.0,
    "rope_full_precision": True,
    "mlp_type": "swiglu",
    "activation": "gelu",
    "mlp_ratio": 4,
    "intermediate_size": None,
    "use_bias": False,
    "clip_qkv": 10.0,
    "weight_sharing": True,
}

PYTORCH_CONCEPTS_CONCEPT_DEFAULTS: dict[str, Any] = {
    # Aligned with upstream `steerling.configs.concept.ConceptConfig` defaults
    # so loading Hub weights succeeds regardless of `config_source`.
    "n_concepts": 33732,
    "n_unknown_concepts": 101196,
    "max_concepts": 16,
    "concept_dim": 4096,
    "use_attention_known": False,
    "use_attention_unknown": False,
    "topk_known": 16,
    "topk_known_features": 32,
    "use_unknown": True,
    "factorize_unknown": True,
    "factorize_rank": 256,
    "unknown_topk": 128,
    "use_epsilon_correction": True,
    "block_size": 4096,
    "pad_multiple": 16,
    "topk_on_logits": False,
    "store_unknown_weights": False,
    "apply_topk_to_unknown": True,
    "inject_layer": 16,
    "inject_alpha": 1.0,
}
# Concept keys to extract from a flat Hub config.json. We deliberately drop the
# bare `block_size`: in the flat HF namespace it is the *model's* block size,
# while the concept block size arrives under `concept_block_size` (which
# `normalize_concept_config` maps back to `block_size`). Without this exclusion
# the extraction would steal the model's `block_size` into the concept config.
_CONCEPT_CONFIG_KEYS = (
    set(PYTORCH_CONCEPTS_CONCEPT_DEFAULTS) | {"concept_block_size"}
) - {"block_size"}


def config_to_dict(config: Any) -> dict[str, Any]:
    """Convert Pydantic/dataclass/mapping config objects to plain dicts."""
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "dict"):
        return dict(config.dict())
    if hasattr(config, "__dict__"):
        return {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
    raise TypeError(f"Unsupported Steerling config type: {type(config)!r}")


def normalize_concept_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize Steerling concept-config aliases used by Hub/package configs."""
    config = dict(config)
    if "concept_block_size" in config:
        config["block_size"] = config.pop("concept_block_size")
    return config


def load_steerling_hub_config(
    model_name_or_path: str = DEFAULT_MODEL_ID,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Download ``config.json`` and return (model_config, concept_config).

    Both are plain dicts. The caller is responsible for extracting the fields
    it needs or passing them through :func:`resolve_steerling_configs`.
    """
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_name_or_path, "config.json")
    with open(config_path) as f:
        raw = json.load(f)

    concept_cfg = dict(raw.pop("concept", {}))
    for key in list(raw):
        if key in _CONCEPT_CONFIG_KEYS:
            concept_cfg[key] = raw.pop(key)
    concept_cfg = normalize_concept_config(concept_cfg)
    return raw, concept_cfg


def resolve_steerling_configs(
    *,
    config_source: SteerlingConfigSource = "hub",
    model_id: str | None = None,
    model_config_overrides: Mapping[str, Any] | None = None,
    concept_config_overrides: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve effective model/concept configs for Steerling wrappers.

    Picks the base configs from ``config_source`` ("pyc" defaults, the
    installed ``steerling`` package defaults, or the Hub ``config.json``),
    then applies the explicit override dictionaries, which always win.
    To toggle the unknown concept head, pass
    ``concept_config_overrides={"use_unknown": False}``.
    """
    if config_source == "pyc":
        model_cfg = deepcopy(PYTORCH_CONCEPTS_MODEL_DEFAULTS)
        concept_cfg = deepcopy(PYTORCH_CONCEPTS_CONCEPT_DEFAULTS)
    elif config_source == "steerling":
        try:
            from steerling.configs.causal_diffusion import CausalDiffusionConfig
            from steerling.configs.concept import ConceptConfig
        except ImportError as exc:
            raise ImportError(
                "config_source='steerling' requires the `steerling` package. "
                "Install it or choose config_source='pyc' or 'hub'."
            ) from exc
        model_cfg = config_to_dict(CausalDiffusionConfig())
        concept_cfg = config_to_dict(ConceptConfig())
    elif config_source == "hub":
        assert model_id is not None, "model_id must be provided when config_source='hub'"
        hub_model_cfg, hub_concept_cfg = load_steerling_hub_config(model_id)
        # Merge under PyC defaults so any key the Hub config omits has a single,
        # well-known fallback instead of being scattered across `dict.get` calls.
        model_cfg = deepcopy(PYTORCH_CONCEPTS_MODEL_DEFAULTS)
        model_cfg.update(hub_model_cfg)
        concept_cfg = deepcopy(PYTORCH_CONCEPTS_CONCEPT_DEFAULTS)
        concept_cfg.update(hub_concept_cfg)
    else:
        raise ValueError("config_source must be one of 'pyc', 'steerling', or 'hub'")

    if model_config_overrides:
        model_cfg.update(model_config_overrides)
    if concept_config_overrides:
        concept_cfg.update(concept_config_overrides)

    concept_cfg = normalize_concept_config(concept_cfg)
    return model_cfg, concept_cfg
