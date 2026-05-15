"""Configuration presets for the local Steerling integration.

The low-level wrapper can either follow the official Steerling package
defaults or keep the PyC defaults that existed before config resolution was
made explicit.  All helpers return plain dictionaries so importing this module
does not require the optional ``steerling`` package.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Dict, Literal, Mapping, Tuple

from ..utils import resolve_hf_token


SteerlingConfigSource = Literal["pyc", "steerling", "hub"]
DEFAULT_MODEL_ID = "guidelabs/steerling-8b"

STEERLING_COMPONENTS = ("backbone", "known_head", "unknown_head", "lm_head")

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
_CONCEPT_CONFIG_KEYS = set(PYTORCH_CONCEPTS_CONCEPT_DEFAULTS) | {
    "concept_block_size",
}


def normalize_steerling_components(
    components: bool | str | list[str] | tuple[str, ...] | set[str] | None,
) -> list[str]:
    """Normalize component selectors used by ``pretrained`` and ``freeze``."""
    if components is True:
        return list(STEERLING_COMPONENTS)
    if components is False or components is None:
        return []
    if isinstance(components, str):
        if components in {"all", "steerling"}:
            return list(STEERLING_COMPONENTS)
        if components in {"none", ""}:
            return []
        return [components]
    return list(components)


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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Download ``config.json`` and return (model_config, concept_config).

    Both are plain dicts. The caller is responsible for extracting the fields
    it needs or passing them through :func:`resolve_steerling_configs`.
    """
    from huggingface_hub import hf_hub_download

    # Late import to avoid a circular dependency with ``steerling_utils``.
    from .steerling_utils import _ensure_hf_login
    _ensure_hf_login()

    config_path = hf_hub_download(
        model_name_or_path,
        "config.json",
        token=resolve_hf_token(),
    )
    with open(config_path) as f:
        raw = json.load(f)

    concept_cfg = dict(raw.pop("concept", {}))
    for key in list(raw):
        if key in _CONCEPT_CONFIG_KEYS:
            concept_cfg[key] = raw.pop(key)
    concept_cfg = normalize_concept_config(concept_cfg)
    return raw, concept_cfg


def _steerling_package_defaults() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from steerling.configs.causal_diffusion import CausalDiffusionConfig
        from steerling.configs.concept import ConceptConfig
    except ImportError as exc:
        raise ImportError(
            "config_source='steerling' requires the `steerling` package. "
            "Install it or choose config_source='pyc' or 'hub'."
        ) from exc

    return (
        config_to_dict(CausalDiffusionConfig()),
        config_to_dict(ConceptConfig()),
    )


def _source_defaults(
    source: SteerlingConfigSource,
    model_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if source == "pyc":
        return (
            deepcopy(PYTORCH_CONCEPTS_MODEL_DEFAULTS),
            deepcopy(PYTORCH_CONCEPTS_CONCEPT_DEFAULTS),
            "pyc",
        )
    if source == "steerling":
        model_cfg, concept_cfg = _steerling_package_defaults()
        return model_cfg, concept_cfg, "steerling"
    if source == "hub":
        assert model_id is not None, "model_id must be provided when config_source='hub'"
        hub_model_cfg, hub_concept_cfg = load_steerling_hub_config(model_id)
        # Merge under PyC defaults so any key the Hub config omits has a
        # single, well-known fallback instead of being scattered across
        # `dict.get(..., …)` call sites.
        merged_model = deepcopy(PYTORCH_CONCEPTS_MODEL_DEFAULTS)
        merged_model.update(hub_model_cfg)
        merged_concept = deepcopy(PYTORCH_CONCEPTS_CONCEPT_DEFAULTS)
        merged_concept.update(hub_concept_cfg)
        return merged_model, merged_concept, "hub"
    raise ValueError(
        "config_source must be one of 'pyc', 'steerling', or 'hub'"
    )


def resolve_steerling_configs(
    *,
    config_source: SteerlingConfigSource = "hub",
    model_id: str | None = None,
    model_config_overrides: Mapping[str, Any] | None = None,
    concept_config_overrides: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Resolve effective model/concept configs for Steerling wrappers.

    Explicit override dictionaries always win over the selected preset.
    To toggle the unknown concept head, pass
    ``concept_config_overrides={"use_unknown": False}``.
    """
    model_cfg, concept_cfg, resolved_source = _source_defaults(config_source, model_id)

    if model_config_overrides:
        model_cfg.update(model_config_overrides)
    if concept_config_overrides:
        concept_cfg.update(concept_config_overrides)

    concept_cfg = normalize_concept_config(concept_cfg)
    return model_cfg, concept_cfg, resolved_source
