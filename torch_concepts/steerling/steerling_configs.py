"""Configuration presets for the local Steerling integration.

The low-level wrapper can either follow the official Steerling package
defaults or keep the PyC defaults that existed before config resolution was
made explicit.  All helpers return plain dictionaries so importing this module
does not require the optional ``steerling`` package.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping



SteerlingConfigSource = Literal["steerling", "hub"]
DEFAULT_MODEL_ID = "guidelabs/steerling-8b"

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


def resolve_steerling_configs(
    *,
    config_source: SteerlingConfigSource = "hub",
    model_id: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve effective model/concept configs for Steerling wrappers.

    Picks the base configs from ``config_source`` (the
    installed ``steerling`` package defaults, or the Hub ``config.json``),
    then applies the explicit override dictionaries, which always win.
    To toggle the unknown concept head, pass
    ``concept_config_overrides={"use_unknown": False}``.
    """
    try:
        from steerling.configs.causal_diffusion import CausalDiffusionConfig
        from steerling.configs.concept import ConceptConfig
    except ImportError as exc:
        raise ImportError(
            "loading configs requires the `steerling` package. "
            "Install it or choose config_source='pyc' or 'hub'."
        ) from exc
    
    if config_source == "steerling":
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
        other_config = {"vocab_size": 100281, 
                        "pad_token_id": 100277, 
                        "bos_token_id": 100278, 
                        "eos_token_id": 100257, 
                        "mask_token_id": 100280, 
                        "endofchunk_token_id": 100279, 
                        "torch_dtype": "bfloat16"}

    elif config_source == "hub":
        assert model_id is not None, "model_id must be provided when config_source='hub'"
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        model_keys = set(config_to_dict(CausalDiffusionConfig()).keys())
        concept_keys = set(config_to_dict(ConceptConfig()).keys())

        model_cfg = {k: v for k, v in cfg.items() if k in model_keys}
        concept_cfg = {k: v for k, v in cfg.items() if k in concept_keys}
        concept_cfg['block_size'] = cfg.pop('concept_block_size')
        other_config = {k: v for k, v in cfg.items() if k not in model_cfg.keys() | concept_cfg.keys()}

    else:
        raise ValueError("config_source must be one of 'steerling' or 'hub'")

    return model_cfg, concept_cfg, other_config
