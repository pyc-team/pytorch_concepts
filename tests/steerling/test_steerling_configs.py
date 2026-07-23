"""Tests for steerling_configs — pure dict logic, no steerling package or network."""

import pytest

from torch_concepts.steerling.steerling_configs import (
    DEFAULT_MODEL_ID,
    PYTORCH_CONCEPTS_CONCEPT_DEFAULTS,
    PYTORCH_CONCEPTS_MODEL_DEFAULTS,
    config_to_dict,
    normalize_concept_config,
    resolve_steerling_configs,
)


# --------------------------------------------------------------------------
# config_to_dict
# --------------------------------------------------------------------------

def test_config_to_dict_none_returns_empty():
    assert config_to_dict(None) == {}


def test_config_to_dict_mapping_is_copied():
    src = {"a": 1, "b": 2}
    out = config_to_dict(src)
    assert out == src and out is not src


def test_config_to_dict_model_dump_object():
    class Pydanticish:
        def model_dump(self):
            return {"x": 1}

    assert config_to_dict(Pydanticish()) == {"x": 1}


def test_config_to_dict_dict_method_object():
    class HasDict:
        def dict(self):
            return {"y": 2}

    assert config_to_dict(HasDict()) == {"y": 2}


def test_config_to_dict_plain_object_drops_private():
    class Plain:
        def __init__(self):
            self.keep = 1
            self._skip = 2

    out = config_to_dict(Plain())
    assert out == {"keep": 1}


def test_config_to_dict_unsupported_raises():
    with pytest.raises(TypeError):
        config_to_dict(42)


# --------------------------------------------------------------------------
# normalize_concept_config
# --------------------------------------------------------------------------

def test_normalize_maps_concept_block_size():
    out = normalize_concept_config({"concept_block_size": 1024})
    assert out["block_size"] == 1024
    assert "concept_block_size" not in out


def test_normalize_without_alias_is_unchanged():
    out = normalize_concept_config({"block_size": 64})
    assert out == {"block_size": 64}


def test_normalize_does_not_mutate_input():
    src = {"concept_block_size": 8}
    normalize_concept_config(src)
    assert src == {"concept_block_size": 8}


# --------------------------------------------------------------------------
# resolve_steerling_configs
# --------------------------------------------------------------------------

def test_resolve_pyc_returns_defaults():
    model_cfg, concept_cfg = resolve_steerling_configs(config_source="pyc")
    assert model_cfg["n_embd"] == PYTORCH_CONCEPTS_MODEL_DEFAULTS["n_embd"]
    assert concept_cfg["n_concepts"] == PYTORCH_CONCEPTS_CONCEPT_DEFAULTS["n_concepts"]


def test_resolve_pyc_is_deep_copied():
    model_cfg, _ = resolve_steerling_configs(config_source="pyc")
    model_cfg["n_embd"] = -1
    # The module-level defaults must be untouched.
    assert PYTORCH_CONCEPTS_MODEL_DEFAULTS["n_embd"] != -1


def test_resolve_overrides_win():
    model_cfg, concept_cfg = resolve_steerling_configs(
        config_source="pyc",
        model_config_overrides={"n_embd": 7},
        concept_config_overrides={"n_concepts": 3},
    )
    assert model_cfg["n_embd"] == 7
    assert concept_cfg["n_concepts"] == 3


def test_resolve_concept_block_size_override_is_normalized():
    _, concept_cfg = resolve_steerling_configs(
        config_source="pyc",
        concept_config_overrides={"concept_block_size": 256},
    )
    assert concept_cfg["block_size"] == 256
    assert "concept_block_size" not in concept_cfg


def test_resolve_invalid_source_raises():
    with pytest.raises(ValueError):
        resolve_steerling_configs(config_source="nonsense")


def test_resolve_hub_requires_model_id():
    # Fires before any network access.
    with pytest.raises(AssertionError):
        resolve_steerling_configs(config_source="hub", model_id=None)


def test_default_model_id_is_steerling():
    assert "steerling" in DEFAULT_MODEL_ID
