"""Tests for the steerling package's public API surface."""

import torch_concepts.steerling as steerling


def test_all_names_are_importable():
    for name in steerling.__all__:
        assert hasattr(steerling, name), f"{name} listed in __all__ but missing"


def test_key_symbols_exported():
    expected = {
        "SteerlingModel",
        "SteerlingLowLevelModel",
        "CausalDiffusionTextBackbone",
        "SteerlingLatentToConcept",
        "MixFactorizedConceptExogenousToConcept",
        "top_concepts",
        "resolve_steerling_configs",
        "DEFAULT_MODEL_ID",
    }
    assert expected.issubset(set(steerling.__all__))


def test_torch_compile_disabled_by_default():
    import os
    # The package sets this on import unless explicitly enabled.
    assert os.environ.get("TORCH_COMPILE_DISABLE") == "1"
