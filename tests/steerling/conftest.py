"""Shared fixtures and constants for the Steerling test suite.

The suite is designed to run **fully offline**: no Hub downloads, no 16 GB
checkpoint, no network. Two ingredients make that possible:

* ``config_source="pyc"`` resolves configs from in-repo dicts (never the Hub).
* ``pretrained_components=None`` builds randomly-initialised modules, so no
  weights are fetched.

Models are shrunk to a couple of layers and a tiny vocab so an end-to-end
forward runs in milliseconds while still exercising the real wiring
(backbone -> concept heads -> mixers -> residual -> LM head -> PGM).

Tests that genuinely need the Hub (tokenizer, real weights, concept-name CSV)
are kept out of the default run; the pure logic + tiny-model integration paths
below cover the wrapper code.
"""

import os

# Must be set before torch / OpenMP import on this platform.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# Fail fast instead of hanging if any test accidentally reaches the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest
import torch

# ---------------------------------------------------------------------------
# Tiny-model configuration (shared by integration tests)
# ---------------------------------------------------------------------------

TINY_MODEL_OVERRIDES = dict(
    n_layers=2,
    n_head=4,
    n_kv_heads=2,
    n_embd=32,
    block_size=64,
    diff_block_size=16,
    vocab_size=48,
)
TINY_CONCEPT_OVERRIDES = dict(
    n_concepts=5,
    n_unknown_concepts=4,
    concept_dim=32,
    factorize_rank=8,
    use_unknown=True,
    factorize_unknown=True,
    use_epsilon_correction=False,
)
KNOWN_NAMES = [f"c{i}" for i in range(TINY_CONCEPT_OVERRIDES["n_concepts"])]
N_EMBD = TINY_MODEL_OVERRIDES["n_embd"]
VOCAB = TINY_MODEL_OVERRIDES["vocab_size"]
N_KNOWN = TINY_CONCEPT_OVERRIDES["n_concepts"]
N_UNKNOWN = TINY_CONCEPT_OVERRIDES["n_unknown_concepts"]


def build_low_level_model(**overrides):
    """Construct a tiny, randomly-initialised SteerlingLowLevelModel offline."""
    from torch_concepts.steerling import SteerlingLowLevelModel

    kwargs = dict(
        config_source="pyc",
        pretrained_components=None,
        freeze_components=None,
        model_config_overrides=dict(TINY_MODEL_OVERRIDES),
        concept_config_overrides=dict(TINY_CONCEPT_OVERRIDES),
    )
    kwargs.update(overrides)
    return SteerlingLowLevelModel(**kwargs)


def build_pgm_model(**overrides):
    """Construct a tiny SteerlingModel offline, stubbing the concept-name CSV."""
    import torch_concepts.steerling.model.steerling as smod
    from torch_concepts.steerling import SteerlingModel

    kwargs = dict(
        config_source="pyc",
        pretrained_components=None,
        freeze_components=None,
        model_config_overrides=dict(TINY_MODEL_OVERRIDES),
        concept_config_overrides=dict(TINY_CONCEPT_OVERRIDES),
    )
    kwargs.update(overrides)

    original = smod.load_steerling_concept_names
    smod.load_steerling_concept_names = lambda *a, **k: list(KNOWN_NAMES)
    try:
        return SteerlingModel(**kwargs)
    finally:
        smod.load_steerling_concept_names = original


@pytest.fixture(scope="module")
def low_model():
    pytest.importorskip("steerling")
    model = build_low_level_model()
    model.eval()
    return model


@pytest.fixture(scope="module")
def pgm_model():
    pytest.importorskip("steerling")
    model = build_pgm_model()
    model.eval()
    return model


@pytest.fixture
def input_ids():
    """A short deterministic token sequence inside the tiny vocab."""
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (1, 6))
