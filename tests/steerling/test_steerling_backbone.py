"""Integration tests for CausalDiffusionTextBackbone + its config helpers."""

import pytest
import torch

pytest.importorskip("steerling")

from torch_concepts.steerling.steerling_backbone import (
    CausalDiffusionTextBackbone,
    _causal_diffusion_config_fields,
    _import_steerling_transformer,
    _to_causal_diffusion_config_kwargs,
)
from torch_concepts.steerling.steerling_configs import resolve_steerling_configs

TINY = dict(n_layers=2, n_head=4, n_kv_heads=2, n_embd=32,
            block_size=64, diff_block_size=16, vocab_size=48)


@pytest.fixture(scope="module")
def tiny_backbone():
    model_cfg, _ = resolve_steerling_configs(config_source="pyc", model_config_overrides=TINY)
    return CausalDiffusionTextBackbone(config=model_cfg)


# --------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------

def test_out_features_and_vocab(tiny_backbone):
    assert tiny_backbone.out_features == TINY["n_embd"]
    assert tiny_backbone.vocab_size == TINY["vocab_size"]


def test_forward_hidden_states(tiny_backbone):
    ids = torch.randint(0, TINY["vocab_size"], (1, 5))
    hidden = tiny_backbone(ids)
    assert hidden.shape == (1, 5, TINY["n_embd"])


def test_forward_logits(tiny_backbone):
    ids = torch.randint(0, TINY["vocab_size"], (1, 5))
    logits = tiny_backbone(ids, return_hidden=False)
    assert logits.shape == (1, 5, TINY["vocab_size"])


def test_vocab_size_override():
    model_cfg, _ = resolve_steerling_configs(config_source="pyc", model_config_overrides=TINY)
    bb = CausalDiffusionTextBackbone(config=model_cfg, vocab_size=20)
    assert bb.vocab_size == 20


# --------------------------------------------------------------------------
# Config-kwargs helpers
# --------------------------------------------------------------------------

def test_config_fields_nonempty():
    _, CausalDiffusionConfig = _import_steerling_transformer()
    fields = _causal_diffusion_config_fields(CausalDiffusionConfig)
    assert "n_embd" in fields and "n_head" in fields


def test_to_config_kwargs_renames_model_type_and_filters():
    _, CausalDiffusionConfig = _import_steerling_transformer()
    raw = {"model_type": "steerling", "n_embd": 32, "not_a_real_field": 123}
    out = _to_causal_diffusion_config_kwargs(CausalDiffusionConfig, raw)
    assert out["model_type"] == "causal_diffusion"
    assert out["n_embd"] == 32
    assert "not_a_real_field" not in out
