"""Integration tests for SteerlingLowLevelModel built tiny & offline."""

import pytest
import torch

pytest.importorskip("steerling")

from torch_concepts.steerling.model import steerling_low as sl
from conftest import (
    build_low_level_model,
    N_EMBD,
    N_KNOWN,
    N_UNKNOWN,
    VOCAB,
)

FORWARD_KEYS = {
    "out_tokens", "known_concepts", "unknown_concepts",
    "known_mixed", "unknown_mixed", "epsilon", "reconstructed_latent",
}


# --------------------------------------------------------------------------
# Forward + shapes
# --------------------------------------------------------------------------

def test_forward_keys_and_shapes(low_model, input_ids):
    out = low_model(input_ids)
    assert set(out) == FORWARD_KEYS
    B, T = input_ids.shape
    assert out["out_tokens"].shape == (B, T, VOCAB)
    assert out["known_concepts"].shape == (B, T, N_KNOWN)
    assert out["unknown_concepts"].shape == (B, T, N_UNKNOWN)
    assert out["reconstructed_latent"].shape == (B, T, N_EMBD)
    assert out["epsilon"].shape == (B, T, N_EMBD)


def test_encode_concepts_shape(low_model, input_ids):
    logits = low_model.encode_concepts(input_ids)
    assert logits.shape == (input_ids.shape[0], input_ids.shape[1], N_KNOWN)


def test_properties(low_model):
    assert low_model.n_known == N_KNOWN
    assert low_model.n_unknown == N_UNKNOWN
    assert low_model.latent_dim == N_EMBD
    assert low_model.embedding_dim == N_EMBD
    assert low_model.vocab_size == VOCAB
    assert low_model.device.type == "cpu"
    assert low_model.known_embeddings.shape == (N_KNOWN, N_EMBD)
    # Unknown head is factorized -> packed (n_unknown + dim, rank).
    assert low_model.unknown_embeddings.shape[0] == N_UNKNOWN + N_EMBD


# --------------------------------------------------------------------------
# use_unknown=False path
# --------------------------------------------------------------------------

def test_no_unknown_head():
    model = build_low_level_model(
        concept_config_overrides=dict(
            n_concepts=5, n_unknown_concepts=4, concept_dim=N_EMBD,
            use_unknown=False, use_epsilon_correction=False,
        )
    )
    model.eval()
    assert model.unknown_concept_head is None
    assert model.n_unknown == 0
    assert model.unknown_embeddings is None
    out = model(torch.randint(0, VOCAB, (1, 4)))
    assert out["unknown_concepts"] is None
    assert out["unknown_mixed"] is None
    assert out["reconstructed_latent"].shape == (1, 4, N_EMBD)


# --------------------------------------------------------------------------
# dtype handling
# --------------------------------------------------------------------------

def test_default_dtype_is_bfloat16(low_model):
    assert low_model.dtype is torch.bfloat16
    assert {p.dtype for p in low_model.parameters()} == {torch.bfloat16}


def test_dtype_override_and_no_global_leak():
    before = torch.get_default_dtype()
    model = build_low_level_model(dtype=torch.float32)
    assert model.dtype is torch.float32
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    # Construction must not leak the global default dtype.
    assert torch.get_default_dtype() is before


# --------------------------------------------------------------------------
# freeze logic (offline — no pretrained load needed)
# --------------------------------------------------------------------------

def test_freeze_components():
    model = build_low_level_model(freeze_components=["known_head"])
    assert all(not p.requires_grad for p in model.known_concept_head.parameters())
    # Backbone left trainable.
    assert any(p.requires_grad for p in model.backbone.parameters())


# --------------------------------------------------------------------------
# print_config
# --------------------------------------------------------------------------

def test_print_config_returns_summary(low_model, capsys):
    info = low_model.print_config()
    assert set(info) == {"summary", "model_cfg", "concept_cfg"}
    assert info["summary"]["n_known"] == N_KNOWN
    assert info["summary"]["vocab_size"] == VOCAB
    # Something was printed.
    assert "configuration" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Construction-time validation
# --------------------------------------------------------------------------

def test_enabled_topk_override_raises():
    with pytest.raises(NotImplementedError):
        build_low_level_model(concept_config_overrides=dict(
            n_concepts=5, n_unknown_concepts=4, concept_dim=N_EMBD, topk_known=8,
        ))


def test_concept_dim_must_match_n_embd():
    with pytest.raises(ValueError):
        build_low_level_model(concept_config_overrides=dict(
            n_concepts=5, n_unknown_concepts=4, concept_dim=N_EMBD + 8,
        ))


def test_concept_names_property(monkeypatch, low_model):
    monkeypatch.setattr(sl, "load_steerling_concept_names", lambda *a, **k: ["x", "y"])
    # cached on first access; clear any cache from a previous test
    if hasattr(low_model, "_concept_names"):
        del low_model._concept_names
    assert low_model.concept_names == ["x", "y"]
