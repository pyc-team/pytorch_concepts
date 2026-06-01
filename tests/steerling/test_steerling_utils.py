"""Tests for steerling_utils — top_concepts and the weight-loading helpers.

Network access is mocked; no steerling package or Hub download required.
"""

import sys
import types

import pytest
import torch

import torch_concepts.steerling.steerling_utils as su


@pytest.fixture
def patched_names(monkeypatch):
    names = ["alpha", "beta", "gamma"]
    monkeypatch.setattr(su, "load_steerling_concept_names", lambda *a, **k: names)
    return names


# --------------------------------------------------------------------------
# top_concepts
# --------------------------------------------------------------------------

def test_top_concepts_1d(patched_names):
    logits = torch.tensor([0.1, 5.0, -2.0])
    df = su.top_concepts(logits, topk=2)
    assert list(df.columns) == ["position", "concept_idx", "concept_name", "probability", "logit"]
    assert len(df) == 2
    # Highest logit first.
    assert df.iloc[0]["concept_name"] == "beta"
    assert df["position"].nunique() == 1


def test_top_concepts_2d_rows_per_position(patched_names):
    logits = torch.randn(4, 3)
    df = su.top_concepts(logits, topk=3)
    assert sorted(df["position"].unique()) == [0, 1, 2, 3]
    assert len(df) == 4 * 3


def test_top_concepts_k_is_clamped(patched_names):
    df = su.top_concepts(torch.randn(3), topk=10)
    assert len(df) == 3  # only 3 concepts available


def test_top_concepts_unknown_index_fallback(patched_names):
    # 5 logits but only 3 names -> indices 3, 4 fall back to "<unknown:idx>".
    df = su.top_concepts(torch.tensor([0.0, 1.0, 2.0, 9.0, 8.0]), topk=5)
    names = set(df["concept_name"])
    assert any(n.startswith("<unknown:") for n in names)


def test_top_concepts_probability_is_sigmoid(patched_names):
    df = su.top_concepts(torch.tensor([0.0]), topk=1)
    assert df.iloc[0]["probability"] == pytest.approx(0.5, abs=1e-6)


def test_top_concepts_rejects_3d(patched_names):
    with pytest.raises(ValueError):
        su.top_concepts(torch.randn(2, 2, 2))


# --------------------------------------------------------------------------
# load_steerling_weights — prefix selection & stripping (mocked I/O)
# --------------------------------------------------------------------------

class _FakeSafeOpen:
    """Minimal stand-in for safetensors.safe_open's context manager."""

    def __init__(self, tensors):
        self._tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return list(self._tensors)

    def get_tensor(self, key):
        return self._tensors[key]


@pytest.fixture
def fake_safetensors(monkeypatch):
    """Install fake huggingface_hub + safetensors modules for the loaders."""
    shard_tensors = {
        "shardA.safetensors": {
            "known_head.linear.weight": torch.ones(2, 2),
            "transformer.block.0.weight": torch.zeros(2),
        },
        "shardB.safetensors": {
            "transformer.block.1.weight": torch.zeros(3),
        },
    }

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda model_id, filename, *a, **k: filename
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = lambda path, framework, device: _FakeSafeOpen(shard_tensors[path])
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)

    return shard_tensors


def test_load_weights_strips_prefix_and_picks_shards(monkeypatch, fake_safetensors):
    weight_map = {
        "known_head.linear.weight": "shardA.safetensors",
        "transformer.block.0.weight": "shardA.safetensors",
        "transformer.block.1.weight": "shardB.safetensors",
    }
    monkeypatch.setattr(su, "_download_weight_map", lambda model_id: weight_map)

    sd = su.load_steerling_weights("any/model", "known_head")
    # Prefix stripped, and only the shard(s) containing the prefix were read.
    assert set(sd) == {"linear.weight"}
    torch.testing.assert_close(sd["linear.weight"], torch.ones(2, 2))


def test_load_weights_trailing_dot_normalized(monkeypatch, fake_safetensors):
    weight_map = {"transformer.block.0.weight": "shardA.safetensors",
                  "transformer.block.1.weight": "shardB.safetensors"}
    monkeypatch.setattr(su, "_download_weight_map", lambda model_id: weight_map)
    sd = su.load_steerling_weights("any/model", "transformer.")  # already dotted
    assert set(sd) == {"block.0.weight", "block.1.weight"}


# --------------------------------------------------------------------------
# _load_lm_head_weights — key fallback order
# --------------------------------------------------------------------------

def test_lm_head_prefers_explicit_key(monkeypatch, fake_safetensors):
    monkeypatch.setattr(
        su, "_download_weight_map",
        lambda model_id: {"known_head.linear.weight": "shardA.safetensors"},
    )
    # No lm-head-ish key present -> KeyError.
    with pytest.raises(KeyError):
        su._load_lm_head_weights("any/model")


def test_lm_head_falls_back_to_tok_emb(monkeypatch):
    tensors = {"shardZ": {"transformer.tok_emb.weight": torch.eye(3)}}
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda model_id, filename, *a, **k: filename
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    st = types.ModuleType("safetensors")
    st.safe_open = lambda path, framework, device: _FakeSafeOpen(tensors[path])
    monkeypatch.setitem(sys.modules, "safetensors", st)
    monkeypatch.setattr(
        su, "_download_weight_map",
        lambda model_id: {"transformer.tok_emb.weight": "shardZ"},
    )
    out = su._load_lm_head_weights("any/model")
    assert set(out) == {"weight"}
    torch.testing.assert_close(out["weight"], torch.eye(3))
