"""Tests for pure module-level helpers in steerling_low and steerling_encoder.

These import the wrapper modules (which do *not* require the steerling package
at import time) and exercise the config-filtering / dtype logic directly.
"""

import pytest
import torch

from torch_concepts.steerling.model import steerling_low as sl
from torch_concepts.steerling import steerling_encoder as se


# --------------------------------------------------------------------------
# _is_topk_enabled
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (None, False),
        (False, False),
        (0, False),
        (-1, False),
        (16, True),
        (True, True),
        ("something", True),
    ],
)
def test_is_topk_enabled(value, expected):
    assert sl._is_topk_enabled(value) is expected


# --------------------------------------------------------------------------
# _sanitize_topk
# --------------------------------------------------------------------------

def test_sanitize_topk_disables_present_keys_only():
    cfg = {"topk_known": 16, "unknown_topk": 128, "apply_topk_to_unknown": True,
           "topk_on_logits": True, "n_concepts": 5}
    out = sl._sanitize_topk(cfg)
    assert out["topk_known"] is None
    assert out["unknown_topk"] is None
    assert out["apply_topk_to_unknown"] is False
    assert out["topk_on_logits"] is False
    # Non-topk keys are untouched; absent topk keys are not added.
    assert out["n_concepts"] == 5
    assert "topk" not in out


def test_sanitize_topk_does_not_mutate_input():
    cfg = {"topk_known": 16}
    sl._sanitize_topk(cfg)
    assert cfg["topk_known"] == 16


# --------------------------------------------------------------------------
# _resolve_dtype
# --------------------------------------------------------------------------

def test_resolve_dtype_explicit_wins():
    assert sl._resolve_dtype(torch.float16, {"torch_dtype": "float64"}) is torch.float16


def test_resolve_dtype_from_torch_dtype_in_cfg():
    assert sl._resolve_dtype(None, {"torch_dtype": torch.float16}) is torch.float16


def test_resolve_dtype_from_string_in_cfg():
    assert sl._resolve_dtype(None, {"torch_dtype": "bfloat16"}) is torch.bfloat16


def test_resolve_dtype_default_when_absent():
    assert sl._resolve_dtype(None, {}) is torch.bfloat16


def test_resolve_dtype_invalid_string_falls_back():
    assert sl._resolve_dtype(None, {"torch_dtype": "not_a_dtype"}) is torch.bfloat16


# --------------------------------------------------------------------------
# _default_dtype context manager
# --------------------------------------------------------------------------

def test_default_dtype_sets_and_restores():
    before = torch.get_default_dtype()
    with sl._default_dtype(torch.float64):
        assert torch.get_default_dtype() is torch.float64
    assert torch.get_default_dtype() is before


def test_default_dtype_restores_on_exception():
    before = torch.get_default_dtype()
    with pytest.raises(RuntimeError):
        with sl._default_dtype(torch.float64):
            raise RuntimeError("boom")
    assert torch.get_default_dtype() is before


# --------------------------------------------------------------------------
# encoder: _concept_head_init_keys
# --------------------------------------------------------------------------

def test_concept_head_init_keys_extracts_named_params():
    class Dummy:
        def __init__(self, a, b=1, *args, c=2, **kwargs):
            pass

    keys = se._concept_head_init_keys(Dummy)
    assert keys == {"a", "b", "c"}  # self / *args / **kwargs excluded


# --------------------------------------------------------------------------
# encoder: _filter_concept_head_config
# --------------------------------------------------------------------------

ALLOWED = {"use_attention", "topk", "topk_features", "factorize", "block_size", "n_concepts"}


def test_filter_known_head_uses_known_aliases():
    cfg = {
        "use_attention_known": True,      # alias -> use_attention (kept)
        "use_attention_unknown": True,    # no known-side alias -> ignored
        "block_size": 64,                 # plain, allowed -> kept
        "inject_layer": 16,               # plain, not allowed -> ignored
    }
    out = se._filter_concept_head_config(cfg, ALLOWED, is_unknown=False)
    assert out == {"use_attention": True, "block_size": 64}


def test_filter_unknown_head_uses_unknown_aliases():
    cfg = {
        "factorize_unknown": True,        # alias -> factorize (kept)
        "use_attention_unknown": False,   # alias -> use_attention (kept)
        "topk_known": 16,                 # known-side, no unknown alias -> ignored
        "n_unknown_concepts": 99,         # alias -> n_concepts (kept)
    }
    out = se._filter_concept_head_config(cfg, ALLOWED, is_unknown=True)
    assert out == {"factorize": True, "use_attention": False, "n_concepts": 99}


def test_filter_alias_to_unsupported_target_is_ignored():
    # use_attention_known -> use_attention, but use_attention not allowed here.
    out = se._filter_concept_head_config(
        {"use_attention_known": True}, allowed_keys={"topk"}, is_unknown=False
    )
    assert out == {}


def test_filter_empty_config_returns_empty():
    assert se._filter_concept_head_config(None, ALLOWED, is_unknown=False) == {}
