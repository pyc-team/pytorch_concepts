"""Integration tests for SteerlingLatentToConcept (tiny ConceptHead, offline).

``config={}`` keeps construction offline — passing ``config=None`` would resolve
the Hub config and download.
"""

import pytest
import torch

pytest.importorskip("steerling")

from torch_concepts.steerling import SteerlingLatentToConcept

IN, OUT, DIM, RANK = 16, 5, 16, 8


def _enc(**kw):
    kw.setdefault("config", {})
    return SteerlingLatentToConcept(in_latent=IN, out_concepts=OUT, embedding_size=DIM, **kw)


# --------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------

def test_dense_forward_3d():
    enc = _enc(factorize=False, use_attention=False)
    out = enc(torch.randn(2, 7, IN))
    assert out.shape == (2, 7, OUT)


def test_dense_forward_2d_is_squeezed():
    enc = _enc(factorize=False, use_attention=False)
    out = enc(torch.randn(3, IN))
    assert out.shape == (3, OUT)


def test_factorized_forward_and_embeddings():
    enc = _enc(factorize=True, factorize_rank=RANK)
    assert enc.factorize is True
    out = enc(torch.randn(2, 4, IN))
    assert out.shape == (2, 4, OUT)
    # Dense view vs packed factorized view.
    assert enc.get_embeddings(factorized=False).shape == (OUT, DIM)
    packed = enc.get_embeddings(factorized=True)
    assert packed.shape == (OUT + DIM, RANK)


def test_attention_forward():
    enc = _enc(use_attention=True)
    assert enc.use_attention is True
    out = enc(torch.randn(2, 5, IN))
    assert out.shape == (2, 5, OUT)


# --------------------------------------------------------------------------
# Config-derived sizing & aliasing
# --------------------------------------------------------------------------

def test_embedding_size_defaults_to_config_concept_dim():
    enc = SteerlingLatentToConcept(in_latent=IN, out_concepts=OUT, config={"concept_dim": DIM})
    assert enc.embedding_size == DIM


def test_known_head_attention_alias():
    enc = SteerlingLatentToConcept(
        in_latent=IN, out_concepts=OUT, embedding_size=DIM, is_unknown=False,
        config={"use_attention_known": True},
    )
    assert enc.use_attention is True


def test_unknown_head_factorize_alias():
    enc = SteerlingLatentToConcept(
        in_latent=IN, out_concepts=OUT, embedding_size=DIM, is_unknown=True,
        config={"factorize_unknown": True, "factorize_rank": RANK},
    )
    assert enc.factorize is True
    assert enc.is_unknown is True


def test_explicit_args_override_config():
    enc = SteerlingLatentToConcept(
        in_latent=IN, out_concepts=OUT, embedding_size=DIM,
        use_attention=False, config={"use_attention_known": True},
    )
    # Explicit use_attention=False wins over the config alias.
    assert enc.use_attention is False
