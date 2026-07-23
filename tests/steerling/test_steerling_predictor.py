"""Tests for MixFactorizedConceptExogenousToConcept — pure tensor layer, no steerling."""

import pytest
import torch

from torch_concepts.steerling import MixFactorizedConceptExogenousToConcept

C, D, V, R = 5, 4, 7, 3


def _dense_mixer(**kw):
    return MixFactorizedConceptExogenousToConcept(
        in_concepts=C, in_exogenous=D, out_concepts=V, **kw
    )


# --------------------------------------------------------------------------
# Dense mixing
# --------------------------------------------------------------------------

def test_dense_mix_no_head_returns_latent():
    mixer = _dense_mixer(factorized=False, add_linear_head=False)
    concepts = torch.randn(2, 6, C)
    emb = torch.randn(C, D)
    out = mixer(concepts, emb)
    assert out.shape == (2, 6, D)
    # Equivalent to a plain matmul.
    torch.testing.assert_close(out, concepts @ emb)


def test_dense_mix_with_linear_head_changes_dim():
    mixer = _dense_mixer(factorized=False, add_linear_head=True)
    out = mixer(torch.randn(3, C), torch.randn(C, D))
    assert out.shape == (3, V)


def test_dense_mix_dimension_mismatch_raises():
    mixer = _dense_mixer(factorized=False, add_linear_head=False)
    with pytest.raises(ValueError):
        mixer(torch.randn(2, C), torch.randn(C + 1, D))


# --------------------------------------------------------------------------
# Factorized mixing
# --------------------------------------------------------------------------

def test_factorized_matches_dense_equivalent():
    mixer = _dense_mixer(factorized=True, add_linear_head=False)
    concepts = torch.randn(2, C)
    coef = torch.randn(C, R)
    basis = torch.randn(D, R)
    packed = torch.cat([coef, basis], dim=0)            # (C + D, R)
    out = mixer(concepts, packed)
    assert out.shape == (2, D)
    # Same as materialising the full (C, D) embedding: E = coef @ basis.T.
    torch.testing.assert_close(out, concepts @ (coef @ basis.T))


def test_factorized_wrong_row_count_raises():
    mixer = _dense_mixer(factorized=True, add_linear_head=False)
    with pytest.raises(ValueError):
        mixer(torch.randn(2, C), torch.randn(C + D + 1, R))


# --------------------------------------------------------------------------
# Cardinality validation
# --------------------------------------------------------------------------

def test_cardinalities_must_sum_to_in_concepts():
    with pytest.raises(ValueError):
        MixFactorizedConceptExogenousToConcept(
            in_concepts=C, in_exogenous=D, out_concepts=V, cardinalities=[1, 1]
        )


def test_non_binary_cardinality_not_implemented():
    # cardinalities sum to in_concepts (passes the sum check) but include a 2.
    with pytest.raises(NotImplementedError):
        MixFactorizedConceptExogenousToConcept(
            in_concepts=3, in_exogenous=D, out_concepts=V, cardinalities=[2, 1],
        )


def test_mix_preserves_dtype():
    mixer = _dense_mixer(factorized=False, add_linear_head=False)
    concepts = torch.randn(2, C)
    emb = torch.randn(C, D, dtype=torch.float64)
    # exogenous is cast to the concepts' dtype.
    assert mixer(concepts, emb).dtype == concepts.dtype
