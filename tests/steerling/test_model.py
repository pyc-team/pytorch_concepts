"""Integration tests for SteerlingModel (PGM-backed) built tiny & offline."""

import pytest
import torch
import torch.nn as nn
from torch.distributions import RelaxedBernoulli, RelaxedOneHotCategorical

pytest.importorskip("steerling")

from torch_concepts.nn import ProbabilisticModel
from torch_concepts.steerling.model import steerling_low as sl
from conftest import build_pgm_model, KNOWN_NAMES, N_EMBD, N_KNOWN, N_UNKNOWN, VOCAB


SPLIT_KEYS = {
    "input", "known_concepts", "unknown_concepts",
    "k_hat", "u_hat", "epsilon", "h_bar", "new_token",
}


# --------------------------------------------------------------------------
# Forward / queries
# --------------------------------------------------------------------------

def test_default_query_split(pgm_model, input_ids):
    out = pgm_model(input_ids)
    parts = pgm_model.split_full_forward(out.probs)
    assert set(parts) == SPLIT_KEYS
    B, T = input_ids.shape
    assert parts["known_concepts"].shape == (B, T, N_KNOWN)
    assert parts["unknown_concepts"].shape == (B, T, N_UNKNOWN)
    assert parts["new_token"].shape == (B, T, VOCAB)
    assert parts["h_bar"].shape == (B, T, N_EMBD)


def test_single_concept_query(pgm_model, input_ids):
    out = pgm_model(input_ids, query=[KNOWN_NAMES[0]])
    assert out.probs.shape == (input_ids.shape[0], input_ids.shape[1], 1)


def test_latent_query(pgm_model, input_ids):
    out = pgm_model(input_ids, query=["h_bar"])
    assert out.probs.shape[-1] == N_EMBD


def test_evidence_contains_embeddings(pgm_model, input_ids):
    evidence = pgm_model._evidence(input_ids)
    assert set(evidence) >= {"input", "K", "U"}
    assert evidence["K"].shape[0] == N_KNOWN


# --------------------------------------------------------------------------
# BaseModel API mirror
# --------------------------------------------------------------------------

def test_model_attribute_is_probabilistic_model(pgm_model):
    assert isinstance(pgm_model.model, ProbabilisticModel)


def test_mirror_attributes(pgm_model):
    assert pgm_model.graph is None
    assert isinstance(pgm_model.latent_encoder, nn.Identity)
    assert pgm_model.train_inference is None
    assert pgm_model.eval_inference is not None


def test_inference_property_selects_eval(pgm_model):
    pgm_model.eval()
    assert pgm_model.inference is pgm_model.eval_inference


def test_pgm_model_is_bfloat16(pgm_model):
    assert {p.dtype for p in pgm_model.parameters()} == {torch.bfloat16}


# --------------------------------------------------------------------------
# Internal annotations builder
# --------------------------------------------------------------------------

def test_build_annotations_labels_and_cardinalities(pgm_model):
    ann = pgm_model.concept_annotations
    assert ann.labels[:N_KNOWN] == KNOWN_NAMES
    assert ann.labels[-1] == "new_token"
    assert len(ann.labels) == N_KNOWN + N_UNKNOWN + 1
    assert ann.cardinalities[-1] == VOCAB
    assert ann.cardinalities[0] == 1


def test_build_annotations_distributions(pgm_model):
    ann = pgm_model.concept_annotations
    assert ann.metadata[KNOWN_NAMES[0]]["distribution"] is RelaxedBernoulli
    assert ann.metadata["new_token"]["distribution"] is RelaxedOneHotCategorical


# --------------------------------------------------------------------------
# Test-time guards
# --------------------------------------------------------------------------

def test_lightning_not_supported():
    with pytest.raises(ValueError, match="Lightning"):
        build_pgm_model(lightning=True)


def test_train_inference_not_supported():
    with pytest.raises(ValueError, match="training inference"):
        build_pgm_model(train_inference=object())


def test_annotations_with_pretrained_head_raises(monkeypatch):
    # Make "pretrained" cheap: no-op the weight loader so we can record a
    # pretrained concept head without hitting the network, then check the guard.
    monkeypatch.setattr(sl.SteerlingLowLevelModel, "_load_steerling_weights",
                        lambda self, *a, **k: None)
    with pytest.raises(ValueError, match="internally"):
        build_pgm_model(pretrained_components=["known_head"], annotations=object())


def test_graph_with_pretrained_head_raises(monkeypatch):
    monkeypatch.setattr(sl.SteerlingLowLevelModel, "_load_steerling_weights",
                        lambda self, *a, **k: None)
    with pytest.raises(ValueError, match="internally"):
        build_pgm_model(pretrained_components=["unknown_head"], graph=object())
