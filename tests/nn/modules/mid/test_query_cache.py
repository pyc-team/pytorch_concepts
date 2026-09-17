"""The per-engine query caches are bounded and evict oldest-first."""
import torch
from torch.distributions import Bernoulli

from torch_concepts import ConceptVariable, EmbeddingVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (BayesianNetwork, DeterministicInference,
                               ParametricCPD, Sequential, LinearEmbeddingToConcept)
from torch_concepts.nn.modules.mid.inference.base import QUERY_CACHE_SIZE, _cache_put


def test_cache_put_evicts_fifo():
    cache = {}
    for i in range(QUERY_CACHE_SIZE + 5):
        _cache_put(cache, (i,), i)
    assert len(cache) == QUERY_CACHE_SIZE
    assert (0,) not in cache and (4,) not in cache      # the 5 oldest are gone
    assert (5,) in cache and (QUERY_CACHE_SIZE + 4,) in cache


def test_engine_caches_stay_bounded():
    k = QUERY_CACHE_SIZE + 10
    z = EmbeddingVariable("z", distribution=Delta, size=8)
    c = ConceptVariable("c", distribution=Bernoulli, size=1,
                        members=[f"c{i}" for i in range(k)])
    pgm = BayesianNetwork(
        variables=[z, c],
        factors=[ParametricCPD(z, parametrization=torch.nn.Identity(), parents=[]),
                 ParametricCPD(c, parametrization={'logits': Sequential(
                     LinearEmbeddingToConcept(in_embeddings=8, out_concepts=k))},
                     parents=[z])],
    )
    engine = DeterministicInference(pgm)
    x = torch.randn(4, 8)
    for i in range(k):                      # a distinct query signature each time
        engine.query([f"c{i}"], evidence={"z": x})
    assert len(engine._label_cache) == QUERY_CACHE_SIZE
    assert len(engine._annotation_cache) == QUERY_CACHE_SIZE

    engine.clear_cache()
    assert not engine._label_cache and not engine._annotation_cache
    engine.query(["c0"], evidence={"z": x})          # still works after a clear
    assert len(engine._label_cache) == 1
