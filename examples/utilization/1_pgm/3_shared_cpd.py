"""Shared concept CPD via a plate variable.

A ``ConceptVariable`` with named ``members`` is a *plate*: ONE variable of size
``N`` produced by ONE ``ParametricCPD`` (the encoder runs once for all members),
while each member stays individually addressable by name. No special CPD type is
needed — it is an ordinary CPD over a single (size-N) variable:

    concepts  = ConceptVariable("concepts", members=["c0", ...], distribution=Bernoulli)
    c_encoder = ParametricCPD(concepts, parametrization=encoder, parents=[embs])

This script verifies:
  1. it is numerically equivalent to N independent CPDs with identical weights;
  2. querying all members runs the encoder exactly once (shared compute);
  3. members are views into one stacked tensor (no memory duplication);
  4. the encoder weights are counted once in ``model.parameters()``;
  5. members are addressable individually (single member; partial evidence);
  6. it works under a different backend (AncestralInference) unchanged.
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ParametricCPD, BayesianNetwork, DeterministicInference, AncestralInference,
    LinearEmbeddingToConcept, LearnablePrior, Sequential,
)

N = 4          # number of concepts in the plate
EMB = 5        # per-concept embedding size
B = 6          # batch
NAMES = [f"c{i}" for i in range(N)]


def build_shared(encoder):
    """PGM with ONE plate variable of N members over a single (N, EMB) embedding."""
    embs = EmbeddingVariable("embs", distribution=Delta, shape=(N, EMB))
    concepts = ConceptVariable("concepts", members=NAMES, distribution=Bernoulli)
    factors = [
        ParametricCPD(embs, parametrization=LearnablePrior(embs.size), parents=[]),
        ParametricCPD(concepts, parametrization=encoder, parents=[embs]),
    ]
    return BayesianNetwork(variables=[embs, concepts], factors=factors)


def build_individual(shared_linear):
    """Equivalent PGM with N independent CPDs, each on its own (EMB,) embedding.

    Each concept's encoder is its own ``LinearEmbeddingToConcept(EMB, 1)`` whose
    weights are copied from the shared encoder, so it is numerically identical.
    """
    emb_vars = [EmbeddingVariable(f"e{i}", distribution=Delta, size=EMB) for i in range(N)]
    concept_vars = [ConceptVariable(name, distribution=Bernoulli, size=1) for name in NAMES]
    factors = [ParametricCPD(e, parametrization=LearnablePrior(e.size), parents=[]) for e in emb_vars]
    for i, concept_var in enumerate(concept_vars):
        enc = LinearEmbeddingToConcept(in_embeddings=EMB, out_concepts=1)
        with torch.no_grad():
            enc.encoder.weight.copy_(shared_linear.encoder.weight)
            enc.encoder.bias.copy_(shared_linear.encoder.bias)
        parametrization = Sequential(enc, nn.Sigmoid(), nn.Flatten())
        factors.append(ParametricCPD(concept_var, parametrization=parametrization, parents=[emb_vars[i]]))
    return BayesianNetwork(variables=[*emb_vars, *concept_vars], factors=factors)


def main():
    seed_everything(42)

    # Shared encoder: per-concept embedding -> per-concept logit. It broadcasts over
    # the N axis of the (B, N, EMB) embedding, then Flatten -> (B, N).
    encoder = Sequential(LinearEmbeddingToConcept(in_embeddings=EMB, out_concepts=1), nn.Sigmoid(), nn.Flatten())

    # Count how many times the shared encoder actually runs.
    runs = {"n": 0}
    inner = encoder[0]
    _orig = inner.forward
    inner.forward = lambda *a, **k: (runs.__setitem__("n", runs["n"] + 1), _orig(*a, **k))[1]

    shared_net = build_shared(encoder)
    shared_linear = shared_net.factors["concepts"].parametrization["probs"][0]   # LinearEmbeddingToConcept
    individual_net = build_individual(shared_linear)

    embs_value = torch.randn(B, N, EMB)
    shared_eng = DeterministicInference(shared_net)
    indiv_eng = DeterministicInference(individual_net)

    # 1) Equivalence: plate members == N independent CPDs with the same weights.
    runs["n"] = 0
    shared_out = shared_eng.query(NAMES, evidence={"embs": embs_value})
    shared_probs = torch.cat([shared_out.params[n]["probs"] for n in NAMES], dim=1)  # (B, N)

    indiv_out = indiv_eng.query(NAMES, evidence={f"e{i}": embs_value[:, i, :] for i in range(N)})
    indiv_probs = torch.cat([indiv_out.params[n]["probs"] for n in NAMES], dim=1)    # (B, N)

    print(f"shared {tuple(shared_probs.shape)} vs individual {tuple(indiv_probs.shape)}")
    assert torch.allclose(shared_probs, indiv_probs, atol=1e-6), "shared != individual"

    # 2) Shared compute: querying all N members ran the encoder once.
    print(f"encoder runs for {N} members: {runs['n']}")
    assert runs["n"] == 1, runs["n"]

    # 3) Member params are views into one stacked tensor (no memory duplication).
    same_storage = shared_out.params["c0"]["probs"].untyped_storage().data_ptr() == \
        shared_out.params["c1"]["probs"].untyped_storage().data_ptr()
    print(f"member params share storage (views): {same_storage}")
    assert same_storage

    # 4) Encoder weights counted once in parameters().
    n_appear = sum(1 for p in shared_net.parameters() if any(p is q for q in shared_linear.parameters()))
    print(f"shared linear params in model.parameters(): {n_appear} (expect {len(list(shared_linear.parameters()))})")
    assert n_appear == len(list(shared_linear.parameters()))

    # 5) Individual addressing: query a single member, with the rest optionally observed.
    runs["n"] = 0
    one = shared_eng.query(["c0"], evidence={"embs": embs_value})
    print(f"query only c0 -> keys {sorted(one.params)}, encoder runs {runs['n']}")
    assert set(one.params) == {"c0"} and runs["n"] == 1

    others = {"c1": torch.ones(B, 1), "c2": torch.zeros(B, 1), "c3": torch.ones(B, 1)}
    partial = shared_eng.query(["c0"], evidence={"embs": embs_value, **others})
    assert tuple(partial.params["c0"]["probs"].shape) == (B, 1)
    print("partial evidence (c0 queried, c1..c3 observed): OK")

    # 6) A different backend works unchanged.
    anc = AncestralInference(shared_net).query(NAMES, evidence={"embs": embs_value})
    print(f"AncestralInference samples: {[tuple(anc.samples[n].shape) for n in NAMES]}")

    print("\nPlate CPD matches N independent CPDs; one shared compute; addressable per member.")


if __name__ == "__main__":
    main()
