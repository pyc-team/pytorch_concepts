"""Shared CPD via a vector-valued concept variable.

With the mid-level API a single CPD can emit *all* of a group's concepts at
once: declare one :class:`ConceptVariable` with a vector event ``shape=(K,)``
and give it one encoder ``8 -> K``. That is the idiomatic "shared CPD" — no
special ``shared=`` flag needed.

This example checks that the shared (vector) formulation is equivalent to the
"individual" one — ``K`` scalar concept variables, each with its own ``8 -> 1``
encoder whose weights are the matching row of the shared encoder — by asserting
both PGMs produce identical concept and task outputs.

Architecture::

    input (D=8)
      └──► concepts  (K=5 Bernoulli)
              └──► task   (OneHotCategorical, 3 classes)
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, OneHotCategorical

from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ParametricCPD,
    BayesianNetwork,
    DeterministicInference,
    LinearEmbeddingToConcept,
    LinearConceptToConcept,
)

latent_dim = 8
n_concepts = 5
n_classes = 3
concept_names = [f"c{i}" for i in range(n_concepts)]


def build_pgm_shared(encoder, task_head):
    """Shared CPD: ONE vector-valued concept variable, one encoder for all K."""
    input_var = EmbeddingVariable("input", distribution=Delta, size=latent_dim)
    concepts_var = ConceptVariable("concepts", distribution=Bernoulli, shape=(n_concepts,))
    task_var = ConceptVariable("task", distribution=OneHotCategorical, size=n_classes)

    return BayesianNetwork(
        variables=[input_var, concepts_var, task_var],
        factors=[
            ParametricCPD(input_var, parametrization={"value": nn.Identity()}, parents=[]),
            ParametricCPD(concepts_var, parametrization={"logits": encoder}, parents=[input_var]),
            ParametricCPD(task_var, parametrization={"logits": task_head}, parents=[concepts_var]),
        ],
    )


def build_pgm_individual(encoder, task_head):
    """Individual CPDs: one scalar concept variable per concept.

    Each concept gets its own ``LinearEmbeddingToConcept(8, 1)`` with weights
    copied from the matching row of the shared encoder, so it is numerically
    equivalent to the shared formulation.
    """
    input_var = EmbeddingVariable("input", distribution=Delta, size=latent_dim)
    concept_vars = ConceptVariable(concept_names, distribution=Bernoulli)
    task_var = ConceptVariable("task", distribution=OneHotCategorical, size=n_classes)

    factors = [ParametricCPD(input_var, parametrization={"value": nn.Identity()}, parents=[])]
    for i, concept_var in enumerate(concept_vars):
        single_enc = LinearEmbeddingToConcept(in_embeddings=latent_dim, out_concepts=1)
        with torch.no_grad():
            single_enc.encoder.weight.copy_(encoder.encoder.weight[i : i + 1])
            single_enc.encoder.bias.copy_(encoder.encoder.bias[i : i + 1])
        factors.append(
            ParametricCPD(concept_var, parametrization={"logits": single_enc}, parents=[input_var])
        )
    factors.append(
        ParametricCPD(task_var, parametrization={"logits": task_head}, parents=[*concept_vars])
    )

    return BayesianNetwork(
        variables=[input_var, *concept_vars, task_var],
        factors=factors,
    )


def main():
    seed_everything(42)

    # Shared encoder (8 -> 5) and task head (5 -> 3) — same weights for both PGMs.
    encoder = LinearEmbeddingToConcept(in_embeddings=latent_dim, out_concepts=n_concepts)
    task_head = LinearConceptToConcept(in_concepts=n_concepts, out_concepts=n_classes)

    pgm_shared = build_pgm_shared(encoder, task_head)
    pgm_individual = build_pgm_individual(encoder, task_head)

    inf_shared = DeterministicInference(pgm_shared)
    inf_individual = DeterministicInference(pgm_individual)

    x = torch.randn(4, latent_dim)
    out_shared = inf_shared.query(["concepts", "task"], evidence={"input": x})
    out_individual = inf_individual.query(concept_names + ["task"], evidence={"input": x})

    # Shared emits all concepts in one (B, K) tensor; individual emits K (B, 1)
    # tensors — stack them to compare.
    concepts_shared = out_shared.params["concepts"]["logits"]
    concepts_individual = torch.cat(
        [out_individual.params[name]["logits"] for name in concept_names], dim=-1
    )
    task_shared = out_shared.params["task"]["logits"]
    task_individual = out_individual.params["task"]["logits"]

    print(f"concepts — shared {tuple(concepts_shared.shape)} vs individual {tuple(concepts_individual.shape)}")
    print(f"task     — shared {tuple(task_shared.shape)} vs individual {tuple(task_individual.shape)}")

    assert torch.allclose(concepts_shared, concepts_individual, atol=1e-6), "concept outputs differ"
    assert torch.allclose(task_shared, task_individual, atol=1e-6), "task outputs differ"
    print("\nShared (vector-variable) CPD matches the individual per-concept CPDs.")


if __name__ == "__main__":
    main()
