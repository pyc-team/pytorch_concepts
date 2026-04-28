"""
Shared CPD example: verify that shared=True produces the same inference
result as individual (shared=False) CPDs when weights are identical.

Architecture::

    input (D=8)
      └──► [c0, c1, c2, c3, c4]   (5 Bernoulli concepts)
              └──► task            (Categorical, 3 classes)
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, OneHotCategorical

from torch_concepts import seed_everything
from torch_concepts import ConceptVariable, LatentVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import (
    ParametricCPD,
    ProbabilisticModel,
    DeterministicInference,
    LinearLatentToConcept,
    LinearConceptToConcept,
)

latent_dim = 8
n_concepts = 5
n_classes = 3
concept_names = [f"c{i}" for i in range(n_concepts)]


def build_pgm_shared(encoder, task_head):
    """Build a PGM using a single shared CPD for all concepts."""
    input_var = LatentVariable("input", distribution=Delta, size=latent_dim)
    concept_vars = ConceptVariable(concept_names, distribution=Bernoulli)
    task_var = ConceptVariable("task", distribution=OneHotCategorical, size=n_classes)

    cpd_input = ParametricCPD("input", parametrization=nn.Identity())
    cpd_concepts = ParametricCPD(
        concept_names,
        parametrization=encoder,
        parents=["input"],
        shared=True,
        shared_name='shared'
    )
    cpd_task = ParametricCPD(
        "task",
        parametrization=task_head,
        parents=concept_names,
    )

    return ProbabilisticModel(
        variables=[input_var] + concept_vars + [task_var],
        factors=[cpd_input, cpd_concepts, cpd_task],
    )


def build_pgm_single(encoder, task_head):
    """Build an equivalent PGM with one CPD per concept (shared=False).

    Each concept gets its own LinearLatentToConcept(8,1) with weights copied
    from the corresponding row of the shared encoder.
    """
    input_var = LatentVariable("input", distribution=Delta, size=latent_dim)
    concept_vars = ConceptVariable(concept_names, distribution=Bernoulli)
    task_var = ConceptVariable("task", distribution=OneHotCategorical, size=n_classes)

    cpd_input = ParametricCPD("input", parametrization=nn.Identity())

    concept_cpds = []
    for i, name in enumerate(concept_names):
        single_enc = LinearLatentToConcept(in_latent=latent_dim, out_concepts=1)
        # Copy the i-th row of weights/bias from the shared encoder.
        with torch.no_grad():
            single_enc.encoder.weight.copy_(encoder.encoder.weight[i : i + 1])
            single_enc.encoder.bias.copy_(encoder.encoder.bias[i : i + 1])
        concept_cpds.append(
            ParametricCPD(name, parametrization=single_enc, parents=["input"])
        )

    cpd_task = ParametricCPD(
        "task",
        parametrization=task_head,
        parents=concept_names,
    )

    return ProbabilisticModel(
        variables=[input_var] + concept_vars + [task_var],
        factors=[cpd_input] + concept_cpds + [cpd_task],
    )


def main():
    seed_everything(42)

    # Shared encoder and task head — same weights for both PGMs.
    encoder = LinearLatentToConcept(in_latent=latent_dim, out_concepts=n_concepts)
    task_head = LinearConceptToConcept(in_concepts=n_concepts, out_concepts=n_classes)

    pgm_single = build_pgm_single(encoder, task_head)
    pgm_shared = build_pgm_shared(encoder, task_head)

    inf_single = DeterministicInference(pgm_single)
    inf_shared = DeterministicInference(pgm_shared)

    print(f"Single CPD map: {pgm_single._shared_cpd_map}")
    print(f"Shared CPD map: {pgm_shared._shared_cpd_map}")

    # ── Forward pass ─────────────────────────────────────────────────
    x = torch.randn(4, latent_dim)

    result_single = inf_single.query(
        concept_names + ["task"],
        evidence={"input": x},
        debug=True,
    )
    result_shared = inf_shared.query(
        concept_names + ["task"],
        evidence={"input": x},
        debug=True,
    )

    print(f"\nSingle result:\n{result_single}")
    print(f"\nShared result:\n{result_shared}")

    assert torch.allclose(result_single.probs, result_shared.probs, atol=1e-6), (
        f"Mismatch!\nSingle:\n{result_single}\nShared:\n{result_shared}"
    )
    print("\nShared CPD output matches single CPD output.")


if __name__ == "__main__":
    main()
