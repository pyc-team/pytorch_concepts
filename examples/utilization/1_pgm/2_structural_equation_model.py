"""Structural equation model (SEM) over a small causal graph.

Causal structure (the classic genotype / smoking / tar / cancer SCM, with
genotype a common cause of both smoking and tar)::

    input ──► genotype ──► smoking
                  │            │
                  └────────────┴──► tar ──► cancer

Each variable is a Bernoulli concept. The mechanisms are deterministic
structural equations expressed with :class:`CallableConceptToConcept`
(``use_bias=False`` so the equation is exact), except ``genotype`` which is a
learnable Sigmoid-linear of the exogenous ``input`` noise.

:class:`AncestralInference` draws a reparameterised (straight-through) sample
per variable in topological order, so ``out.samples`` holds a hard 0/1
realisation of every node.
"""

import torch
from torch.distributions import Bernoulli

from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import ParametricCPD, BayesianNetwork, AncestralInference, \
    CallableConceptToConcept, LearnablePrior


def main():
    seed_everything(42)
    n_samples = 1000

    # Variable setup: an exogenous noise root + four binary concepts.
    input_var = EmbeddingVariable("input", distribution=Delta, size=1)
    genotype_var = ConceptVariable("genotype", distribution=Bernoulli)
    smoking_var = ConceptVariable("smoking", distribution=Bernoulli)
    tar_var = ConceptVariable("tar", distribution=Bernoulli)
    cancer_var = ConceptVariable("cancer", distribution=Bernoulli)

    # One structural equation (parametrization module) per variable.
    layers = {
        # genotype: learnable predisposition driven by the exogenous noise.
        "genotype": torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Sigmoid()),
        # smoking := 1[genotype].
        "smoking": CallableConceptToConcept(lambda g: (g > 0.5).float(), use_bias=False),
        # tar := genotype OR smoking. Parents are concatenated along the last
        # dim, so the callable receives a (batch, 2) tensor.
        "tar": CallableConceptToConcept(
            lambda gs: torch.logical_or(gs[:, 0] > 0.5, gs[:, 1] > 0.5).float().unsqueeze(-1),
            use_bias=False),
        # cancer := tar.
        "cancer": CallableConceptToConcept(lambda t: t, use_bias=False),
    }

    # ParametricCPD setup — wire each structural equation to its variable.
    input_cpd = ParametricCPD(input_var, parametrization=LearnablePrior(input_var.size), parents=[])
    genotype_cpd = ParametricCPD(genotype_var, parametrization=layers['genotype'], parents=[input_var])
    smoking_cpd = ParametricCPD(smoking_var, parametrization=layers['smoking'], parents=[genotype_var])
    tar_cpd = ParametricCPD(tar_var, parametrization=layers['tar'], parents=[genotype_var, smoking_var])
    cancer_cpd = ParametricCPD(cancer_var, parametrization=layers['cancer'], parents=[tar_var])

    concept_model = BayesianNetwork(
        variables=[input_var, genotype_var, smoking_var, tar_var, cancer_var],
        factors=[input_cpd, genotype_cpd, smoking_cpd, tar_cpd, cancer_cpd],
    )

    # Inference: ancestral sampling through the structural equations.
    inference_engine = AncestralInference(concept_model)
    initial_input = {'input': torch.randn((n_samples, 1))}
    query_concepts = ["genotype", "smoking", "tar", "cancer"]

    results = inference_engine.query(query_concepts, evidence=initial_input)

    for name in query_concepts:
        samples = results.samples[name]
        print(f"{name.capitalize()} samples (first 5): {samples[:5].flatten().tolist()}")
        print(f"  P({name}=1) ≈ {samples.mean().item():.3f}")

    # === Interventions / CACE (TODO: not yet wired for the mid-level API) ===
    # The do-operator and causal effect estimation will be added once the
    # intervention machinery is available for BayesianNetwork. Sketch:
    #
    # # Force smoking to 0, then to 1, and compare the cancer distribution.
    # with intervention(strategies=DoIntervention(constants=0.0),
    #                   target_concepts=["smoking"]):
    #     cancer_do_0 = inference_engine.query(["cancer"], evidence=initial_input).samples["cancer"]
    # with intervention(strategies=DoIntervention(constants=1.0),
    #                   target_concepts=["smoking"]):
    #     cancer_do_1 = inference_engine.query(["cancer"], evidence=initial_input).samples["cancer"]
    # ace = cace_score(cancer_do_0, cancer_do_1)
    # print(f"ACE of smoking on cancer: {ace:.3f}")

    return


if __name__ == "__main__":
    main()
