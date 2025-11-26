import torch
from torch.distributions import RelaxedBernoulli, Normal, RelaxedOneHotCategorical

from torch_concepts import EndogenousVariable, ExogenousVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import ParametricCPD, ProbabilisticModel, AncestralSamplingInference, \
    CallableCC, UniformPolicy, DoIntervention, intervention
from torch_concepts.nn.functional import cace_score
from bp_with_conditional import BPInference


def main():

    batch_size = 3
    emb_size = 2

    # Variable setup
    emb = ExogenousVariable("emb", parents=[], distribution=Delta)
    a = EndogenousVariable("a", parents=["emb"], distribution=RelaxedBernoulli)
    b = EndogenousVariable("b", parents=["emb"], size=3, distribution=RelaxedOneHotCategorical)
    c = EndogenousVariable("c", parents=["a", "b"], distribution=RelaxedBernoulli)

    # ParametricCPD setup
    emb_cpd = ParametricCPD("emb", parametrization=torch.nn.Identity())
    a_cpd = ParametricCPD("a",
                                 parametrization=torch.nn.Sequential(torch.nn.Linear(emb_size, a.size),
                                                                     torch.nn.Sigmoid()))
    b_cpd = ParametricCPD("b",
                                parametrization=torch.nn.Sequential(torch.nn.Linear(emb_size, b.size),
                                                                     torch.nn.Sigmoid()))
    c_cpd = ParametricCPD("c",
                            parametrization=torch.nn.Sequential(torch.nn.Linear(a.size + b.size, c.size),
                                                                torch.nn.Sigmoid()))

    concept_model = ProbabilisticModel(variables=[emb, a, b, c],
                                        parametric_cpds=[emb_cpd, a_cpd, b_cpd, c_cpd])


    # Inference Initialization
    inference_engine = BPInference(concept_model)


    initial_input = {'emb': torch.randn((batch_size, emb_size))}
    query_concepts = ["a", "b", "c"]

    results = inference_engine.query(query_concepts, evidence=initial_input)

    print(results)
    exit()

    print("Genotype Predictions (first 5 samples):")
    print(results[:, 0][:5])
    print("Smoking Predictions (first 5 samples):")
    print(results[:, 1][:5])
    print("Tar Predictions (first 5 samples):")
    print(results[:, 2][:5])
    print("Cancer Predictions (first 5 samples):")
    print(results[:, 3][:5])

    # Original predictions (observational)
    original_results = inference_engine.query(
        query_concepts=["genotype", "smoking", "tar", "cancer"],
        evidence=initial_input
    )

    # Intervention: Force smoking to 0 (prevent smoking)
    smoking_strategy_0 = DoIntervention(
        model=concept_model.parametric_cpds,
        constants=0.0
    )
    with intervention(
            policies=UniformPolicy(out_features=1),
            strategies=smoking_strategy_0,
            target_concepts=["smoking"]
    ):
        intervened_results = inference_engine.query(
            query_concepts=["genotype", "smoking", "tar", "cancer"],
            evidence=initial_input
        )
        cancer_do_smoking_0 = intervened_results[:, 3]

    # Intervention: Force smoking to 1 (promote smoking)
    smoking_strategy_1 = DoIntervention(
        model=concept_model.parametric_cpds,
        constants=1.0
    )
    with intervention(
            policies=UniformPolicy(out_features=1),
            strategies=smoking_strategy_1,
            target_concepts=["smoking"]
    ):
        intervened_results = inference_engine.query(
            query_concepts=["genotype", "smoking", "tar", "cancer"],
            evidence=initial_input
        )
        cancer_do_smoking_1 = intervened_results[:, 3]

    ace_cancer_do_smoking = cace_score(cancer_do_smoking_0, cancer_do_smoking_1)
    print(f"ACE of smoking on cancer: {ace_cancer_do_smoking:.3f}")

    return


if __name__ == "__main__":
    main()
