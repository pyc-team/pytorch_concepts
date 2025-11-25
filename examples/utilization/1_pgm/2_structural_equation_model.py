import torch
from torch.distributions import RelaxedBernoulli, Normal

from torch_concepts import EndogenousVariable, ExogenousVariable
from torch_concepts.nn import ParametricCPD, ProbabilisticModel, AncestralSamplingInference, \
    CallableCC, UniformPolicy, DoIntervention, intervention
from torch_concepts.nn.functional import cace_score


def main():
    n_samples = 1000

    # Variable setup
    exogenous_var = ExogenousVariable("exogenous", parents=[], distribution=RelaxedBernoulli)
    genotype_var = EndogenousVariable("genotype", parents=["exogenous"], distribution=RelaxedBernoulli)
    smoking_var = EndogenousVariable("smoking", parents=["genotype"], distribution=RelaxedBernoulli)
    tar_var = EndogenousVariable("tar", parents=["genotype", "smoking"], distribution=RelaxedBernoulli)
    cancer_var = EndogenousVariable("cancer", parents=["tar"], distribution=RelaxedBernoulli)

    # ParametricCPD setup
    exogenous_cpd = ParametricCPD("exogenous", parametrization=torch.nn.Sigmoid())
    genotype_cpd = ParametricCPD("genotype",
                                 parametrization=torch.nn.Sequential(torch.nn.Linear(1, 1),
                                                                     torch.nn.Sigmoid()))
    smoking_cpd = ParametricCPD(["smoking"],
                                parametrization=CallableCC(lambda x: (x>0.5).float(), use_bias=False))
    tar_cpd = ParametricCPD("tar",
                            parametrization=CallableCC(lambda x: torch.logical_or(x[:, 0]>0.5, x[:, 1]>0.5).float().unsqueeze(-1),
                                                       use_bias=False))
    cancer_cpd = ParametricCPD("cancer",
                               parametrization=CallableCC(lambda x: x, use_bias=False))
    concept_model = ProbabilisticModel(variables=[exogenous_var, genotype_var, smoking_var, tar_var, cancer_var],
                                       parametric_cpds=[exogenous_cpd, genotype_cpd, smoking_cpd, tar_cpd, cancer_cpd])

    # Inference Initialization
    inference_engine = AncestralSamplingInference(concept_model, temperature=1.0, log_probs=False)
    initial_input = {'exogenous': torch.randn((n_samples, 1))}
    query_concepts = ["genotype", "smoking", "tar", "cancer"]

    results = inference_engine.query(query_concepts, evidence=initial_input)

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
