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
                                                                     torch.nn.Softmax(dim=-1)))
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

if __name__ == "__main__":
    main()
