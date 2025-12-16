import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from torch.distributions import RelaxedBernoulli, Normal, RelaxedOneHotCategorical
from torch_concepts import EndogenousVariable, ExogenousVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import ParametricCPD, ProbabilisticModel, AncestralSamplingInference, \
    CallableCC, UniformPolicy, DoIntervention, intervention
from torch_concepts.nn.functional import cace_score
from bp_with_conditional import BPInference


def main():


    # Fix torch random seed for reproducibility
    torch.manual_seed(22)



    hidden_size = 100
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    n = 300
    n_epochs = 10000

    #BP parameters
    alpha = 1. # no dumping / residual factor
    n_iters = 5


    # Deterministic CPTs
    data = {'emb': torch.randint(0,3, (n, 2)).float()}
    data["a"] = (data["emb"][:, 0:1] % 2 == 0 ).long()
    data["b"] = torch.nn.functional.one_hot((data["emb"][:, 0] % 3).long(), num_classes=3).float()
    data['c'] = data['a'] * data['b'][:, 2:3]



    # Variable setup
    emb = ExogenousVariable("emb", parents=[], size=2, distribution=Delta)
    a = EndogenousVariable("a", parents=["emb"], distribution=RelaxedBernoulli)
    b = EndogenousVariable("b", parents=["emb"], size=3, distribution=RelaxedOneHotCategorical)
    c = EndogenousVariable("c", parents=["a", "b"], distribution=RelaxedBernoulli)

    # ParametricCPD setup
    emb_cpd = ParametricCPD("emb", parametrization=torch.nn.Identity())
    a_cpd = ParametricCPD("a",
                                 parametrization=torch.nn.Sequential(torch.nn.Linear(emb.size, hidden_size),
                                                                     torch.nn.ReLU(),
                                                                     torch.nn.Linear(hidden_size, a.size),
                                                                     torch.nn.Sigmoid()))
    b_cpd = ParametricCPD("b",
                                parametrization=torch.nn.Sequential(torch.nn.Linear(emb.size, hidden_size),
                                                                     torch.nn.ReLU(),
                                                                     torch.nn.Linear(hidden_size, b.size),
                                                                     torch.nn.Softmax(dim=-1)))
    c_cpd = ParametricCPD("c",
                            parametrization=torch.nn.Sequential(torch.nn.Linear(a.size + b.size, hidden_size),
                                                                torch.nn.ReLU(),
                                                                torch.nn.Linear(hidden_size, c.size),
                                                                torch.nn.Sigmoid()))

    concept_model = ProbabilisticModel(variables=[emb, a, b, c],
                                        parametric_cpds=[emb_cpd, a_cpd, b_cpd, c_cpd])


    # Inference Initialization
    inference_engine = BPInference(concept_model, iters = n_iters, alpha= alpha)



    initial_input = {'emb': data['emb']}
    query_concepts = ["a", "b", "c"]

    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.001)
    concept_model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        cy_pred = inference_engine.query(query_concepts, observed=initial_input, evidence=None)
        loss = 0
        for i, key in enumerate(query_concepts):
            pred = cy_pred[key]
            gt = data[key]
            if concept_model.concept_to_variable[key].distribution is RelaxedBernoulli:
                loss += torch.nn.CrossEntropyLoss()(pred, gt.squeeze(-1).long())
            elif concept_model.concept_to_variable[key].distribution  is RelaxedOneHotCategorical:
                loss += torch.nn.CrossEntropyLoss()(pred, gt.argmax(dim=1))

        if epoch % 100 == 0:
            for i, key in enumerate(query_concepts):
                pred = cy_pred[key]
                gt = data[key]
                if concept_model.concept_to_variable[key].distribution is RelaxedBernoulli:
                    accuracy = (pred.argmax(dim=1) == gt.squeeze(-1).long()).float().mean().item()
                elif concept_model.concept_to_variable[key].distribution  is RelaxedOneHotCategorical:
                    accuracy = (pred.argmax(dim=1) == gt.argmax(dim=1)).float().mean().item()
                print(f"Epoch {epoch}: Concept {key} Accuracy: {accuracy:.2f}")


        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()
