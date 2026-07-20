"""
This example is equivalent to the previous one (1.1_cbm.py), but it uses plate notation for the concepts variable.
The plate notation allows to instantiate a single nn.Module which produces the parameters for multiple conditionally 
independent variables.
This allows to:
- make inference faster: the inference engine do not has to loop over the conditionally independent variables, 
but it can compute the parameters for all of them at once.
- easier implementation: we can specify only one variable (e.g., "concepts") which includes mulitple conditionally 
independent variables (e.g., "c1" and "c2"), instead of specifying each variable separately.
"""

import torch
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli, OneHotCategorical, Normal

from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.data import ToyDataset
from torch_concepts.nn import LinearEmbeddingToConcept, LinearConceptToConcept, \
    ParametricCPD, BayesianNetwork, DeterministicInference, LearnablePrior, Sequential


def main():
    seed_everything(42)

    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5

    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]

    y_train = torch.cat([y_train, 1 - y_train], dim=1)

    # Variable setup
    input_var = EmbeddingVariable("input", distribution=Delta, size=x_train.shape[1])
    latent_var = EmbeddingVariable("latent", distribution=Delta, size=latent_dims)
    concepts = ConceptVariable("concepts", distribution=Bernoulli, size=1, members=["c1", "c2"])
    tasks = ConceptVariable("xor", distribution=OneHotCategorical, size=2)

    # ParametricCPD setup
    input_cpd = ParametricCPD(input_var, parametrization=LearnablePrior(x_train.shape[1]),
                              parents=[])  # learnable prior parametrization is automatically set

    backbone = ParametricCPD(latent_var,
                             parametrization=torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], latent_dims),
                                                                 torch.nn.LeakyReLU()), parents=[input_var])

    c_encoder = ParametricCPD(
        concepts, 
        parametrization={'logits': Sequential(LinearEmbeddingToConcept(in_embeddings=latent_dims, out_concepts=2))}, 
        parents=[latent_var]
    )

    y_predictor = ParametricCPD(
        tasks, 
        parametrization={'logits': Sequential(LinearConceptToConcept(in_concepts=2, out_concepts=2))}, 
        parents=[concepts]
    )

    # ProbabilisticModel Initialization
    concept_model = BayesianNetwork(
        variables=[input_var, latent_var, concepts, tasks],
        factors=[input_cpd, backbone, c_encoder, y_predictor]
    )

    # Inference Initialization
    inference_engine = DeterministicInference(concept_model, activate_before_propagation=True)
    evidence = {'input': x_train}
    query_concepts = {"concepts": c_train, "xor": y_train}

    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    concept_model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        cy_pred = inference_engine.query(
            query=query_concepts,
            evidence=evidence
        )
        c_pred = cy_pred.logits['concepts']
        y_pred = cy_pred.logits['xor']

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.detach() > 0.)
            concept_accuracy = accuracy_score(c_train, c_pred.detach() > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    return


if __name__ == "__main__":
    main()
