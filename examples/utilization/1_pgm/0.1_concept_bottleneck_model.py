import torch
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli, OneHotCategorical

from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.data import ToyDataset
from torch_concepts.nn import LinearEmbeddingToConcept, LinearConceptToConcept, ParametricCPD, BayesianNetwork, DeterministicInference


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

    y_train = torch.cat([y_train, 1-y_train], dim=1)

    # Variable setup
    input_var = EmbeddingVariable("input", distribution=Delta, size=x_train.shape[1])
    latent_var = EmbeddingVariable("latent", distribution=Delta, size=latent_dims)
    concepts = ConceptVariable(['c1', 'c2'], distribution=Bernoulli)
    tasks = ConceptVariable("xor", distribution=OneHotCategorical, size=2)

    # ParametricCPD setup
    input_cpd = ParametricCPD(input_var, parents=[]) # learnable prior parametrization is automatically set
    backbone = ParametricCPD(latent_var, parametrization=torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], latent_dims), torch.nn.LeakyReLU()), parents=[input_var])
    c_encoder = ParametricCPD(concepts, parametrization=LinearEmbeddingToConcept(in_embeddings=latent_dims, out_concepts=1), parents=[latent_var])
    y_predictor = ParametricCPD(tasks, parametrization=LinearConceptToConcept(in_concepts=2, out_concepts=2), parents=[*concepts])

    # ProbabilisticModel Initialization
    concept_model = BayesianNetwork(
        variables=[input_var, latent_var, *concepts, tasks],
        factors=[input_cpd, backbone, *c_encoder, y_predictor]
    )

    # Inference Initialization
    inference_engine = DeterministicInference(concept_model)
    evidence = {'input': x_train}
    query_concepts = {"c1": c_train[:, 0], "c2": c_train[:, 1], "xor": y_train}

    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    concept_model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        cy_pred = inference_engine.query(
            query = query_concepts,
            evidence = evidence
        )
        c_pred = torch.cat([cy_pred.params['c1']['probs'], cy_pred.params['c2']['probs']], dim=1)
        y_pred = cy_pred.params['xor']['probs']

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.detach() > 0.5)
            concept_accuracy = accuracy_score(c_train, c_pred.detach() > 0.5)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    # print("=== Interventions ===")
    # print(cy_pred.logits[:5])

    # int_policy_c = RandomPolicy(out_concepts=concept_model.concept_to_variable["c1"].size, scale=100)
    # int_strategy_c = DoIntervention(model=concept_model.parametric_cpds, constants=-10)
    # with intervention(policies=int_policy_c,
    #                   strategies=int_strategy_c,
    #                   target_concepts=["c1", "c2"],
    #                   quantiles=1):
    #     # intervention affect the layer output 
    #     # -> the parametrization of the distribution 
    #     # -> the logits for discrete variables
    #     cy_pred = inference_engine.query(query_concepts, evidence=initial_input, return_logits=True)
    #     print(cy_pred.logits[:5])

    return


if __name__ == "__main__":
    main()
