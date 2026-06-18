import torch
import torch
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli, OneHotCategorical

from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.data import ToyDataset
from torch_concepts.nn import LinearEmbeddingToConcept, LinearConceptToConcept, \
    ParametricCPD, BayesianNetwork, AncestralInference, LearnablePrior, Sequential, \
    RandomPolicy, DoIntervention, intervention, InterventionModule


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
    concepts = ConceptVariable("concepts", members=['c1', 'c2'], distribution=Bernoulli) # shared concept variable for both c1 and c2
    tasks = ConceptVariable("xor", distribution=OneHotCategorical, size=2)

    # ParametricCPD setup
    input_cpd = ParametricCPD(input_var, parents=[], 
        parametrization=LearnablePrior(input_var.size)
    )
    backbone = ParametricCPD(latent_var, parents=[input_var],
        parametrization=torch.nn.Sequential(
            torch.nn.Linear(x_train.shape[1], latent_dims), 
            torch.nn.LeakyReLU()
        )
    )
    c_encoder = ParametricCPD(concepts, parents=[latent_var],
        parametrization=Sequential(
            # one shared encoder maps the latent to ALL members at once
            LinearEmbeddingToConcept(in_embeddings=latent_dims, out_concepts=concepts.size),
            torch.nn.Sigmoid()
        )
    )
    y_predictor = ParametricCPD(tasks, parents=[concepts],
        parametrization=Sequential(
            LinearConceptToConcept(in_concepts=2, out_concepts=2), 
            torch.nn.Softmax(dim=-1)
        )
    )
    
    # ProbabilisticModel Initialization
    concept_model = BayesianNetwork(
        variables=[input_var, latent_var, concepts, tasks],
        factors=[input_cpd, backbone, c_encoder, y_predictor]
    )

    # Inference Initialization
    inference_engine = AncestralInference(concept_model)
    evidence = {'input': x_train}
    query_concepts = {"concepts": c_train, "xor": y_train}

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
        c_pred = cy_pred.params['concepts']['probs']
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

    print("=== Interventions ===")
    print(cy_pred.params['concepts']['probs'][:3])

    int_policy_c = RandomPolicy(scale=100)
    int_strategy_c = DoIntervention(constants=1)
    c_encoder.parametrization["probs"] = InterventionModule(
        c_encoder.parametrization["probs"],
        intervention_strategy=int_strategy_c,
        intervention_policy=int_policy_c,
        out_concepts_to_intervene_on=[1]
    )
    cy_pred = inference_engine.query(
        query=query_concepts,
        evidence=evidence
    )
    print(cy_pred.params['concepts']['probs'][:3])

    with intervention(
            concept_model,
            intervention_strategy=int_strategy_c,
            intervention_policy=int_policy_c,
            variable_to_intervene_on="concepts",
            parameter_to_intervene_on="probs",
            members_to_intervene_on=["c1"]
    ):
        cy_pred = inference_engine.query(
            query=query_concepts,
            evidence=evidence
        )
        print(cy_pred.params['concepts']['probs'][:3])

    cy_pred = inference_engine.query(
        query=query_concepts,
        evidence=evidence
    )
    print(cy_pred.params['concepts']['probs'][:3])



    return


if __name__ == "__main__":
    main()
