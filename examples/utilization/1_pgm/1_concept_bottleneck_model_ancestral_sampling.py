import torch
from sklearn.metrics import accuracy_score
from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli

from torch_concepts import Annotations, AxisAnnotation, Variable, LatentVariable, EndogenousVariable
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.nn import ProbEncoderFromEmb, ProbPredictor, ParametricCPD, ProbabilisticModel, \
    RandomPolicy, DoIntervention, intervention, AncestralSamplingInference


def main():
    latent_dims = 10
    n_epochs = 1000
    n_samples = 1000
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    y_train = torch.cat([y_train, 1-y_train], dim=1)

    concept_names = ['c1', 'c2']
    task_names = ['xor']

    # Variable setup
    latent_var = LatentVariable("emb", parents=[], size=latent_dims)
    concepts = EndogenousVariable(concept_names, parents=["emb"], distribution=RelaxedBernoulli)
    tasks = EndogenousVariable("xor", parents=concept_names, distribution=RelaxedOneHotCategorical, size=2)

    # ParametricCPD setup
    backbone = ParametricCPD("emb", parametrization=torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], latent_dims), torch.nn.LeakyReLU()))
    c_encoder = ParametricCPD(["c1", "c2"], parametrization=ProbEncoderFromEmb(in_features_embedding=latent_dims, out_features=concepts[0].size))
    y_predictor = ParametricCPD("xor", parametrization=ProbPredictor(in_features_logits=sum(c.size for c in concepts), out_features=tasks.size))

    # ProbabilisticModel Initialization
    concept_model = ProbabilisticModel(variables=[latent_var, *concepts, tasks], parametric_cpds=[backbone, *c_encoder, y_predictor])

    # Inference Initialization
    inference_engine = AncestralSamplingInference(concept_model, temperature=1.)
    initial_input = {'emb': x_train}
    query_concepts = ["c1", "c2", "xor"]

    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    concept_model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        cy_pred = inference_engine.query(query_concepts, evidence=initial_input)
        c_pred = cy_pred[:, :c_train.shape[1]]
        y_pred = cy_pred[:, c_train.shape[1]:]

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + 0 * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred > 0.5)
            concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    print("=== Interventions ===")
    print(cy_pred[:5])

    int_policy_c = RandomPolicy(out_features=concept_model.concept_to_variable["c1"].size, scale=100)
    int_strategy_c = DoIntervention(model=concept_model.parametric_cpds, constants=-10)
    with intervention(policies=int_policy_c,
                      strategies=int_strategy_c,
                      target_concepts=["c1", "c2"]):
        cy_pred = inference_engine.query(query_concepts, evidence=initial_input)
        print(cy_pred[:5])

    return


if __name__ == "__main__":
    main()
