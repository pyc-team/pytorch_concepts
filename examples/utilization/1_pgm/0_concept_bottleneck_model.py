import torch
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli, RelaxedOneHotCategorical

from torch_concepts import Annotations, AxisAnnotation, Variable, InputVariable, EndogenousVariable
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.nn import LinearZC, LinearCC, ParametricCPD, ProbabilisticModel, \
    RandomPolicy, DoIntervention, intervention, DeterministicInference, LazyConstructor


def main():
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
    concept_names = ['c1', 'c2']

    y_train = torch.cat([y_train, 1-y_train], dim=1)

    # Variable setup
    input_var = InputVariable("input", parents=[], size=latent_dims)
    concepts = EndogenousVariable(concept_names, parents=["input"], distribution=Bernoulli)
    tasks = EndogenousVariable("xor", parents=concept_names, distribution=RelaxedOneHotCategorical, size=2)

    # ParametricCPD setup
    backbone = ParametricCPD("input", parametrization=torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], latent_dims), torch.nn.LeakyReLU()))
    c_encoder = ParametricCPD(["c1", "c2"], parametrization=LazyConstructor(LinearZC))
    y_predictor = ParametricCPD("xor", parametrization=LinearCC(in_features_endogenous=2, out_features=2))

    # ProbabilisticModel Initialization
    concept_model = ProbabilisticModel(variables=[input_var, *concepts, tasks], parametric_cpds=[backbone, *c_encoder, y_predictor])

    # Inference Initialization
    inference_engine = DeterministicInference(concept_model)
    initial_input = {'input': x_train}
    query_concepts = ["c1", "c2", "xor"]

    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    concept_model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        cy_pred = inference_engine.query(query_concepts, evidence=initial_input, debug=True)
        c_pred = cy_pred[:, :c_train.shape[1]]
        y_pred = cy_pred[:, c_train.shape[1]:]

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred > 0.)
            concept_accuracy = accuracy_score(c_train, c_pred > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    print("=== Interventions ===")
    print(cy_pred[:5])

    int_policy_c = RandomPolicy(out_features=concept_model.concept_to_variable["c1"].size, scale=100)
    int_strategy_c = DoIntervention(model=concept_model.parametric_cpds, constants=-10)
    with intervention(policies=int_policy_c,
                      strategies=int_strategy_c,
                      target_concepts=["c1", "c2"],
                      quantiles=1):
        cy_pred = inference_engine.query(query_concepts, evidence=initial_input, debug=True)
        print(cy_pred[:5])

    return


if __name__ == "__main__":
    main()
