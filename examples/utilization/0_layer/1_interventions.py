import torch
from sklearn.metrics import accuracy_score

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data.datasets import ToyDataset
from torch_concepts.nn import LinearZC, LinearCC, GroundTruthIntervention, \
    UncertaintyInterventionPolicy, intervention, DoIntervention, DistributionIntervention, UniformPolicy, RandomPolicy


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
    concept_names = [dataset.concept_names[i] for i in concept_idx]
    task_names = [dataset.concept_names[i] for i in task_idx]

    c_train = torch.concat([c_train, c_train, c_train], dim=1)
    n_features = x_train.shape[1]

    c_annotations = Annotations({1: AxisAnnotation(concept_names+['C3', 'C4', 'C5', 'C6'])})
    y_annotations = Annotations({1: AxisAnnotation(task_names)})

    encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
    )
    encoder_layer = LinearZC(in_features=latent_dims, out_features=c_annotations.shape[1])
    y_predictor = LinearCC(in_features_endogenous=c_annotations.shape[1], out_features=y_annotations.shape[1])

    # all models in a ModuleDict for easier intervention
    model = torch.nn.ModuleDict({
        "encoder": encoder,
        "encoder_layer": encoder_layer,
        "y_predictor": y_predictor,
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_pred = encoder_layer(input=emb)
        y_pred = y_predictor(endogenous=c_pred)

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

    int_policy_c = UniformPolicy(out_features=c_train.shape[1])
    int_strategy_c = GroundTruthIntervention(model=encoder_layer, ground_truth=torch.logit(c_train, eps=1e-6))

    print("Uncertainty + Ground Truth Intervention:")
    with intervention(policies=int_policy_c,
                      strategies=int_strategy_c,
                      target_concepts=[0, 1]) as new_encoder_layer:
        emb = model["encoder"](x_train)
        c_pred = new_encoder_layer(input=emb)
        y_pred = model["y_predictor"](endogenous=c_pred)
        print("\nConcept predictions (first 5):")
        print(c_pred[:5])
        print("\nGround truth (first 5):")
        print(torch.logit(c_train, eps=1e-6)[:5])

    int_policy_c = UniformPolicy(out_features=c_train.shape[1])
    int_strategy_c = DoIntervention(model=model["encoder_layer"], constants=-10)

    print("Do Intervention + Uniform Policy:")
    with intervention(
            policies=int_policy_c,
            strategies=int_strategy_c,
            target_concepts=[1],
    ) as new_encoder_layer:
        emb = model["encoder"](x_train)
        c_pred = new_encoder_layer(input=emb)
        y_pred = model["y_predictor"](endogenous=c_pred)
        print("\nConcept predictions (first 5):")
        print(c_pred[:5, :2])

    int_policy_c = RandomPolicy(out_features=c_train.shape[1])
    int_strategy_c = DoIntervention(model=encoder_layer, constants=-10)

    print("Do Intervention + Random Policy:")
    with intervention(
            policies=int_policy_c,
            strategies=int_strategy_c,
            target_concepts=[0, 1],
            quantiles=0.5
    ) as new_encoder_layer:
        emb = model["encoder"](x_train)
        c_pred = new_encoder_layer(input=emb)
        y_pred = model["y_predictor"](endogenous=c_pred)
        print("\nConcept predictions (first 5):")
        print(c_pred[:5, :2])

    int_strategy_c = DistributionIntervention(model=encoder_layer, dist=torch.distributions.Normal(loc=50, scale=1))

    print("Distribution Intervention:")
    with intervention(
            policies=int_policy_c,
            strategies=int_strategy_c,
            target_concepts=[1, 3],
            quantiles=.5
    ) as new_encoder_layer:
        emb = model["encoder"](x_train)
        c_pred = new_encoder_layer(input=emb)
        y_pred = model["y_predictor"](c_pred)
        print("\nConcept predictions (first 5):")
        print(c_pred[:5])

    return


if __name__ == "__main__":
    main()
