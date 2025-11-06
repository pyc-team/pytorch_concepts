import torch
from fontTools.subset import subset
from sklearn.metrics import accuracy_score
from torch.distributions import Normal

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data import ToyDataset
from torch_concepts.nn import ProbEncoderFromEmb, ProbPredictor, intervention, GroundTruthIntervention, \
    UncertaintyInterventionPolicy, intervention, DoIntervention, DistributionIntervention, UniformPolicy, RandomPolicy


def main():
    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    c_train = torch.concat([c_train, c_train, c_train], dim=1)
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]

    c_annotations = Annotations({1: AxisAnnotation(concept_names+['C3', 'C4', 'C5', 'C6'])})
    y_annotations = Annotations({1: AxisAnnotation(task_names)})

    encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
    )
    encoder_layer = ProbEncoderFromEmb(in_features_embedding=latent_dims, out_features=c_annotations.shape[1])
    y_predictor = ProbPredictor(in_features_logits=c_annotations.shape[1], out_features=y_annotations.shape[1])

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
        c_pred = encoder_layer(embedding=emb)
        y_pred = y_predictor(logits=c_pred)

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


    print(c_pred[:5])
    print(y_pred[:5])
    model = torch.nn.ModuleDict({
        "encoder": encoder,
        "encoder_layer": encoder_layer,
        "y_predictor": y_predictor,
    })
    quantile = 0.8
    int_policy_c = UniformPolicy(out_annotations=c_annotations, subset=["C1", "C4", "C5", "C6"])
    int_strategy_c = GroundTruthIntervention(model=model, ground_truth=torch.logit(c_train, eps=1e-6))
    int_policy_y = UncertaintyInterventionPolicy(out_annotations=y_annotations, subset=["xor"])
    int_strategy_y = DoIntervention(model=model, constants=100)
    print("Uncertainty + DoIntervention")
    with intervention(policies=[int_policy_c, int_policy_y],
                      strategies=[int_strategy_c, int_strategy_y],
                      on_layers=["encoder_layer.encoder", "y_predictor.predictor"],
                      quantiles=[quantile, 1]):
        emb = model["encoder"](x_train)
        c_pred = model["encoder_layer"](emb)
        y_pred = model["y_predictor"](c_pred)
        print(c_pred[:5])
        print(y_pred[:5])

    print("Do Intervention + UniformPolicy")
    int_policy_c = UniformPolicy(out_annotations=c_annotations, subset=["C1", "C2", "C6"])
    int_strategy_c = DoIntervention(model=model, constants=-10)
    with intervention(policies=[int_policy_c],
                      strategies=[int_strategy_c],
                      on_layers=["encoder_layer.encoder"],
                      quantiles=[quantile]):
        emb = model["encoder"](x_train)
        c_pred = model["encoder_layer"](emb)
        y_pred = model["y_predictor"](c_pred)
        print(c_pred[:5])

    print("Do Intervention + RandomPolicy")
    int_policy_c = RandomPolicy(out_annotations=c_annotations, scale=100, subset=["C1", "C2", "C6"])
    int_strategy_c = DoIntervention(model=model, constants=-10)
    with intervention(policies=[int_policy_c],
                      strategies=[int_strategy_c],
                      on_layers=["encoder_layer.encoder"],
                      quantiles=[quantile]):
        emb = model["encoder"](x_train)
        c_pred = model["encoder_layer"](emb)
        y_pred = model["y_predictor"](c_pred)
        print(c_pred[:5])

    print("Distribution Intervention")
    int_strategy_c = DistributionIntervention(model=model, dist=torch.distributions.Normal(loc=0, scale=1))
    with intervention(policies=[int_policy_c],
                      strategies=[int_strategy_c],
                      on_layers=["encoder_layer.encoder"],
                      quantiles=[quantile]):
        emb = model["encoder"](x_train)
        c_pred = model["encoder_layer"](emb)
        y_pred = model["y_predictor"](c_pred)
        print(c_pred[:5])

    print("Single Intervention")
    with intervention(policies=[int_policy_c],
                      strategies=[int_strategy_c],
                      on_layers=["encoder_layer.encoder"],
                      quantiles=[quantile]):
        emb = model["encoder"](x_train)
        c_pred = model["encoder_layer"](emb)
        y_pred = model["y_predictor"](c_pred)
        print(c_pred[:5])
        print(y_pred[:5])

    return


if __name__ == "__main__":
    main()
