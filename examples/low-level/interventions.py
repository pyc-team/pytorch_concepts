import torch
from sklearn.metrics import accuracy_score
from torch.distributions import Normal

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data import ToyDataset
from torch_concepts.nn import ProbEncoder, ProbPredictor, intervene_in_dict, ConstantTensorIntervention, \
    ConstantLikeIntervention, DistributionIntervention


def main():
    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]

    c_annotations = Annotations({1: AxisAnnotation(concept_names)})
    y_annotations = Annotations({1: AxisAnnotation(task_names)})

    encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
    )
    encoder_layer = ProbEncoder(latent_dims, c_annotations)
    y_predictor = ProbPredictor(encoder_layer.out_features, y_annotations)

    # all models in a ModuleDict for easier intervention
    model = torch.nn.ModuleDict({
        "encoder": encoder,
        "encoder_layer": encoder_layer,
        "y_predictor": y_predictor,
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_pred = encoder_layer(emb)
        y_pred = y_predictor(c_pred)

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred > 0.5)
            concept_accuracy = accuracy_score(c_train, c_pred.concept_probs > 0.5)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")


    print(c_pred[:5])
    with intervene_in_dict(model, {
        "encoder_layer": ConstantTensorIntervention(model, torch.zeros_like(c_train))("encoder_layer"),
    }):
        emb = model["encoder"](x_train)
        c_pred = model["encoder_layer"](emb)
        # y_pred = model["y_predictor"](c_pred)
        print(c_pred[:5])

    with intervene_in_dict(model, {
        "encoder_layer": ConstantLikeIntervention(model, fill=11.0)("encoder_layer"),
    }):
        emb = model["encoder"](x_train)
        c_pred = model["encoder_layer"](c_train)
        # y_pred = model["y_predictor"](c_pred)
        print(c_pred[:5])

    with intervene_in_dict(model, {
        "encoder_layer": DistributionIntervention(model, Normal(0.0, 1.0))("encoder_layer"),
    }):
        emb = model["encoder"](x_train)
        c_pred = model["encoder_layer"](c_train)
        # y_pred = model["y_predictor"](c_pred)
        print(c_pred[:5])

    return


if __name__ == "__main__":
    main()
