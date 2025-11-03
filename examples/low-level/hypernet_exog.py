import torch
from sklearn.metrics import accuracy_score

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data import ToyDataset
from torch_concepts.nn import ExogEncoder, ProbEncoderFromEmb, HyperLinearPredictor


def main():
    latent_dims = 20
    n_epochs = 2000
    n_samples = 1000
    concept_reg = 0.5
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]

    c_annotations = Annotations({1: AxisAnnotation(concept_names)})
    y_annotations = Annotations({1: AxisAnnotation(task_names)})
    cy_annotations = c_annotations.join_union(y_annotations, axis=1)

    encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(latent_dims, latent_dims),
        torch.nn.LeakyReLU(),
    )
    encoder_layer = ProbEncoderFromEmb(in_features_embedding=latent_dims,
                                       out_annotations=c_annotations)
    exog_encoder = ExogEncoder(in_features_embedding=latent_dims,
                               out_annotations=y_annotations,
                               embedding_size=latent_dims)
    y_predictor = HyperLinearPredictor(in_features_logits=c_annotations.shape[1],
                                       in_features_exogenous=latent_dims,
                                       embedding_size=latent_dims,
                                       out_annotations=y_annotations)
    model = torch.nn.Sequential(encoder, exog_encoder, encoder_layer, y_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_pred = encoder_layer(embedding=emb)
        emb_rule = exog_encoder(embedding=emb)
        y_pred = y_predictor(logits=c_pred, exogenous=emb_rule)

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

    return


if __name__ == "__main__":
    main()
