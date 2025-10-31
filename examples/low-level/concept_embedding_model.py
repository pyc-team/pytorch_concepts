import torch
from sklearn.metrics import accuracy_score

from torch_concepts import Annotations, AxisAnnotation
from torch_concepts.data import ToyDataset
from torch_concepts.nn import MixProbExogPredictor, ExogEncoder, ProbEncoderFromExog


def main():
    latent_dims = 10
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5
    embedding_size = 7
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
    exog_encoder = ExogEncoder(in_features_embedding=latent_dims,
                               out_annotations=c_annotations,
                               embedding_size=embedding_size)
    c_encoder = ProbEncoderFromExog(in_features_exogenous=embedding_size,
                                    out_annotations=c_annotations)
    y_predictor = MixProbExogPredictor(in_features_logits=c_annotations.shape[1],
                                       in_features_exogenous=embedding_size,
                                       out_annotations=y_annotations)
    model = torch.nn.Sequential(encoder, exog_encoder, c_encoder, y_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        exog = exog_encoder(embedding=emb)
        c_pred = c_encoder(exogenous=exog)
        y_pred = y_predictor(logits=c_pred, exogenous=exog)

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
