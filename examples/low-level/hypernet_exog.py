import torch
from sklearn.metrics import accuracy_score

from torch_concepts import Annotations, AxisAnnotation, ConceptTensor
from torch_concepts.data import ToyDataset
from torch_concepts.nn import ExogEncoder, ProbEncoder, HyperNetLinearPredictor


def main():
    latent_dims = 5
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
    )
    exog_encoder_c = ExogEncoder(latent_dims, c_annotations, embedding_size=5)
    exog_encoder_y = ExogEncoder(latent_dims, y_annotations, embedding_size=5)
    encoder_layer = ProbEncoder(exog_encoder_c.out_features, c_annotations, exogenous=True)
    y_predictor = HyperNetLinearPredictor((exog_encoder_y.out_features, encoder_layer.out_features), 
                                          y_annotations)
    model = torch.nn.Sequential(encoder, exog_encoder_c, exog_encoder_y, encoder_layer, y_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        exog_emb_c = exog_encoder_c(emb)
        exog_emb_y = exog_encoder_y(emb)

        c_pred = encoder_layer(exog_emb_c)

        y_pred = y_predictor(c_pred, exog_emb_y)

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.concept_probs > 0.5)
            concept_accuracy = accuracy_score(c_train, c_pred.concept_probs > 0.5)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    return


if __name__ == "__main__":
    main()
