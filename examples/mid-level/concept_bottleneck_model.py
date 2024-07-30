import torch
from sklearn.metrics import accuracy_score

from torch_concepts.data import CompletenessDataset
from torch_concepts.nn import ConceptBottleneck


def main():
    emb_size = 20
    n_epochs = 500
    n_samples = 1000
    data = CompletenessDataset(n_samples=n_samples, n_features=100, n_concepts=4, n_tasks=2)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]

    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, emb_size), torch.nn.LeakyReLU())
    bottleneck = ConceptBottleneck(emb_size, n_concepts, concept_names)
    y_predictor = torch.nn.Sequential(torch.nn.Linear(n_concepts, emb_size),
                                      torch.nn.LeakyReLU(),
                                      torch.nn.Linear(emb_size, n_classes),
                                      torch.nn.Sigmoid())
    model = torch.nn.Sequential(encoder, bottleneck, y_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form = torch.nn.BCELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        preds = bottleneck(emb)
        c_pred = preds['next']
        y_pred = y_predictor(c_pred)

        # compute loss
        concept_loss = loss_form(c_pred, c_train)
        task_loss = loss_form(y_pred, y_train)
        loss = concept_loss + 0.5 * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.2f}")

    task_accuracy = accuracy_score(y_train, y_pred > 0.5)
    concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")
    print(f"Concept names: {bottleneck.concept_names}")
    print(f"Concepts: {c_pred.describe()}")

    return


if __name__ == "__main__":
    main()
