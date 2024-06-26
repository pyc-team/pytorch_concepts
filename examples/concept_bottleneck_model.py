import torch
from sklearn.metrics import accuracy_score

from torch_concepts.data import xor
from torch_concepts.nn import ConceptLinear, MLPReasoner


def main():
    emb_size = 5
    n_epochs = 500
    n_samples = 1000
    x_train, c_train, y_train = xor(n_samples)
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]

    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, emb_size),
        torch.nn.LeakyReLU(),
        ConceptLinear(emb_size, n_concepts),
        MLPReasoner(n_concepts, n_classes, emb_size, 2),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form_c = torch.nn.BCELoss()
    loss_form_y = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        preds = model(x_train)
        y_pred = preds["y_pred"]
        c_pred = preds["c_pred"]

        # compute loss
        concept_loss = loss_form_c(c_pred, c_train)
        task_loss = loss_form_y(y_pred, y_train)
        loss = concept_loss + 0.5 * task_loss

        loss.backward()
        optimizer.step()

    task_accuracy = accuracy_score(y_train, y_pred > 0)
    concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")

    return


if __name__ == "__main__":
    main()
