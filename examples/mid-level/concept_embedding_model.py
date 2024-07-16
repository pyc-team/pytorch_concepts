import torch
from sklearn.metrics import accuracy_score

from torch_concepts.data import xor
from torch_concepts.nn import MixConceptEmbeddingBottleneck


def main():
    emb_size = 6
    n_epochs = 500
    n_samples = 1000
    x_train, c_train, y_train = xor(n_samples)
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]
    concept_names = ["C1", "C2"]

    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, emb_size), torch.nn.LeakyReLU())
    bottleneck = MixConceptEmbeddingBottleneck(emb_size, n_concepts, emb_size, concept_names)
    y_predictor = torch.nn.Sequential(torch.nn.Flatten(),
                                      torch.nn.Linear(n_concepts * emb_size, emb_size),
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
        c_next = preds['next']
        c_pred = preds['c_pred']
        y_pred = y_predictor(c_next)

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
