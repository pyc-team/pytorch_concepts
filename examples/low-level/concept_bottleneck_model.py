import torch
from sklearn.metrics import accuracy_score

from torch_concepts.data import ToyDataset
from torch_concepts.nn import ConceptEncoder


def main():
    emb_size = 5
    n_epochs = 500
    n_samples = 1000
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]

    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, emb_size), torch.nn.LeakyReLU())
    c_scorer = ConceptEncoder(in_features=emb_size, out_concept_dimensions={1: concept_names})
    y_predictor = torch.nn.Sequential(torch.nn.Linear(n_concepts, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, n_classes))
    model = torch.nn.Sequential(encoder, c_scorer, y_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_pred = c_scorer(emb)
        y_pred = y_predictor(c_pred)

        # compute loss
        concept_loss = loss_form(c_pred, c_train)
        task_loss = loss_form(y_pred, y_train)
        loss = concept_loss + 0.5 * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.2f}")

    task_accuracy = accuracy_score(y_train, y_pred > 0)
    concept_accuracy = accuracy_score(c_train, c_pred > 0)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")
    print(f"Concept names: {c_scorer.concept_names}")
    print(f"Concept 1 (by name): {c_pred.extract_by_concept_names({1: ['C1']})[:5]}")
    print(f"Concept 2 (by name): {c_pred.extract_by_concept_names({1: ['C2']})[:5]}")
    print(f"Concepts (by name): {c_pred.describe()}")

    return


if __name__ == "__main__":
    main()
