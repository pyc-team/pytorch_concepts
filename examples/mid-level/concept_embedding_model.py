import torch
from sklearn.metrics import accuracy_score

from torch_concepts.data import CompletenessDataset
from torch_concepts.nn import ConceptEmbeddingBottleneck, concept_embedding_mixture


def main():
    latent_dims = 20
    concept_emb_size = 2*latent_dims
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5
    data = CompletenessDataset(n_samples=n_samples, n_hidden_concepts=20, n_concepts=4, n_tasks=2)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]

    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, latent_dims), torch.nn.LeakyReLU())
    bottleneck = ConceptEmbeddingBottleneck(in_features=latent_dims, out_concept_dimensions={1: concept_names, 2: concept_emb_size})
    y_predictor = torch.nn.Sequential(torch.nn.Flatten(),
                                      torch.nn.Linear(n_concepts * latent_dims, latent_dims),
                                      torch.nn.LeakyReLU(),
                                      torch.nn.Linear(latent_dims, n_classes),
                                      torch.nn.Sigmoid())
    model = torch.nn.Sequential(encoder, bottleneck, y_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
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
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_reg*concept_loss + task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.2f}")

    task_accuracy = accuracy_score(y_train, y_pred > 0.5)
    concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")
    print(f"Concept names: {bottleneck.concept_names}")
    print(f"Concepts: {c_pred}")

    return


if __name__ == "__main__":
    main()
