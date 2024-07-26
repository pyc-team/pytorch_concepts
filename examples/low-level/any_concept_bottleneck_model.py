import torch
from sklearn.metrics import accuracy_score

from torch_concepts.data import ToyDataset
from torch_concepts.nn import ConceptEncoder


def main():
    emb_size = 20
    n_epochs = 500
    n_samples = 1000
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]

    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, emb_size), torch.nn.LeakyReLU(),
                                  torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU())
    y_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size, n_classes))
    black_box = torch.nn.Sequential(encoder, y_predictor)

    optimizer = torch.optim.AdamW(black_box.parameters(), lr=0.01)
    task_loss = torch.nn.BCEWithLogitsLoss()
    black_box.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate task predictions
        emb = encoder(x_train)
        y_pred = y_predictor(emb)

        # compute loss
        loss = task_loss(y_pred, y_train)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.2f}")

    task_accuracy = accuracy_score(y_train, y_pred > 0)
    print(f"Task accuracy: {task_accuracy:.2f}")

    # once the model is trained, we create an autoencoder which maps
    # black-box embeddings to concepts and back
    concept_encoder = torch.nn.Sequential(
        torch.nn.Linear(emb_size, emb_size),
        torch.nn.LeakyReLU(),
        ConceptEncoder(emb_size, n_concepts, concept_names=concept_names)
    )
    concept_decoder = torch.nn.Sequential(
        torch.nn.Linear(n_concepts, emb_size),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(emb_size, emb_size),
        torch.nn.LeakyReLU(),
    )
    concept_autoencoder = torch.nn.Sequential(concept_encoder, concept_decoder)
    optimizer = torch.optim.AdamW(concept_autoencoder.parameters(), lr=0.01)
    concept_loss = torch.nn.BCEWithLogitsLoss()
    reconstruction_loss = torch.nn.MSELoss()
    concept_autoencoder.train()
    black_box.eval() # we can freeze the black-box model!
    for epoch in range(3000):
        optimizer.zero_grad()

        # generate concept predictions
        emb = encoder(x_train)
        c_pred = concept_encoder(emb)
        emb_pred = concept_decoder(c_pred)
        y_pred = y_predictor(emb_pred)

        # compute loss
        concept_loss_value = concept_loss(c_pred, c_train)
        reconstruction_loss_value = reconstruction_loss(emb_pred, emb)
        task_loss_value = task_loss(y_pred, y_train)
        loss = concept_loss_value + reconstruction_loss_value + 0.01*task_loss_value

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.2f} "
                  f"(concept {concept_loss_value.item():.2f}, "
                  f"task {task_loss_value.item():.2f}, "
                  f"rec. {reconstruction_loss_value.item():.2f})")

    task_accuracy = accuracy_score(y_train, y_pred > 0)
    concept_accuracy = accuracy_score(c_train, c_pred > 0)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")

    return


if __name__ == "__main__":
    main()
