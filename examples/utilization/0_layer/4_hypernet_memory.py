import torch
from sklearn.metrics import accuracy_score

from torch_concepts import seed_everything
from torch_concepts.data import ToyDataset
from torch_concepts.nn import (
    LinearLatentToConcept,
    HyperlinearConceptExogenousToConcept,
    SelectorLatentToExogenous,
)


def main():
    latent_dims = 30
    exog_dims = 30
    memory_size = 11
    n_epochs = 2000
    n_samples = 1000
    concept_reg = 0.5

    seed_everything(42)

    dataset = ToyDataset(dataset='xor', seed=42, n_gen=n_samples)
    x_train = dataset.input_data
    concept_idx = list(dataset.graph.edge_index[0].unique().numpy())
    task_idx = list(dataset.graph.edge_index[1].unique().numpy())
    c_train = dataset.concepts[:, concept_idx]
    y_train = dataset.concepts[:, task_idx]
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_tasks = y_train.shape[1]

    encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(latent_dims, latent_dims),
        torch.nn.LeakyReLU(),
    )
    concept_encoder = LinearLatentToConcept(
        in_latent=latent_dims,
        out_concepts=n_concepts
    )
    selector = SelectorLatentToExogenous(
        in_latent=latent_dims,
        memory_size=memory_size,
        out_exogenous=exog_dims,
        out_concepts=n_tasks
    )
    task_predictor = HyperlinearConceptExogenousToConcept(
        in_concepts=n_concepts,
        in_exogenous=exog_dims,
        hidden_size=latent_dims
    )
    model = torch.nn.Sequential(encoder, selector, concept_encoder, task_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        latent = encoder(x_train)
        c_pred = concept_encoder(latent=latent)
        exog = selector(latent=latent, sampling=False)
        exog = torch.nn.functional.leaky_relu(exog)
        y_pred = task_predictor(concepts=c_pred, exogenous=exog)

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(y_train, y_pred.detach() > 0.)
            concept_accuracy = accuracy_score(c_train, c_pred.detach() > 0.)

            exog_sampled = selector(latent=latent, sampling=True)
            exog_sampled = torch.nn.functional.leaky_relu(exog_sampled)
            y_pred_sampled = task_predictor(concepts=c_pred, exogenous=exog_sampled)

            task_accuracy_sampling = accuracy_score(y_train, y_pred_sampled.detach() > 0.)
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f} | Task Acc w/ Sampling: {task_accuracy_sampling:.2f}")

    return


if __name__ == "__main__":
    main()
