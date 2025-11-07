# To run interventions on SCBM, make sure to instal torchmin from https://github.com/rfeinman/pytorch-minimize.git
# Add the project root to PYTHONPATH sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_concepts.data import CompletenessDataset
from torch_concepts.nn import StochasticConceptBottleneck
from torch.distributions import RelaxedBernoulli
from torch_concepts.utils import compute_temperature


def main():
    latent_dims = 20
    n_epochs = 500
    n_samples = 1000
    concept_reg = 1.0
    cov_reg = 1.0
    num_monte_carlo = 100
    level = 0.99
    data = CompletenessDataset(n_samples=n_samples, n_concepts=4, n_tasks=2)
    x_train, c_train, y_train, concept_names, task_names = (
        data.data,
        data.concept_labels,
        data.target_labels,
        data.concept_attr_names,
        data.task_attr_names,
    )
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]

    encoder = torch.nn.Sequential(
        torch.nn.Linear(n_features, latent_dims), torch.nn.LeakyReLU()
    )

    bottleneck = StochasticConceptBottleneck(
        latent_dims, concept_names, num_monte_carlo=num_monte_carlo, level=level
    )
    y_predictor = torch.nn.Sequential(
        torch.nn.Linear(n_concepts, latent_dims),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(latent_dims, n_classes),
        torch.nn.Sigmoid(),
    )
    model = torch.nn.Sequential(encoder, bottleneck, y_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_pred, _ = bottleneck(emb)
        c_pred_av = c_pred.mean(-1)
        # Hard MC concepts
        temp = compute_temperature(epoch, n_epochs).to(c_pred.device)
        c_pred_relaxed = RelaxedBernoulli(temp, probs=c_pred).rsample()
        c_pred_hard = (c_pred_relaxed > 0.5).int()
        c_pred_hard = c_pred_hard - c_pred_relaxed.detach() + c_pred_relaxed
        y_pred = 0
        for i in range(num_monte_carlo):
            c_i = c_pred_hard[:, :, i]
            y_pred += y_predictor(c_i)
        y_pred /= num_monte_carlo

        # MC concept loss
        bce_loss = F.binary_cross_entropy(
            c_pred, c_train.unsqueeze(-1).expand_as(c_pred).float(), reduction="none"
        )  # [B,C,MCMC]
        intermediate_concepts_loss = -torch.sum(bce_loss, dim=1)  # [B,MCMC]
        mcmc_loss = -torch.logsumexp(
            intermediate_concepts_loss, dim=1
        )  # [B], logsumexp for numerical stability due to shift invariance
        concept_loss = torch.mean(mcmc_loss)
        # Regularization loss
        c_triang_cov = bottleneck.predict_sigma(emb)
        c_triang_inv = torch.inverse(c_triang_cov)
        prec_matrix = torch.matmul(
            torch.transpose(c_triang_inv, dim0=1, dim1=2), c_triang_inv
        )
        prec_loss = prec_matrix.abs().sum(dim=(1, 2)) - prec_matrix.diagonal(
            offset=0, dim1=1, dim2=2
        ).abs().sum(-1)

        if prec_matrix.size(1) > 1:
            prec_loss = prec_loss / (prec_matrix.size(1) * (prec_matrix.size(1) - 1))
        cov_loss = prec_loss.mean(-1)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_reg * concept_loss + task_loss + cov_reg * cov_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.2f}")

    task_accuracy = accuracy_score(y_train, y_pred > 0.5)
    concept_accuracy = accuracy_score(c_train, c_pred_av > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")
    print(f"Concepts: {c_pred_av}")

    return


if __name__ == "__main__":
    main()
