import torch
from sklearn.metrics import accuracy_score

from torch_concepts.data import ToyDataset
from torch_concepts.nn import ConceptEncoder, ProbabilisticConceptEncoder
import torch_concepts.nn.functional as CF


def main():
    latent_dims = 20
    concept_emb_size = 2*latent_dims
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5
    kl_reg = 1
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]

    intervention_indexes = torch.ones_like(c_train).bool()

    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, latent_dims), torch.nn.LeakyReLU())
    c_encoder = ProbabilisticConceptEncoder(in_features=latent_dims, out_concept_dimensions={1: concept_names, 2: concept_emb_size})
    c_scorer = ConceptEncoder(in_features=concept_emb_size, out_concept_dimensions={1: concept_names}, reduce_dim=2)
    y_predictor = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(latent_dims*n_concepts, latent_dims),
                                      torch.nn.LeakyReLU(), torch.nn.Linear(latent_dims, n_classes))
    model = torch.nn.Sequential(encoder, c_encoder, c_scorer, y_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        q_c_emb = c_encoder(emb)
        c_emb = q_c_emb.rsample()
        c_pred = c_scorer(c_emb)
        c_intervened = CF.intervene(c_pred, c_train, intervention_indexes)
        c_mix = CF.concept_embedding_mixture(c_emb, c_intervened)
        y_pred = y_predictor(c_mix)

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        kl_loss = torch.distributions.kl_divergence(c_encoder.p_z.base_dist, q_c_emb.base_dist).mean()
        loss = concept_loss + concept_reg * task_loss + kl_reg * kl_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Task Loss {task_loss.item():.2f}, Concept Loss {concept_loss.item():.2f}, KL Loss {kl_loss.item():.2f}")

    task_accuracy = accuracy_score(y_train, y_pred > 0)
    concept_accuracy = accuracy_score(c_train, c_pred > 0)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")
    print(f"Concept names: {c_encoder.concept_names}")
    print(f"Concepts: {c_pred}")

    return


if __name__ == "__main__":
    main()
