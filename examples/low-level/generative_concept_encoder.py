import torch
from sklearn.metrics import accuracy_score

from torch_concepts.base import ConceptTensor
from torch_concepts.data import xor
from torch_concepts.nn import ConceptScorer, ConceptEncoder, GenerativeConceptEncoder
import torch_concepts.nn.functional as CF


def main():
    emb_size = 20
    n_epochs = 500
    n_samples = 1000
    x_train, c_train, y_train = xor(n_samples)
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]
    concept_names = [f"C{i}" for i in range(n_concepts)]

    concepts_train = ConceptTensor.concept(c_train, concept_names)
    intervention_indexes = ConceptTensor.concept(torch.ones_like(c_train).bool(), concept_names)

    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, emb_size), torch.nn.LeakyReLU())
    c_sampler = GenerativeConceptEncoder(emb_size, n_concepts, 2*emb_size, concept_names)
    c_scorer = ConceptScorer(2*emb_size, concept_names)
    y_predictor = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(emb_size*n_concepts, emb_size),
                                      torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, n_classes))
    model = torch.nn.Sequential(encoder, c_sampler, c_scorer, y_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_emb = c_sampler(emb)
        c_pred = c_scorer(c_emb)
        c_intervened = CF.intervene(c_pred, concepts_train, intervention_indexes)
        c_mix = CF.concept_embedding_mixture(c_emb, c_intervened)
        y_pred = y_predictor(c_mix)

        # compute loss
        concept_loss = loss_form(c_pred, c_train)
        task_loss = loss_form(y_pred, y_train)
        kl_loss = torch.distributions.kl_divergence(c_sampler.p_z, c_sampler.qz_x).mean()
        loss = concept_loss + 0.5 * task_loss + 1 * kl_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Task Loss {task_loss.item():.2f}, Concept Loss {concept_loss.item():.2f}, KL Loss {kl_loss.item():.2f}")

    task_accuracy = accuracy_score(y_train, y_pred > 0)
    concept_accuracy = accuracy_score(c_train, c_pred > 0)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")
    print(f"Concept names: {c_sampler.concept_names}")
    print(f"Concept 1 (by name): {c_pred.extract_by_concept_names(['C0'])[:5]}")
    print(f"Concept 2 (by name): {c_pred.extract_by_concept_names(['C1'])[:5]}")

    return


if __name__ == "__main__":
    main()
