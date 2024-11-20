import torch
from sklearn.metrics import accuracy_score

from torch_concepts.data import ToyDataset
from torch_concepts.nn import ConceptLayer, ConceptMemory
from torch_concepts.nn.functional import selection_eval, linear_memory_eval


def main():
    latent_dims = 5
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]
    memory_size = 7
    memory_concept_states = 3

    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, latent_dims), torch.nn.LeakyReLU())
    c_encoder = ConceptLayer(in_features=latent_dims, out_concept_dimensions={1: n_concepts})
    classifier_selector = ConceptLayer(in_features=latent_dims, out_concept_dimensions={1: n_classes, 2: memory_size})
    concept_memory = ConceptMemory(memory_size=memory_size, emb_size=latent_dims,
                                   out_concept_dimensions={1: n_concepts, 2: n_classes, 3: memory_concept_states})
    model = torch.nn.Sequential(encoder, c_encoder, classifier_selector, concept_memory)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_pred = c_encoder(emb).sigmoid()
        classifier_selector_logits = classifier_selector(emb)
        prob_per_classifier = torch.softmax(classifier_selector_logits, dim=-1)
        concept_weights = concept_memory()
        y_per_classifier = linear_memory_eval(concept_weights, c_pred).sigmoid()
        y_pred = selection_eval(prob_per_classifier, y_per_classifier)
        
        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.2f}")

    task_accuracy = accuracy_score(y_train, y_pred > 0.5)
    concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")
    print(f"Concept names: {c_encoder.concept_names}")

    return


if __name__ == "__main__":
    main()
