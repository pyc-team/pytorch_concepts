import torch
from sklearn.metrics import accuracy_score

from torch_concepts.data import ToyDataset
from torch_concepts.nn import Annotate
from torch_concepts.nn.functional import selection_eval, logic_rule_eval, logic_memory_reconstruction, logic_rule_explanations
import torch_concepts.nn.functional as CF


def main():
    latent_dims = 5
    concept_emb_size = 2*latent_dims
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, class_names = (data.data, data.concept_labels, data.target_labels,
                                                             data.concept_attr_names, data.task_attr_names)
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]
    intervention_indexes = torch.ones_like(c_train).bool()
    memory_size = 7
    # memory_states = ["positive", "negative", "irrelevant"]

    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, latent_dims), torch.nn.LeakyReLU())
    concept_emb_bottleneck = torch.nn.Sequential(
        torch.nn.Linear(latent_dims, n_concepts*concept_emb_size),
        torch.nn.Unflatten(-1, (n_concepts, concept_emb_size)),
        Annotate(concept_names, 1),
    )
    concept_score_bottleneck = torch.nn.Sequential(
        torch.nn.Linear(concept_emb_size, 1),
        torch.nn.Flatten(),
        Annotate(concept_names, 1),
    )
    classifier_selector = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(concept_emb_size//2*n_concepts, n_classes*memory_size),
        torch.nn.Unflatten(-1, (n_classes, memory_size)),
        Annotate(class_names, 1),
    )
    latent_concept_memory = torch.nn.Embedding(memory_size, latent_dims)
    concept_memory_decoder = torch.nn.Sequential(
        # the memory decoder maps to the concept space which has also bias
        torch.nn.Linear(latent_dims, (n_concepts+1)*n_classes),
        torch.nn.Unflatten(-1, (n_concepts+1, n_classes)),
        Annotate([concept_names+["BIAS"], class_names], [1, 2]),
    )
    model = torch.nn.Sequential(encoder, concept_emb_bottleneck, concept_score_bottleneck,
                                classifier_selector, latent_concept_memory, concept_memory_decoder)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_emb = concept_emb_bottleneck(emb)
        c_pred = concept_score_bottleneck(c_emb).sigmoid()
        c_intervened = CF.intervene(c_pred, c_train, intervention_indexes)
        c_mix = CF.concept_embedding_mixture(c_emb, c_intervened)
        classifier_selector_logits = classifier_selector(c_mix)
        prob_per_classifier = torch.softmax(classifier_selector_logits, dim=-1)
        memory_weights = concept_memory_decoder(latent_concept_memory.weight)
        # add batch dimension
        memory_weights = memory_weights.unsqueeze(dim=0)
        concept_weights = memory_weights[:, :, :n_concepts]
        bias = memory_weights[:, :, -1]

        y_per_classifier = CF.linear_equation_eval(concept_weights, c_pred, bias)
        y_pred = selection_eval(prob_per_classifier, y_per_classifier).sigmoid()

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

    explanations = CF.linear_equation_explanations(concept_weights, bias,
                                                   {1: concept_names,
                                            2: class_names})
    print(f"Learned rules: {explanations}")

    return


if __name__ == "__main__":
    main()
