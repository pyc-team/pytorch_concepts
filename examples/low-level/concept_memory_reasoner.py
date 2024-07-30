import torch
from sklearn.metrics import accuracy_score

from torch_concepts.base import ConceptTensor
from torch_concepts.data import ToyDataset
from torch_concepts.nn import ConceptEncoder
from torch_concepts.nn.base import LogicMemory
from torch_concepts.nn.functional import selection_eval


def main():
    emb_size = 5
    n_epochs = 500
    n_samples = 1000
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]
    n_rules = 5

    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, emb_size), torch.nn.LeakyReLU())
    c_encoder = ConceptEncoder(in_features=emb_size, n_concepts=n_concepts, concept_names=concept_names)
    rule_selector_logits = ConceptEncoder(in_features=emb_size, n_concepts=n_classes*n_rules, emb_size=1)
    rulebook = LogicMemory(n_concepts=n_concepts, n_tasks=n_classes, memory_size=n_rules, rule_emb_size=emb_size, concept_names=concept_names)

    model = torch.nn.Sequential(encoder, c_encoder, rule_selector_logits, rulebook)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_form = torch.nn.BCELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_pred = ConceptTensor.concept(torch.sigmoid(c_encoder(emb)), concept_names=concept_names)
        prob_per_rule = torch.softmax(rule_selector_logits(emb).view(-1, n_classes, n_rules), dim=-1)
        y_per_rule, c_rec_per_rule = rulebook.forward(c_pred)
        c_rec_per_rule = (torch.where(c_train[:, None, None, :] == 1, c_rec_per_rule, 1-c_rec_per_rule)
                            .prod(dim=-1)).pow(y_train[:, :, None])
        y_pred = selection_eval(prob_per_rule, y_per_rule, c_rec_per_rule)

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
    print(f"Concept names: {c_encoder.concept_names}")
    print(f"Concept 1 (by name): {c_pred.extract_by_concept_names(['C1'])[:5]}")
    print(f"Concept 2 (by name): {c_pred.extract_by_concept_names(['C2'])[:5]}")
    print(f"Concepts (by name): {c_pred.extract_by_concept_names(['C1', 'C2'])[:5]}")
    print(f"Learned rules:", rulebook.extract_rules())

    return


if __name__ == "__main__":
    main()
