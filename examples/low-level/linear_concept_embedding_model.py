import torch
from sklearn.metrics import accuracy_score

from torch_concepts.data import ToyDataset
from torch_concepts.nn import Annotate
import torch_concepts.nn.functional as CF


def main():
    latent_dims = 6
    concept_emb_size = 2*latent_dims
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.5
    data = ToyDataset('xor', size=n_samples, random_state=42)
    x_train, c_train, y_train, concept_names, task_names = (data.data, data.concept_labels, data.target_labels,
                                                            data.concept_attr_names, data.task_attr_names)
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]

    intervention_indexes = torch.ones_like(c_train).bool()

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
    # it is the module predicting the concept importance for each concept for all classes
    # its input is B x C x E, where B is the batch size, C is the number of concepts, and E is the embedding size
    # its output is B x C x T, where T is the number of tasks
    concept_importance_predictor = torch.nn.Sequential(
        torch.nn.Linear(concept_emb_size//2, n_classes),
        Annotate([concept_names, task_names], [1, 2])
    )
    # it is the module predicting the class bias for each class
    # its input is B x C x E, where B is the batch size, C is the number of concepts, and E is the embedding size
    # its output is B x T, where T is the number of tasks
    class_bias_predictor = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(n_concepts * concept_emb_size//2, n_classes),
        Annotate([task_names], 1)
    )

    model = torch.nn.Sequential(encoder, concept_emb_bottleneck, concept_score_bottleneck,
                                concept_importance_predictor, class_bias_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_emb = concept_emb_bottleneck(emb)
        c_pred = concept_score_bottleneck(c_emb)
        c_intervened = CF.intervene(c_pred, c_train, intervention_indexes)
        c_mix = CF.concept_embedding_mixture(c_emb, c_intervened)
        c_imp = concept_importance_predictor(c_mix)
        y_bias = class_bias_predictor(c_mix)
        y_pred = CF.linear_equation_eval(c_imp, c_pred, y_bias)

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        loss = concept_loss + concept_reg * task_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.2f}")

    task_accuracy = accuracy_score(y_train, y_pred > 0)
    concept_accuracy = accuracy_score(c_train, c_pred > 0)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")
    print(f"Concepts: {c_pred}")

    return


if __name__ == "__main__":
    main()
