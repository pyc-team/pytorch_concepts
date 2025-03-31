import torch
from sklearn.metrics import accuracy_score

from torch_concepts.data import ToyDataset
from torch_concepts.nn import Annotate
import torch_concepts.nn.functional as CF
from torch_concepts.utils import get_most_common_expl


def main():
    latent_dims = 8
    concept_emb_size = 2*latent_dims
    n_epochs = 500
    n_samples = 1000
    concept_reg = 0.1
    data = ToyDataset('xor', size=n_samples, random_state=42)
    (x_train, c_train, y_train,
     concept_names, task_names) = (data.data, data.concept_labels,
                                   data.target_labels,
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
        torch.nn.Sigmoid(),
        Annotate(concept_names, 1),
    )
    # it is the module predicting the concept importance for each concept for all classes
    # its input is B x C x E, where B is the batch size, C is the number of concepts, and E is the embedding size
    # its output is B x C x T, where T is the number of tasks
    concept_importance_predictor = torch.nn.Sequential(
        torch.nn.Linear(concept_emb_size//2, concept_emb_size//2),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(concept_emb_size//2, n_classes),
        Annotate([concept_names, task_names], [1, 2])
    )
    # it is the module predicting the class bias for each class
    # its input is B x C x E, where B is the batch size, C is the number of concepts, and E is the embedding size
    # its output is B x T, where T is the number of tasks
    class_bias_predictor = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(n_concepts * concept_emb_size//2, concept_emb_size//2),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(concept_emb_size//2, n_classes),
        Annotate([task_names], 1)
    )

    model = torch.nn.Sequential(encoder, concept_emb_bottleneck, concept_score_bottleneck,
                                concept_importance_predictor, class_bias_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        c_emb = concept_emb_bottleneck(emb)
        c_pred = concept_score_bottleneck(c_emb)
        c_intervened = CF.intervene(c_pred, c_train, intervention_indexes)
        c_mix = CF.concept_embedding_mixture(c_emb, c_intervened)
        c_weights = concept_importance_predictor(c_mix)
        y_bias = class_bias_predictor(c_mix)
        # add memory size
        c_weights, y_bias = c_weights.unsqueeze(1), y_bias.unsqueeze(1)
        # remove memory size
        y_pred = CF.linear_equation_eval(c_weights, c_pred, y_bias)[:, :, 0].sigmoid()

        # compute loss
        concept_loss = loss_fn(c_pred, c_train)
        task_loss = loss_fn(y_pred, y_train)
        concept_norm = torch.norm(c_weights, p=1)
        bias_norm = torch.norm(y_bias, p=2)
        loss = (concept_loss + concept_reg * task_loss +
                1e-6 * concept_norm + 1e-4 * bias_norm)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.2f}")

    task_accuracy = accuracy_score(y_train, y_pred > 0.5)
    concept_accuracy = accuracy_score(c_train, c_pred > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")
    print(f"Concepts: {c_pred}")

    explanations = CF.linear_equation_explanations(c_weights, y_bias,
                                                   {1: concept_names,
                                                    2: task_names})

    print(f"Explanations: {explanations}")

    global_explanations = get_most_common_expl(explanations, y_pred)
    print(f"Global explanations: {global_explanations}")

    return


if __name__ == "__main__":
    main()
