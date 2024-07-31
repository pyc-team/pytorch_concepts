import torch
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from torch_concepts.base import ConceptTensor
from torch_concepts.data import CompletenessDataset
from torch_concepts.nn import ProbabilisticConceptEncoder, ConceptEncoder
from torch_concepts.nn import functional as F


def main():
    target_coverage = 0.5
    emb_size = 5
    n_epochs = 1000
    n_samples = 10000
    data = CompletenessDataset(n_samples=n_samples, n_features=100, n_concepts=4, n_tasks=2)
    x_train, c_train, y_train, concept_names, task_names = data.data, data.concept_labels, data.target_labels, data.concept_attr_names, data.task_attr_names
    n_features = x_train.shape[1]
    n_concepts = c_train.shape[1]
    n_classes = y_train.shape[1]

    encoder = torch.nn.Sequential(torch.nn.Linear(n_features, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU())
    prob_encoder = ProbabilisticConceptEncoder(in_features=emb_size, out_concept_dimensions={1: emb_size})
    c_scorer = ConceptEncoder(in_features=emb_size, out_concept_dimensions={1: concept_names})
    y_predictor = torch.nn.Sequential(torch.nn.Linear(n_concepts, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, n_classes))
    model = torch.nn.Sequential(encoder, prob_encoder, c_scorer, y_predictor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    y_loss = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # generate concept and task predictions
        emb = encoder(x_train)
        q_c_hidden = prob_encoder(emb)
        c_hidden = q_c_hidden.rsample()
        c_pred = c_scorer(c_hidden)
        y_pred = y_predictor(c_pred)

        # compute loss
        concept_loss = y_loss(c_pred, c_train)
        task_loss = y_loss(y_pred, y_train)
        kl_loss = torch.distributions.kl_divergence(prob_encoder.p_z.base_dist, q_c_hidden.base_dist).mean()
        loss = concept_loss + 0.5 * task_loss + 1 * kl_loss

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Task Loss {task_loss.item():.2f}, Concept Loss {concept_loss.item():.2f}, KL Loss {kl_loss.item():.2f}")

    c_abs_logits = ConceptTensor.concept(c_pred.abs().clone(), {1: concept_names})
    c_pred = ConceptTensor.concept(c_pred.sigmoid(), {1: concept_names})

    task_accuracy = accuracy_score(y_train.ravel(), y_pred.ravel() > 0)
    concept_accuracy = accuracy_score(c_train.ravel(), c_pred.ravel() > 0.5)
    print(f"Task accuracy: {task_accuracy:.2f}")
    print(f"Concept accuracy: {concept_accuracy:.2f}")

    # train a model to use as a proxy of a human oracle
    human_model = DecisionTreeClassifier(max_depth=8, random_state=42)
    human_model.fit(x_train, c_train)
    c_pred_human = torch.hstack([torch.FloatTensor(c[:, 1]).unsqueeze(-1) for c in human_model.predict_proba(x_train)])
    human_accuracy = accuracy_score(c_train.ravel(), c_pred_human.ravel() > 0.5)
    print(f"Human accuracy: {human_accuracy:.2f}")

    # find OOD samples
    # compute mahalanobis distance of training data to q_c_hidden.base_dist
    diff = c_hidden - q_c_hidden.base_dist.mean
    distance = torch.sqrt((diff ** 2) / (q_c_hidden.base_dist.scale ** 2))
    distance = ConceptTensor.concept(-distance.mean(dim=1).unsqueeze(1))
    threshold = F.selective_calibration(distance, target_coverage)
    confident_indist = F.confidence_selection(distance, threshold).squeeze()
    coverage_rates = confident_indist.float().mean(dim=0)
    print(f"Coverage rates: {coverage_rates}")

    # find confident predictions
    threshold = F.selective_calibration(c_abs_logits, target_coverage)
    confident_preds = F.confidence_selection(c_abs_logits, threshold)
    print(f"The thresholds for rejection are {threshold}")
    coverage_rates = confident_preds.float().mean(dim=0)
    print(f"Coverage rates: {coverage_rates}")

    # accuracy of confident samples
    c_preds_accepted = c_pred.ravel()[confident_preds.ravel()]
    c_train_accepted = c_train.ravel()[confident_preds.ravel()]
    concept_accuracy_accepted = accuracy_score(c_train_accepted, c_preds_accepted > 0.5)
    print(f"Concept accuracy of accepted samples (high concept probs): {concept_accuracy_accepted:.2f}")

    # accuracy of rejected samples
    c_preds_rejected = c_pred.ravel()[~confident_preds.ravel()]
    c_train_rejected = c_train.ravel()[~confident_preds.ravel()]
    concept_accuracy_rejected = accuracy_score(c_train_rejected, c_preds_rejected > 0.5)
    print(f"Concept accuracy of rejected samples (low concept probs): {concept_accuracy_rejected:.2f}")

    # accuracy of confident in distribution samples
    c_preds_accepted = c_pred[confident_indist].ravel()
    c_train_accepted = c_train[confident_indist].ravel()
    concept_accuracy_accepted = accuracy_score(c_train_accepted, c_preds_accepted > 0.5)
    print(f"Concept accuracy of accepted samples (IID): {concept_accuracy_accepted:.2f}")

    # accuracy of rejected OOD samples
    c_preds_rejected = c_pred[~confident_indist].ravel()
    c_train_rejected = c_train[~confident_indist].ravel()
    concept_accuracy_rejected = accuracy_score(c_train_rejected, c_preds_rejected > 0.5)
    print(f"Concept accuracy of rejected samples (OOD): {concept_accuracy_rejected:.2f}")

    # combine confident predictions and confident in distribution samples
    confident_samples = confident_preds & confident_indist.unsqueeze(1).repeat(1, n_concepts)
    coverage_rates = confident_samples.float().mean(dim=0)
    print(f"Coverage rates: {coverage_rates}")
    c_preds_accepted = c_pred[confident_samples].ravel()
    c_train_accepted = c_train[confident_samples].ravel()
    concept_accuracy_accepted = accuracy_score(c_train_accepted, c_preds_accepted > 0.5)
    print(f"Concept accuracy of accepted samples (high concept probs and IID): {concept_accuracy_accepted:.2f}")

    # reject low confidence predictions and OOD samples
    c_preds_rejected = c_pred[~confident_samples].ravel()
    c_train_rejected = c_train[~confident_samples].ravel()
    concept_accuracy_rejected = accuracy_score(c_train_rejected, c_preds_rejected > 0.5)
    print(f"Concept accuracy of rejected samples (low concept probs and OOD): {concept_accuracy_rejected:.2f}")

    return


if __name__ == "__main__":
    main()
