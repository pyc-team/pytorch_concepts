import torch

from sklearn.metrics import roc_auc_score
from typing import Callable, List, Union


def completeness_score(
    y_true,
    y_pred_blackbox,
    y_pred_whitebox,
    scorer=roc_auc_score,
    average='macro',
):
    """
    Calculate the completeness score for the given predictions and true labels.
    Main reference: `"On Completeness-aware Concept-Based Explanations in
    Deep Neural Networks" <https://arxiv.org/abs/1910.07969>`_

    Parameters:
        y_true (torch.Tensor): True labels.
        y_pred_blackbox (torch.Tensor): Predictions from the blackbox model.
        y_pred_whitebox (torch.Tensor): Predictions from the whitebox model.
        scorer (function): Scoring function to evaluate predictions. Default is
            roc_auc_score.
        average (str): Type of averaging to use. Default is 'macro'.

    Returns:
        float: Completeness score.
    """
    # Convert to numpy for sklearn metrics
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_blackbox_np = y_pred_blackbox.cpu().detach().numpy()
    y_pred_whitebox_np = y_pred_whitebox.cpu().detach().numpy()

    # Compute accuracy or other score using scorer
    blackbox_score = scorer(y_true_np, y_pred_blackbox_np, average=average)
    whitebox_score = scorer(y_true_np, y_pred_whitebox_np, average=average)

    return (whitebox_score) / (blackbox_score + 1e-10)


def intervention_score(
    y_predictor: torch.nn.Module,
    c_pred: torch.Tensor,
    c_true: torch.Tensor,
    y_true: torch.Tensor,
    intervention_groups: List[List[int]],
    activation: Callable = torch.sigmoid,
    scorer: Callable = roc_auc_score,
    average: str = 'macro',
    auc: bool = True,
) -> Union[float, List[float]]:
    """
    Compute the effect of concept interventions on downstream task predictions.

    Given  set of intervention groups, the intervention score measures the
    effectiveness of each intervention group on the model's task predictions.

    Main reference: `"Concept Bottleneck
    Models" <https://arxiv.org/abs/2007.04612>`_

    Parameters:
        y_predictor (torch.nn.Module): Model that predicts downstream task
            abels.
        c_pred (torch.Tensor): Predicted concept values.
        c_true (torch.Tensor): Ground truth concept values.
        y_true (torch.Tensor): Ground truth task labels.
        intervention_groups (List[List[int]]): List of intervention groups.
        activation (Callable): Activation function to apply to the model's
            predictions. Default is torch.sigmoid.
        scorer (Callable): Scoring function to evaluate predictions. Default is
            roc_auc_score.
        average (str): Type of averaging to use. Default is 'macro'.
        auc (bool): Whether to return the average score across all intervention
            groups. Default is True.

    Returns:
        Union[float, List[float]]: The intervention effectiveness for each
            intervention group or the average score across all groups.
    """
    # Convert to numpy for sklearn metrics
    y_true_np = y_true.cpu().detach().numpy()

    # Re-compute the model's predictions for each intervention group
    intervention_effectiveness = []
    for group in intervention_groups:
        # Intervene on the concept values
        c_pred_group = c_pred.clone()
        c_pred_group[:, group] = c_true[:, group]

        # Compute the new model's predictions
        y_pred_group = activation(y_predictor(c_pred_group))

        # Compute the new model's task performance
        intervention_effectiveness.append(scorer(
            y_true_np,
            y_pred_group.cpu().detach().numpy(),
            average=average,
        ))

    # Compute the area under the curve of the intervention curve
    if auc:
        intervention_effectiveness = (
            sum(intervention_effectiveness) / len(intervention_groups)
        )
    return intervention_effectiveness


def cace_score(y_pred_c0, y_pred_c1):
    """
    Compute the Average Causal Effect (ACE) also known as the Causal Concept
    Effect (CaCE) score.

    The ACE/CaCE score measures the causal effect of a concept on the
    predictions of a model. It is computed as the absolute difference between
    the expected predictions when the concept is inactive (c0) and active (c1).

    Main reference: `"Explaining Classifiers with Causal Concept Effect
    (CaCE)" <https://arxiv.org/abs/1907.07165>`_

    Parameters:
        y_pred_c0 (torch.Tensor): Predictions of the model when the concept is
            inactive. Shape: (batch_size, num_classes).
        y_pred_c1 (torch.Tensor): Predictions of the model when the concept is
            active. Shape: (batch_size, num_classes).

    Returns:
        torch.Tensor: The ACE/CaCE score for each class. Shape: (num_classes,).
    """
    if y_pred_c0.shape != y_pred_c1.shape:
        raise RuntimeError(
            "The shapes of y_pred_c0 and y_pred_c1 must be the same but got "
            f"{y_pred_c0.shape} and {y_pred_c1.shape} instead."
        )
    return y_pred_c1.mean(dim=0) - y_pred_c0.mean(dim=0)
