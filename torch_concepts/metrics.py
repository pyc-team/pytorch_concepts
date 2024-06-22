import torch
from sklearn.metrics import f1_score


def completeness_score(y_true, y_pred_blackbox, y_pred_whitebox, scorer=f1_score, average='macro'):
    """
    Calculate the completeness score for the given predictions and true labels.
    Main reference: `"On Completeness-aware Concept-Based Explanations in Deep Neural Networks" <https://arxiv.org/abs/1910.07969>`_

    Parameters:
        y_true (torch.Tensor): True labels.
        y_pred_blackbox (torch.Tensor): Predictions from the blackbox model.
        y_pred_whitebox (torch.Tensor): Predictions from the whitebox model.
        scorer (function): Scoring function to evaluate predictions. Default is f1_score.
        average (str): Type of averaging to use. Default is 'macro'.

    Returns:
        float: Completeness score.
    """
    # Convert to numpy for sklearn metrics
    y_true_np = y_true.cpu().numpy()
    y_pred_blackbox_np = y_pred_blackbox.cpu().numpy()
    y_pred_whitebox_np = y_pred_whitebox.cpu().numpy()

    # Compute class frequencies in y_true
    class_frequencies = torch.bincount(y_true) / len(y_true)
    random_accuracy = torch.sum(class_frequencies ** 2).item()

    # Compute accuracy or other score using scorer
    blackbox_score = scorer(y_true_np, y_pred_blackbox_np, average=average)
    whitebox_score = scorer(y_true_np, y_pred_whitebox_np, average=average)

    completeness = (whitebox_score - random_accuracy) / (blackbox_score - random_accuracy)
    return completeness
