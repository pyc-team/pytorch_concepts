import torch

from sklearn.metrics import f1_score


def completeness_score(
    y_true,
    y_pred_blackbox,
    y_pred_whitebox,
    scorer=f1_score,
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
        scorer (function): Scoring function to evaluate predictions.
            Default is f1_score.
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

    completeness = \
        (whitebox_score - random_accuracy) / (blackbox_score - random_accuracy)
    return completeness


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
