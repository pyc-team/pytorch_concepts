from collections import defaultdict
from typing import List, Union, Tuple

import torch

from torch_concepts.base import ConceptTensor


def intervene(c_pred: ConceptTensor, c_true: ConceptTensor, indexes: ConceptTensor) -> ConceptTensor:
    """
    Intervene on concept embeddings.

    Args:
        c_pred (ConceptTensor): Predicted concepts.
        c_true (ConceptTensor): Ground truth concepts.
        indexes (ConceptTensor): Boolean ConceptTensor indicating which concepts to intervene on.

    Returns:
        ConceptTensor: Intervened concept scores.
    """
    if c_true is None or indexes is None:
        return c_pred

    if c_true is not None and indexes is not None:
        if indexes.max() >= c_pred.shape[1]:
            raise ValueError("Intervention indices must be less than the number of concepts.")

    return ConceptTensor.concept(torch.where(indexes, c_true, c_pred), c_true.concept_names)


def concept_embedding_mixture(c_emb: ConceptTensor, c_scores: ConceptTensor):
    """
    Mixes concept embeddings and concept predictions.
    Main reference: `"Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_

    Args:
        c_emb (ConceptTensor): Concept embeddings with shape (batch_size, n_concepts, emb_size).
        c_scores (ConceptTensor): Concept scores with shape (batch_size, n_concepts).

    Returns:
        ConceptTensor: Mix of concept embeddings and concept scores with shape (batch_size, n_concepts, emb_size//2)
    """
    emb_size = c_emb[0].shape[1] // 2
    c_mix = c_scores.unsqueeze(-1) * c_emb[:, :, :emb_size] + (1 - c_scores.unsqueeze(-1)) * c_emb[:, :, emb_size:]
    return ConceptTensor.concept(c_mix, c_scores.concept_names)


def intervene_on_concept_graph(c_adj: ConceptTensor, indexes: List[Union[int, str]]) -> ConceptTensor:
    """
    Intervene on a ConceptTensor adjacency matrix by zeroing out specified concepts representing parent nodes.

    Args:
        c_adj: ConceptTensor adjacency matrix.
        indexes: List of concept names or indices to zero out.

    Returns:
        ConceptTensor: Intervened ConceptTensor adjacency matrix.
    """
    # Check if the tensor is a square matrix
    if c_adj.shape[0] != c_adj.shape[1]:
        raise ValueError("The ConceptTensor must be a square matrix (it represents an adjacency matrix).")

    # Get indices for concepts to zero out
    if isinstance(indexes[0], str):
        indices = [c_adj.concept_names.index(name) for name in indexes if name in c_adj.concept_names]
        if len(indices) != len(indexes):
            raise ValueError("Some concept names are not found in the tensor's concept names.")
    else:
        indices = indexes

    # Zero out specified columns
    concept_names = c_adj.concept_names
    c_adj = c_adj.clone()
    c_adj[:, indices] = 0

    return ConceptTensor.concept(c_adj, concept_names)


def selection_eval(selection_weights: torch.Tensor, *predictions: torch.Tensor) -> torch.Tensor:
    """
    Evaluate predictions as a weighted product based on selection weights.

    Args:
        selection_weights (Tensor): Selection weights with at least two dimensions (D1, ..., Dn).
        predictions (Tensor): Arbitrary number of prediction tensors, each with the same shape as selection_weights (D1, ..., Dn).

    Returns:
        Tensor: Weighted product sum with shape (D1, ...).
    """
    if len(predictions) == 0:
        raise ValueError("At least one prediction tensor must be provided.")

    product = selection_weights
    for pred in predictions:
        product = product * pred

    result = product.sum(dim=-1)

    return result


def linear_memory_eval(concept_weights: ConceptTensor, c_pred: ConceptTensor) -> ConceptTensor:
    """
    Use concept weights to make predictions.

    Args:
        concept_weights: parameters representing the weights of multiple linear models with shape (memory_size, n_concepts, n_classes, 1).
        c_pred: concept predictions with shape (batch_size, n_concepts).

    Returns:
        ConceptTensor: Predictions made by the linear models with shape (batch_size, n_classes, memory_size).
    """
    return torch.einsum('mcys,bc->bym', concept_weights, c_pred)


def logic_memory_eval(concept_weights: ConceptTensor, c_pred: ConceptTensor, memory_idxs: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use concept weights to make predictions based on logic rules.

    Args:
        concept_weights: concept weights with shape (memory_size, n_concepts, n_tasks, n_roles) with n_roles=3.
        c_pred: concept predictions with shape (batch_size, n_concepts).
        memory_idxs: Indices of rules to evaluate with shape (batch_size, n_tasks). Default is None (evaluate all).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of rule predictions with shape (batch_size, n_tasks, memory_size) and concept reconstructions with shape (batch_size, n_tasks, memory_size, n_concepts).
    """
    memory_size = concept_weights.size(0)
    n_tasks = concept_weights.size(2)
    pos_polarity, neg_polarity, irrelevance = concept_weights[..., 0], concept_weights[..., 1], concept_weights[..., 2]

    if memory_idxs is None:  # cast all to (batch_size, memory_size, n_concepts, n_tasks)
        x = c_pred.unsqueeze(1).unsqueeze(-1).expand(-1, memory_size, -1, n_tasks)
        pos_polarity = pos_polarity.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        neg_polarity = neg_polarity.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        irrelevance = irrelevance.unsqueeze(0).expand(x.size(0), -1, -1, -1)
    else:  # cast all to (batch_size, n_tasks, 1, n_concepts)
        x = c_pred.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, n_tasks)

    # TODO: incorporate t-norms?
    y_per_rule = (irrelevance + (1 - x) * neg_polarity + x * pos_polarity).prod(dim=2)  # batch_size, mem_size, n_tasks

    c_rec_per_classifier = 0.5 * irrelevance + pos_polarity  # batch_size, mem_size, n_tasks, n_concepts
    return y_per_rule.permute(0, 2, 1), c_rec_per_classifier


def logic_memory_reconstruction(c_rec_per_classifier: torch.Tensor, c_true: ConceptTensor, y_true: ConceptTensor) -> torch.Tensor:
    """
    Reconstruct tasks based on concept reconstructions, ground truth concepts and ground truth tasks.

    Args:
        c_rec_per_classifier: concept reconstructions with shape (batch_size, memory_size, n_tasks, n_concepts).
        c_true: concept ground truth with shape (batch_size, n_concepts).
        y_true: task ground truth with shape (batch_size, n_tasks).

    Returns:
        torch.Tensor: Reconstructed tasks with shape (batch_size, n_tasks, memory_size).
    """
    reconstruction_mask = torch.where(c_true[:, None, :, None] == 1, c_rec_per_classifier, 1 - c_rec_per_classifier)
    c_rec_per_classifier = reconstruction_mask.prod(dim=2).pow(y_true[:, :, None])
    return c_rec_per_classifier.permute(0, 2, 1)


def logic_memory_explanations(concept_logic_weights: ConceptTensor, concept_names: List[str], task_names: List[str]) -> dict:
    """
    Extracts rules from rule embeddings as strings.

    Args:
        concept_logic_weights: Rule embeddings with shape (memory_size, n_concepts, n_tasks, 3).
        concept_names: Concept names.
        task_names: Task names.
    Returns:
        Dict[str, Dict[str, str]]: Rules as strings.
    """
    rules_str = defaultdict(dict)  # task, memory_size
    memory_size = concept_logic_weights.size(0)
    n_concepts = concept_logic_weights.size(1)
    n_tasks = concept_logic_weights.size(2)
    concept_logic_probs = torch.softmax(concept_logic_weights, dim=-1)  # memory_size, n_concepts, n_tasks, 3
    concept_roles = torch.argmax(concept_logic_probs, dim=-1)  # memory_size, n_concepts, n_tasks
    for task_idx in range(n_tasks):
        for mem_idx in range(memory_size):
            rule = [("~ " if concept_roles[mem_idx, concept_idx, task_idx] == 1 else "") + concept_names[concept_idx]
                    for concept_idx in range(n_concepts)
                        if concept_roles[mem_idx, concept_idx, task_idx] != 2]
            rules_str[task_names[task_idx]][f"Rule {mem_idx}"] = " & ".join(rule)
    return dict(rules_str)


def selective_calibration(c_confidence: ConceptTensor, target_coverage: float) -> ConceptTensor:
    """
    Selects concepts based on confidence scores and target coverage.

    Args:
        c_confidence: Concept confidence scores.
        target_coverage: Target coverage.

    Returns:
        ConceptTensor: Thresholds to select confident predictions.
    """
    theta = torch.quantile(c_confidence, 1 - target_coverage, dim=0, keepdim=True)
    return ConceptTensor.concept(theta, c_confidence.concept_names)


def confidence_selection(c_confidence: ConceptTensor, theta: ConceptTensor) -> ConceptTensor:
    """
    Selects concepts with confidence above a selected threshold.

    Args:
        c_confidence: Concept confidence scores.
        theta: Threshold to select confident predictions.

    Returns:
        ConceptTensor: mask selecting confident predictions.
    """
    c_confident_mask = torch.where(c_confidence > theta, True, False)
    return ConceptTensor.concept(c_confident_mask, c_confidence.concept_names)
