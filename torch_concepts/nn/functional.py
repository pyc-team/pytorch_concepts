import torch

from collections import defaultdict
from typing import List, Union, Tuple, Dict

from torch_concepts.base import ConceptTensor

def _default_concept_names(shape: List[int]) -> Dict[int, List[str]]:
    concept_names = {}
    for dim in range(len(shape)):
        concept_names[dim+1] = [
            f"concept_{dim+1}_{i}" for i in range(shape[dim])
        ]
    return concept_names


def intervene(
    c_pred: torch.Tensor,
    c_true: torch.Tensor,
    indexes: torch.Tensor,
    concept_names: Dict[int, List[str]] = None,
) -> ConceptTensor:
    """
    Intervene on concept embeddings.

    Args:
        c_pred (Tensor): Predicted concepts.
        c_true (Tensor): Ground truth concepts.
        indexes (Tensor): Boolean ConceptTensor indicating which concepts to
            intervene on.
        concept_names (List[str]): Concept names.

    Returns:
        ConceptTensor: Intervened concept scores.
    """
    if c_pred.shape != c_true.shape:
        raise ValueError(
            "Predicted and true concepts must have the same shape."
        )

    if hasattr(c_pred, 'concept_names'):
        concept_names = c_pred.concept_names.copy()
    elif hasattr(c_true, 'concept_names'):
        concept_names = c_true.concept_names.copy()

    if concept_names is None:
        concept_names = _default_concept_names(c_pred.shape[1:])

    if c_true is None or indexes is None:
        return ConceptTensor.concept(c_pred, concept_names)

    if c_true is not None and indexes is not None:
        if indexes.max() >= c_pred.shape[1]:
            raise ValueError(
                "Intervention indices must be less than the number of concepts."
            )

    return ConceptTensor.concept(torch.where(indexes, c_true, c_pred), concept_names)


def concept_embedding_mixture(
    c_emb: torch.Tensor,
    c_scores: torch.Tensor,
    concept_names: Dict[int, List[str]] = None,
) -> ConceptTensor:
    """
    Mixes concept embeddings and concept predictions.
    Main reference: `"Concept Embedding Models: Beyond the
    Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_

    Args:
        c_emb (Tensor): Concept embeddings with shape (batch_size, n_concepts,
            emb_size).
        c_scores (Tensor): Concept scores with shape (batch_size, n_concepts).
        concept_names (List[str]): Concept names.

    Returns:
        ConceptTensor: Mix of concept embeddings and concept scores with shape
            (batch_size, n_concepts, emb_size//2)
    """
    if hasattr(c_emb, 'concept_names'):
        concept_names = c_emb.concept_names.copy()
        concept_names = {1: concept_names[1]}
    elif hasattr(c_scores, 'concept_names'):
        concept_names = c_scores.concept_names.copy()
        concept_names = {1: concept_names[1]}

    if concept_names is None:
        concept_names = _default_concept_names([c_emb.shape[1]])

    emb_size = c_emb[0].shape[1] // 2
    c_mix = (
        c_scores.unsqueeze(-1) * c_emb[:, :, :emb_size] +
        (1 - c_scores.unsqueeze(-1)) * c_emb[:, :, emb_size:]
    )
    return ConceptTensor.concept(c_mix, concept_names)


def intervene_on_concept_graph(
    c_adj: torch.Tensor,
    indexes: List[Union[int, str]],
    concept_names: Dict[int, List[str]] = None,
) -> ConceptTensor:
    """
    Intervene on a ConceptTensor adjacency matrix by zeroing out specified
    concepts representing parent nodes.

    Args:
        c_adj: torch.Tensor adjacency matrix.
        indexes: List of concept names or indices to zero out.
        concept_names: List of concept names.

    Returns:
        ConceptTensor: Intervened ConceptTensor adjacency matrix.
    """
    if hasattr(c_adj, 'concept_names'):
        concept_names = c_adj.concept_names.copy()

    if concept_names is None:
        concept_names = {}
        for dim in range(len(c_adj.shape)):
            concept_names[dim] = [
                f"concept_{dim}_{i}" for i in range(c_adj.shape[dim])
            ]

    # Check if the tensor is a square matrix
    if c_adj.shape[0] != c_adj.shape[1]:
        raise ValueError(
            "The ConceptTensor must be a square matrix (it represents an "
            "adjacency matrix)."
        )

    # Get indices for concepts to zero out
    if isinstance(indexes[0], str):
        indexes_from_str = [
            concept_names[1].index(name)
            for name in indexes if name in concept_names[1]
        ]
        if len(indexes_from_str) != len(indexes):
            raise ValueError(
                "Some concept names are not found in the tensor's concept "
                "names."
            )
    else:
        indexes_from_str = indexes

    # Zero out specified columns
    c_adj = c_adj.clone()
    c_adj[:, indexes_from_str] = 0

    return ConceptTensor.concept(c_adj, concept_names)


def selection_eval(
    selection_weights: torch.Tensor,
    *predictions: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate predictions as a weighted product based on selection weights.

    Args:
        selection_weights (Tensor): Selection weights with at least two
            dimensions (D1, ..., Dn).
        predictions (Tensor): Arbitrary number of prediction tensors, each with
            the same shape as selection_weights (D1, ..., Dn).

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


def linear_memory_eval(
    concept_weights: torch.Tensor,
    c_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Use concept weights to make predictions.

    Args:
        concept_weights: parameters representing the weights of multiple linear
            models with shape (memory_size, n_concepts, n_classes, 1).
        c_pred: concept predictions with shape (batch_size, n_concepts).

    Returns:
        Tensor: Predictions made by the linear models with shape (batch_size,
            n_classes, memory_size).
    """
    return torch.einsum('mcys,bc->bym', concept_weights, c_pred)


def logic_memory_eval(
    concept_weights: torch.Tensor,
    c_pred: torch.Tensor,
    memory_idxs: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use concept weights to make predictions based on logic rules.

    Args:
        concept_weights: concept weights with shape (memory_size, n_concepts,
            n_tasks, n_roles) with n_roles=3.
        c_pred: concept predictions with shape (batch_size, n_concepts).
        memory_idxs: Indices of rules to evaluate with shape (batch_size,
            n_tasks). Default is None (evaluate all).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of rule predictions with shape
            (batch_size, n_tasks, memory_size) and concept reconstructions with
            shape (batch_size, n_tasks, memory_size, n_concepts).
    """
    c_pred = (
        c_pred.to_standard_tensor()
        if isinstance(c_pred, ConceptTensor) else c_pred
    )
    concept_weights = (
        concept_weights.to_standard_tensor()
        if isinstance(concept_weights, ConceptTensor) else concept_weights
    )

    memory_size = concept_weights.size(0)
    n_tasks = concept_weights.size(2)
    pos_polarity, neg_polarity, irrelevance = (
        concept_weights[..., 0],
        concept_weights[..., 1],
        concept_weights[..., 2],
    )

    if memory_idxs is None:
        # cast all to (batch_size, memory_size, n_concepts, n_tasks)
        x = c_pred.unsqueeze(1).unsqueeze(-1).expand(
            -1,
            memory_size,
            -1,
            n_tasks,
        )
        pos_polarity = pos_polarity.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        neg_polarity = neg_polarity.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        irrelevance = irrelevance.unsqueeze(0).expand(x.size(0), -1, -1, -1)
    else:  # cast all to (batch_size, n_tasks, 1, n_concepts)
        x = c_pred.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, n_tasks)

    # TODO: incorporate t-norms?
    # batch_size, mem_size, n_tasks
    y_per_rule = (
        irrelevance + (1 - x) * neg_polarity + x * pos_polarity
    ).prod(dim=2)

    # batch_size, mem_size, n_tasks, n_concepts
    c_rec_per_classifier = 0.5 * irrelevance + pos_polarity
    return y_per_rule.permute(0, 2, 1), c_rec_per_classifier


def logic_memory_reconstruction(
    c_rec_per_classifier: torch.Tensor,
    c_true: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct tasks based on concept reconstructions, ground truth concepts
    and ground truth tasks.

    Args:
        c_rec_per_classifier: concept reconstructions with shape (batch_size,
            memory_size, n_tasks, n_concepts).
        c_true: concept ground truth with shape (batch_size, n_concepts).
        y_true: task ground truth with shape (batch_size, n_tasks).

    Returns:
        torch.Tensor: Reconstructed tasks with shape (batch_size, n_tasks,
            memory_size).
    """
    reconstruction_mask = torch.where(
        c_true[:, None, :, None] == 1,
        c_rec_per_classifier,
        1 - c_rec_per_classifier,
    )
    c_rec_per_classifier = reconstruction_mask.prod(dim=2).pow(
        y_true[:, :, None]
    )
    return c_rec_per_classifier.permute(0, 2, 1)


def logic_memory_explanations(
    concept_logic_weights: torch.Tensor,
    concept_names: Dict[int, List[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Extracts rules from rule embeddings as strings.

    Args:
        concept_logic_weights: Rule embeddings with shape (memory_size,
            n_concepts, n_tasks, 3).
        concept_names: Concept and task names.

    Returns:
        Dict[str, Dict[str, str]]: Rules as strings.
    """
    if len(concept_logic_weights.shape) != 4 or (
        concept_logic_weights.shape[3] != 3
    ):
        raise ValueError(
            "The concept logic weights must have 4 dimensions (memory_size, "
            "n_concepts, n_tasks, 3)."
        )

    if hasattr(concept_logic_weights, 'concept_names'):
        names = concept_logic_weights.concept_names.copy()
        c_names = names[1]
        t_names = names[2]

    else:
        names = _default_concept_names(concept_logic_weights.shape[1:3])
        if concept_names is None:
            c_names = names[1]
            t_names = names[2]
        else:
            c_names = concept_names[1]
            t_names = concept_names[2]

    rules_str = defaultdict(dict)  # task, memory_size
    memory_size = concept_logic_weights.size(0)
    n_concepts = concept_logic_weights.size(1)
    n_tasks = concept_logic_weights.size(2)
    # memory_size, n_concepts, n_tasks, 3
    concept_logic_probs = torch.softmax(concept_logic_weights, dim=-1)
    # memory_size, n_concepts, n_tasks
    concept_roles = torch.argmax(concept_logic_probs, dim=-1)
    for task_idx in range(n_tasks):
        for mem_idx in range(memory_size):
            rule = [
                (
                    "~ " if concept_roles[mem_idx, concept_idx, task_idx] == 1
                    else ""
                ) + c_names[concept_idx]
                for concept_idx in range(n_concepts)
                if concept_roles[mem_idx, concept_idx, task_idx] != 2
            ]
            rules_str[t_names[task_idx]][f"Rule {mem_idx}"] = " & ".join(rule)
    return dict(rules_str)


def selective_calibration(
    c_confidence: torch.Tensor,
    target_coverage: float,
    concept_names: Dict[int, List[str]] = None,
) -> ConceptTensor:
    """
    Selects concepts based on confidence scores and target coverage.

    Args:
        c_confidence: Concept confidence scores.
        target_coverage: Target coverage.
        concept_names: Concept names.

    Returns:
        ConceptTensor: Thresholds to select confident predictions.
    """
    if hasattr(c_confidence, 'concept_names'):
        concept_names = {1: c_confidence.concept_names.copy()[1]}
    if concept_names is None:
        concept_names = _default_concept_names([c_confidence.shape[1]])

    theta = torch.quantile(
        c_confidence, 1 - target_coverage,
        dim=0,
        keepdim=True,
    )
    return ConceptTensor.concept(theta, concept_names)


def confidence_selection(
    c_confidence: torch.Tensor,
    theta: torch.Tensor,
    concept_names: Dict[int, List[str]] = None,
) -> ConceptTensor:
    """
    Selects concepts with confidence above a selected threshold.

    Args:
        c_confidence: Concept confidence scores.
        theta: Threshold to select confident predictions.
        concept_names: Concept names.

    Returns:
        ConceptTensor: mask selecting confident predictions.
    """
    if hasattr(c_confidence, 'concept_names'):
        concept_names = c_confidence.concept_names.copy()
    if concept_names is None:
        concept_names = _default_concept_names([c_confidence.shape[1]])

    c_confident_mask = torch.where(c_confidence > theta, True, False)
    return ConceptTensor.concept(c_confident_mask, concept_names)
