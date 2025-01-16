import numpy as np
import torch

from collections import defaultdict, Counter
from typing import List, Union, Tuple, Dict

from torch_concepts.semantic import CMRSemantic, Semantic, ProductTNorm


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
) -> torch.Tensor:
    """
    Intervene on concept embeddings.

    Args:
        c_pred (Tensor): Predicted concepts.
        c_true (Tensor): Ground truth concepts.
        indexes (Tensor): Boolean Tensor indicating which concepts to intervene
            on.

    Returns:
        Tensor: Intervened concepts.
    """
    if c_pred.shape != c_true.shape:
        raise ValueError(
            "Predicted and true concepts must have the same shape."
        )

    if c_true is None or indexes is None:
        return c_pred

    if c_true is not None and indexes is not None:
        if indexes.max() >= c_pred.shape[1]:
            raise ValueError(
                "Intervention indices must be less than the number of concepts."
            )

    return torch.where(indexes, c_true, c_pred)


def concept_embedding_mixture(
    c_emb: torch.Tensor,
    c_scores: torch.Tensor,
) -> torch.Tensor:
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
        Tensor: Mix of concept embeddings and concept scores with shape
            (batch_size, n_concepts, emb_size//2)
    """
    emb_size = c_emb[0].shape[1] // 2
    c_mix = (
        c_scores.unsqueeze(-1) * c_emb[:, :, :emb_size] +
        (1 - c_scores.unsqueeze(-1)) * c_emb[:, :, emb_size:]
    )
    return c_mix


def intervene_on_concept_graph(
    c_adj: torch.Tensor,
    indexes: List[int],
) -> torch.Tensor:
    """
    Intervene on a Tensor adjacency matrix by zeroing out specified
    concepts representing parent nodes.

    Args:
        c_adj: torch.Tensor adjacency matrix.
        indexes: List of indices to zero out.

    Returns:
        Tensor: Intervened Tensor adjacency matrix.
    """
    # Check if the tensor is a square matrix
    if c_adj.shape[0] != c_adj.shape[1]:
        raise ValueError(
            "The Tensor must be a square matrix (it represents an "
            "adjacency matrix)."
        )

    # Zero out specified columns
    c_adj = c_adj.clone()
    c_adj[:, indexes] = 0

    return c_adj


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


def linear_equation_eval(
    concept_weights: torch.Tensor,
    c_pred: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    Function to evaluate a set of linear equations with concept predictions.
    In this case we have one equation (concept_weights) for each sample in the batch.

    Args:
        concept_weights: Parameters representing the weights of multiple
            linear models with shape (batch_size, n_concepts, n_classes).
        c_pred: Concept predictions with shape (batch_size, n_concepts).
        bias: Bias term to add to the linear models.

    Returns:
        Tensor: Predictions made by the linear models with shape (batch_size,
            n_classes).
    """
    assert concept_weights.shape[1] == c_pred.shape[1]
    assert bias.shape[1] == concept_weights.shape[2]
    y_pred = torch.einsum('bcy,bc->by', concept_weights, c_pred)
    if bias is not None:
        y_pred += bias
    return y_pred


def linear_memory_eval(
        concept_weights: torch.Tensor,
        c_pred: torch.Tensor,
        bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    Function to evaluate a memory of linear equations with concept predictions.
    The number of equation correspond to the memory size, and it is
    not the same as the number of sample in the batch here.

    Args:
        concept_weights: parameters representing the weights of multiple linear
            models with shape (memory_size, n_concepts, n_classes)
        c_pred: concept predictions with shape (batch_size, n_concepts).
        bias: Bias term to add to the linear models (memory_size, n_classes).
    Returns:
        Tensor: Predictions made by the linear models with shape (batch_size,
                n_classes, memory_size)
    """
    if bias is not None:
        assert (concept_weights.shape[0] == bias.shape[0]
                and concept_weights.shape[2] == bias.shape[1])
    assert c_pred.shape[1] == concept_weights.shape[1]
    y_pred = torch.einsum('mcy,bc->bym', concept_weights, c_pred)
    if bias is not None:
        # the bias is (m,y) while y_pred is (bym) so we invert bias dimension
        y_pred += bias.T
    return y_pred


def linear_memory_explanations(
    concept_weights: torch.Tensor,
    concept_names: Dict[int, List[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Extract linear equations from decoded equations embeddings as strings.
    Args:
        concept_weights: Rule embeddings with shape (memory_size, n_concepts,
            n_tasks). It also has the bias term as last concept
        concept_names: Concept and task names.
    Returns:
        Dict[str, Dict[str, str]]: Equations as strings.
    """
    if len(concept_weights.shape) != 3:
        raise ValueError(
            "The concept weights must have 3 dimensions (memory_size, "
            "n_concepts, n_tasks)."
        )

    if hasattr(concept_weights, 'concept_names'):
        names = concept_weights.concept_names.copy()
        c_names = names[1]
        t_names = names[2]

    else:
        names = _default_concept_names(concept_weights.shape[1:3])
        if concept_names is None:
            c_names = names[1]
            t_names = names[2]
        else:
            c_names = concept_names[1]
            t_names = concept_names[2]

    equations_str = defaultdict(dict)  # task, memory_size
    memory_size = concept_weights.size(0)
    n_concepts = concept_weights.size(1)
    n_tasks = concept_weights.size(2)
    for task_idx in range(n_tasks):
        for mem_idx in range(memory_size):
            rule = [
                f"{c_names[concept_idx]} * {concept_weights[mem_idx, concept_idx, task_idx]:.2f}"
                for concept_idx in range(n_concepts)
            ]
            equations_str[t_names[task_idx]][f"Equation {mem_idx}"] = " + ".join(rule)
    return dict(equations_str)


def logic_rule_eval(
    concept_weights: torch.Tensor,
    c_pred: torch.Tensor,
    memory_idxs: torch.Tensor = None,
    semantic = CMRSemantic()
) -> torch.Tensor:
    """
    Use concept weights to make predictions based on logic rules.

    Args:
        concept_weights: concept weights with shape (batch_size,
            memory_size, n_concepts, n_tasks, n_roles) with n_roles=3.
        c_pred: concept predictions with shape (batch_size, n_concepts).
        memory_idxs: Indices of rules to evaluate with shape (batch_size,
            n_tasks). Default is None (evaluate all).

    Returns:
        torch.Tensor: Rule predictions with shape (batch_size, n_tasks, memory_size)
    """

    assert len(concept_weights.shape) == 5, \
        ("Size error, concept weights should be batch_size x memory_size "
         f"x n_concepts x n_tasks x n_roles. Received {concept_weights.shape}")
    memory_size = concept_weights.size(1)
    n_tasks = concept_weights.size(3)

    # to avoid numerical problem
    concept_weights = concept_weights * 0.999

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
    else:  # cast all to (batch_size, memory_size=1, n_concepts, n_tasks)
        # TODO: memory_idxs never used!
        x = c_pred.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, n_tasks)

    # batch_size, mem_size, n_tasks
    y_per_rule = semantic.disj(
        irrelevance,
        semantic.conj((1 - x), neg_polarity),
        semantic.conj(x, pos_polarity)
    )
    assert (y_per_rule < 1.0).all(), "y_per_rule should be in [0, 1]"

    # performing a conj while iterating over concepts of y_per_rule
    y_per_rule = semantic.conj(*[y for y in y_per_rule.split(1, dim=2)]).squeeze(dim=2)

    return y_per_rule.permute(0, 2, 1)


def logic_memory_reconstruction(
    concept_weights: torch.Tensor,
    c_true: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct tasks based on concept reconstructions, ground truth concepts
    and ground truth tasks.

    Args:
        concept_weights: concept reconstructions with shape (batch_size,
            memory_size, n_concepts, n_tasks).
        c_true: concept ground truth with shape (batch_size, n_concepts).
        y_true: task ground truth with shape (batch_size, n_tasks).

    Returns:
        torch.Tensor: Reconstructed tasks with shape (batch_size, n_tasks,
            memory_size).
    """
    pos_polarity, neg_polarity, irrelevance = (
        concept_weights[..., 0],
        concept_weights[..., 1],
        concept_weights[..., 2],
    )

    # batch_size, mem_size, n_tasks, n_concepts
    c_rec_per_classifier = 0.5 * irrelevance + pos_polarity

    reconstruction_mask = torch.where(
        c_true[:, None, :, None] == 1,
        c_rec_per_classifier,
        1 - c_rec_per_classifier,
    )
    c_rec_per_classifier = reconstruction_mask.prod(dim=2).pow(
        y_true[:, None, :]
    )
    return c_rec_per_classifier.permute(0, 2, 1)


def logic_rule_explanations(
    concept_logic_weights: torch.Tensor,
    concept_names: Dict[int, List[str]] = None,
) -> List[Dict[str, Dict[str, str]]]:
    """
    Extracts rules from rule concept weights as strings.

    Args:
        concept_logic_weights: Rule embeddings with shape (memory_size,
            n_concepts, n_tasks, 3).
        concept_names: Concept and task names.

    Returns:
        List[Dict[str, Dict[str, str]]]: Rules as strings.
    """
    if len(concept_logic_weights.shape) != 5 or (
        concept_logic_weights.shape[-1] != 3
    ):
        raise ValueError(
            "The concept logic weights must have 4 dimensions "
            "(batch_size, memory_size, n_concepts, n_tasks, 3)."
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

    batch_size = concept_logic_weights.size(0)
    memory_size = concept_logic_weights.size(1)
    n_concepts = concept_logic_weights.size(2)
    n_tasks = concept_logic_weights.size(3)
    # memory_size, n_concepts, n_tasks
    concept_roles = torch.argmax(concept_logic_weights, dim=-1)
    rule_list = []
    for sample_id in range(batch_size):
        rules_str = defaultdict(dict)  # task, memory_size
        for task_id in range(n_tasks):
            for mem_id in range(memory_size):
                rule = []
                for concept_id in range(n_concepts):
                    role = concept_roles[sample_id, mem_id, concept_id, task_id]
                    if role == 0:
                        rule.append(c_names[concept_id])
                    if role == 1:
                        rule.append(f"~ {c_names[concept_id]}")
                    else:
                        continue
                rules_str[t_names[task_id]][f"Rule {mem_id}"] = " & ".join(rule)
        rule_list.append(dict(rules_str))
    return rule_list


def selective_calibration(
    c_confidence: torch.Tensor,
    target_coverage: float,
) -> torch.Tensor:
    """
    Selects concepts based on confidence scores and target coverage.

    Args:
        c_confidence: Concept confidence scores.
        target_coverage: Target coverage.

    Returns:
        Tensor: Thresholds to select confident predictions.
    """
    theta = torch.quantile(
        c_confidence, 1 - target_coverage,
        dim=0,
        keepdim=True,
    )
    return theta


def confidence_selection(
    c_confidence: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    Selects concepts with confidence above a selected threshold.

    Args:
        c_confidence: Concept confidence scores.
        theta: Threshold to select confident predictions.

    Returns:
        Tensor: mask selecting confident predictions.
    """
    return torch.where(c_confidence > theta, True, False)


def soft_select(values, temperature, dim=1):
    softmax_scores = torch.log_softmax(values, dim=dim)
    soft_scores = torch.sigmoid(softmax_scores - temperature *
                               softmax_scores.mean(dim=dim, keepdim=True))
    return soft_scores
