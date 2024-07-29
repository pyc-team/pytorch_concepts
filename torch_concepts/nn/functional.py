from typing import List, Union

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
