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
    return ConceptTensor.concept(torch.where(indexes, c_true, c_pred))


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
    return c_mix