"""
Concept-activation-vector (CAV) encoders for Post-hoc Concept Bottleneck
Models.

Post-hoc CBMs (Yuksekgonul et al., ICLR 2023,
https://arxiv.org/abs/2205.15480) build their bottleneck from a *pretrained*
backbone: each concept ``i`` is a vector ``v_i`` (and intercept ``b_i``) in
the backbone's embedding space — typically fitted post-hoc with an SVM or a
logistic-regression probe on a concept dataset — and the concept score of an
input embedding ``f(x)`` is its (normalised) signed distance to the concept
hyperplane::

    s_i(x) = (<f(x), v_i> + b_i) / ||v_i||

The scores are *raw real values* (not probabilities): the downstream
interpretable predictor consumes them as-is.

This module provides:

* :class:`ConceptActivationVectors` — the shared, optionally trainable table
  of concept vectors and intercepts;
* :class:`CAVEmbeddingToConcept` — the encoder layer producing the concept
  scores from an embedding.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn

from ..base.layer import BaseConceptLayer


class ConceptActivationVectors(nn.Module):
    """
    Learnable (or frozen) table of concept activation vectors.

    Holds one concept vector ``v_i`` of size ``embedding_size`` and one
    intercept ``b_i`` per concept. Vectors fitted externally (e.g. per-concept
    SVM / logistic-regression probes on a pretrained backbone's embeddings, as
    in the original PCBM pipeline) can be passed in and frozen; when omitted,
    the table is randomly initialised and can be trained end-to-end (training
    the scores with a BCE-with-logits loss makes each row exactly a
    logistic-regression probe).

    Args:
        n_concepts: Number of concepts.
        embedding_size: Dimensionality of the backbone embedding space.
        vectors: Optional pre-fitted concept vectors of shape
            ``(n_concepts, embedding_size)``.
        intercepts: Optional pre-fitted intercepts of shape ``(n_concepts,)``.
            Defaults to zeros.
        trainable: Whether the vectors/intercepts receive gradients.
            Default False (the post-hoc setting: the concept bank is given).

    References:
        Yuksekgonul et al. "Post-hoc Concept Bottleneck Models", ICLR 2023.
        https://arxiv.org/abs/2205.15480
    """

    def __init__(
        self,
        n_concepts: int,
        embedding_size: int,
        vectors: Optional[torch.Tensor] = None,
        intercepts: Optional[torch.Tensor] = None,
        trainable: bool = False,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.embedding_size = embedding_size

        if vectors is None:
            vectors = torch.randn(n_concepts, embedding_size)
            vectors = vectors / math.sqrt(embedding_size)
        else:
            vectors = torch.as_tensor(vectors, dtype=torch.get_default_dtype())
            if vectors.shape != (n_concepts, embedding_size):
                raise ValueError(
                    f"Expected concept vectors of shape "
                    f"({n_concepts}, {embedding_size}), got "
                    f"{tuple(vectors.shape)}."
                )
        if intercepts is None:
            intercepts = torch.zeros(n_concepts)
        else:
            intercepts = torch.as_tensor(
                intercepts, dtype=torch.get_default_dtype()
            ).reshape(-1)
            if intercepts.shape != (n_concepts,):
                raise ValueError(
                    f"Expected concept intercepts of shape ({n_concepts},), "
                    f"got {tuple(intercepts.shape)}."
                )

        self.vectors = nn.Parameter(vectors.clone(), requires_grad=trainable)
        self.intercepts = nn.Parameter(
            intercepts.clone(), requires_grad=trainable
        )

    def scores(
        self,
        embeddings: torch.Tensor,
        concept_idx: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Concept scores of ``embeddings``: normalised signed distances.

        Args:
            embeddings: Backbone embeddings of shape (batch, embedding_size).
            concept_idx: Optional indices restricting which concepts to score.

        Returns:
            torch.Tensor: Scores of shape (batch, n_selected_concepts).
        """
        vectors, intercepts = self.vectors, self.intercepts
        if concept_idx is not None:
            vectors = vectors[concept_idx]
            intercepts = intercepts[concept_idx]
        norms = vectors.norm(p=2, dim=1).clamp_min(1e-12)
        return (embeddings @ vectors.T + intercepts) / norms


class CAVEmbeddingToConcept(BaseConceptLayer):
    """
    Encoder producing concept scores from a shared CAV table.

    A thin layer around :class:`ConceptActivationVectors`: given the backbone
    embedding, returns the raw concept scores of the (optionally selected)
    concepts. In a Post-hoc CBM this parametrizes the ``value`` of the
    deterministic (Delta) concept-score variables.

    Args:
        cavs: The shared :class:`ConceptActivationVectors` table.
        concept_idx: Optional indices selecting which concepts this layer
            scores (used by the per-concept building path). ``None`` (default)
            scores every concept in the table.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import (
        ...     CAVEmbeddingToConcept, ConceptActivationVectors,
        ... )
        >>>
        >>> cavs = ConceptActivationVectors(n_concepts=4, embedding_size=16)
        >>> encoder = CAVEmbeddingToConcept(cavs)
        >>> scores = encoder(torch.randn(8, 16))
        >>> print(scores.shape)
        torch.Size([8, 4])

    References:
        Yuksekgonul et al. "Post-hoc Concept Bottleneck Models", ICLR 2023.
        https://arxiv.org/abs/2205.15480
    """

    def __init__(
        self,
        cavs: ConceptActivationVectors,
        concept_idx: Optional[List[int]] = None,
    ):
        n_selected = (
            cavs.n_concepts if concept_idx is None else len(concept_idx)
        )
        super().__init__(
            out_concepts=n_selected,
            in_embeddings=cavs.embedding_size,
        )
        self.cavs = cavs
        self.concept_idx = (
            list(concept_idx) if concept_idx is not None else None
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute concept scores from backbone embeddings.

        Args:
            embeddings: Backbone embeddings of shape (batch, embedding_size).

        Returns:
            torch.Tensor: Raw concept scores of shape (batch, m).
        """
        return self.cavs.scores(embeddings, self.concept_idx)
