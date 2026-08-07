"""
Anchor-based distance predictors for Probabilistic Concept Bottleneck Models.

ProbCBM (Kim et al., ICML 2023, https://arxiv.org/abs/2306.01574) predicts
concepts and classes from *distances in embedding space* rather than from
linear logits:

* each concept has a learnable **positive** and **negative** anchor embedding,
  and the concept probability is a two-way softmax over the (scaled) distances
  between the predicted concept embedding and the two anchors;
* each class has a learnable anchor in a class-embedding space, and class
  probabilities are a softmax over the (scaled) negative distances between a
  projection of the concept embeddings and the class anchors.

This module provides those pieces as PyC layers:

* :class:`ConceptAnchors` — the shared table of positive/negative concept
  anchors plus the learnable distance scale;
* :class:`AnchorEmbeddingToConcept` — concept logits from embedding-to-anchor
  distances;
* :class:`ConceptAnchorProjection` — the shared class-head trunk: interpolate
  the anchors with the concept activations and project the result to the
  class-embedding space;
* :class:`AnchorConceptToConcept` — task logits from class-anchor distances.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.layer import BaseConceptLayer


class ConceptAnchors(nn.Module):
    """
    Learnable positive/negative anchor embeddings, one pair per concept.

    Holds the ``(n_concepts, 2, embedding_size)`` anchor table (index 0 along
    the second axis is the *negative* anchor, index 1 the *positive* one) and
    the learnable ``negative_scale`` used to turn anchor distances into concept
    logits. Anchors are L2-normalised at use, matching the reference ProbCBM
    implementation. The module is meant to be *shared*: the concept predictors
    (:class:`AnchorEmbeddingToConcept`) and the class-head trunk
    (:class:`ConceptAnchorProjection`) reference the same instance, so
    interventions replace predicted embeddings with exactly the anchors the
    concept probabilities are measured against.

    Args:
        n_concepts: Number of (binary) concepts.
        embedding_size: Dimensionality of each anchor embedding.
        init_negative_scale: Initial value of the distance scale. Default 5.
    """

    def __init__(
        self,
        n_concepts: int,
        embedding_size: int,
        init_negative_scale: float = 5.0,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.embedding_size = embedding_size
        anchors = torch.empty(n_concepts, 2, embedding_size)
        nn.init.trunc_normal_(anchors, std=1.0 / math.sqrt(embedding_size))
        self.anchors = nn.Parameter(anchors)
        self.negative_scale = nn.Parameter(
            torch.tensor([float(init_negative_scale)])
        )

    @property
    def normalized(self) -> torch.Tensor:
        """L2-normalised anchors, shape ``(n_concepts, 2, embedding_size)``."""
        return F.normalize(self.anchors, p=2, dim=-1)

    def interpolate(self, values: torch.Tensor) -> torch.Tensor:
        """Anchor embeddings interpolated by concept activations.

        For activation ``v_i`` of concept ``i`` (a probability, a relaxed
        sample, or a hard 0/1 value under intervention),
        returns ``v_i * anchor_i^+ + (1 - v_i) * anchor_i^-`` — exactly the
        ground-truth anchor embedding when ``v_i`` is a hard label, which is
        how ProbCBM performs interventions.

        Args:
            values: Concept activations of shape (batch, n_concepts).

        Returns:
            torch.Tensor: Interpolated embeddings of shape
                (batch, n_concepts, embedding_size).
        """
        anchors = self.normalized
        values = values.unsqueeze(-1)
        return values * anchors[:, 1, :] + (1.0 - values) * anchors[:, 0, :]


class AnchorEmbeddingToConcept(BaseConceptLayer):
    """
    Concept predictor computing logits from embedding-to-anchor distances.

    For each concept, the logit is ``s * (d(z, a^-) - d(z, a^+))`` where ``z``
    is the (sampled or mean) concept embedding, ``a^+``/``a^-`` the concept's
    positive/negative anchors and ``s`` the shared learnable
    ``negative_scale``. This is the two-way softmax over scaled distances of
    the reference ProbCBM (``use_neg_concept=True``) expressed as a single
    Bernoulli logit: ``sigmoid(logit) = softmax([-s·d^-, -s·d^+])[+]``.

    Args:
        anchors: The shared :class:`ConceptAnchors` table.
        concept_idx: Optional indices selecting which concepts this layer
            predicts (used by the per-concept building path). ``None`` (the
            default) predicts all concepts in the table at once.
        eps: Stabiliser added inside the square root. Default 1e-6.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import AnchorEmbeddingToConcept, ConceptAnchors
        >>>
        >>> anchors = ConceptAnchors(n_concepts=4, embedding_size=16)
        >>> predictor = AnchorEmbeddingToConcept(anchors)
        >>> logits = predictor(torch.randn(8, 4 * 16))
        >>> print(logits.shape)
        torch.Size([8, 4])

    References:
        Kim et al. "Probabilistic Concept Bottleneck Models", ICML 2023.
        https://arxiv.org/abs/2306.01574
    """

    def __init__(
        self,
        anchors: ConceptAnchors,
        concept_idx: Optional[List[int]] = None,
        eps: float = 1e-6,
    ):
        n_selected = (
            anchors.n_concepts if concept_idx is None else len(concept_idx)
        )
        super().__init__(
            out_concepts=n_selected,
            in_embeddings=n_selected * anchors.embedding_size,
        )
        self.anchors = anchors
        self.concept_idx = (
            list(concept_idx) if concept_idx is not None
            else None
        )
        self.eps = eps

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict concept logits from concept embeddings.

        Args:
            embeddings: Concept embeddings of shape (batch, m * embedding_size)
                or (batch, m, embedding_size), where m is the number of
                concepts this layer predicts.

        Returns:
            torch.Tensor: Concept logits of shape (batch, m).
        """
        batch_size = embeddings.shape[0]
        z = embeddings.reshape(batch_size, -1, self.anchors.embedding_size)
        anchors = self.anchors.normalized
        if self.concept_idx is not None:
            anchors = anchors[self.concept_idx]
        # (batch, m, 2): distance of each embedding to its (neg, pos) anchors.
        distance = torch.sqrt(
            (z.unsqueeze(2) - anchors.unsqueeze(0)).pow(2).sum(-1) + self.eps
        )
        return self.anchors.negative_scale * (
            distance[..., 0] - distance[..., 1]
        )


class ConceptAnchorProjection(nn.Module):
    """
    Shared trunk of the ProbCBM class head.

    Interpolates the concept anchors with the concept activations
    (:meth:`ConceptAnchors.interpolate`) and projects the concatenated result
    into the class-embedding space with a single linear map. Also holds the
    learnable class distance ``scale`` shared by every task head. Meant to be
    shared across all :class:`AnchorConceptToConcept` heads so multi-task
    models use one class-embedding space, as in the reference implementation.

    Args:
        anchors: The shared :class:`ConceptAnchors` table.
        class_embedding_size: Dimensionality of the class-embedding space.
        init_scale: Initial value of the class distance scale. Default 5.
    """

    def __init__(
        self,
        anchors: ConceptAnchors,
        class_embedding_size: int,
        init_scale: float = 5.0,
    ):
        super().__init__()
        self.anchors = anchors
        self.class_embedding_size = class_embedding_size
        self.projection = nn.Linear(
            anchors.n_concepts * anchors.embedding_size,
            class_embedding_size,
        )
        self.scale = nn.Parameter(torch.tensor([float(init_scale)]))

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Project concept activations into the class-embedding space.

        Args:
            concepts: Concept activations of shape (batch, n_concepts).

        Returns:
            torch.Tensor: Class embeddings of shape
                (batch, class_embedding_size).
        """
        mixed = self.anchors.interpolate(concepts)
        return self.projection(mixed.flatten(start_dim=1))


class AnchorConceptToConcept(BaseConceptLayer):
    """
    Distance-based task predictor of ProbCBM.

    Projects the concept activations into the class-embedding space through
    the shared :class:`ConceptAnchorProjection` trunk and computes logits from
    the distances to learnable per-class anchors. A categorical task with
    cardinality ``k`` uses ``k`` anchors and logits ``-s * d_k``, so a softmax
    over the logits recovers ProbCBM's ``softmax(-s * distance)``. A binary
    task (cardinality 1) uses a (negative, positive) anchor pair and the
    single logit ``s * (d^- - d^+)``, the two-anchor special case.

    Args:
        projection: The shared :class:`ConceptAnchorProjection` trunk.
        cardinality: Cardinality of each task handled by this layer (1 for a
            binary task). Default 1.
        n_heads: Number of tasks handled by this layer (>1 on the plate
            building path, where one layer predicts all tasks). Default 1.
        eps: Stabiliser added inside the square root. Default 1e-10.

    References:
        Kim et al. "Probabilistic Concept Bottleneck Models", ICML 2023.
        https://arxiv.org/abs/2306.01574
    """

    def __init__(
        self,
        projection: ConceptAnchorProjection,
        cardinality: int = 1,
        n_heads: int = 1,
        eps: float = 1e-10,
    ):
        super().__init__(
            out_concepts=n_heads * cardinality,
            in_concepts=projection.anchors.n_concepts,
        )
        self.projection = projection
        self.cardinality = cardinality
        self.n_heads = n_heads
        self.eps = eps
        n_anchor_states = cardinality if cardinality > 1 else 2
        self.class_anchors = nn.Parameter(
            torch.randn(
                n_heads,
                n_anchor_states,
                projection.class_embedding_size,
            )
        )

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Predict task logits from concept activations.

        Args:
            concepts: Concept activations of shape (batch, n_concepts).

        Returns:
            torch.Tensor: Task logits of shape (batch, n_heads * cardinality).
        """
        class_embedding = self.projection(concepts)
        # (batch, n_heads, n_anchor_states): distance to every class anchor.
        diff = (
            class_embedding.unsqueeze(1).unsqueeze(1) -
            self.class_anchors.unsqueeze(0)
        )
        distance = torch.sqrt(diff.pow(2).mean(-1) + self.eps)
        scale = self.projection.scale
        if self.cardinality == 1:
            # Binary task: single logit from the (neg, pos) anchor pair.
            return scale * (distance[..., 0] - distance[..., 1])
        return (-scale * distance).flatten(start_dim=1)
