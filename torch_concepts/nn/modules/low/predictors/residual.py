"""
Residual task predictor for hybrid Post-hoc Concept Bottleneck Models
(PCBM-h).

The hybrid variant of the Post-hoc CBM (Yuksekgonul et al., ICLR 2023,
https://arxiv.org/abs/2205.15480) predicts the task as the sum of an
*interpretable* linear head over the concept scores and a *residual* linear
head over the raw backbone embedding::

    y = W_c s(x) + b_c  +  r(f(x))

The residual is fitted sequentially, after the interpretable head, to recover
the accuracy the concept bottleneck loses; disabling it at evaluation time
recovers the purely interpretable predictions.
"""
from typing import Union

import torch
import torch.nn as nn

from torch_concepts import Annotations
from ..base.layer import BaseConceptLayer


class ResidualConceptEmbeddingToConcept(BaseConceptLayer):
    """
    PCBM-h task head: interpretable linear head plus embedding residual.

    Computes ``c2y(concepts) + residual(embeddings)``. The ``residual_use``
    flag toggles the residual term at inference time, so the same trained
    model can produce hybrid (residual on) or purely interpretable (residual
    off) predictions, as in the reference implementation.

    Attributes:
        c2y (nn.Linear): Interpretable head over the concept scores.
        residual (nn.Linear): Residual head over the backbone embedding.
        residual_use (bool): Whether the residual term is added. Default True.

    Args:
        in_concepts: Number of input concept scores.
        in_embeddings: Dimensionality of the backbone embedding.
        out_concepts: Number of output task logits.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import ResidualConceptEmbeddingToConcept
        >>>
        >>> head = ResidualConceptEmbeddingToConcept(
        ...     in_concepts=4, in_embeddings=16, out_concepts=2,
        ... )
        >>> logits = head(
        ...     concepts=torch.randn(8, 4), embeddings=torch.randn(8, 16),
        ... )
        >>> print(logits.shape)
        torch.Size([8, 2])

    References:
        Yuksekgonul et al. "Post-hoc Concept Bottleneck Models", ICLR 2023.
        https://arxiv.org/abs/2205.15480
    """

    def __init__(
        self,
        in_concepts: Union[int, Annotations],
        in_embeddings: Union[int, Annotations],
        out_concepts: Union[int, Annotations],
    ):
        super().__init__(
            out_concepts=out_concepts,
            in_concepts=in_concepts,
            in_embeddings=in_embeddings,
        )
        self.c2y = nn.Linear(self.in_concepts_shape, self.out_concepts_shape)
        self.residual = nn.Linear(
            self.in_embeddings_shape,
            self.out_concepts_shape,
        )
        self.residual_use = True

    def forward(
        self,
        concepts: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict task logits from concept scores and backbone embeddings.

        Args:
            concepts: Concept scores of shape (batch, in_concepts).
            embeddings: Backbone embeddings of shape (batch, in_embeddings).

        Returns:
            torch.Tensor: Task logits of shape (batch, out_concepts).
        """
        out = self.c2y(concepts)
        if self.residual_use:
            out = out + self.residual(embeddings)
        return out
