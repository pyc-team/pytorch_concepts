"""
Concept Activation Vectors (CAV) encoder.

A Concept Activation Vector (Kim et al., 2018) is a unit vector in the
activation space of a trained network that points towards a user-defined
concept. It is obtained post hoc: activations of concept-positive and
concept-negative examples are separated with a binary linear classifier, and
the CAV is the unit normal to its decision boundary.

:class:`CAVEmbeddingToConcept` fits one CAV per concept and then acts as a
frozen concept encoder: its forward pass returns the signed distance of an
embedding to each concept boundary (positive means the concept is present).

The TCAV testing machinery (the directional-derivative sensitivity of a
downstream head along the CAVs, reduced to the TCAV score) is stateless and
lives in :mod:`torch_concepts.nn.functional` as
:func:`~torch_concepts.nn.functional.tcav_score`.
"""
import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from torch_concepts import Annotations
from ..base.layer import BaseConceptLayer


class CAVEmbeddingToConcept(BaseConceptLayer):
    """
    Concept encoder based on Concept Activation Vectors (Kim et al., 2018).

    The layer is constructed unfitted and trained post hoc with :meth:`fit`,
    which fits one logistic-regression probe per concept on frozen
    activations and stores the unit-normalized probe weights as CAVs. The
    CAVs are buffers, not parameters: they are invisible to optimizers and
    are never updated by the main loss, but they move with ``.to(device)``
    and survive ``state_dict`` round-trips.

    The forward pass returns the signed distance of each embedding to each
    concept's decision boundary, ``x @ cav_j + bias_j``: its sign equals the
    probe's prediction (positive means concept present) and its gradient
    w.r.t. the input is exactly the unit CAV, matching the directional
    derivative used by TCAV.

    Attributes:
        cavs (torch.Tensor): Buffer of shape (out_concepts, in_embeddings)
            holding the unit-norm CAVs (zeros before :meth:`fit`).
        bias (torch.Tensor): Buffer of shape (out_concepts,) holding the
            probe intercepts rescaled by the same normalization.

    Args:
        in_embeddings: Number of input embedding features.
        out_concepts: Number of output concept representations.
        **fit_kwargs: Additional keyword arguments for
            :class:`sklearn.linear_model.LogisticRegression`
            (``max_iter`` defaults to 1000).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import CAVEmbeddingToConcept
        >>>
        >>> _ = torch.manual_seed(0)
        >>> encoder = CAVEmbeddingToConcept(in_embeddings=16, out_concepts=2)
        >>> embeddings = torch.randn(64, 16)
        >>> labels = (embeddings[:, :2] > 0).float()
        >>> accuracy = encoder.fit(embeddings, labels)
        >>> concepts = encoder(embeddings)
        >>> print(concepts.shape)
        torch.Size([64, 2])

    References:
        Kim et al. "Interpretability Beyond Feature Attribution: Quantitative
        Testing with Concept Activation Vectors (TCAV)", ICML 2018.
        https://proceedings.mlr.press/v80/kim18d
    """

    def __init__(
        self,
        in_embeddings: Union[int, Annotations],
        out_concepts: Union[int, Annotations],
        **fit_kwargs,
    ):
        """
        Initialize the encoder.

        Args:
            in_embeddings: Number of input embedding features.
            out_concepts: Number of output concept representations.
            **fit_kwargs: Additional keyword arguments for
                :class:`sklearn.linear_model.LogisticRegression`
                (``max_iter`` defaults to 1000).
        """
        super().__init__(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        self.fit_kwargs = {"max_iter": 1000, **fit_kwargs}
        n, d = self.out_concepts_shape, self.in_embeddings_shape
        self.register_buffer("cavs", torch.zeros(n, d))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(
        self,
        embeddings: torch.Tensor,
        concept_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fit one CAV per concept on frozen activations.

        Each concept's binary labels are separated with a logistic-regression
        probe; the CAV is the probe's weight vector normalized to unit norm
        (pointing towards the concept-positive side), and the bias is the
        intercept rescaled by the same factor.

        Every label column is an independent one-vs-rest probe, so
        categorical concepts are supported by passing their one-hot state
        columns (k columns for a k-state concept — e.g. construct the layer
        with an :class:`~torch_concepts.Annotations` whose cardinalities
        sum to the label width, and one CAV is fit per state).

        Args:
            embeddings: Activations of shape (..., in_embeddings).
            concept_labels: Binary concept labels of shape
                (..., out_concepts), one column per concept (or per
                categorical state).

        Returns:
            torch.Tensor: Per-concept probe training accuracy of shape
            (out_concepts,) — the paper's check that the concept is
            linearly separable at this layer.
        """
        if embeddings.shape[-1] != self.in_embeddings_shape:
            raise ValueError(
                f"embeddings have {embeddings.shape[-1]} features, expected "
                f"in_embeddings={self.in_embeddings_shape}."
            )
        if concept_labels.shape[-1] != self.out_concepts_shape:
            raise ValueError(
                f"concept_labels have {concept_labels.shape[-1]} columns, "
                f"expected out_concepts={self.out_concepts_shape}."
            )
        x = embeddings.reshape(-1, self.in_embeddings_shape)
        y = concept_labels.reshape(-1, self.out_concepts_shape)
        if x.size(0) != y.size(0):
            raise ValueError(
                f"embeddings and concept_labels disagree on the number of "
                f"samples: {x.size(0)} vs {y.size(0)}."
            )
        # .float(): numpy cannot represent bfloat16; fp32/fp64 pass through
        to_np = lambda t: t.detach().cpu().numpy() \
            if t.dtype != torch.bfloat16 else t.detach().cpu().float().numpy()
        x_np, y_np = to_np(x), to_np(y)

        accuracies = torch.zeros(self.out_concepts_shape)
        for j in range(self.out_concepts_shape):
            if len(np.unique(y_np[:, j])) > 2:
                raise ValueError(
                    f"Concept column {j} has more than 2 distinct values; "
                    f"CAV probes are binary. Encode categorical concepts as "
                    f"one-hot state columns (see fit's docstring)."
                )
            try:
                probe = LogisticRegression(**self.fit_kwargs).fit(
                    x_np, y_np[:, j]
                )
            except ValueError as err:
                raise ValueError(
                    f"Fitting the probe for concept column {j} failed: {err}"
                ) from err
            weight = torch.from_numpy(probe.coef_[0])
            intercept = float(probe.intercept_[0])
            norm = weight.norm()
            self.cavs[j] = (weight / norm).to(self.cavs)
            self.bias[j] = intercept / norm
            accuracies[j] = probe.score(x_np, y_np[:, j])
        self.fitted.fill_(True)
        return accuracies.to(self.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embeddings into signed distances to the concept boundaries.

        Args:
            embeddings: Input embeddings of shape (..., in_embeddings).

        Returns:
            torch.Tensor: Concept scores of shape (..., out_concepts);
            positive values mean the concept is predicted present.
        """
        if not self.fitted:
            raise RuntimeError(
                "CAVEmbeddingToConcept has not been fitted; call fit() on "
                "concept-labeled activations first."
            )
        # .to(embeddings): buffers are fp32; keeps AMP fp16/bf16 activations
        return (
            embeddings @ self.cavs.t().to(embeddings)
            + self.bias.to(embeddings)
        )


class ConceptActivationVectors(nn.Module):
    """
    Shared bank of per-concept activation vectors (the Post-hoc CBM concept bank).

    Holds one CAV per concept — a direction in the backbone's embedding space —
    with a per-concept intercept, and turns an embedding into the *normalised
    signed distance* to each concept hyperplane,
    ``s_i = (<f(x), v_i> + b_i) / ||v_i||`` (positive means the concept is
    predicted present). This is the concept bank of the Post-hoc CBM
    (Yuksekgonul et al., ICLR 2023): the CAVs are normally fitted post hoc — one
    linear probe (SVM / logistic regression) per concept on the frozen backbone —
    and passed in via ``vectors`` / ``intercepts``, then frozen
    (``trainable=False``). Left trainable, the same table can be learned in place
    as a bank of logistic-regression probes.

    This differs from :class:`CAVEmbeddingToConcept`, which fits *itself* from
    labelled activations with sklearn and stores frozen buffers: this table is
    initialised from externally-fitted (or random) vectors and is meant to be
    *shared* across a model's per-concept score layers, exactly as ProbCBM shares
    one :class:`~torch_concepts.nn.ConceptAnchors` table.

    Args:
        n_concepts: Number of (binary) concepts.
        embedding_size: Dimensionality of the backbone embedding.
        vectors: Optional pre-fitted CAVs of shape ``(n_concepts,
            embedding_size)``. Randomly initialised when omitted.
        intercepts: Optional pre-fitted intercepts of shape ``(n_concepts,)``.
            Zeros when omitted.
        trainable: If True (default) the table is an ``nn.Parameter`` bank; if
            False it is registered as buffers (the frozen post-hoc setting).
    """

    def __init__(
        self,
        n_concepts: int,
        embedding_size: int,
        vectors: Optional[torch.Tensor] = None,
        intercepts: Optional[torch.Tensor] = None,
        trainable: bool = True,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.embedding_size = embedding_size

        if vectors is None:
            vectors = torch.empty(n_concepts, embedding_size)
            nn.init.normal_(vectors, std=1.0 / math.sqrt(embedding_size))
        else:
            vectors = torch.as_tensor(vectors, dtype=torch.float).reshape(
                n_concepts, embedding_size
            )
        if intercepts is None:
            intercepts = torch.zeros(n_concepts)
        else:
            intercepts = torch.as_tensor(
                intercepts, dtype=torch.float
            ).reshape(n_concepts)

        if trainable:
            self.vectors = nn.Parameter(vectors)
            self.intercepts = nn.Parameter(intercepts)
        else:
            self.register_buffer("vectors", vectors)
            self.register_buffer("intercepts", intercepts)

    def forward(
        self,
        embeddings: torch.Tensor,
        concept_idx: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Normalised signed distances of ``embeddings`` to the concept hyperplanes.

        Args:
            embeddings: Backbone embeddings of shape (..., embedding_size).
            concept_idx: Optional indices selecting a subset of concepts.
                ``None`` (default) scores every concept in the bank.

        Returns:
            torch.Tensor: Concept scores of shape (..., m), where ``m`` is the
            number of selected concepts.
        """
        v = self.vectors
        b = self.intercepts
        if concept_idx is not None:
            v = v[concept_idx]
            b = b[concept_idx]
        norm = v.norm(dim=-1)                       # (m,)
        unit = (v / norm.unsqueeze(-1)).to(embeddings)
        return embeddings @ unit.t() + (b / norm).to(embeddings)


class CAVScoreEncoder(BaseConceptLayer):
    """
    CPD-facing concept-score layer over a shared :class:`ConceptActivationVectors`.

    Thin adaptor so a Post-hoc CBM's concept-score CPDs read from one shared CAV
    bank: its forward returns the bank's normalised signed distances for the
    concepts this layer is responsible for (``concept_idx``, or all of them).
    Mirrors :class:`~torch_concepts.nn.AnchorEmbeddingToConcept` over
    :class:`~torch_concepts.nn.ConceptAnchors`.

    Args:
        cavs: The shared :class:`ConceptActivationVectors` bank.
        concept_idx: Optional indices selecting which concepts this layer scores
            (used by the per-concept building path). ``None`` scores all.
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
        """Concept scores of shape (..., m) from the shared CAV bank."""
        return self.cavs(embeddings, self.concept_idx)
