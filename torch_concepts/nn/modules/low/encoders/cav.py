"""
Concept Activation Vectors (CAV) encoder.

A Concept Activation Vector (Kim et al., 2018) is a unit vector in the
activation space of a trained network that points towards a user-defined
concept. It is obtained post hoc: activations of concept-positive and
concept-negative examples are separated with a binary linear classifier, and
the CAV is the unit normal to its decision boundary.

:class:`CAVEmbeddingToConcept` is that bank of CAVs. It can be filled in three
ways: fitted here with :meth:`fit` (the TCAV recipe), handed pre-fitted vectors
from elsewhere, or left trainable and learned in place. A Post-hoc CBM
(Yuksekgonul et al., ICLR 2023) uses the last two.
"""
import numpy as np
import torch
import torch.nn as nn

from ..base.layer import BaseConceptLayer
from sklearn.linear_model import LogisticRegression
from torch_concepts import Annotations
from typing import Optional, Union


class CAVEmbeddingToConcept(BaseConceptLayer):
    """
    Concept encoder based on Concept Activation Vectors (Kim et al., 2018).

    The forward pass returns the *geometric margin* of each embedding to each
    concept's decision boundary, ``(x . v_j + b_j) / ||v_j||`` — what is
    sometimes called the "concept score" or "activation value". Positive values
    mean the concept is predicted present.

    The CAVs can come from three places:

    * :meth:`fit`, which learns one logistic-regression probe per concept on
      frozen activations and stores its unit-normalized weights (the TCAV
      recipe);
    * ``cavs`` / ``bias``, pre-fitted elsewhere — how a Post-hoc CBM supplies
      its concept bank;
    * gradient descent, with ``trainable=True``, which makes each CAV a
      logistic-regression probe learned in place when supervised with
      BCE-with-logits against concept labels.

    A layer that has been given none of these holds zeros, and its forward pass
    raises rather than returning meaningless scores.

    Attributes:
        cavs (torch.Tensor): Shape (out_concepts, in_embeddings), the CAVs.
            Unit-norm as :meth:`fit` leaves them, though training moves
            them off the unit sphere, so :meth:`forward` re-normalizes.
        bias (torch.Tensor): Shape (out_concepts,), the probe intercepts.
        fitted (torch.Tensor): Whether the CAVs are meaningful yet.

    Args:
        in_embeddings: Number of input embedding features.
        out_concepts: Number of output concept representations.
        cavs: Optional pre-fitted CAVs of shape (out_concepts, in_embeddings).
            Zeros when omitted, pending a call to :meth:`fit`.
        bias: Optional pre-fitted intercepts of shape (out_concepts,). Zeros
            when omitted.
        trainable: If True, the CAVs and intercepts are ``nn.Parameter``s; if
            False (default) they are buffers, the frozen post-hoc setting.
        **fit_kwargs: Additional keyword arguments for
            :class:`sklearn.linear_model.LogisticRegression`
            (``max_iter`` defaults to 1000).

    References:
        Kim et al. "Interpretability Beyond Feature Attribution: Quantitative
        Testing with Concept Activation Vectors (TCAV)", ICML 2018.
        https://proceedings.mlr.press/v80/kim18d

        Yuksekgonul et al. "Post-hoc Concept Bottleneck Models", ICLR 2023.
        https://openreview.net/forum?id=nA5AZ8CEyow
    """

    def __init__(
        self,
        in_embeddings: Union[int, Annotations],
        out_concepts: Union[int, Annotations],
        cavs: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        trainable: bool = False,
        **fit_kwargs,
    ):
        super().__init__(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        self.fit_kwargs = {"max_iter": 1000}
        self.fit_kwargs.update(fit_kwargs)
        n, d = self.out_concepts_shape, self.in_embeddings_shape

        # Note for self: zeros are the "nothing here yet" state, marked by
        # ``fitted`` so forward refuses rather than divide by a zero norm.
        given = cavs is not None
        cavs = torch.zeros(n, d) if not given else \
            torch.as_tensor(cavs, dtype=torch.float).reshape(n, d)
        bias = torch.zeros(n) if bias is None else \
            torch.as_tensor(bias, dtype=torch.float).reshape(n)

        if trainable:
            self.cavs = nn.Parameter(cavs)
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer("cavs", cavs)
            self.register_buffer("bias", bias)
        self.register_buffer("fitted", torch.tensor(given))

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
                    X=x_np,
                    y=y_np[:, j],
                )
            except ValueError as err:
                raise ValueError(
                    f"Fitting the probe for concept column {j} failed: {err}"
                ) from err
            # Now store the unit-normalized probe weights and the intercept
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
        Encode embeddings into geometric margins to the concept boundaries.

        Args:
            embeddings: Input embeddings of shape (..., in_embeddings).

        Returns:
            torch.Tensor: Concept scores of shape (..., out_concepts);
            positive values mean the concept is predicted present.
        """
        if not self.fitted:
            raise RuntimeError(
                "CAVEmbeddingToConcept has not been fitted; call fit() on "
                "concept-labeled activations, or pass pre-fitted cavs."
            )
        norm = self.cavs.norm(dim=-1)                       # (out_concepts,)
        # .to(embeddings): buffers are fp32; keeps AMP fp16/bf16 activations
        unit = (self.cavs / norm.unsqueeze(-1)).to(embeddings)
        return embeddings @ unit.t() + (self.bias / norm).to(embeddings)
