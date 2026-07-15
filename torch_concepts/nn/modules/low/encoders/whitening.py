"""
Concept Whitening layers.

Concept Whitening (CW) replaces a normalization layer (e.g. BatchNorm) with a
module that (1) whitens the latent representation — zero mean, identity
covariance — and (2) rotates it with an orthogonal matrix so that each of the
first ``n_concepts`` axes aligns with a pre-defined concept.

Two layers are provided:

- :class:`ConceptWhitening`: a plain ``torch.nn.Module`` faithful to the
  paper — a concept-agnostic, drop-in normalization layer that preserves the
  embedding dimension and whose axes can be aligned to concepts.
- :class:`WhitenedEmbeddingToConcept`: a
  :class:`~torch_concepts.nn.BaseConceptLayer` encoder that wraps
  :class:`ConceptWhitening` and returns only the concept axes, for CBM-style
  pipelines.

This is a minimal 2D-embedding adaptation of the official implementation
(`IterNormRotation`) released with the paper, which builds on IterNorm:
- https://github.com/zhiCHEN96/ConceptWhitening
- https://github.com/huangleiBuaa/IterNorm

Differences from the original: inputs are flat embeddings ``(..., d)`` instead
of conv feature maps, a single whitening group is used, and only the ``mean``
concept-activation mode is implemented. Whitening uses plain differentiable
ops, so autograd replaces the hand-written backward of the original.
"""
from typing import Union

import torch

from torch_concepts import Annotations
from ..base.layer import BaseConceptLayer


class ConceptWhitening(torch.nn.Module):
    """
    Concept Whitening layer (Chen, Bei & Rudin, 2020).

    The layer whitens input embeddings with iterative normalization
    (Newton–Schulz iterations, as in IterNorm) and applies a learned
    orthogonal rotation ``R`` so that axis ``j`` of the output responds to
    concept ``j``. The embedding dimension is preserved, so the layer can
    replace a BatchNorm anywhere in a backbone or
    :class:`~torch_concepts.nn.Sequential` without introducing a bottleneck.

    The layer itself is concept-agnostic: any output axis can be aligned via
    :meth:`align`. Which (and how many) axes are designated as concepts is
    decided by the caller — typically :class:`WhitenedEmbeddingToConcept`,
    which designates the first ``out_concepts`` axes and slices them.

    The concept axes are not supervised by the main loss: the whitening
    matrix and the rotation are buffers, invisible to the optimizer. The
    rotation is aligned separately — gradients of the concept-alignment
    objective are accumulated on auxiliary concept batches (forward passes
    inside :meth:`align`) and ``R`` is then updated with a Cayley-transform
    curvilinear search on the Stiefel manifold (Wen & Yin, 2013), exactly as
    in the original implementation. As in the original training script,
    alignment runs in eval mode (:meth:`align` takes care of this for the
    layer itself), so concept batches do not pollute the running whitening
    statistics.

    Typical training loop::

        cw = ConceptWhitening(in_features=64)
        # ... regular main-objective steps use cw(x) as a normalization layer

        # periodically, align axes on auxiliary concept datasets:
        for j, concept_batch in enumerate(concept_loaders):
            with cw.align(j):
                cw(concept_batch)   # accumulates alignment gradients
        cw.update_rotation_matrix()

    Args:
        in_features: Dimension of the input (and output) embeddings.
        num_iterations: Newton–Schulz iterations for whitening (T in IterNorm).
        eps: Ridge added to the covariance for numerical stability.
        momentum: Momentum for running whitening statistics and for the
            accumulated alignment gradient.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import ConceptWhitening
        >>>
        >>> cw = ConceptWhitening(in_features=16)
        >>> x = torch.randn(128, 16)
        >>> z = cw(x)             # whitened + rotated, shape (128, 16)
        >>> concepts = z[:, :4]   # concept activations, if axes 0-3 aligned
        >>> print(z.shape)
        torch.Size([128, 16])

    References:
        - Chen, Bei & Rudin. "Concept whitening for interpretable image
        recognition", Nature Machine Intelligence 2020. https://www.nature.com/articles/s42256-020-00265-z

        - Huang et al. "Iterative Normalization: Beyond Standardization towards
        Efficient Whitening", CVPR 2019. https://arxiv.org/abs/1904.03441

        - Wen & Yin. "A feasible method for optimization with orthogonality
        constraints", Mathematical Programming 2013. https://link.springer.com/article/10.1007/s10107-012-0584-1
    """

    def __init__(
        self,
        in_features: int,
        num_iterations: int = 5,
        eps: float = 1e-5,
        momentum: float = 0.05,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.eps = eps
        self.momentum = momentum

        # concept index whose alignment gradient is being accumulated;
        # -1 means normal operation (no accumulation)
        self.mode = -1

        d = in_features
        # running whitening statistics (IterNorm)
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_wm", torch.eye(d))
        # orthogonal rotation and alignment-gradient accumulators (CW)
        self.register_buffer("running_rot", torch.eye(d))
        self.register_buffer("sum_G", torch.zeros(d, d))
        self.register_buffer("counter", torch.full((d,), 1e-3))

    def _whiten(self, x: torch.Tensor) -> torch.Tensor:
        """ZCA-whiten a (m, d) batch via Newton–Schulz iterations."""
        d = x.size(-1)
        eye = torch.eye(d, dtype=x.dtype, device=x.device)
        if self.training:
            mean = x.mean(0)
            xc = x - mean
            sigma = xc.t() @ xc / x.size(0) + self.eps * eye
            # trace-normalize so the Newton–Schulz iteration converges
            inv_trace = sigma.diagonal().sum().reciprocal()
            sigma_n = sigma * inv_trace
            p = eye
            for _ in range(self.num_iterations):
                p = 1.5 * p - 0.5 * torch.matrix_power(p, 3) @ sigma_n
            wm = p * inv_trace.sqrt()  # = sigma^{-1/2}
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(
                    mean.detach(), alpha=self.momentum
                )
                self.running_wm.mul_(1 - self.momentum).add_(
                    wm.detach(), alpha=self.momentum
                )
        else:
            xc = x - self.running_mean
            wm = self.running_wm
        return xc @ wm  # wm is symmetric

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Whiten and rotate embeddings.

        Args:
            embeddings: Input of shape (..., in_features).

        Returns:
            torch.Tensor: Whitened, concept-aligned output of shape
            (..., in_features).
        """
        shape = embeddings.shape
        x = embeddings.reshape(-1, shape[-1])
        x_hat = self._whiten(x)

        if self.mode >= 0:
            # accumulate the gradient of the alignment loss
            # -mean(x_hat @ R[:, mode]) w.r.t. column `mode` of R
            with torch.no_grad():
                self.sum_G[:, self.mode] = (
                    self.momentum * (-x_hat.mean(0))
                    + (1.0 - self.momentum) * self.sum_G[:, self.mode]
                )
                self.counter[self.mode] += 1

        return (x_hat @ self.running_rot).reshape(shape)

    def align(self, axis_index: int):
        """
        Context manager: forward passes inside accumulate alignment
        gradients for output axis ``axis_index`` instead of behaving
        normally.

        The layer is switched to eval mode for the duration of the context
        (and restored on exit), so that concept batches are whitened with
        the running statistics without updating them — as in the original
        implementation.
        """
        if not 0 <= axis_index < self.in_features:
            raise ValueError(
                f"axis_index must be in [0, {self.in_features})."
            )
        layer = self

        class _AlignContext:
            def __enter__(self):
                self.was_training = layer.training
                layer.eval()
                layer.mode = axis_index

            def __exit__(self, *args):
                layer.mode = -1
                layer.train(self.was_training)

        return _AlignContext()

    @torch.no_grad()
    def update_rotation_matrix(self, num_updates: int = 2):
        """
        Update the rotation matrix from the accumulated alignment gradients
        using a Cayley-transform curvilinear search (Wen & Yin, 2013).

        Port of ``IterNormRotation.update_rotation_matrix`` from the official
        Concept Whitening repository, for a single whitening group.
        """
        G = self.sum_G / self.counter  # G[:, j] / counter[j]
        R = self.running_rot.clone()
        eye = torch.eye(R.size(0), dtype=R.dtype, device=R.device)
        c1, c2 = 1e-4, 0.9
        for _ in range(num_updates):
            tau, alpha, beta = 1000.0, 0.0, 1e8
            A = G @ R.t() - R @ G.t()  # skew-symmetric ascent direction
            dF_0 = -0.5 * (A ** 2).sum()
            # binary line search for a step satisfying Armijo–Wolfe
            cnt = 0
            while True:
                Q = torch.linalg.solve(eye + 0.5 * tau * A, eye - 0.5 * tau * A)
                Y_tau = Q @ R
                F_X = (G * R).sum()
                F_Y_tau = (G * Y_tau).sum()
                dF_tau = -torch.trace(
                    G.t()
                    @ torch.linalg.solve(eye + 0.5 * tau * A, A)
                    @ (0.5 * (R + Y_tau))
                )
                if F_Y_tau > F_X + c1 * tau * dF_0 + 1e-18:
                    beta = tau
                    tau = (beta + alpha) / 2
                elif dF_tau + 1e-18 < c2 * dF_0:
                    alpha = tau
                    tau = (beta + alpha) / 2
                else:
                    break
                cnt += 1
                if cnt > 500:
                    break
            Q = torch.linalg.solve(eye + 0.5 * tau * A, eye - 0.5 * tau * A)
            R = Q @ R
        self.running_rot.copy_(R)
        self.counter.fill_(1e-3)


class WhitenedEmbeddingToConcept(BaseConceptLayer):
    """
    Concept encoder based on Concept Whitening.

    Wraps a :class:`ConceptWhitening` layer and returns only the concept-aligned axes, 
    so it can be used as a CBM-style bottleneck encoder in PyC pipelines. Note that,
    unlike the paper's setting where the task head sees the full whitened
    embedding, this discards the residual ``in_embeddings - out_concepts``
    axes; use the wrapped :attr:`encoder` directly to keep them.

    Alignment is delegated to the wrapped layer::

        enc = WhitenedEmbeddingToConcept(in_embeddings=64, out_concepts=3)
        with enc.align(0):
            enc(concept_batch)
        enc.update_rotation_matrix()

    Args:
        in_embeddings: Number of input embedding features.
        out_concepts: Number of output concept representations.
        **kwargs: Additional keyword arguments for :class:`ConceptWhitening`.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import WhitenedEmbeddingToConcept
        >>>
        >>> enc = WhitenedEmbeddingToConcept(in_embeddings=16, out_concepts=4)
        >>> x = torch.randn(8, 16)
        >>> concepts = enc(x)
        >>> print(concepts.shape)
        torch.Size([8, 4])

    References:
        Chen, Bei & Rudin. "Concept whitening for interpretable image
        recognition", Nature Machine Intelligence 2020. https://www.nature.com/articles/s42256-020-00265-z
    """

    def __init__(
        self,
        in_embeddings: Union[int, Annotations],
        out_concepts: Union[int, Annotations],
        **kwargs,
    ):
        super().__init__(
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        if self.out_concepts_shape > self.in_embeddings_shape:
            raise ValueError(
                f"out_concepts ({self.out_concepts_shape}) cannot exceed "
                f"in_embeddings ({self.in_embeddings_shape})."
            )
        # (..., in_embeddings) -> (..., in_embeddings), sliced in forward
        self.encoder = ConceptWhitening(
            in_features=self.in_embeddings_shape,
            **kwargs,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embeddings into concept activations.

        Args:
            embeddings: Input embeddings of shape (..., in_embeddings).

        Returns:
            torch.Tensor: Concept activations of shape (..., out_concepts).
        """
        return self.encoder(embeddings)[..., : self.out_concepts_shape]

    def align(self, concept_index: int):
        """See :meth:`ConceptWhitening.align`."""
        if not 0 <= concept_index < self.out_concepts_shape:
            raise ValueError(
                f"concept_index must be in [0, {self.out_concepts_shape})."
            )
        return self.encoder.align(concept_index)

    def update_rotation_matrix(self, num_updates: int = 2):
        """See :meth:`ConceptWhitening.update_rotation_matrix`."""
        self.encoder.update_rotation_matrix(num_updates)
