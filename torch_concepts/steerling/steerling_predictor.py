"""Steerling concept-mixing layers.

The main layer in this module, ``MixFactorizedConceptExogenousToConcept``,
combines concept scores from ``SteerlingLatentToConcept`` with concept
embeddings returned by ``get_embeddings``. It produces either reconstructed
latent features or token/output logits through an optional linear head.
"""

import logging

import torch
import torch.nn as nn

from torch_concepts.nn.modules.low.base.layer import BasePredictor

logger = logging.getLogger(__name__)


class MixFactorizedConceptExogenousToConcept(BasePredictor):
    r"""Mix concept scores with Steerling concept embeddings.

    This layer is the PyC-compatible concept-embedding mixing step used after
    :class:`~torch_concepts.steerling.steerling_encoder.SteerlingLatentToConcept`.
    It deliberately accepts only tensors in ``forward(concepts, exogenous)`` so
    it can be inserted into a PyC graph without passing dictionaries, custom
    output objects, or hidden mutable state between layers.

    The layer reconstructs a latent feature vector by contracting concept
    scores with concept embeddings. ``concepts`` are expected to already be
    scores or activations. If ``add_linear_head=True``, the reconstructed
    latent is passed through an internal ``nn.Linear`` head. If
    ``add_linear_head=False``, :meth:`forward` returns the reconstructed latent
    directly.

    Two exogenous embedding layouts are supported:

    Dense layout, ``factorized=False``:
        ``concepts`` has shape ``(..., C)`` and ``exogenous`` has shape
        ``(C, D)`` or ``(..., C, D)``.  The mix is ``concepts @ exogenous``
        and returns ``(..., D)``.

    Packed factorized layout, ``factorized=True``:
        ``concepts`` has shape ``(..., C)`` and ``exogenous`` has shape
        ``(C + D, R)`` or ``(..., C + D, R)``.  The first ``C`` rows are the
        concept coefficients ``coef`` with shape ``(..., C, R)``; the remaining
        ``D`` rows are the basis matrix ``basis`` with shape ``(..., D, R)``.
        The mix is ``(concepts @ coef) @ basis.T`` and returns ``(..., D)``.

    Args:
        in_concepts: Number of input concepts ``C``.
        in_exogenous: Reconstructed latent/embedding dimension ``D``.
        out_concepts: Output dimension of the optional linear prediction head.
        factorized: Whether ``exogenous`` uses the packed factorized layout.
        add_linear_head: If ``True``, apply ``nn.Linear(D, out_concepts)`` in
            :meth:`forward`. If ``False``, return the mixed latent exactly like
            :meth:`mix`.
        bias: Bias flag for the optional linear prediction head.
        cardinalities: Optional concept cardinalities. Only Bernoulli concepts
            are supported here, so every cardinality must be ``1``.

    Example:
        Dense embeddings::

            mixer = MixFactorizedConceptExogenousToConcept(
                in_concepts=C,
                in_exogenous=D,
                out_concepts=V,
                factorized=False,
            )
            scores = concept_head(h)          # (..., C)
            embeddings = concept_head.get_embeddings(factorized=False)  # (C, D)
            token_logits = mixer(scores, embeddings)  # (..., V)

        Packed factorized embeddings::

            mixer = MixFactorizedConceptExogenousToConcept(
                in_concepts=C,
                in_exogenous=D,
                out_concepts=V,
                factorized=True,
            )
            scores = concept_head(h)          # (..., C)
            packed = concept_head.get_embeddings(factorized=True)  # (C + D, R)
            token_logits = mixer(scores, packed)  # (..., V)
    """

    def __init__(
        self,
        in_concepts: int,
        in_exogenous: int,
        out_concepts: int,
        cardinalities: list[int] | None = None,
        factorized: bool = False,
        add_linear_head: bool = True,
        bias: bool = False,
    ):
        """Initialize the mixer.

        The constructor does not store embeddings.  Embeddings are supplied as
        the ``exogenous`` tensor at call time, which keeps the layer stateless
        with respect to the concept encoder and allows dense/factorized
        embeddings to be routed through the PyC graph explicitly.

        Raises:
            ValueError: If ``cardinalities`` do not sum to ``in_concepts`` or
            if any cardinality is not ``1``.
        """
        super().__init__(
            in_concepts=in_concepts,
            in_exogenous=in_exogenous,
            out_concepts=out_concepts
        )
        cardinalities = [1] * in_concepts if cardinalities is None else cardinalities
        if sum(cardinalities) != in_concepts:
            raise ValueError(
                "Cardinalities must sum to in_concepts: "
                f"{sum(cardinalities)} != {in_concepts}."
            )
        if any(cardinality != 1 for cardinality in cardinalities):
            raise NotImplementedError(
                "MixFactorizedConceptExogenousToConcept only supports "
                "Bernoulli concepts for now; all cardinalities must be 1."
            )
        self.in_concepts = in_concepts
        self.in_exogenous = in_exogenous
        self.out_concepts = out_concepts
        self.cardinalities = cardinalities
        self.factorized = factorized
        self.add_linear_head = add_linear_head
        if add_linear_head:
            self.predictor = nn.Linear(in_exogenous, out_concepts, bias=bias)
        else:
            self.predictor = None

    def _as_factorized(self, exogenous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split packed factorized embeddings into ``coef`` and ``basis``.

        The packed convention is ``exogenous[..., :C, :] == coef`` and
        ``exogenous[..., C:, :] == basis``, where ``C`` is ``self.in_concepts``.
        The total row count must be ``C + D`` so the method can recover both
        tensors unambiguously from a single PyC-compatible tensor.

        Args:
            exogenous: Packed factorized tensor with shape
                ``(..., in_concepts + in_exogenous, rank)``.

        Returns:
            Tuple ``(coef, basis)`` with shapes ``(..., C, R)`` and
            ``(..., D, R)``.

        Raises:
            TypeError: If ``exogenous`` is not a tensor.
            ValueError: If the packed row dimension is not ``C + D``.
        """
        if isinstance(exogenous, torch.Tensor):
            expected_rows = self.in_concepts + self.in_exogenous
            if exogenous.shape[-2] != expected_rows:
                raise ValueError(
                    "Packed factorized exogenous has the wrong number of rows: "
                    f"{exogenous.shape[-2]} != {expected_rows} "
                    f"(in_concepts + in_exogenous)."
                )
            coef = exogenous[..., :self.in_concepts, :]
            basis = exogenous[..., self.in_concepts:, :]
            return coef, basis
        raise TypeError(
            "Factorized mixing expects exogenous to be a packed tensor with "
            "shape (..., in_concepts + in_exogenous, rank)."
        )

    def _mix_dense(self, concepts: torch.Tensor, exogenous: torch.Tensor) -> torch.Tensor:
        """Compute dense mixing, equivalent to ``concepts @ embeddings``.

        Args:
            concepts: Concept scores or activations with shape ``(..., C)``.
            exogenous: Dense embedding tensor with shape ``(C, D)`` or
                ``(..., C, D)``.

        Returns:
            Mixed latent tensor with shape ``(..., D)``.

        Raises:
            ValueError: If the concept dimension ``C`` does not match the
            embedding row dimension.
        """
        if concepts.shape[-1] != exogenous.shape[-2]:
            raise ValueError(
                "Concept and exogenous dimensions do not match: "
                f"{concepts.shape[-1]} != {exogenous.shape[-2]}."
            )
        exogenous = exogenous.to(concepts.dtype)
        return torch.matmul(concepts.unsqueeze(-2), exogenous).squeeze(-2)

    def _mix_factorized(self, concepts: torch.Tensor, exogenous: torch.Tensor) -> torch.Tensor:
        """Compute mixing from packed low-rank embedding components.

        This performs the same contraction as dense mixing without materializing
        the full ``(C, D)`` embedding matrix:

        ``mixed_rank = concepts @ coef``
        ``mixed = mixed_rank @ basis.T``

        Args:
            concepts: Concept scores or activations with shape ``(..., C)``.
            exogenous: Packed factorized tensor with shape
                ``(..., C + D, R)``.

        Returns:
            Mixed latent tensor with shape ``(..., D)``.

        Raises:
            TypeError: If ``exogenous`` is not a tensor.
            ValueError: If the packed tensor cannot be split into compatible
            ``coef`` and ``basis`` components.
        """
        coef, basis = self._as_factorized(exogenous)
        if concepts.shape[-1] != coef.shape[-2]:
            raise ValueError(
                "Concept and factorized coefficient dimensions do not match: "
                f"{concepts.shape[-1]} != {coef.shape[-2]}."
            )

        coef = coef.to(concepts.dtype)
        basis = basis.to(concepts.dtype)
        rank = coef.shape[-1]
        if basis.shape[-1] == rank:
            basis_t = basis.transpose(-2, -1)
        elif basis.shape[-2] == rank:
            basis_t = basis
        else:
            raise ValueError(
                "Basis rank dimension does not match coefficient rank: "
                f"basis shape {tuple(basis.shape)}, rank {rank}."
            )

        mixed_rank = torch.matmul(concepts.unsqueeze(-2), coef).squeeze(-2)
        return torch.matmul(mixed_rank.unsqueeze(-2), basis_t).squeeze(-2)

    def mix(
        self,
        concepts: torch.Tensor,
        exogenous: torch.Tensor,
    ) -> torch.Tensor:
        """Return reconstructed latent features before the optional linear head.

        Args:
            concepts: Concept scores or activations with shape ``(..., C)``.
            exogenous: Dense embedding tensor ``(..., C, D)`` when
                ``factorized=False``, or packed factorized tensor
                ``(..., C + D, R)`` when ``factorized=True``.

        Returns:
            Mixed latent tensor with shape ``(..., D)``.

        Raises:
            ValueError: If the mixed latent's last dimension is not
            ``in_exogenous``.
        """
        if self.factorized:
            mixed = self._mix_factorized(concepts, exogenous)
        else:
            mixed = self._mix_dense(concepts, exogenous)
        if mixed.shape[-1] != self.in_exogenous:
            raise ValueError(
                "Mixed exogenous dimension does not match the original input: "
                f"{mixed.shape[-1]} != {self.in_exogenous}."
            )
        return mixed

    def forward(
        self,
        concepts: torch.Tensor,
        exogenous: torch.Tensor,
    ) -> torch.Tensor:
        """Mix concepts with embeddings and optionally apply the linear head.

        This is the public PyTorch/PyC layer entrypoint.  It has exactly two
        tensor inputs: concept scores and exogenous embeddings.  The output is
        either the mixed latent ``(..., D)`` or, when ``add_linear_head=True``,
        the linear-head output ``(..., out_concepts)``.

        Args:
            concepts: Concept scores or activations with shape ``(..., C)``.
            exogenous: Dense or packed factorized embedding tensor.  See the
                class docstring for the exact layout.

        Returns:
            ``(..., out_concepts)`` if the linear head is enabled; otherwise
            ``(..., D)``.
        """
        c_mix = self.mix(concepts, exogenous)
        if self.add_linear_head and self.predictor is not None:
            return self.predictor(c_mix)
        return c_mix
