"""Steerling concept layers.

``SteerlingEmbeddingToConcept`` wraps the official Steerling ``ConceptHead`` and
adapts it to the PyC ``BaseConceptLayer`` interface, mapping transformer hidden
states to dense concept logits and exposing the concept embeddings.

``SteerlingConceptEmbeddingMixer`` is the matching feature-reconstruction step: a
plain ``nn.Module`` that turns those concept logits and embeddings back into a
latent feature (the sigmoid-weighted, top-k sum of embeddings that feeds the
language-model head), reusing the live ``ConceptHead`` for fidelity.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from torch_concepts.nn.modules.low.base.layer import BaseConceptLayer

logger = logging.getLogger(__name__)


def _import_concept_head():
    try:
        import steerling.models.interpretable.concept_head as _ch
        from steerling.models.interpretable.concept_head import ConceptHead
        # # Allow dense logits for all heads (including >50k unknown concepts).
        # _ch.LARGE_CONCEPT_THRESHOLD = float("inf")
        return ConceptHead
    except ImportError as exc:
        raise ImportError(
            "SteerlingEmbeddingToConcept requires the `steerling` package. "
            "Install it with: pip install steerling  (requires Python >= 3.13)"
        ) from exc


class SteerlingEmbeddingToConcept(BaseConceptLayer):
    """PyC wrapper around Steerling's embedding-to-concept head.
    """

    def __init__(
        self,
        in_embeddings: int = 4096,
        out_concepts: int = 33732,
        **kwargs
    ):
        super().__init__(in_embeddings=in_embeddings, out_concepts=out_concepts)

        ConceptHead = _import_concept_head()

        # Initialize the ConceptHead with the resolved kwargs
        self.head: nn.Module = ConceptHead(
            n_embd=in_embeddings, 
            n_concepts=out_concepts, 
            **kwargs
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embeddings(
        self,
        idxs: torch.Tensor | list[int] | int | None = None,
        cat_factorized: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return concept embeddings for downstream mixing.

        Args:
            idxs: Concept indices to select. ``None`` returns all
                ``out_concepts`` (padding rows excluded).
            cat_factorized: Only affects the factorized layout. When ``True``,
                return ``coef`` and ``basis`` packed into a single tensor
                (concatenated along the concept axis) so a downstream mixer can
                unpack them; when ``False`` (default) return them as a tuple.

        Returns:
            Dense layout: ``(out_concepts, in_embeddings)``.
            Factorized layout, ``cat_factorized=False``: ``(coef, basis)`` with
            shapes ``(out_concepts, rank)`` and ``(in_embeddings, rank)``; the
            full embedding is ``coef @ basis.T``.
            Factorized layout, ``cat_factorized=True``: packed
            ``(out_concepts + in_embeddings, rank)`` where rows ``[:out_concepts]``
            are ``coef`` and rows ``[out_concepts:]`` are ``basis``.
        """
        idxs = slice(self.out_concepts) if idxs is None else idxs
        if self.head.factorize:
            coef = self.head.embedding_coef.weight[idxs]
            basis = self.head.embedding_basis.weight
            if cat_factorized:
                return torch.cat([coef, basis], dim=0)
            return coef, basis
        return self.head.concept_embedding.weight[idxs]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dense raw concept logits.

        Three scoring paths, by head config:

        - **Linear, dense** (``use_attention=False, factorize=False``): a single
          ``concept_predictor`` matmul.
        - **Linear, factorized** (``use_attention=False, factorize=True``): scored in
          rank-``r`` space as ``(latent @ down.T) @ up.T``, so the full ``(C, D)``
          predictor weight is **never materialized** — this is what keeps the large
          unknown head cheap.
        - **Attention** (``use_attention=True``): scores the query against the concept
          embeddings via ``blocked_logits``. This path **materializes the dense
          ``(C, D)`` embedding** (``_get_embedding_weight``) even when ``factorize=True``,
          because attention scores against the embeddings directly and the rank-space
          trick does not apply. For a large factorized head this re-introduces the
          ``(C, D)`` allocation the factorized-linear path avoids. Not exercised by the
          current hub config (``use_attention=False`` for both heads), but be aware of
          the cost if you enable it.

        Precision note: logits are computed and clamped in the model dtype (bf16),
        not fp32. Steerling computes logits and the subsequent sigmoid in fp32 and
        casts to bf16 only at the final weighted sum (see
        ``ConceptHead.linear_features_topk_factorized``). We deliberately stay in
        bf16: the downstream sigmoid happens externally in bf16 anyway, so an fp32
        clamp here would just allocate a transient fp32 ``(B, T, C)`` copy (which
        grows with sequence length) for a precision gain thrown away one step later.
        The resulting divergence from Steerling is at the bf16-rounding level
        (~1e-3 relative), below the model's own bf16 quantization noise.

        To match Steerling bit-for-bit instead, you would: (1) return fp32 logits
        here, upcasting before the matmul — ``(h_r.float() @ up_w.float().T)`` /
        ``(latent.float() @ W.float().T)``; (2) apply the external sigmoid in fp32;
        (3) in the mixer, ``torch.topk`` on the fp32 weights and cast to bf16 only at
        the weighted-sum einsum. Even then the dense (known) head's top-k ranking can
        differ at the k-boundary, because Steerling ranks on a bf16 matmul in pass 1
        but weights from an fp32 einsum in pass 2 — two logit computations the
        single-logit decomposition here does not reproduce.

        Args:
            latent: Hidden states with shape ``(batch, in_latent)`` or
                ``(batch, sequence, in_latent)``.

        Returns:
            Dense logits with shape ``(batch, out_concepts)`` for 2-D input or
            ``(batch, sequence, out_concepts)`` for 3-D input.
        """
        squeeze = latent.dim() == 2
        if squeeze:
            latent = latent.unsqueeze(1)

        if self.head.use_attention:
            embeddings = self.head._get_embedding_weight()[:self.out_concepts]
            query = self.head.concept_query_projection(latent)
            logits = self.head.blocked_logits(
                query,
                embeddings,
                block_size=int(self.head.block_size),
            ).to(latent.dtype)
        elif self.head.factorize:
            # Score in rank-r space instead of materializing the full (C, D) predictor
            # weight: logits = latent @ (up @ down).T = (latent @ down.T) @ up.T.
            down_w = self.head.predictor_down.weight              # (r, D)
            up_w = self.head.predictor_up.weight[:self.out_concepts]  # (C, r)
            h_r = latent @ down_w.T                               # (B, T, r)
            logits = (h_r @ up_w.T).clamp(-15, 15)                # (B, T, C)
        else:
            logits = self.head.concept_predictor(latent)[..., :self.out_concepts].clamp(-15, 15)

        return logits.squeeze(1) if squeeze else logits


class SteerlingConceptEmbeddingMixer(nn.Module):
    r"""Reconstruct a latent feature from concept logits and concept embeddings.

    This is the feature-reconstruction half of Steerling's ``ConceptHead``: given
    concept **weights** (activation applied externally) it computes
    ``features = topk(weights) @ embeddings`` exactly as the head does on its feature
    path. It outputs a latent embedding (not concepts), so it is a plain
    ``nn.Module`` rather than a PyC concept layer.

    The module is stateless w.r.t. parameters — embeddings are supplied at call
    time — but holds a reference to the source ``ConceptHead`` to (a) reuse its
    ``blocked_mix`` weighted sum and (b) read its config flags (``factorize``,
    ``topk_features``, ``topk``).

    Two embedding layouts are accepted in :meth:`forward`:

    Dense (``head.factorize == False``):
        ``embeddings`` has shape ``(C, D)``.

    Packed factorized (``head.factorize == True``):
        ``embeddings`` is the packed tensor ``(C + D, R)`` from the encoder's
        ``embeddings(cat_factorized=True)``: rows ``[:C]`` are the coefficients
        ``coef`` ``(C, R)`` and rows ``[C:]`` are the basis ``(D, R)``. The dense
        embedding ``E = coef @ basis.T`` is reconstructed before mixing.

    Args:
        n_concepts: Number of concepts ``C`` (used to split the packed factorized
            embedding).
        head: The Steerling ``ConceptHead`` whose ``blocked_mix`` and top-k config
            are reused.
    """

    def __init__(self, n_concepts: int, head: nn.Module):
        super().__init__()
        self.n_concepts = n_concepts
        self.head = head

    def _feature_k(self) -> int | None:
        """Number of concepts that build the feature.

        Uses ``topk_features`` (what the head's feature path sums), not the loss-only
        ``topk`` that ``ConceptHead.topk_with_cutoff`` hardcodes. ``None`` means no
        top-k (sum over all concepts).
        """
        if self.head.topk_features is not None:
            return self.head.topk_features  
        else:
            return self.head.topk

    def forward(
        self,
        concepts: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the latent feature from concept weights and embeddings.

        Sparse top-k mix: select the top-k concept weights, gather only those ``k``
        embeddings, and contract — so the cost is ``O(k)`` per token, not ``O(C)``.
        For the factorized layout this also avoids ever materializing the dense
        ``(C, D)`` embedding, since only the ``k`` selected ``coef`` rows are gathered
        before the single shared-basis projection.

        This gather path is independent of the head's ``use_attention`` setting — it
        stays ``O(k)`` and never builds ``(C, D)`` regardless. (The
        ``use_attention=True`` cost lives upstream, in
        :meth:`SteerlingEmbeddingToConcept.forward`, which materializes the dense
        ``(C, D)`` embedding to score the query; see that method's docstring.)

        Args:
            concepts: Concept **weights** (activation already applied) with shape
                ``(B, T, C)`` (or ``(B, C)``).
            embeddings: Dense ``(C, D)`` or packed factorized ``(C + D, R)`` tensor.

        Returns:
            Reconstructed latent features with shape ``(B, T, D)`` (or ``(B, D)``).
        """
        squeeze = concepts.dim() == 2
        if squeeze:
            concepts = concepts.unsqueeze(1)

        factorize = self.head.factorize
        if factorize:
            coef = embeddings[: self.n_concepts]    # (C, R)
            basis = embeddings[self.n_concepts :]   # (D, R)
            out_dtype = coef.dtype
        else:
            E = embeddings                          # (C, D)
            out_dtype = E.dtype

        weights = concepts.to(out_dtype)
        C = weights.size(-1)

        k = self._feature_k()
        if k is not None and k < C:
            # Sparse path: keep only the top-k weights and gather their embeddings.
            w_sel, topi = torch.topk(weights, k, dim=-1)   # (B, T, k)
            if factorize:
                coef_sel = coef[topi]                       # (B, T, k, R)
                wc = torch.einsum("btk,btkr->btr", w_sel, coef_sel)
                features = wc @ basis.T                     # (B, T, D)
            else:
                E_sel = E[topi]                             # (B, T, k, D)
                features = torch.einsum("btk,btkd->btd", w_sel, E_sel)
        else:
            # No top-k: dense weighted sum over all concepts.
            if factorize:
                features = (weights @ coef) @ basis.T
            else:
                features = weights @ E

        return features.squeeze(1) if squeeze else features



class SteerlingResidualCorrection(nn.Module):
    r"""Aggregate parent tensors into an additive-reconstruction correction.

    This is an **aggregate operation** for a PGM factor: it consumes the ordered
    set of parent tensors ``[target, *parts]`` directly — no concatenation — and
    returns a single correction :math:`\varepsilon`. The first element is the
    reconstruction ``target``; the remaining ``n_terms`` elements are the ``parts``
    whose sum approximates it. The downstream node forms
    ``h_bar = sum(parts) + epsilon``, so the choice of :math:`\varepsilon` controls
    both the *value* of ``h_bar`` and how gradient flows back to each part.

    The Steerling use: ``target`` is the backbone hidden state ``h`` and the parts
    are the concept reconstructions (``k_hat`` and optionally ``u_hat``). With
    ``residual_mode="block_parts"`` this makes ``h_bar == h`` exactly while routing
    gradient only through the backbone — reproducing the upstream
    ``composed = unk_for_lm + known_features (+ epsilon)`` path.

    Two orthogonal mechanisms combine into :math:`\varepsilon`:

    1.  A *target-residual* term selected by ``residual_mode``:

        - ``"block_parts"``: :math:`\varepsilon = \text{target} - \sum_i p_i`.
          In ``h_bar = sum(parts) + epsilon`` each part appears as :math:`+1`
          (downstream sum) and :math:`-1` (inside :math:`\varepsilon`); they cancel,
          so ``h_bar == target`` and gradient flows only through ``target``.
        - ``"keep_parts"``: :math:`\varepsilon = \text{target} - \sum_i p_i.\mathrm{detach}()`.
          Same value (``h_bar == target``), but detaching the parts inside
          :math:`\varepsilon` leaves their downstream gradient intact (coefficient
          :math:`+1`) — the natural reconstruction gradient.
        - ``"off"``: :math:`\varepsilon = 0`. ``h_bar == sum(parts)``. ``target`` is
          still accepted (and ignored) so the aggregate's input set is unchanged
          across modes.

    2.  Optional *stop-gradient* terms: for each index ``i`` in ``stop_grad_parts``,
        :math:`p_i.\mathrm{detach}() - p_i` is added to :math:`\varepsilon`. This adds
        zero to the value but cancels the gradient through :math:`p_i` downstream.

    Args:
        input_size (int): Feature size of ``target`` and each part.
        n_terms (int): Number of reconstruction parts (excludes ``target``).
        residual_mode (str): ``"block_parts"``, ``"keep_parts"``, or ``"off"``.
            Defaults to ``"block_parts"``.
        stop_grad_parts (sequence of int, optional): Indices into ``parts`` that
            receive a stop-gradient term. Defaults to none.

    Input / output:
        - Input: an ordered sequence ``[target, *parts]`` of ``n_terms + 1`` tensors,
          each of shape ``(*, input_size)``.
        - Output: :math:`\varepsilon` of shape ``(*, input_size)``.

    Behavior table (``n_terms=2``), where ``h_bar = sum(parts) + epsilon``:

    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+
    | ``residual_mode`` | ``stop_grad_parts``| grad ``part[0]``| grad ``part[1]``| value of ``h_bar``                    |
    +===================+====================+=================+=================+=======================================+
    | ``"block_parts"`` | ``()``             | 0               | 0               | ``target``                            |
    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+
    | ``"keep_parts"``  | ``()``             | 1               | 1               | ``target``                            |
    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+
    | ``"keep_parts"``  | ``(1,)``           | 1               | 0               | ``target``                            |
    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+
    | ``"keep_parts"``  | ``(0,)``           | 0               | 1               | ``target``                            |
    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+
    | ``"off"``         | ``()``             | 1               | 1               | ``part[0] + part[1]`` (target unused) |
    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+
    | ``"off"``         | ``(1,)``           | 1               | 0               | ``part[0] + part[1].detach()``        |
    +-------------------+--------------------+-----------------+-----------------+---------------------------------------+

    For ``n_terms=1`` only the ``stop_grad_parts=()`` rows apply
    (``"block_parts"`` → grad 0, ``"keep_parts"`` → grad 1, ``"off"`` → grad 1 with
    ``h_bar == part``).

    Example:
        >>> # ResNet-style residual: y = part + (target - part) == target
        >>> agg = SteerlingResidualCorrection(input_size=4, n_terms=1, residual_mode="block_parts")
        >>> target = torch.randn(2, 4)
        >>> part = torch.randn(2, 4, requires_grad=True)
        >>> epsilon = agg([target, part])
        >>> y = part + epsilon
        >>> torch.allclose(y, target)
        True
    """

    _RESIDUAL_MODES = ("block_parts", "keep_parts", "off")

    def __init__(
        self,
        input_size: int,
        n_terms: int,
        residual_mode: str = "block_parts",
        stop_grad_parts=None,
    ):
        super().__init__()
        if n_terms < 1:
            raise ValueError(f"n_terms must be >= 1, got {n_terms}.")
        if residual_mode not in self._RESIDUAL_MODES:
            raise ValueError(
                f"residual_mode must be one of {self._RESIDUAL_MODES}, got {residual_mode!r}."
            )
        self.input_size = input_size
        self.n_terms = n_terms
        self.residual_mode = residual_mode
        self.stop_grad_parts = tuple(stop_grad_parts) if stop_grad_parts else ()
        for idx in self.stop_grad_parts:
            if not 0 <= idx < n_terms:
                raise ValueError(
                    f"stop_grad_parts index {idx} out of range [0, {n_terms})."
                )

    def forward(self, inputs) -> torch.Tensor:
        """Aggregate the parent set ``[target, *parts]`` into :math:`\\varepsilon`.

        Args:
            inputs: Ordered sequence of ``n_terms + 1`` tensors. ``inputs[0]`` is the
                reconstruction ``target``; ``inputs[1:]`` are the ``parts``.

        Returns:
            The correction :math:`\\varepsilon` of shape ``(*, input_size)``.
        """
        target, *parts = inputs
        if len(parts) != self.n_terms:
            raise ValueError(
                f"Expected target + {self.n_terms} parts, got {len(inputs)} inputs."
            )
        if self.residual_mode == "block_parts":
            epsilon = target - sum(parts)
        elif self.residual_mode == "keep_parts":
            epsilon = target - sum(p.detach() for p in parts)
        else:  # "off"
            epsilon = torch.zeros_like(target)
        for idx in self.stop_grad_parts:
            epsilon = epsilon + (parts[idx].detach() - parts[idx])
        return epsilon