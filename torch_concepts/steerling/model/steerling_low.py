
"""Low-level Steerling model assembled from explicit PyTorch modules.

``SteerlingLowLevelModel`` wires the backbone, known/unknown concept heads,
concept-embedding mixers, and language-model head into the same data flow used
by Steerling:

``input_ids -> hidden -> concept logits -> concept features -> token logits``.

The model returns intermediate tensors as a dictionary so callers can inspect
known concepts, unknown concepts, and reconstructed latent features.
"""

import logging
import gc
from contextlib import contextmanager

import torch
import torch.nn as nn

from ..steerling_backbone import CausalDiffusionTextBackbone
from ..steerling_configs import (
    DEFAULT_MODEL_ID,
    SteerlingConfigSource,
    resolve_steerling_configs,
)
from ..steerling_layers import (
    SteerlingEmbeddingToConcept,
    SteerlingConceptEmbeddingMixer,
    SteerlingResidualCorrection
)
from ...nn import LinearEmbeddingToConcept 
from ..steerling_utils import (
    load_steerling_weights,
    _load_lm_head_weights,
    load_steerling_concept_names,
    top_concepts,
)

logger = logging.getLogger(__name__)

# Steerling components that can be pretrained-loaded and/or frozen.
_ALL_COMPONENTS = ("backbone", "known_head", "unknown_head", "lm_head")


@contextmanager
def _default_dtype(dtype: torch.dtype):
    """Temporarily set the global default float dtype, then restore it.

    Used to build the (large) Steerling modules directly in their target dtype
    so that loading the bf16 checkpoint does not transiently allocate a float32
    copy of the 8B backbone.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


class SteerlingLowLevelModel(nn.Module):
    """Low-level Steerling concept-bottleneck language model.

    Instantiates and wires all low-level Steerling modules internally.
    Use :meth:`forward` for end-to-end prediction or the convenience methods
    for intermediate representations.

    Args:
        pretrained_components: List of component names to load from the
            Steerling Hub checkpoint, a subset of ``"backbone"``,
            ``"known_head"``, ``"unknown_head"``, ``"lm_head"``. ``None``
            loads nothing.
        freeze_components: List of component names to freeze after loading;
            ``None`` freezes nothing.
        use_unknown: Whether to build the unsupervised "unknown" concept head.
            When ``False``, the wrapper mirrors upstream's no-unknown-head
            path (``composed ≈ hidden``) so the LM head sees the raw
            backbone state.
        model_id: Hugging Face model id or local path for Steerling weights.
        config_source: Config source passed to ``resolve_steerling_configs``.
        n_concepts, n_unknown_concepts, concept_dim, use_epsilon_correction:
            Common concept-config knobs.  ``None`` (the default) reads the
            value from the resolved config; any non-``None`` value wins over
            ``concept_config_overrides``.
        model_config_overrides: Optional model config overrides.
        concept_config_overrides: Optional concept config overrides — the
            escape hatch for any concept key without a dedicated kwarg (e.g.
            ``factorize_unknown``, ``use_attention_known``).  Passing top-k
            keys here (``topk_known``, ``unknown_topk``, ...) raises
            :class:`NotImplementedError` — top-k inference is not implemented
            yet.
        dtype: dtype the modules are built in. ``None`` (default) resolves to
            the config's ``torch_dtype``, then to ``bfloat16`` (Steerling ships
            bf16 weights), so loading does not allocate a transient float32 copy.

    Modules are constructed and loaded on CPU; move the model afterward with
    the standard ``model.to(device)`` pattern.

    Example::

        model = SteerlingLowLevelModel().to("cuda")

        # End-to-end: tokens → concept bottleneck → next-token logits
        out = model(input_ids)
        logits = out["out_tokens"]                  # (B, T, vocab)

        # Just concept activations
        concepts = model.encode_concepts(input_ids) # (B, T, n_known)

        # Concept-based hidden-state reconstruction
        h_bar = out["reconstructed_latent"]         # (B, T, D)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        config_source: SteerlingConfigSource = "hub",  # keep 'hub' to match model_id
        pretrained_components: list[str] | None = _ALL_COMPONENTS,
        freeze_components: list[str] | None = _ALL_COMPONENTS,
        use_epsilon_correction: bool = False,
        # ---------- individual config overrides ----------
        # (win over default config from 'config_source') 
    ):
        super().__init__()
        self.model_id = model_id
        self.config_source = config_source
        self.pretrained_components = list(pretrained_components or [])
        self.freeze_components = list(freeze_components or [])
        self.use_epsilon_correction = use_epsilon_correction

        # fetch steerling configs (hub JSON or local file)
        self.model_cfg, self.concept_cfg, self.other_cfg = resolve_steerling_configs(
            config_source=config_source,
            model_id=model_id,
        )
        
        self.vocab_size = self.other_cfg.get("vocab_size")

        # put here the code to override the config with the kwargs passed to the constructor
        #
        #
        #

        # Build the Steerling modules directly in the target dtype.
        # Steerling ships bf16 weights; constructing in float32 and loading the
        # bf16 checkpoint would transiently allocate a float32 copy of the 8B
        # backbone (~2x peak memory) plus a redundant down-cast afterwards.
        dtype = getattr(torch, self.other_cfg.get("torch_dtype"))
        with _default_dtype(dtype):
            self._build_modules()

        # Load and freeze pretrained weights.
        self._load_steerling_weights(model_id, pretrained=self.pretrained_components)
        self._freeze_steerling_weights(freeze=self.freeze_components)


    def _build_modules(self,) -> None:
        """Instantiate, load, and freeze all Steerling submodules.

        Called from :meth:`__init__` inside a :func:`_default_dtype` context so
        every parameter is created in ``self.dtype`` (bf16 by default).
        """
        # Backbone: tokens -> hidden states.
        self.backbone = CausalDiffusionTextBackbone(
            config=self.model_cfg,
            vocab_size=self.vocab_size,
            tokenizer_model_id=self.model_id,
        )

        # Concept encoders.
        # known (supervised): h → k dense concept logits
        # dense + linear + topk
        self.known_head = SteerlingEmbeddingToConcept(
            in_embeddings=self.model_cfg['n_embd'],
            out_concepts=self.concept_cfg['n_concepts'],
            concept_dim=self.concept_cfg['concept_dim'],
            is_unknown=False,
            use_attention=self.concept_cfg['use_attention_known'],
            topk=self.concept_cfg['topk_known'],
            topk_features=self.concept_cfg['topk_known_features'],
            block_size=self.concept_cfg['block_size'],
            pad_multiple=self.concept_cfg['pad_multiple'],
            store_unknown_weights=False,
            apply_topk_to_unknown=False,
            topk_on_logits=False, # differently from Steerling original implementation, we don't support topk on logits as the activation is a inference concern.
        )

        # Unknown concept head (optional)
        # factorized + linear + topk
        if self.concept_cfg['use_unknown']:
            if self.concept_cfg['n_unknown_concepts'] is None:
                raise ValueError("n_unknown_concepts must be set when use_unknown=True")

            self.unknown_head: SteerlingEmbeddingToConcept | None = SteerlingEmbeddingToConcept(
                in_embeddings=self.model_cfg['n_embd'],
                out_concepts=self.concept_cfg['n_unknown_concepts'],
                concept_dim=self.concept_cfg['concept_dim'],
                is_unknown=True,
                use_attention=self.concept_cfg['use_attention_unknown'],
                topk=self.concept_cfg['unknown_topk'],
                block_size=self.concept_cfg['block_size'],
                pad_multiple=self.concept_cfg['pad_multiple'],
                store_unknown_weights=False,
                apply_topk_to_unknown=self.concept_cfg['apply_topk_to_unknown'],
                topk_on_logits=False, # differently from Steerling original implementation, we don't support topk on logits as the activation is a inference concern.
                factorize=self.concept_cfg['factorize_unknown'],
                factorize_rank=self.concept_cfg['factorize_rank'],
            )
        else:
            self.unknown_head = None

        # Concept-logit + embedding mixing: reconstruct the latent feature from the
        # concept logits and embeddings, reusing each head's blocked_mix / top-k config.
        self.known_concept_mixer = SteerlingConceptEmbeddingMixer(
            n_concepts=self.concept_cfg['n_concepts'],
            head=self.known_head.head,
        )
        if self.unknown_head is not None:
            self.unknown_concept_mixer = SteerlingConceptEmbeddingMixer(
                n_concepts=self.concept_cfg['n_unknown_concepts'],
                head=self.unknown_head.head,
            )
        else:
            self.unknown_concept_mixer = None

        # Epsilon correction term for h_bar = k_hat + (u_hat) + epsilon.
        # `use_epsilon` toggles whether the residual recovers `h` exactly
        # (`block_parts`) or leaves `h_bar` as the pure concept
        # reconstruction (`off`).  `stop_grad_parts` detaches `u_hat`
        # specifically when unknown concepts are present and epsilon is
        # off, so the unknown branch doesn't backprop into the backbone.
        #   (has_unknown, use_epsilon)  →  (mode,           stop_grad_parts)
        #   (True,  True)               →  ("block_parts",  ())
        #   (True,  False)              →  ("off",          (1,))  stop-grad on u_hat
        #   (False, True)               →  ("block_parts",  ())
        #   (False, False)              →  ("off",          ())   h_bar = k_hat
        self.epsilon_correction = SteerlingResidualCorrection(
            input_size=self.concept_cfg['concept_dim'],
            n_terms=2 if self.unknown_head is not None else 1,
            residual_mode="block_parts" if self.use_epsilon_correction else "off",
            stop_grad_parts=(1,) if self.unknown_head is not None and not self.use_epsilon_correction else (),
        )

        # LM head: reconstructed latent → next-token logits
        # Alias the backbone's LM head.  Under upstream ``weight_sharing=True``
        # (Steerling and hub default) its weight is tied to ``transformer.tok_emb.weight``
        # via ``_tie_weights``, so a single underlying Parameter is shared by
        # ``self.backbone.transformer.lm_head``, and the backbone's input embedding.
        assert self.model_cfg['n_embd'] == self.concept_cfg['concept_dim'], (
            "Steerling constrains concept_dim == n_embd (both 4096 by default)."
        )
        self.lm_head = LinearEmbeddingToConcept(
            in_embeddings=self.concept_cfg['concept_dim'], 
                # Steerling constrains concept_dim == n_embd (both 4096 by default), so either key
                # works here. We use concept_dim to stay in the concept-feature space.
            out_concepts=self.vocab_size,
            bias=False
        )


    def _load_steerling_weights(self, model_id: str, pretrained: list | None = None):
        """Load selected pretrained Steerling components."""
        pretrained = pretrained or []

        if "backbone" in pretrained:
            backbone_sd = load_steerling_weights(model_id, "transformer", device="cpu")
            # When the backbone uses weight sharing (the Steerling default),
            # the checkpoint stores only `tok_emb.weight`; the transformer's
            # `lm_head.weight` is the same tensor at runtime.  Treat that
            # specific missing key as expected.
            weight_sharing = bool(self.model_cfg.get("weight_sharing", False))
            expected_missing: tuple[str, ...] = (
                ("lm_head.weight",) if weight_sharing else ()
            )
            self._load_state_dict(
                self.backbone.transformer,
                backbone_sd,
                "backbone",
                expected_missing=expected_missing,
            )
            self._discard_state_dict(backbone_sd)
            if weight_sharing:
                # Re-tie defensively in case the upstream module rebuilt
                # `lm_head.weight` during construction without sharing.
                transformer = self.backbone.transformer
                if (
                    hasattr(transformer, "lm_head")
                    and hasattr(transformer, "tok_emb")
                    and transformer.lm_head.weight is not transformer.tok_emb.weight
                ):
                    transformer.lm_head.weight = transformer.tok_emb.weight
            logger.info("Loaded pretrained weights into backbone.")

        if "known_head" in pretrained:
            known_sd = load_steerling_weights(model_id, "known_head", device="cpu")
            self._load_state_dict(self.known_head.head, known_sd, "known_head")
            self._discard_state_dict(known_sd)
            logger.info("Loaded pretrained weights into known concept head.")

        if "unknown_head" in pretrained:
            if self.unknown_head is None:
                logger.info("Skipped unknown concept head weights because use_unknown=False.")
            else:
                unknown_sd = load_steerling_weights(model_id, "unknown_head", device="cpu")
                self._load_state_dict(self.unknown_head.head, unknown_sd, "unknown_head")
                self._discard_state_dict(unknown_sd)
                logger.info("Loaded pretrained weights into unknown concept head.")

        if "lm_head" in pretrained:
            weight_sharing = bool(self.model_cfg.get("weight_sharing", False))
            if weight_sharing and "backbone" in pretrained:
                # Under weight sharing the checkpoint omits `lm_head.weight` (tied
                # to `tok_emb.weight`). Tie our LM head to the backbone's shared
                # lm_head tensor so it uses the trained weights, not its init.
                self.lm_head.encoder.weight = self.backbone.transformer.lm_head.weight
                logger.info("Tied LM head weight to the backbone's shared lm_head.")
            else:
                lm_head_sd = _load_lm_head_weights(model_id, device="cpu")
                self._load_state_dict(self.lm_head, lm_head_sd, "lm_head")
                self._discard_state_dict(lm_head_sd)
                logger.info("Loaded pretrained weights into LM head.")


    @staticmethod
    def _load_state_dict(
        module: nn.Module,
        state_dict: dict,
        name: str,
        *,
        allow_partial: bool = False,
        expected_missing: tuple[str, ...] | list[str] = (),
    ) -> None:
        """Strict load by default; accept missing/unexpected keys only when
        explicitly requested via ``allow_partial`` (or for keys that are
        documented as legitimately missing, via ``expected_missing``).

        A non-empty missing/unexpected list almost always means the wrapper
        was built with a config that doesn't match the checkpoint
        (factorize_unknown, use_attention_*).  Failing loudly is better than
        silent weight corruption.

        ``expected_missing`` covers known exceptions (e.g. weight-tied
        ``lm_head.weight`` inside the backbone, where the checkpoint stores
        only ``tok_emb.weight``).
        """
        incompatible = module.load_state_dict(state_dict, strict=False)
        expected = set(expected_missing)
        unexpected_missing = [k for k in incompatible.missing_keys if k not in expected]
        if not (unexpected_missing or incompatible.unexpected_keys):
            return
        if allow_partial:
            logger.warning(
                "Loaded %s with missing keys=%s and unexpected keys=%s.",
                name,
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )
            return
        raise RuntimeError(
            f"Loading pretrained weights for {name!r} produced a key mismatch "
            "with the wrapped module. This usually means the wrapper config "
            "(e.g. factorize_unknown, use_attention_*) does not match the "
            "checkpoint. "
            f"missing_keys={unexpected_missing}, "
            f"unexpected_keys={incompatible.unexpected_keys}. "
            "Pass `allow_partial=True` to bypass for debugging."
        )

    @staticmethod
    def _discard_state_dict(state_dict: dict) -> None:
        """Release loaded checkpoint tensors after they are copied to modules."""
        state_dict.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def _freeze_steerling_weights(self, freeze: list | None = None):
        """Freeze selected Steerling components."""
        freeze = freeze or []

        if "backbone" in freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Froze backbone parameters.")

        if "known_head" in freeze:
            for p in self.known_head.parameters():
                p.requires_grad = False
            logger.info("Froze known concept head parameters.")

        if "unknown_head" in freeze:
            if self.unknown_head is None:
                logger.info("Skipped freezing unknown concept head because use_unknown=False.")
            else:
                for p in self.unknown_head.parameters():
                    p.requires_grad = False
                logger.info("Froze unknown concept head parameters.")

        if "lm_head" in freeze:
            for p in self.lm_head.parameters():
                p.requires_grad = False
            logger.info("Froze LM head parameters.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Current device of the model parameters."""
        return next(self.parameters()).device

    @property
    def tokenizer(self):
        """The Steerling tokenizer (lazy-loaded via the backbone)."""
        return self.backbone.tokenizer

    @property
    def n_known(self) -> int:
        """Number of supervised (known) concepts."""
        return self.known_head.out_concepts

    @property
    def n_unknown(self) -> int:
        """Number of unsupervised (unknown) concepts."""
        return 0 if self.unknown_head is None else self.unknown_head.out_concepts

    @property
    def latent_size(self) -> int:
        """Transformer hidden dimension (``n_embd``)."""
        return self.backbone.out_features
    
    @property
    def embedding_size(self) -> int:
        """Concept embedding dimension (``concept_dim``)."""
        return self.concept_cfg['concept_dim']

    def known_embeddings(self) -> torch.Tensor:
        """Known concept embeddings ``(n_known, embedding_dim)``."""
        return self.known_head.embeddings()

    def unknown_embeddings(self, cat_factorized: bool = False) -> torch.Tensor | None:
        """Unknown concept embeddings (packed factorized when applicable)."""
        if self.unknown_head is None:
            return None
        return self.unknown_head.embeddings(cat_factorized=cat_factorized)

    @property
    def known_embeddings_shape(self) -> torch.Size:
        """Shape of :meth:`known_embeddings`, without materializing it."""
        return self.known_head.embeddings_shape()

    def unknown_embeddings_shape(self, cat_factorized: bool = False) -> torch.Size | None:
        """Shape of :meth:`unknown_embeddings`, without materializing it."""
        if self.unknown_head is None:
            return None
        return self.unknown_head.embeddings_shape(cat_factorized=cat_factorized)

    @property
    def concept_names(self) -> list[str]:
        """Ordered list of known-concept names, cached on first access."""
        if not hasattr(self, "_concept_names"):
            self._concept_names = load_steerling_concept_names()
        return self._concept_names

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """End-to-end forward through the concept bottleneck.

        Mirrors the upstream Steerling reference path
        (``InterpretableCausalDiffusionLM.forward``) by decomposing the
        reconstructed latent as ``h_bar = k_hat + u_hat + epsilon``, where
        ``epsilon`` is computed by a :class:`~torch_concepts.nn.ResidualCorrectionOp`
        configured at construction from the ``use_unknown`` and
        ``use_epsilon_correction`` concept-config flags.

        Args:
            input_ids: Token ids, shape ``(B, T)``.

        Returns:
            Dict with ``out_tokens``, ``known_concepts``,
            ``unknown_concepts``, ``known_mixed``, ``unknown_mixed``,
            ``epsilon``, and ``reconstructed_latent``.
        """
        h = self.backbone(input_ids)
        k = self.known_head(h)
        k_embs = self.known_embeddings()
        k_hat = self.known_concept_mixer(torch.sigmoid(k), k_embs)

        u = u_hat = None
        if self.unknown_head is not None:
            # Detach so unknown loss can't back-prop into the transformer.
            u = self.unknown_head(h.detach())
            u_embs = self.unknown_embeddings(cat_factorized=True)
            u_hat = self.unknown_concept_mixer(torch.sigmoid(u), u_embs)
            parts = (k_hat, u_hat)
        else:
            parts = (k_hat,)

        epsilon = self.epsilon_correction([h, *parts])
        h_bar = sum(parts) + epsilon

        return {
            "out_tokens": self.lm_head(h_bar),
            "known_concepts": k,
            "unknown_concepts": u,
            "known_mixed": k_hat,
            "unknown_mixed": u_hat,
            "epsilon": epsilon,
            "reconstructed_latent": h_bar,
        }

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def build_input(
        self, prompt: str, n_new_tokens: int
    ) -> tuple[torch.Tensor, torch.BoolTensor, torch.BoolTensor]:
        """Build the ``[prompt | MASK × N]`` input tensor for generation.

        Mirrors the reference Steerling generation setup: a single prompt, no
        padding (Steerling generates one prompt at a time — its backbone has no
        attention-mask input, so batched/unequal-length prompts are not
        supported).

        Returns:
            input_ids: Shape ``(1, T)``.
            prompt_mask: ``True`` for prompt positions, shape ``(T,)``.
            gen_mask: ``True`` for generation positions, shape ``(T,)``.
        """
        if not isinstance(prompt, str):
            raise ValueError(
                "build_input accepts a single prompt string; batched / multiple "
                "prompts are not supported by Steerling (its backbone has no "
                "attention-mask input, so it generates one prompt at a time)."
            )

        tokenizer = self.tokenizer
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        total_len = prompt_len + n_new_tokens

        input_ids = torch.full((1, total_len), tokenizer.mask_token_id, dtype=torch.long)
        input_ids[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long)

        prompt_mask = torch.zeros(total_len, dtype=torch.bool)
        prompt_mask[:prompt_len] = True
        gen_mask = ~prompt_mask.clone()

        return input_ids, prompt_mask, gen_mask

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        n_new_tokens: int,
        topk_concepts: int | None = None,
        verbose: bool = True,
    ) -> str:
        """Procedurally unmask and generate tokens after a text prompt.

        At each step the model scores all still-masked positions, picks the
        one with the highest confidence (max token probability), fills it
        with the argmax token, and repeats until all ``n_new_tokens``
        positions are filled.

        Args:
            prompt: Text prompt to condition on.
            n_new_tokens: Number of tokens to generate.
            topk_concepts: If set, print this many top known concepts for each
                newly filled token.
            verbose: Print each decoding step to stdout.

        Returns:
            The generated continuation (excluding the prompt).
        """
        tokenizer = self.tokenizer
        mask_id = tokenizer.mask_token_id

        # Special tokens that must never be generated (they would otherwise be
        # picked as the argmax — the mask token in particular, which a masked
        # diffusion LM scores highly at unfilled positions, leaving the slot
        # still masked). Mirrors the reference Steerling generation loop.
        banned_ids = [mask_id]
        if tokenizer.pad_token_id is not None:
            banned_ids.append(tokenizer.pad_token_id)

        input_ids, _, _ = self.build_input(prompt, n_new_tokens)
        input_ids = input_ids.to(self.device)

        prompt_len = (input_ids[0] != mask_id).sum().item()

        if verbose:
            print(f"\nGenerating {n_new_tokens} tokens one at a time:")

        for step in range(n_new_tokens):
            # 1. Forward through the concept bottleneck
            out = self.forward(input_ids)
            token_logits = out["out_tokens"]                           # (1, T, vocab)

            # 2. Pick the most confident masked position, take argmax.
            # Confidence = max softmax probability per position — the
            # standard masked-diffusion convention (MaskGIT and successors).
            masked_positions = (input_ids[0] == mask_id).nonzero(as_tuple=False).squeeze(-1)
            if masked_positions.numel() == 0:
                break

            masked_logits = token_logits[0, masked_positions].clone()  # (n_masked, vocab)
            masked_logits[:, banned_ids] = float("-inf")               # never emit mask / pad
            masked_probs = torch.softmax(masked_logits.float(), dim=-1)
            confidences = masked_probs.max(dim=-1).values             # (n_masked,)
            best = confidences.argmax()
            seq_idx = masked_positions[best].item()
            chosen_token = masked_logits[best].argmax().item()

            # 3. Fill the chosen position
            input_ids[0, seq_idx] = chosen_token
            if verbose:
                decoded = tokenizer.decode([chosen_token])
                print(f"  step {step + 1}: position {seq_idx} → {decoded!r}")
                if topk_concepts is not None:
                    concepts = top_concepts(
                        out["known_concepts"][0, seq_idx],
                        topk=topk_concepts,
                    )
                    print(concepts.to_string(index=False))

        generated_ids = input_ids[0, prompt_len:].tolist()
        generated_text = tokenizer.decode(generated_ids)

        if verbose:
            print(f"\n{prompt}{generated_text}")

        return generated_text

    def __repr__(self) -> str:
        return (
            f"SteerlingLowLevelModel("
            f"n_known={self.n_known}, "
            f"n_unknown={self.n_unknown}, "
            f"latent_dim={self.latent_size}, "
            f"vocab={self.vocab_size}, "
            f"factorize_unknown={self.concept_cfg['factorize_unknown']}, "
            f"use_epsilon_correction={self.use_epsilon_correction}, "
            f"pretrained={self.pretrained_components}, "
            f"frozen={self.freeze_components}, "
            f"config_source={self.config_source!r})"
        )
