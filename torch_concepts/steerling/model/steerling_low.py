
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
from typing import Any

import torch
import torch.nn as nn

from ..steerling_backbone import CausalDiffusionTextBackbone
from ..steerling_configs import (
    DEFAULT_MODEL_ID,
    SteerlingConfigSource,
    normalize_steerling_components,
    resolve_steerling_configs,
)
from ..steerling_encoder import SteerlingLatentToConcept
from torch_concepts.nn import ResidualCorrectionOp

from ..steerling_predictor import MixFactorizedConceptExogenousToConcept
from ..steerling_utils import (
    load_steerling_backbone_weights,
    load_steerling_concept_names,
    load_steerling_known_head_weights,
    load_steerling_lm_head_weights,
    load_steerling_unknown_head_weights,
    prepare_generation_sequence,
    print_concepts,
)

logger = logging.getLogger(__name__)


# Top-k inference is not implemented yet (see steerling_encoder.py).  These
# are the resolved-config keys that, if non-disabled, would request it.
_TOPK_CONFIG_KEYS = (
    "topk",
    "topk_features",
    "topk_known",
    "topk_known_features",
    "unknown_topk",
    "apply_topk_to_unknown",
    "topk_on_logits",
)


def _is_topk_enabled(value) -> bool:
    if value is None or value is False:
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return bool(value)


def _sanitize_topk(concept_cfg: dict) -> dict:
    """Return a copy of ``concept_cfg`` with every top-k key disabled."""
    sanitized = dict(concept_cfg)
    for key in _TOPK_CONFIG_KEYS:
        if key in sanitized:
            sanitized[key] = (
                False if key in ("apply_topk_to_unknown", "topk_on_logits") else None
            )
    return sanitized


class SteerlingLowLevelModel(nn.Module):
    """Low-level Steerling concept-bottleneck language model.

    Instantiates and wires all low-level Steerling modules internally.
    Use :meth:`forward` for end-to-end prediction or the convenience methods
    for intermediate representations.

    Args:
        pretrained_components: Components to load from the Steerling Hub
            checkpoint. Accepts ``True``/``False`` or component names.
        freeze_components: Components whose parameters should be frozen after
            loading.
        use_unknown: Whether to build the unsupervised "unknown" concept head.
            When ``False``, the wrapper mirrors upstream's no-unknown-head
            path (``composed ≈ hidden``) so the LM head sees the raw
            backbone state.
        model_id: Hugging Face model id or local path for Steerling weights.
        config_source: Config source passed to ``resolve_steerling_configs``.
        model_config_overrides: Optional model config overrides.
        concept_config_overrides: Optional concept config overrides.  Passing
            top-k keys here (``topk_known``, ``unknown_topk``, ...) raises
            :class:`NotImplementedError` — top-k inference is not implemented
            yet.
        n_concepts, n_unknown_concepts, concept_dim, factorize_unknown,
        use_epsilon_correction, use_attention_known, use_attention_unknown:
            Elevated concept-config knobs.  ``None`` (the default) reads the
            value from the resolved config; any non-``None`` value overrides
            ``concept_config_overrides``.
        Modules are constructed and loaded on CPU. Move the model afterward
        with the standard PyTorch ``model.to(device)`` pattern.

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
        pretrained_components: bool | str | list[str] | tuple[str, ...] | None = (
            "backbone",
            "known_head",
            "unknown_head",
            "lm_head",
        ),
        freeze_components: bool | str | list[str] | tuple[str, ...] | None = (
            "backbone",
            "known_head",
            "unknown_head",
            "lm_head",
        ),
        use_unknown: bool | None = None,
        model_id: str = DEFAULT_MODEL_ID,
        config_source: SteerlingConfigSource = "hub",
        model_config_overrides: dict[str, Any] | None = None,
        concept_config_overrides: dict[str, Any] | None = None,
        n_concepts: int | None = None,
        n_unknown_concepts: int | None = None,
        concept_dim: int | None = None,
        factorize_unknown: bool | None = None,
        use_epsilon_correction: bool | None = None,
        use_attention_known: bool | None = None,
        use_attention_unknown: bool | None = None,
    ):
        super().__init__()

        # Fold elevated kwargs into the user-supplied concept_config_overrides
        # (elevated kwargs win).  Keep the original dict around so the topk
        # check below sees only user input.
        elevated = {
            "use_unknown": use_unknown,
            "n_concepts": n_concepts,
            "n_unknown_concepts": n_unknown_concepts,
            "concept_dim": concept_dim,
            "factorize_unknown": factorize_unknown,
            "use_epsilon_correction": use_epsilon_correction,
            "use_attention_known": use_attention_known,
            "use_attention_unknown": use_attention_unknown,
        }
        elevated = {key: value for key, value in elevated.items() if value is not None}
        user_overrides = dict(concept_config_overrides or {})

        # Top-k inference is not implemented yet — raise loudly if the user
        # tried to opt in via overrides.  Values coming from the package's
        # defaults / Hub config are normalized further below.
        bad_topk = [key for key in _TOPK_CONFIG_KEYS if _is_topk_enabled(user_overrides.get(key))]
        if bad_topk:
            raise NotImplementedError(
                "Top-k inference is not yet implemented in the PyC "
                "SteerlingLatentToConcept encoder. Got non-disabled top-k "
                f"keys in `concept_config_overrides`: {bad_topk}. Pass them "
                "as None/False (or omit them) until top-k is wired up."
            )

        merged_overrides = {**user_overrides, **elevated}

        self.pretrained_components = normalize_steerling_components(pretrained_components)
        self.freeze_components = normalize_steerling_components(freeze_components)
        self.model_cfg, self.concept_cfg, self.config_source = resolve_steerling_configs(
            config_source=config_source,
            model_id=model_id,
            model_config_overrides=model_config_overrides,
            concept_config_overrides=merged_overrides,
        )

        # Resolve use_unknown from the merged config (elevated kwarg already
        # folded in above; otherwise the config-source default wins).
        use_unknown = bool(self.concept_cfg.get("use_unknown", True))
        latent_dim = int(self.model_cfg["n_embd"])
        n_known = int(self.concept_cfg["n_concepts"])
        n_unknown_resolved = int(self.concept_cfg.get("n_unknown_concepts") or 0)
        embedding_dim = int(self.concept_cfg["concept_dim"])
        self.factorize_unknown = bool(self.concept_cfg.get("factorize_unknown", False))
        self.config_use_epsilon_correction = bool(
            self.concept_cfg.get("use_epsilon_correction", False)
        )
        if self.config_use_epsilon_correction and embedding_dim != latent_dim:
            raise ValueError(
                "use_epsilon_correction requires concept_dim to match n_embd: "
                f"{embedding_dim} != {latent_dim}."
            )
        self.embedding_dim = embedding_dim

        # The encoders never see top-k values — the wrapper enforces dense
        # inference for now.
        encoder_concept_cfg = _sanitize_topk(self.concept_cfg)

        # Backbone: tokens -> hidden states.
        self.backbone = CausalDiffusionTextBackbone(
            config=self.model_cfg,
            model_id=model_id,
        )
        self.vocab_size = self.backbone.vocab_size

        # Concept encoders.
        # known (supervised): h → k dense concept logits
        self.known_concept_head = SteerlingLatentToConcept(
            in_latent=latent_dim,
            out_concepts=n_known,
            embedding_size=embedding_dim,
            is_unknown=False,
            factorize=False,
            config=encoder_concept_cfg,
        )
        # unknown (unsupervised): h → u dense concept logits
        if use_unknown and n_unknown_resolved == 0:
            raise ValueError("n_unknown_concepts must be set when use_unknown=True")
        if use_unknown:
            self.unknown_concept_head = SteerlingLatentToConcept(
                in_latent=latent_dim,
                out_concepts=n_unknown_resolved,
                embedding_size=embedding_dim,
                is_unknown=True,
                factorize=self.factorize_unknown,
                config=encoder_concept_cfg,
            )
        else:
            self.unknown_concept_head = None

        # Concept-logit + embedding mixing.
        # We use the mixing layer, but stop at the concept embeddings without the linear head.
        self.known_concept_mixer = MixFactorizedConceptExogenousToConcept(
            in_concepts=n_known,
            in_exogenous=embedding_dim,
            out_concepts=self.vocab_size,
            factorized=False,
            add_linear_head=False,
            bias=False
        )
        if self.unknown_concept_head is not None:
            self.unknown_concept_mixer = MixFactorizedConceptExogenousToConcept(
                in_concepts=n_unknown_resolved,
                in_exogenous=embedding_dim,
                out_concepts=self.vocab_size,
                factorized=self.factorize_unknown,
                add_linear_head=False,
                bias=False
            )
        else:
            self.unknown_concept_mixer = None

        # Alias the backbone's LM head.  Under upstream ``weight_sharing=True``
        # (Steerling default) its weight is tied to ``transformer.tok_emb.weight``
        # via ``_tie_weights``, so a single underlying Parameter is shared by
        # ``self.lm_head``, ``self.backbone.transformer.lm_head``, and the
        # backbone's input embedding.
        self.lm_head = self.backbone.transformer.lm_head

        # Epsilon correction term for h_bar = k_hat + u_hat + epsilon.
        #   (has_unknown, use_epsilon)  →  (mode,           stop_grad_parts)
        #   (True,  True)               →  ("block_parts",  ())
        #   (True,  False)              →  ("off",          (1,))  stop-grad on u_hat
        #   (False, True)               →  ("block_parts",  ())
        #   (False, False)              →  ("keep_parts",   ())
        self.has_unknown = self.unknown_concept_head is not None
        use_eps = self.config_use_epsilon_correction
        self.epsilon_correction = ResidualCorrectionOp(
            input_size=embedding_dim,
            n_terms=2 if self.has_unknown else 1,
            residual_mode=(
                "block_parts" if use_eps
                else ("off" if self.has_unknown else "keep_parts")
            ),
            stop_grad_parts=(1,) if self.has_unknown and not use_eps else (),
        )

        # Load and freeze pretrained weights.
        self._load_steerling_weights(model_id, pretrained=self.pretrained_components)
        self._freeze_steerling_weights(freeze=self.freeze_components)


    def _load_steerling_weights(self, model_id: str, pretrained: list | None = None):
        """Load selected pretrained Steerling components."""
        pretrained = pretrained or []

        if "backbone" in pretrained:
            backbone_sd = load_steerling_backbone_weights(model_id, device="cpu")
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
            self.backbone._model_id = model_id
            self.backbone._tokenizer_model_id = model_id
            logger.info("Loaded pretrained weights into backbone.")

        if "known_head" in pretrained:
            known_sd = load_steerling_known_head_weights(model_id, device="cpu")
            self._load_state_dict(self.known_concept_head.head, known_sd, "known_head")
            self._discard_state_dict(known_sd)
            logger.info("Loaded pretrained weights into known concept head.")

        if "unknown_head" in pretrained:
            if self.unknown_concept_head is None:
                logger.info("Skipped unknown concept head weights because use_unknown=False.")
            else:
                unknown_sd = load_steerling_unknown_head_weights(model_id, device="cpu")
                self._load_state_dict(self.unknown_concept_head.head, unknown_sd, "unknown_head")
                self._discard_state_dict(unknown_sd)
                logger.info("Loaded pretrained weights into unknown concept head.")

        if "lm_head" in pretrained:
            weight_sharing = bool(self.model_cfg.get("weight_sharing", False))
            if weight_sharing and "backbone" in pretrained:
                # `self.lm_head` aliases `transformer.lm_head`, whose weight is
                # tied to `tok_emb.weight`.  The backbone load above (followed
                # by the defensive re-tie) already populated it; an explicit
                # load would just overwrite the shared tensor with itself.
                logger.info(
                    "Skipped LM head weight load (tied to tok_emb, loaded with backbone)."
                )
            else:
                lm_head_sd = load_steerling_lm_head_weights(model_id, device="cpu")
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
            for p in self.known_concept_head.parameters():
                p.requires_grad = False
            logger.info("Froze known concept head parameters.")

        if "unknown_head" in freeze:
            if self.unknown_concept_head is None:
                logger.info("Skipped freezing unknown concept head because use_unknown=False.")
            else:
                for p in self.unknown_concept_head.parameters():
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
        return self.known_concept_head.out_concepts

    @property
    def n_unknown(self) -> int:
        """Number of unsupervised (unknown) concepts."""
        return 0 if self.unknown_concept_head is None else self.unknown_concept_head.out_concepts

    @property
    def latent_dim(self) -> int:
        """Transformer hidden dimension (``n_embd``)."""
        return self.backbone.out_features

    @property
    def known_embeddings(self) -> torch.Tensor:
        """Known concept embeddings ``(n_known, embedding_dim)``.

        Computed live from the current encoder weights, so the result stays
        in sync if the concept head is fine-tuned.
        """
        return self.known_concept_head.get_embeddings(factorized=False)

    @property
    def unknown_embeddings(self) -> torch.Tensor | None:
        """Unknown concept embeddings (packed factorized when applicable).

        Returns ``None`` when ``use_unknown=False``.  Computed live so the
        result stays in sync with the unknown concept head's parameters.
        """
        if self.unknown_concept_head is None:
            return None
        return self.unknown_concept_head.get_embeddings(
            factorized=self.unknown_concept_head.factorize
        )

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
        k = self.known_concept_head(h)
        k_hat = self.known_concept_mixer(torch.sigmoid(k), self.known_embeddings)

        u = u_hat = None
        if self.has_unknown:
            # Detach so unknown loss can't back-prop into the transformer.
            u = self.unknown_concept_head(h.detach())
            u_hat = self.unknown_concept_mixer(torch.sigmoid(u), self.unknown_embeddings)

        parts = (k_hat,) if u_hat is None else (k_hat, u_hat)
        epsilon = self.epsilon_correction.compute(h, *parts)
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

    def prepare_input(self, prompt: str, n_new_tokens: int):
        """Prepare input tensors for generation with a text prompt.

        Args:
            prompt: Text prompt to condition on.
            n_new_tokens: Number of new tokens to generate after the prompt.
        Returns:
            input_ids: Token ids with prompt + mask tokens, shape ``(1, T)``.
            prompt_mask: Boolean mask for prompt positions, shape ``(T,)``.
            gen_mask: Boolean mask for generation positions, shape ``(T,)``.
        """
        return prepare_generation_sequence(self.tokenizer, prompt, n_new_tokens)

    def encode_concepts(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return known-concept logits for the given token ids.

        Args:
            input_ids: Token ids, shape ``(B, T)``.

        Returns:
            Known-concept logits ``(B, T, n_known)`` before sigmoid.
        """
        h = self.backbone(input_ids)                # (B, T, D)
        return self.known_concept_head(h)

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

        input_ids, _, _ = self.prepare_input(prompt, n_new_tokens)
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

            masked_logits = token_logits[0, masked_positions]         # (n_masked, vocab)
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
                    concepts = print_concepts(
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
            f"latent_dim={self.latent_dim}, "
            f"vocab={self.vocab_size}, "
            f"factorize_unknown={self.factorize_unknown}, "
            f"use_epsilon_correction={self.config_use_epsilon_correction}, "
            f"pretrained={self.pretrained_components}, "
            f"frozen={self.freeze_components}, "
            f"config_source={self.config_source!r})"
        )
