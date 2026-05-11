
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

from ..steerling_backbone import SteerlingBackbone
from ..steerling_configs import (
    DEFAULT_MODEL_ID,
    SteerlingConfigSource,
    normalize_steerling_components,
    resolve_steerling_configs,
)
from ..steerling_encoder import SteerlingLatentToConcept
from ..steerling_predictor import MixFactorizedConceptExogenousToConcept
from ..steerling_utils import (
    load_steerling_backbone_weights,
    load_steerling_known_head_weights,
    load_steerling_lm_head_weights,
    load_steerling_unknown_head_weights,
    prepare_generation_sequence,
)

logger = logging.getLogger(__name__)


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
        compact: Use one combined concept mixer plus linear head. This is only
            supported when the enabled concept embeddings share the same dense
            layout.
        model_id: Hugging Face model id or local path for Steerling weights.
        config_source: Config source passed to ``resolve_steerling_configs``.
        model_config_overrides: Optional model config overrides.
        concept_config_overrides: Optional concept config overrides.
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
        use_unknown: bool = True,
        compact: bool = False,
        model_id: str = DEFAULT_MODEL_ID,
        config_source: SteerlingConfigSource = "hub",
        model_config_overrides: dict[str, Any] | None = None,
        concept_config_overrides: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.pretrained_components = normalize_steerling_components(pretrained_components)
        self.freeze_components = normalize_steerling_components(freeze_components)
        self.model_cfg, self.concept_cfg, self.config_source = resolve_steerling_configs(
            config_source=config_source,
            use_unknown=use_unknown,
            model_id=model_id,
            model_config_overrides=model_config_overrides,
            concept_config_overrides=concept_config_overrides
        )

        latent_dim = int(self.model_cfg["n_embd"])
        n_known = int(self.concept_cfg["n_concepts"])
        n_unknown = int(self.concept_cfg.get("n_unknown_concepts", 0))
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

        # Backbone: tokens -> hidden states.
        self.backbone = SteerlingBackbone(
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
            config=self.concept_cfg,
        )
        # unknown (unsupervised): h → u dense concept logits
        if use_unknown and n_unknown == 0:
            raise ValueError("n_unknown_concepts must be set when use_unknown=True")
        if use_unknown:
            self.unknown_concept_head = SteerlingLatentToConcept(
                in_latent=latent_dim,
                out_concepts=n_unknown,
                embedding_size=embedding_dim,
                is_unknown=True,
                factorize=self.factorize_unknown,
                config=self.concept_cfg,
            )
        else:
            self.unknown_concept_head = None

        # Concept-logit + embedding mixing combined with LM head.
        self.compact = compact
        if compact:
            if self.factorize_unknown:
                raise NotImplementedError(
                    "compact=True is not supported when factorize_unknown=True. "
                    "Use compact=False for the default Steerling-8B configuration."
                )
            self.decoder = MixFactorizedConceptExogenousToConcept(
                in_concepts=n_known + (n_unknown if use_unknown else 0),
                in_exogenous=embedding_dim,
                out_concepts=self.vocab_size,
                factorized=self.factorize_unknown,
                add_linear_head=True,
                bias=False
            )
        else:
            # Concept-logit + embedding mixing, split from the concept encoder.
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
                    in_concepts=n_unknown,
                    in_exogenous=embedding_dim,
                    out_concepts=self.vocab_size,
                    factorized=self.factorize_unknown,
                    add_linear_head=False,
                    bias=False
                )
            else:
                self.unknown_concept_mixer = None

            self.lm_head = nn.Linear(
                embedding_dim,
                self.vocab_size,
                bias=False,
            )

        # Load and freeze pretrained weights.
        self._load_steerling_weights(model_id, pretrained=self.pretrained_components)
        self._freeze_steerling_weights(freeze=self.freeze_components)
        self.register_buffer(
            "known_embeddings",
            self.known_concept_head.get_embeddings(factorized=False).detach(),
        )
        self.register_buffer(
            "unknown_embeddings",
            (
                self.unknown_concept_head.get_embeddings(
                    factorized=self.unknown_concept_head.factorize
                ).detach()
                if self.unknown_concept_head is not None
                else None
            ),
        )


    def _load_steerling_weights(self, model_id: str, pretrained: list | None = None):
        """Load selected pretrained Steerling components."""
        pretrained = pretrained or []

        if "backbone" in pretrained:
            backbone_sd = load_steerling_backbone_weights(model_id, device="cpu")
            self._load_state_dict(self.backbone.transformer, backbone_sd, "backbone")
            self._discard_state_dict(backbone_sd)
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
            lm_head_sd = load_steerling_lm_head_weights(model_id, device="cpu")
            lm_head = self.decoder.predictor if self.compact else self.lm_head
            self._load_state_dict(lm_head, lm_head_sd, "lm_head")
            self._discard_state_dict(lm_head_sd)
            logger.info("Loaded pretrained weights into LM head.")


    @staticmethod
    def _load_state_dict(module: nn.Module, state_dict: dict, name: str) -> None:
        incompatible = module.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(
                "Loaded %s with missing keys=%s and unexpected keys=%s.",
                name,
                incompatible.missing_keys,
                incompatible.unexpected_keys,
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
            lm_head = self.decoder.predictor if self.compact else self.lm_head
            for p in lm_head.parameters():
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        unknown_path: bool = True,
        use_epsilon_correction: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """End-to-end forward through the concept bottleneck.

        Args:
            input_ids: Token ids, shape ``(B, T)``.
            unknown_path: Include the unknown/discovered concept head when it
                is configured.
            use_epsilon_correction: If ``True``, add the residual correction
                used by upstream Steerling to recover the backbone hidden state.
                Defaults to ``False`` for concept-faithful inference.

        Returns:
            Dict with ``out_tokens``, ``known_concepts``,
            ``unknown_concepts``, and ``reconstructed_latent``.
        """
        h = self.backbone(input_ids).float()

        k = self.known_concept_head(h)
        k_embeddings = self.known_concept_head.get_embeddings(factorized=False)

        u = None
        u_embeddings = None
        if unknown_path and self.unknown_concept_head is not None:
            u = self.unknown_concept_head(h)
            u_embeddings = self.unknown_concept_head.get_embeddings(
                factorized=self.unknown_concept_head.factorize
            )

        if self.compact:
            concepts = (
                k
                if u is None
                else torch.cat([k, u], dim=-1)
            )
            embeddings = (
                k_embeddings
                if u_embeddings is None
                else torch.cat([k_embeddings, u_embeddings], dim=0)
            )
            h_bar = self.decoder.mix(torch.sigmoid(concepts), embeddings)
            if use_epsilon_correction and u is not None:
                h_bar = h
            out_tokens = self.decoder.predictor(h_bar)
        else:
            k_hat = self.known_concept_mixer.mix(torch.sigmoid(k), k_embeddings)
            if u is None:
                h_bar = k_hat
            else:
                u_hat = self.unknown_concept_mixer.mix(torch.sigmoid(u), u_embeddings)
                h_bar = h if use_epsilon_correction else k_hat + u_hat
            out_tokens = self.lm_head(h_bar)

        return {
            "out_tokens": out_tokens,
            "known_concepts": k,
            "unknown_concepts": u,
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
        h = self.backbone(input_ids).float()         # (B, T, D)
        return self.known_concept_head(h)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        n_new_tokens: int,
        unknown_path: bool = True,
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
            unknown_path: Whether to include the unknown concept head in the
                forward pass.
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
            out = self.forward(input_ids, unknown_path=unknown_path)
            token_logits = out["out_tokens"]                           # (1, T, vocab)

            # 2. Pick the most confident masked position, take argmax
            masked_positions = (input_ids[0] == mask_id).nonzero(as_tuple=False).squeeze(-1)
            if masked_positions.numel() == 0:
                break

            masked_logits = token_logits[0, masked_positions]         # (n_masked, vocab)
            confidences = masked_logits.max(dim=-1).values            # (n_masked,)
            best = confidences.argmax()
            seq_idx = masked_positions[best].item()
            chosen_token = masked_logits[best].argmax().item()

            # 3. Fill the chosen position
            input_ids[0, seq_idx] = chosen_token
            if verbose:
                decoded = tokenizer.decode([chosen_token])
                print(f"  step {step + 1}: position {seq_idx} → {decoded!r}")

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
            f"config_source={self.config_source!r})"
        )
