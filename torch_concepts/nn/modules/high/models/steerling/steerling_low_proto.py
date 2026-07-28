
"""Low-level Steerling model assembled from explicit PyTorch modules.

``SteerlingLowLevelModel`` wires the backbone, known/unknown concept heads,
concept-embedding mixers, and language-model head into the same data flow used
by Steerling:

``input_ids -> hidden -> concept logits -> concept features -> token logits``.

The model returns intermediate tensors as a dictionary so callers can inspect
known concepts, unknown concepts, and reconstructed latent features.
"""

import logging
import torch
from . import SteerlingLowLevelModel
from .steerling_utils import (
    load_steerling_concept_names,
    top_concepts,
)

logger = logging.getLogger(__name__)

# Steerling components that can be pretrained-loaded and/or frozen.
_ALL_COMPONENTS = ("backbone", "known_head", "unknown_head", "lm_head")


class SteerlingLowLevelModelPrototypes(SteerlingLowLevelModel):
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        proto_h_dict: dict,
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

        for concept_id, concept_proto_h in proto_h_dict.items():
            distances = torch.cdist(concept_proto_h, h.squeeze(0), p=2)
            weights = torch.softmax(-distances, dim=0).T
            scores = torch.linspace(-15, 15, concept_proto_h.shape[0], dtype=distances.dtype, device=distances.device)
            concept_score = weights @ scores
            k[0, :, concept_id] = concept_score

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

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        n_new_tokens: int,
        topk_concepts: int = 5,
        prototypes: dict | None = None,
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
            prototypes: Dictionary of concept-id -> list strings each of which represents a prototype of the concept.
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

        print("Prompt:")
        print(prompt)

        concept_names = load_steerling_concept_names()
        proto_h_dict = {}
        for concept_id, proto_list in prototypes.items():
            print(f"\nLoading prototype of concept {concept_names[concept_id]}:")
            proto_h_list = []
            for i, proto in enumerate(proto_list):
                print(f"  prototype {i}: {proto}")
                proto_input_ids, _, _ = self.build_input(proto, n_new_tokens=0)
                proto_input_ids = proto_input_ids.to(self.device)
                proto_h = self.backbone(proto_input_ids)
                proto_h_list.append(proto_h[0].mean(dim=0))
            proto_h_dict[concept_id] = torch.stack(proto_h_list)

        if verbose:
            print(f"\n\nGenerating {n_new_tokens} tokens one at a time:")

        for step in range(n_new_tokens):
            # 1. Forward through the concept bottleneck
            out = self.forward(input_ids, proto_h_dict=proto_h_dict)
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
                print(f"  step {step + 1}: position {seq_idx} → {decoded!r}\n")

                concepts = top_concepts(
                    out["known_concepts"][0, :seq_idx].max(dim=0)[0],
                    topk=topk_concepts,
                )
                print("\n  Chunks' concepts")
                print(concepts.to_string(index=False))

                concepts = top_concepts(
                    out["known_concepts"][0, seq_idx],
                    topk=topk_concepts,
                )
                print("\n  Token's concepts")
                print(concepts.to_string(index=False))

                print("\n  Concept prototype")
                for concept_id in proto_h_dict.keys():
                    concept_proto_name = concept_names[concept_id]
                    concept_chunk_activation = out["known_concepts"][0, :seq_idx].max(dim=0)[0][concept_id]
                    concept_token_activation = out["known_concepts"][0, seq_idx][concept_id]
                    print(f"  Concept {concept_proto_name} chunk: {concept_chunk_activation}")
                    print(f"  Concept {concept_proto_name} token: {concept_token_activation}")

                print("\n\n")

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
