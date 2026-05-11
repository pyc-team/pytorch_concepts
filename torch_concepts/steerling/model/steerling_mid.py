"""
Steerling mid-level model — PGM-backed concept bottleneck LM.

:class:`SteerlingMidLevelModel` exposes the same interface as
:class:`SteerlingLowLevelModel` but routes all computation through a
:class:`~torch_concepts.nn.ProbabilisticModel` + 
:class:`~torch_concepts.nn.DeterministicInference` graph, enabling
fine-grained concept queries and future interventions.

Internal PGM graph::

    input  ──► k  (known concepts)    ──► k_hat ──┐
           └──► u  (unknown concepts) ──► u_hat ──► h_bar ──► new_token

Data-flow::

    input_ids  ──► backbone  ──► h         (B, T, D)       [evidence]
    h          ──► k_cpd     ──► k         (B, T, n_known)
    h          ──► u_cpd     ──► u         (B, T, n_unknown)
    k          ──► k_hat_cpd ──► k_hat     (B, T, D)
    u          ──► u_hat_cpd ──► u_hat     (B, T, D)
    k_hat,u_hat──► h_bar_cpd ──► h_bar     (B, T, D)
    h_bar      ──► token_cpd ──► new_token (B, T, vocab)

Mid-level modules used:
    - :class:`SteerlingBackbone`
    - :class:`SteerlingLatentToConcept`           (known + unknown)
    - :class:`MixFactorizedConceptExogenousToConcept`
    - :class:`LatentFusion`
    - ``nn.Linear`` LM head
    - :class:`ParametricCPD`, :class:`ProbabilisticModel`,
      :class:`DeterministicInference`
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributions import RelaxedBernoulli, RelaxedOneHotCategorical

from torch_concepts.distributions import Delta
from torch_concepts import ConceptVariable, LatentVariable, ExogenousVariable
from torch_concepts.nn import ParametricCPD, ProbabilisticModel, DeterministicInference, BaseInference

from ...steerling.model.steerling_low import SteerlingLowLevelModel
from ..steerling_utils import load_steerling_concept_names, prepare_generation_sequence
from ..steerling_configs import DEFAULT_MODEL_ID, SteerlingConfigSource

logger = logging.getLogger(__name__)


class LatentFusion(nn.Module):
    """Sum one or two latent contributions routed through PyC as ``latent``."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] == self.latent_dim:
            return latent
        if latent.shape[-1] != 2 * self.latent_dim:
            raise ValueError(
                "Expected one or two latent contributions: "
                f"{latent.shape[-1]} != {self.latent_dim} or {2 * self.latent_dim}."
            )
        left, right = latent.split(self.latent_dim, dim=-1)
        return left + right


class SteerlingMidLevelModel(SteerlingLowLevelModel):
    """PGM-backed Steerling concept-bottleneck language model.

    Same interface as :class:`SteerlingLowLevelModel`, but all computation
    is routed through a :class:`ProbabilisticModel` so individual variables
    (concepts, latents, tokens) can be queried or intervened upon directly.

    Args:
        device: Target device (``"cuda"``, ``"cpu"``, etc.).

    Example::

        model = SteerlingMidLevelModel(device="cuda")
        model.eval()

        # End-to-end: tokens → concept bottleneck → next-token logits
        out = model(input_ids)
        parts = model.split_full_forward(out.probs)

        # Query a single named concept
        act = model(input_ids, query=["food"]).probs   # (1, T, 1)

        # Concept-based hidden-state reconstruction
        h_bar = model(input_ids, query=["h_bar"]).probs  # (1, T, D)

        # Generation
        model.generate("As an Italian living abroad I miss", n_new_tokens=20)
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
        device: str = "cuda",
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
    ):
        if compact:
            raise NotImplementedError(
                "compact=True is currently supported only by "
                "SteerlingLowLevelModel, not by the mid-level PyC graph."
            )
        super().__init__(
            pretrained_components=pretrained_components,
            freeze_components=freeze_components,
            use_unknown=use_unknown,
            compact=False,
            model_id=model_id,
            config_source=config_source,
            model_config_overrides=model_config_overrides,
            concept_config_overrides=concept_config_overrides,
            device=device,
        )
        self.use_unknown = use_unknown

        # ── PGM variables ─────────────────────────────────────────────────
        emb_dim = self.embedding_dim
        self.known_names   = load_steerling_concept_names() # list[str]
        if len(self.known_names) != self.n_known:
            raise ValueError(
                "SteerlingMidLevelModel requires n_concepts to match the "
                f"known-concept CSV ({len(self.known_names)}), got {self.n_known}."
            )
        if use_unknown:
            self.unknown_names = [f"unsup_{i}" for i in range(self.unknown_concept_head.out_concepts)]
        else:
            self.unknown_names = []

        h = LatentVariable("input", size=self.latent_dim, distribution=Delta)
        k = ConceptVariable(
            self.known_names, 
            distribution=RelaxedBernoulli, 
            dist_kwargs={"temperature": 0.5}
        )
        k_embs = ExogenousVariable("K", size=emb_dim, distribution=Delta)
        if use_unknown:
            u = ConceptVariable(
                self.unknown_names,
                distribution=RelaxedBernoulli,
                dist_kwargs={"temperature": 0.5},
            )
            u_embs = ExogenousVariable("U", size=emb_dim, distribution=Delta)
        k_mix = LatentVariable("k_hat", size=emb_dim, distribution=Delta)
        if use_unknown:
            u_mix = LatentVariable("u_hat", size=emb_dim, distribution=Delta)
        h_bar = LatentVariable("h_bar", size=emb_dim, distribution=Delta)
        new_token = ConceptVariable(
            "new_token",
            size=self.vocab_size,
            distribution=RelaxedOneHotCategorical,
            dist_kwargs={"temperature": 0.5},
        )
        self._split_query_specs = [
            ("input", self.latent_dim),
            ("known_concepts", self.n_known),
            *([("unknown_concepts", self.n_unknown)] if use_unknown else []),
            ("k_hat", emb_dim),
            *([("u_hat", emb_dim)] if use_unknown else []),
            ("h_bar", emb_dim),
            ("new_token", self.vocab_size),
        ]
        self.default_query = (
            ["input"]
            + self.known_names
            + self.unknown_names
            + ["k_hat"]
            + (["u_hat"] if use_unknown else [])
            + ["h_bar", "new_token"]
        )


        # ── CPDs ──────────────────────────────────────────────────────────
        backbone_cpd = ParametricCPD("input", parents=[], parametrization=nn.Identity())
        k_cpd = ParametricCPD(
            self.known_names, 
            parents=["input"], 
            parametrization=self.known_concept_head, 
            shared=True, 
            shared_name="known_concepts"
        )
        k_embs_cpd = ParametricCPD("K", parents=[], parametrization=nn.Identity())
        if use_unknown:
            u_cpd = ParametricCPD(
                self.unknown_names, 
                parents=["input"], 
                parametrization=self.unknown_concept_head, 
                shared=True, 
                shared_name="unknown_concepts"
            )
            u_embs_cpd = ParametricCPD("U", parents=[], parametrization=nn.Identity())
        k_hat_cpd = ParametricCPD(
            "k_hat", 
            parents=self.known_names + ["K"], 
            parametrization=self.known_concept_mixer
        )
        if use_unknown:
            u_hat_cpd = ParametricCPD(
                "u_hat", 
                parents=self.unknown_names + ["U"], 
                parametrization=self.unknown_concept_mixer
            )
        h_bar_cpd = ParametricCPD(
            "h_bar",
            parents=(["k_hat", "u_hat"] if use_unknown else ["k_hat"]),
            parametrization=LatentFusion(emb_dim),
        )
        new_token_cpd = ParametricCPD("new_token", parents=["h_bar"], parametrization=self.lm_head)

        # ── ProbabilisticModel + inference engine ─────────────────────────
        variables = [h, *k]
        factors = [backbone_cpd, k_cpd]
        if use_unknown:
            variables += [*u]
            factors += [u_cpd]
        variables += [k_embs]
        factors += [k_embs_cpd]
        if use_unknown:
            variables += [u_embs]
            factors += [u_embs_cpd]
        variables += [k_mix]
        factors += [k_hat_cpd]
        if use_unknown:
            variables += [u_mix]
            factors += [u_hat_cpd]
        variables += [h_bar, new_token]
        factors += [h_bar_cpd, new_token_cpd]
        self.pgm = ProbabilisticModel(variables=variables, factors=factors)
        self.inference = inference(self.pgm, **(inference_kwargs or {}))

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _evidence(self, input_ids: torch.Tensor) -> dict:
        """Build full evidence dict: backbone hidden states + both embedding matrices."""
        hidden = self.backbone(input_ids.to(self._device)).float()
        evidence = {"input": hidden}
        evidence["K"] = self.known_embeddings
        if self.use_unknown:
            evidence["U"] = self.unknown_embeddings
        return evidence

    # ------------------------------------------------------------------
    # Forward — mirrors SteerlingLowLevelModel.forward()
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        query: Optional[list[str]] = None
    ):
        """End-to-end forward through the PGM concept bottleneck.

        Args:
            input_ids: Token ids, shape ``(B, T)``.
            query: Optional variable names to query. By default all
                token-aligned variables are queried.

        Returns:
            ``InferenceOutput`` from the configured PyC inference engine.
        """
        evidence = self._evidence(input_ids)

        if query is None:
            query = self.default_query

        out = self.inference.query(query, evidence=evidence)
        return out

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def split_full_forward(self, out: torch.Tensor) -> dict:
        """Split the concatenated output of :meth:`forward` for full queries into named tensors.

        Args:
            out: Concatenated tensor from :meth:`forward`,
                using the default query order.

        Returns:
            Dict mapping output groups to tensor slices. Concept outputs are
            returned as ``known_concepts`` and ``unknown_concepts`` chunks.
        """
        expected = sum(size for _, size in self._split_query_specs)
        if out.shape[-1] != expected:
            raise ValueError(
                "Expected output from the default full mid-level query with "
                f"last dimension {expected}, got {out.shape[-1]}."
            )
        pieces = {}
        offset = 0
        for name, size in self._split_query_specs:
            pieces[name] = out[..., offset:offset + size]
            offset += size
        return pieces

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        n_new_tokens: int,
        verbose: bool = True,
    ) -> str:
        """Procedurally unmask and generate tokens via the PGM bottleneck.

        At each step the PGM is queried for ``new_token`` probabilities at
        all positions; the most confident masked position is filled with the
        argmax token and the process repeats.

        Args:
            prompt: Text prompt to condition on.
            n_new_tokens: Number of tokens to generate.
            verbose: Print each decoding step to stdout.

        Returns:
            The generated continuation (excluding the prompt).
        """
        tokenizer = self.tokenizer
        mask_id = tokenizer.mask_token_id

        input_ids, _, _ = self.prepare_input(prompt, n_new_tokens)
        input_ids = input_ids.to(self._device)

        prompt_len = (input_ids[0] != mask_id).sum().item()

        if verbose:
            print(f"\nGenerating {n_new_tokens} tokens one at a time:")

        for step in range(n_new_tokens):
            # 1. Query token probabilities through the PGM
            token_probs = self.forward(input_ids, query=["new_token"]).probs

            # 2. Pick the most confident masked position, take argmax
            masked_positions = (input_ids[0] == mask_id).nonzero(as_tuple=False).squeeze(-1)
            if masked_positions.numel() == 0:
                break

            masked_probs = token_probs[0, masked_positions]           # (n_masked, vocab)
            confidences  = masked_probs.max(dim=-1).values            # (n_masked,)
            best         = confidences.argmax()
            seq_idx      = masked_positions[best].item()
            chosen_token = masked_probs[best].argmax().item()

            # 3. Fill the chosen position
            input_ids[0, seq_idx] = chosen_token
            if verbose:
                decoded = tokenizer.decode([chosen_token])
                print(f"  step {step + 1}: position {seq_idx} → {decoded!r}")

        generated_ids  = input_ids[0, prompt_len:].tolist()
        generated_text = tokenizer.decode(generated_ids)

        if verbose:
            print(f"\n{prompt}{generated_text}")

        return generated_text

    def __repr__(self) -> str:
        return (
            f"SteerlingMidLevelModel("
            f"n_known={self.n_known}, "
            f"n_unknown={self.n_unknown}, "
            f"latent_dim={self.latent_dim}, "
            f"vocab={self.vocab_size})"
        )
