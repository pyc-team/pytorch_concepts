"""
Steerling mid-level model — PGM-backed concept bottleneck LM.

:class:`SteerlingMidLevelModel` exposes the same interface as
:class:`SteerlingLowLevelModel` but routes all computation through a
:class:`~torch_concepts.nn.ProbabilisticModel` + 
:class:`~torch_concepts.nn.DeterministicInference` graph, enabling
fine-grained concept queries and future interventions.

Internal PGM graph::

                                      K ──┐
                                          ▼
    input  ──► k  (known concepts)    ──► k_hat ──────────────┐
           ├──► u  (unknown concepts) ──► u_hat ──────────────┤
           │                              ▲                   │
           │                          U ──┘                   │
           └────────────────────────────────────► epsilon ────► h_bar ──► new_token

Data-flow::

    input_ids        ──► backbone     ──► h          (B, T, D)        [evidence]
    known_embeddings                  ──► K          (n_known,   D)   [evidence]
    unknown_embeddings                ──► U          (n_unknown, D)   [evidence]
    h                ──► k_cpd        ──► k          (B, T, n_known)
    h                ──► u_cpd        ──► u          (B, T, n_unknown)
    k, K             ──► k_hat_cpd    ──► k_hat      (B, T, D)
    u, U             ──► u_hat_cpd    ──► u_hat      (B, T, D)
    h, k_hat, u_hat  ──► epsilon_cpd  ──► eps        (B, T, D)
    k_hat,u_hat,eps  ──► h_bar_cpd    ──► h_bar      (B, T, D)
    h_bar            ──► token_cpd    ──► new_token  (B, T, vocab)

Mid-level modules used:
    - :class:`CausalDiffusionTextBackbone`
    - :class:`SteerlingLatentToConcept`           (known + unknown)
    - :class:`MixFactorizedConceptExogenousToConcept`
    - :class:`ResidualCorrectionOp`
    - :class:`SumOp`
    - ``nn.Linear`` LM head
    - :class:`ParametricCPD`, :class:`ProbabilisticModel`,
      :class:`DeterministicInference`
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import RelaxedBernoulli, RelaxedOneHotCategorical

from torch_concepts.distributions import Delta
from torch_concepts import ConceptVariable, LatentVariable, ExogenousVariable
from torch_concepts.nn import (
    BaseInference,
    DeterministicInference,
    ParametricCPD,
    ProbabilisticModel,
    SumOp,
)

from ...steerling.model.steerling_low import SteerlingLowLevelModel
from ..steerling_utils import (
    load_steerling_concept_names,
    top_concepts,
)

logger = logging.getLogger(__name__)


class SteerlingMidLevelModel(SteerlingLowLevelModel):
    """PGM-backed Steerling concept-bottleneck language model.

    Same interface as :class:`SteerlingLowLevelModel`, but all computation
    is routed through a :class:`ProbabilisticModel` so individual variables
    (concepts, latents, tokens) can be queried or intervened upon directly.

    Internal PGM graph::

                                          K ──┐
                                              ▼
        input  ──► k  (known concepts)    ──► k_hat ──────────────┐
               ├──► u  (unknown concepts) ──► u_hat ──────────────┤
               │                              ▲                   │
               │                          U ──┘                   │
               └────────────────────────────────────► epsilon ────► h_bar ──► new_token

    Data-flow::

        input_ids        ──► backbone     ──► h          (B, T, D)        [evidence]
        known_embeddings                  ──► K          (n_known,   D)   [evidence]
        unknown_embeddings                ──► U          (n_unknown, D)   [evidence]
        h                ──► k_cpd        ──► k          (B, T, n_known)
        h                ──► u_cpd        ──► u          (B, T, n_unknown)
        k, K             ──► k_hat_cpd    ──► k_hat      (B, T, D)
        u, U             ──► u_hat_cpd    ──► u_hat      (B, T, D)
        h, k_hat, u_hat  ──► epsilon_cpd  ──► eps        (B, T, D)
        k_hat,u_hat,eps  ──► h_bar_cpd    ──► h_bar      (B, T, D)
        h_bar            ──► token_cpd    ──► new_token  (B, T, vocab)

    Example::

        model = SteerlingMidLevelModel().to("cuda")
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
        *args,
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # The low-level wrapper has resolved `use_unknown` from the merged
        # concept config (elevated kwarg + config_source).
        use_unknown = bool(self.concept_cfg.get("use_unknown", True))
        self.use_unknown = use_unknown

        # ── PGM variables ─────────────────────────────────────────────────
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
        k_embs = ExogenousVariable("K", size=self.embedding_dim, distribution=Delta)
        k_mix = LatentVariable("k_hat", size=self.embedding_dim, distribution=Delta)

        if use_unknown:
            u = ConceptVariable(
                self.unknown_names,
                distribution=RelaxedBernoulli,
                dist_kwargs={"temperature": 0.5},
            )
            u_embs = ExogenousVariable("U", size=self.embedding_dim, distribution=Delta)
            u_mix = LatentVariable("u_hat", size=self.embedding_dim, distribution=Delta)

        epsilon_var = LatentVariable("epsilon", size=self.embedding_dim, distribution=Delta)
        h_bar = LatentVariable("h_bar", size=self.embedding_dim, distribution=Delta)
        new_token = ConceptVariable(
            "new_token",
            size=self.vocab_size,
            distribution=RelaxedOneHotCategorical,
            dist_kwargs={"temperature": 0.5},
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
        k_hat_cpd = ParametricCPD(
            "k_hat", 
            parents=self.known_names + ["K"], 
            parametrization=self.known_concept_mixer
        )

        if use_unknown:
            u_cpd = ParametricCPD(
                self.unknown_names,
                parents=["input"],
                parametrization=self.unknown_concept_head,
                shared=True,
                shared_name="unknown_concepts"
            )
            u_embs_cpd = ParametricCPD("U", parents=[], parametrization=nn.Identity())
            u_hat_cpd = ParametricCPD(
                "u_hat",
                parents=self.unknown_names + ["U"],
                parametrization=self.unknown_concept_mixer
            )

        # The epsilon CPD owns the Steerling residual correction term.
        # Its forward expects the concatenation of (input, k_hat, [u_hat])
        # along the last dim and applies the four-case logic baked in at
        # construction time.
        epsilon_cpd = ParametricCPD(
            "epsilon",
            parents=["input", "k_hat"] + (["u_hat"] if use_unknown else []),
            parametrization=self.epsilon_correction,
        )
        h_bar_cpd = ParametricCPD(
            "h_bar",
            parents=["k_hat"] + (["u_hat"] if use_unknown else []) + ["epsilon"],
            parametrization=SumOp(
                input_size=self.embedding_dim,
                n_terms=3 if use_unknown else 2,
            ),
        )
        new_token_cpd = ParametricCPD("new_token", parents=["h_bar"], parametrization=self.lm_head)

        # ── ProbabilisticModel + inference engine ─────────────────────────
        variables = [
            h, *k, k_embs, k_mix,
            *([*u, u_embs, u_mix] if use_unknown else []),
            epsilon_var, h_bar, new_token,
        ]
        factors = [
            backbone_cpd, k_cpd, k_embs_cpd, k_hat_cpd,
            *([u_cpd, u_embs_cpd, u_hat_cpd] if use_unknown else []),
            epsilon_cpd, h_bar_cpd, new_token_cpd,
        ]
        self.pgm = ProbabilisticModel(variables=variables, factors=factors)
        self.inference = inference(self.pgm, **(inference_kwargs or {}))

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _evidence(self, input_ids: torch.Tensor) -> dict:
        """Build full evidence dict: backbone hidden states + both embedding matrices."""
        hidden = self.backbone(input_ids)
        evidence = {"input": hidden}
        evidence["K"] = self.known_embeddings
        if self.use_unknown:
            evidence["U"] = self.unknown_embeddings
        return evidence

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def _split_query_specs(self):
        return [            
            ("input", self.latent_dim),
            ("known_concepts", self.n_known),
            *([("unknown_concepts", self.n_unknown)] if self.use_unknown else []),
            ("k_hat", self.embedding_dim),
            *([("u_hat", self.embedding_dim)] if self.use_unknown else []),
            ("epsilon", self.embedding_dim),
            ("h_bar", self.embedding_dim),
            ("new_token", self.vocab_size),
        ]   
    
    @property
    def default_query(self):
        return (
            ["input"]
            + self.known_names
            + self.unknown_names
            + ["k_hat"]
            + (["u_hat"] if self.use_unknown else [])
            + ["epsilon", "h_bar", "new_token"]
        )


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
        topk_concepts: int | None = None,
        verbose: bool = True,
    ) -> str:
        """Procedurally unmask and generate tokens via the PGM bottleneck.

        At each step the PGM is queried for ``new_token`` probabilities at
        all positions; the most confident masked position is filled with the
        argmax token and the process repeats.

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

        input_ids, _, _ = self.build_input(prompt, n_new_tokens)
        input_ids = input_ids.to(self.device)

        prompt_len = (input_ids[0] != mask_id).sum().item()

        if verbose:
            print(f"\nGenerating {n_new_tokens} tokens one at a time:")

        for step in range(n_new_tokens):
            # 1. Query token probabilities through the PGM
            query = ["new_token"] + (self.known_names if topk_concepts is not None else [])
            out = self.forward(input_ids, query=query).probs
            token_probs = out[..., :self.vocab_size]

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
                if topk_concepts is not None:
                    concepts = top_concepts(
                        out[0, seq_idx, self.vocab_size:],
                        topk=topk_concepts,
                    )
                    print(concepts.to_string(index=False))

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
