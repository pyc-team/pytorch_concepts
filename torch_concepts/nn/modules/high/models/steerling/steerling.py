"""
Steerling model — PGM-backed concept bottleneck LM (recommended entry point).

:class:`SteerlingModel` is the high-level, test-time interface to Steerling.
It builds on :class:`SteerlingLowLevelModel` but routes all computation through
a :class:`~torch_concepts.nn.BayesianNetwork`, so concepts, latents, and
tokens can be queried by name.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, OneHotCategorical

import pandas as pd

from ......distributions import Delta
from ......concept_graph import ConceptGraph
from ......annotations import Annotations
from ....mid.variable import ConceptVariable, EmbeddingVariable
from ....mid.graph.bayesian_network import BayesianNetwork
from ....mid.factors.cpd import ParametricCPD
from ....mid.inference.base import BaseInference
from ....mid.inference.torch.deterministic import DeterministicInference
from ....low.priors import FixedPrior, TiedPrior
from .....modules.outputs import ModelOutput

from .steerling_low import SteerlingLowLevelModel
from .steerling_utils import (
    load_steerling_concept_names,
    top_concepts,
)

logger = logging.getLogger(__name__)


class SteerlingModel(SteerlingLowLevelModel):
    """PGM-backed Steerling concept-bottleneck language model.

    Same construction interface as :class:`SteerlingLowLevelModel`, but wraps
    its modules in a :class:`~torch_concepts.nn.BayesianNetwork` so individual
    variables (concepts, latents, tokens) can be queried by name. Unlike the
    low-level model, :meth:`forward` returns a
    :class:`~torch_concepts.nn.ModelOutput` and takes an optional ``query``.

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

        model = SteerlingModel().to("cuda")
        model.eval()

        # End-to-end: the default query returns known concepts + next token.
        # Every queried variable that reports logits shares one annotated
        # tensor, sliced by variable name.
        out = model(input_ids=input_ids)
        concept_logits = out.logits["concepts"]      # (1, T, n_known)
        token_logits   = out.logits["new_token"]     # (1, T, vocab)

        # Query a single named concept (its logits)
        logits = model(input_ids=input_ids, query=["food"]).logits["food"]  # (1, T, 1)

        # Concept-based hidden-state reconstruction (query the latent explicitly).
        # The latents are Delta variables, so they report `value`, not `logits`.
        out = model(input_ids=input_ids, query=["h_bar"])
        h_bar = out.value["h_bar"]  # (1, T, D)

        # Generation
        model.generate("As an Italian living abroad I miss", n_new_tokens=20)
    """

    def __init__(
        self,
        annotations: Optional[Annotations] = None,
        inference: Optional[BaseInference] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        graph=None,
        lightning: bool = False,
        *args,
        **kwargs,
    ):
        # SteerlingModel is a test-time model: it has no training 
        # engine and builds its concept annotations internally 
        # from the (pretrained) concept heads.
        if lightning:
            raise ValueError(
                "SteerlingModel is a test-time model and does not support "
                "Lightning training; pass lightning=False (the default)."
            )
        if train_inference is not None:
            raise ValueError(
                "SteerlingModel is a test-time model; a training inference "
                "engine is not available. Leave train_inference=None — the "
                "evaluation inference is used for all queries."
            )
        super().__init__(*args, **kwargs)

        # Concept annotations are derived from the concept heads, whose concept
        # structure is fixed once those heads are pretrained. Refuse a
        # caller-supplied annotations/graph in that case.
        concept_pretrained = [
            component
            for component in ("known_head", "unknown_head")
            if component in self.pretrained_components
        ]
        if (annotations is not None or graph is not None) and concept_pretrained:
            raise ValueError(
                f"Cannot pass `annotations` or `graph` when concept heads are pretrained "
                f"({concept_pretrained}); they are built internally."
            )

        # ── concept names ─────────────────────────────────────────────────
        self.known_names   = load_steerling_concept_names() # list[str]
        if len(self.known_names) != self.n_known:
            raise ValueError(
                "SteerlingModel requires n_concepts to match the "
                f"known-concept CSV ({len(self.known_names)}), got {self.n_known}."
            )
        if self.concept_cfg['use_unknown']:
            self.unknown_names = [f"unknown_{i}" for i in range(self.unknown_head.out_concepts)]
        else:
            self.unknown_names = []

        # ── High-level BaseModel API mirror ───────────────────────────────
        # SteerlingModel builds a bipartite concept→token graph, 
        # a backbone, and concept annotations built internally from the concept heads.
        # NOTE: these are not used for now, but are retained for the high-level BaseModel API mirror.
        self.concept_annotations = self._build_annotations()
        self.task_names = ["new_token"]
        self.graph = self._build_graph()

        self.use_unknown = self.concept_cfg["use_unknown"]

        # ── PGM variables ─────────────────────────────────────────────────
        # FIXME: the `input` token should be a `Categorical` over the vocabulary, which 
        # is not yet supported in PyC. We cannot use `OneHotCategorical`, as (B,T,vocab) 
        # would be inefficient considering the backbone expects integer ids.
        # Here, we model it as a Delta over the index and always require evidence over 'inputs'.
        input = EmbeddingVariable("input", distribution=Delta, size=1)
        h = EmbeddingVariable("h", distribution=Delta, size=self.latent_size)
        k = ConceptVariable("concepts", members=self.known_names, distribution=Bernoulli)
        # Concept-embedding matrices are fixed model state (``TiedPrior`` roots,
        # below), not evidence; their shape is fixed by the pretrained heads.
        k_embs = EmbeddingVariable("embeddings", distribution=Delta, shape=tuple(self.known_embeddings_shape))
        k_hat = EmbeddingVariable("k_hat", distribution=Delta, size=self.embedding_size)

        if self.use_unknown:
            u = ConceptVariable("unknown_concepts", members=self.unknown_names, distribution=Bernoulli)
            u_embs = EmbeddingVariable(
                "unknown_embeddings",
                distribution=Delta,
                shape=tuple(self.unknown_embeddings_shape(cat_factorized=True)),
            )
            u_hat = EmbeddingVariable("u_hat", distribution=Delta, size=self.embedding_size)

        epsilon = EmbeddingVariable("epsilon", distribution=Delta, size=self.embedding_size)
        h_bar = EmbeddingVariable("h_bar", distribution=Delta, size=self.embedding_size)
        new_token = ConceptVariable("new_token", distribution=OneHotCategorical, size=self.vocab_size)

        # ── CPDs ──────────────────────────────────────────────────────────
        # FIXME: placeholder prior for `input` until PyC supports a Categorical over the vocabulary.
        # (see above)
        input_cpd = ParametricCPD(variable=input, parents=[], parametrization=FixedPrior(torch.zeros(1)))
        h_cpd = ParametricCPD(
            variable=h,
            parents=[input],
            parametrization=self.backbone,
            # NOTE: `input` carries token ids shaped ``(B, T, 1)``; the backbone expects ``(B, T)``.
            # So the the aggregate squeezes the event axis so the backbone runs on the intact ``(B, T)`` sequence 
            # and emits ``(B, T, D)``.
            aggregate=lambda parent_values: next(iter(parent_values.values())).squeeze(-1),
        )
        k_cpd = ParametricCPD(
            variable=k, 
            parents=[h], 
            parametrization={"logits": self.known_head}
        )
        k_embs_cpd = ParametricCPD(
            variable=k_embs,
            parents=[],
            parametrization=TiedPrior(self.known_embeddings, broadcast=False)
        )
        k_hat_cpd = ParametricCPD(
            variable=k_hat,
            parents=[k, k_embs],
            parametrization=self.known_concept_mixer,
        )

        if self.use_unknown:
            u_cpd = ParametricCPD(
                variable=u, 
                parents=[h], 
                parametrization={"logits": self.unknown_head}
            )
            u_embs_cpd = ParametricCPD(
                variable=u_embs,
                parents=[],
                parametrization=TiedPrior(lambda: self.unknown_embeddings(cat_factorized=True), broadcast=False),
            )
            u_hat_cpd = ParametricCPD(
                variable=u_hat,
                parents=[u, u_embs],
                parametrization=self.unknown_concept_mixer,
            )

        # epsilon_correction consumes the ordered list [h, k_hat, (u_hat)] =
        # [target, *parts]; the aggregate hands it exactly that list.
        epsilon_cpd = ParametricCPD(
            variable=epsilon,
            parents=[h, k_hat] + ([u_hat] if self.use_unknown else []),
            parametrization=self.epsilon_correction,
            aggregate=lambda parent_values: list(parent_values.values()),
        )
        # h_bar = sum(parts) + epsilon = k_hat (+ u_hat) + epsilon.
        h_bar_cpd = ParametricCPD(
            variable=h_bar,
            parents=[k_hat] + ([u_hat] if self.use_unknown else []) + [epsilon],
            parametrization=nn.Identity(),
            aggregate=lambda parent_values: sum(parent_values.values()),
        )
        new_token_cpd = ParametricCPD(
            variable=new_token, 
            parents=[h_bar], 
            parametrization={"logits": self.lm_head}
        )

        # ── BayesianNetwork + inference engine ────────────────────────────
        variables = [
            input, h, k, k_embs, k_hat,
            *([u, u_embs, u_hat] if self.use_unknown else []),
            epsilon, h_bar, new_token,
        ]
        factors = [
            input_cpd, h_cpd, k_cpd, k_embs_cpd, k_hat_cpd,
            *([u_cpd, u_embs_cpd, u_hat_cpd] if self.use_unknown else []),
            epsilon_cpd, h_bar_cpd, new_token_cpd,
        ]
        self.model = BayesianNetwork(variables=variables, factors=factors)
        # Test-time model: a single (evaluation) inference engine, no training inference.
        self.eval_inference = inference(self.model, **(inference_kwargs or {}))
        self.train_inference = None

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _default_evidence(self, input_ids: torch.Tensor) -> dict:
        """Build the evidence dict for a forward pass.

        Only ``input`` is observed: the token ids, shaped ``(B, T, 1)`` to match
        the variable's ``(1,)`` event, kept integer for the backbone's embedding
        lookup. The concept-embedding matrices are model state (root priors), not
        evidence.
        """
        return {"input": input_ids.unsqueeze(-1)}

    def _default_query(self) -> list[str]:
        return ["concepts",  "new_token"]
    
    def _build_annotations(self) -> Annotations:
        """Build concept annotations from the (pretrained) concept heads.

        Mirrors the high-level :class:`BaseModel`, which receives an
        :class:`~torch_concepts.annotations.Annotations` describing the concept
        variables.  SteerlingModel's concepts are fixed — the known-concept CSV,
        the ``unsup_i`` unknown concepts, and the categorical ``new_token`` — so
        the annotation is constructed internally rather than supplied by a
        datamodule.

        The ``types`` + cardinalities below describe the concepts; the model
        itself owns how each is modelled (``Bernoulli`` / sigmoid for the
        binary known/unknown concepts and ``OneHotCategorical`` / softmax
        for the categorical ``new_token``), set explicitly on its variables.

        Returns:
            Annotations: axis-1 annotation (labels, types, cardinalities).
        """
        from torch_concepts.annotations import Annotations

        labels = list(self.known_names) + list(self.unknown_names) + ["new_token"]
        cardinalities = (
            [1] * self.n_known
            + [1] * len(self.unknown_names)
            + [self.vocab_size]
        )
        types = ['binary'] * (self.n_known + len(self.unknown_names)) + ['categorical']
        return Annotations(
            labels=labels,
            cardinalities=cardinalities,
            types=types,
        )

    def _build_graph(self) -> ConceptGraph:
        """Build the bipartite concept→token graph.

        Mirrors :class:`~torch_concepts.nn.modules.high.base.bipartite.BipartiteMixin`:
        every concept (known + unknown) points to the single ``new_token`` task,
        and ``new_token`` has no outgoing edges. Nodes are the concept-annotation
        labels, in annotation order.

        Returns:
            ConceptGraph: the concept→token bipartite adjacency.
        """
        labels = list(self.concept_names)
        missing = [t for t in self.task_names if t not in labels]
        assert not missing, (
            f"All task_names must be annotation labels; {missing} are not in {labels}."
        )
        task_set = set(self.task_names)
        concept_idx = [i for i, name in enumerate(labels) if name not in task_set]
        task_idx = [i for i, name in enumerate(labels) if name in task_set]

        source = torch.repeat_interleave(torch.tensor(concept_idx), len(task_idx))
        target = torch.tensor(task_idx).repeat(len(concept_idx))
        edge_index = torch.stack([source, target])
        edge_weight = torch.ones(edge_index.shape[1])

        return ConceptGraph.from_sparse(
            edge_index,
            edge_weight,
            n_nodes=len(labels),
            node_names=labels,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def inference(self):
        """Active inference engine.

        Mirrors :class:`~torch_concepts.nn.modules.high.base.model.BaseModel`,
        which selects the engine by ``self.training``.  This is a test-time
        model with no training engine, so the evaluation engine is always
        returned.
        """
        if self.training and self.train_inference is not None:
            return self.train_inference
        return self.eval_inference

    # ------------------------------------------------------------------
    # Forward - mirror high-level PyC models
    # ------------------------------------------------------------------

    def forward(
        self,
        query: Optional[list[str]] = None,
        evidence: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Run inference over the concept-bottleneck PGM.

        Args:
            input_ids: Token ids ``(B, T)``. Required when ``evidence`` is None.
            query: Variable names to return. Defaults to :attr:`default_query`.
            evidence: Observed variables. Defaults to the evidence built from
                ``input_ids``.

        Returns:
            ModelOutput: quantity-keyed ``params`` (``out.logits``,
            ``out.value``, ...), each an annotated tensor sliceable by variable
            name and token-aligned ``(B, T, ...)``.
        """
        if query is None:
            query = self._default_query()
        if evidence is None:
            if input_ids is None:
                raise ValueError(
                    "forward requires explicit `input_ids` " \
                    "when `evidence` is not provided."
                )
            evidence = self._default_evidence(input_ids)

        # FIXME: placeholder assert for `input` until PyC supports a Categorical 
        # over the vocabulary and simple interface to mix inferences (see above)
        # (events should be passed to backbone, params should be passed to other layers)
        assert "input" in evidence, (
            "evidence must always include 'input' (the token sequence)."
        )

        result = self.inference.query(query, evidence=evidence)
        return ModelOutput(
            params=result.params,
            guide_params=result.guide_params,
            samples=result.samples,
            probabilities=result.probabilities,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

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

        # Special tokens that must never be generated (see the low-level
        # generate): the mask token especially, which the LM scores highly at
        # unfilled positions and would leave the slot still masked.
        banned_ids = [mask_id]
        if tokenizer.pad_token_id is not None:
            banned_ids.append(tokenizer.pad_token_id)

        input_ids, _, _ = self.build_input(prompt, n_new_tokens)
        input_ids = input_ids.to(self.device)

        prompt_len = (input_ids[0] != mask_id).sum().item()

        if verbose:
            print(f"\nGenerating {n_new_tokens} tokens one at a time:")

        for step in range(n_new_tokens):
            # 1. Query token (and optionally concept) parameters through the PGM
            query = ["new_token"] + (["concepts"] if topk_concepts is not None else [])
            out = self.forward(query=query, input_ids=input_ids)
            logits = out.logits["new_token"].clone()                  # (1, T, vocab)
            logits[..., banned_ids] = float("-inf")                   # never emit mask / pad
            token_probs = torch.softmax(logits, dim=-1)

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
                    # `top_concepts` takes raw logits and applies the sigmoid itself.
                    concepts = top_concepts(out.logits["concepts"][0, seq_idx], topk=topk_concepts)
                    print(concepts.to_string(index=False))

        generated_ids  = input_ids[0, prompt_len:].tolist()
        generated_text = tokenizer.decode(generated_ids)

        if verbose:
            print(f"\n{prompt}{generated_text}")

        return generated_text

    def __repr__(self) -> str:
        return (
            f"SteerlingModel("
            f"n_known={self.n_known}, "
            f"n_unknown={self.n_unknown}, "
            f"latent_size={self.latent_size}, "
            f"vocab={self.vocab_size})"
        )
