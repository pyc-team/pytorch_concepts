from typing import Union

import torch

from .....annotations import Annotations
from ..base.layer import BaseConceptLayer
from ..dense_layers import MLP


class RuleMemory(torch.nn.Module):
    """Learnable rule memory decoded into categorical role probabilities.

    During training the decoded roles remain soft probabilities. During eval,
    ``hard_at_eval=True`` converts each 3-way role categorical to its argmax
    one-hot mode.

    References:
        Debot et al. "Interpretable Concept-Based Memory Reasoning", NeurIPS 2024.
        https://arxiv.org/abs/2407.15527
    """

    def __init__(
        self,
        n_tasks: int,
        n_rules: int,
        n_concepts: int,
        latent_size: int = 64,
        hidden_layers: int = 1,
        hard_at_eval: bool = False,
    ):
        super().__init__()
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative.")
        self.hard_at_eval = hard_at_eval
        self.shape = (n_tasks, n_rules, n_concepts, 3)
        width = n_rules * n_concepts * 3
        self.memory = torch.nn.Embedding(n_tasks, latent_size)
        decoder = (
            torch.nn.Linear(latent_size, width)
            if hidden_layers == 0
            else MLP(
                input_size=latent_size,
                hidden_size=width,
                output_size=width,
                n_layers=hidden_layers,
                activation="leaky_relu",
            )
        )
        self.decoder = torch.nn.Sequential(
            decoder,
            torch.nn.Unflatten(-1, (n_rules, n_concepts, 3)),
        )

    def forward(self):
        pred = torch.softmax(self.decoder(self.memory.weight), dim=-1)
        if (not self.training) and self.hard_at_eval:
            idx = pred.argmax(dim=-1)
            pred = torch.nn.functional.one_hot(
                idx, num_classes=pred.shape[-1]
            ).to(pred.dtype)
        assert torch.all((pred >= 0) & (pred <= 1)), (
            "Decoded memory should be in [0, 1]"
        )
        return pred


class RuleConceptEmbeddingToConcept(BaseConceptLayer):
    """Compute CMR task probabilities from concepts and rule embeddings.

    The embedding input packs the flattened selector followed by the flattened
    rule-role tensor. This layer inherently returns probabilities, not
    logits.

    ``rec_weight`` selects which of CMR's two task paths this layer computes.
    At ``0`` each rule contributes its task-satisfaction probability alone --
    the ordinary prediction. Above ``0`` that probability is additionally
    scaled by the rule's reconstruction probability raised to ``rec_weight``,
    penalising rules that mark concepts irrelevant; larger weights make
    reconstruction more influential.

    Args:
        out_concepts: Number or annotation of output tasks.
        in_concepts: Number or annotation of input concepts.
        in_embeddings: Width of the packed rule embedding; must equal
            ``out_concepts * n_rules * (1 + 3 * in_concepts)``.
        n_rules: Number of rules stored per task.
        rec_weight: Non-negative exponent on each rule's reconstruction
            probability. ``0`` disables reconstruction.

    References:
        Debot et al. "Interpretable Concept-Based Memory Reasoning", NeurIPS 2024.
        https://arxiv.org/abs/2407.15527
    """

    def __init__(
        self,
        out_concepts: Union[int, Annotations],
        in_concepts: Union[int, Annotations] = None,
        in_embeddings: Union[int, Annotations] = None,
        *,
        n_rules: int,
        rec_weight: float = 0.0,
    ):
        super().__init__(
            out_concepts=out_concepts,
            in_concepts=in_concepts,
            in_embeddings=in_embeddings,
        )
        if self.in_concepts_shape is None or self.in_embeddings_shape is None:
            raise ValueError("CMR rule predictors require concepts and embeddings.")
        if n_rules <= 0:
            raise ValueError("n_rules must be positive.")
        if rec_weight < 0:
            raise ValueError("rec_weight must be non-negative.")
        self.n_rules = n_rules
        self.rec_weight = rec_weight
        expected = self.out_concepts_shape * n_rules * (
            1 + 3 * self.in_concepts_shape
        )
        if self.in_embeddings_shape != expected:
            raise ValueError(
                f"in_embeddings must be {expected} for "
                f"{self.out_concepts_shape} tasks, {n_rules} rules, and "
                f"{self.in_concepts_shape} concepts; got {self.in_embeddings_shape}."
            )

    def _unpack_embeddings(self, embeddings: torch.Tensor):
        # TODO: in the future, we can consider extending the number of inputs for layers
        # so that one can pass args separately instaed of packing them into a single tensor. 
        # For now, we unpack here.
        selector_size = self.out_concepts_shape * self.n_rules
        selector, roles = embeddings.split(
            [selector_size, self.in_embeddings_shape - selector_size], dim=-1
        )
        selector = selector.unflatten(
            -1, (self.out_concepts_shape, self.n_rules)
        )
        roles = roles.unflatten(
            -1,
            (
                self.out_concepts_shape,
                self.n_rules,
                self.in_concepts_shape,
                3,
            ),
        )
        return selector, roles

    def forward(
        self,
        concepts: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score every rule against the concepts and mix them by the selector.

        Args:
            concepts: Concept *probabilities* in [0, 1] of shape
                ``(..., in_concepts)``. Detached here, so no gradient flows
                back to the concept encoder through this layer.
            embeddings: Packed rule embeddings of shape
                ``(..., in_embeddings)``, where ``in_embeddings`` equals
                ``out_concepts * n_rules * (1 + 3 * in_concepts)``.

        Returns:
            torch.Tensor: Task probabilities of shape ``(..., out_concepts)``,
            clamped away from 0 and 1 by ``eps``.
        """
        selector, roles = self._unpack_embeddings(embeddings)
        c = concepts.detach().unsqueeze(-2).unsqueeze(-2)
        per_rule = (
            c * roles[..., 0]
            + (1.0 - c) * roles[..., 1]
            + roles[..., 2]
        ).prod(dim=-1)
        if self.rec_weight:
            # An irrelevant concept reconstructs with probability 0.5, so a
            # rule that ignores concepts is scored down.
            reconstruction_per_rule = (
                c * roles[..., 0]
                + (1.0 - c) * roles[..., 1]
                + 0.5 * roles[..., 2]
            ).prod(dim=-1)
            per_rule = per_rule * torch.pow(
                reconstruction_per_rule + 1e-6, self.rec_weight
            )
        pred = (per_rule * selector).sum(dim=-1)
        eps = 0.0001
        pred = eps + (1 - 2 * eps) * pred
        return pred
