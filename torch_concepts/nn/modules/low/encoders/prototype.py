"""Prototype-grounded encoders for label-free concept supervision."""
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from torch_concepts import Annotations
from torch_concepts.nn.modules.low.base.layer import BaseConceptLayer


class PrototypeEmbeddingToConcept(BaseConceptLayer):
    """Map compatible embeddings to concept logits with frozen prototypes.

    This layer grounds each concept state in a fixed prototype from the same
    embedding space as the input. It is therefore suitable for settings such as
    CLIP, where image embeddings and text embeddings can be compared directly.
    Binary concepts use one positive-state prototype; categorical concepts use
    one prototype per state. ``state_prototypes`` must consequently contain one
    row per flattened output state in ``out_concepts`` order.

    The layer has no learned projection or bias. Its prototypes are buffers, so
    task-only optimisation cannot move their external semantic grounding.

    Args:
        in_embeddings: Embedding dimension, or its annotations.
        out_concepts: Binary/categorical output concept annotations.
        state_prototypes: Tensor with shape ``(out_concepts.size,
            in_embeddings)`` in the shared embedding space.
        temperature: Positive scale applied to similarity scores.
        similarity: Optional scorer with signature ``(embeddings, prototypes)
            -> scores``. Defaults to cosine similarity.
        prototype_labels: Optional semantic labels, one per prototype, retained
            in checkpoints for inspection.
    """

    def __init__(
        self,
        in_embeddings: Union[int, Annotations],
        out_concepts: Annotations,
        state_prototypes: torch.Tensor,
        *,
        temperature: float = 12.0,
        similarity: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        prototype_labels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(in_embeddings=in_embeddings, out_concepts=out_concepts)
        if any(kind == "continuous" for kind in out_concepts.types):
            raise ValueError(
                "PrototypeEmbeddingToConcept supports binary and categorical "
                "concepts only."
            )
        if state_prototypes.ndim != 2:
            raise ValueError(
                "state_prototypes must have shape "
                "(n_state_logits, embedding_size)."
            )
        expected_shape = (self.out_concepts_shape, self.in_embeddings_shape)
        if tuple(state_prototypes.shape) != expected_shape:
            raise ValueError(
                "Expected state_prototypes with shape "
                f"{expected_shape}, got {tuple(state_prototypes.shape)}."
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.register_buffer("state_prototypes", state_prototypes.detach().clone())
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.similarity = similarity or self._cosine_similarity

        labels = tuple(prototype_labels or self._default_prototype_labels(out_concepts))
        if len(labels) != self.out_concepts_shape:
            raise ValueError(
                f"Expected {self.out_concepts_shape} prototype labels, got {len(labels)}."
            )
        self.prototype_labels = labels

    @staticmethod
    def _cosine_similarity(
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Return one cosine similarity per prototype."""
        return F.normalize(embeddings, dim=-1) @ F.normalize(prototypes, dim=-1).T

    @staticmethod
    def _default_prototype_labels(annotations: Annotations) -> tuple[str, ...]:
        """Derive stable state labels from concept annotations."""
        labels = []
        for name in annotations.labels:
            concept = annotations.concept(name)
            if concept.is_binary:
                labels.append(name)
            else:
                labels.extend(f"{name}:{state}" for state in concept.states)
        return tuple(labels)

    def get_extra_state(self) -> dict:
        """Store non-tensor prototype semantics in a state dictionary."""
        return {"prototype_labels": self.prototype_labels}

    def set_extra_state(self, state: dict) -> None:
        self.prototype_labels = tuple(state["prototype_labels"])

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return scaled prototype-similarity logits."""
        if embeddings.shape[-1] != self.in_embeddings_shape:
            raise ValueError(
                f"Expected embeddings with final size {self.in_embeddings_shape}, got "
                f"{embeddings.shape[-1]}."
            )
        scores = self.similarity(embeddings, self.state_prototypes.to(embeddings))
        expected_shape = (*embeddings.shape[:-1], self.out_concepts_shape)
        if tuple(scores.shape) != expected_shape:
            raise ValueError(
                "similarity must return one score per prototype, with shape "
                f"{expected_shape}; got {tuple(scores.shape)}."
            )
        return scores * self.temperature.to(scores)
