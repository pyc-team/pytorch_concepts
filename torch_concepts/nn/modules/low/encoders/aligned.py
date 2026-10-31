"""Linear projection fitted to reference concept activation patterns."""
import math
import random
from typing import Union

import torch

from torch_concepts import Annotations
from ....functional import cosine_cubed_similarity
from .linear import LinearEmbeddingToConcept


class AlignedEmbeddingToConcept(LinearEmbeddingToConcept):
    """Bias-free linear encoder fitted with LF-CBM's cosine-cubed objective.

    ``fit`` consumes frozen sample-by-feature embeddings and sample-by-concept
    reference scores. Adam stops at the first validation non-improvement and
    restores the best weights. ``alignment_scores_`` contains the resulting
    per-concept validation similarities on CPU; ``history_`` and ``best_step_``
    record selection. Move the module to the training device before fitting.
    The fitted encoder is frozen and remains usable as a regular PyC layer.
    """

    def __init__(self, in_embeddings: int, out_concepts: Union[int, Annotations]):
        super().__init__(in_embeddings, out_concepts, bias=False)

    def fit(self, train_embeddings: torch.Tensor, train_scores: torch.Tensor,
            val_embeddings: torch.Tensor, val_scores: torch.Tensor, *,
            steps=1000, batch_size=50000, lr=.001, validation_interval=50, seed=0) -> "AlignedEmbeddingToConcept":
        """Fit from seeded weights, without differentiating through input data.

        Each split needs at least two observations. ``steps`` bounds Adam
        updates; ``batch_size`` bounds sampled training rows per update.
        Validation runs at step zero, every ``validation_interval`` updates,
        and the final step. Optimizer defaults follow the LF-CBM paper.
        """
        from ...loss import CosineCubedLoss

        if min(steps, batch_size, validation_interval) < 1 or lr <= 0:
            raise ValueError("Projection steps, batch sizes, validation interval, and learning rate must be positive.")
        for embeddings, scores in ((train_embeddings, train_scores), (val_embeddings, val_scores)):
            if (embeddings.ndim != 2 or scores.ndim != 2 or len(embeddings) < 2
                    or embeddings.shape[1] != self.encoder.in_features
                    or scores.shape != (len(embeddings), self.encoder.out_features)):
                raise ValueError("Expected matching sample-by-feature embeddings and sample-by-concept scores.")
        weight = self.encoder.weight
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            initial = torch.empty(weight.shape, dtype=weight.dtype).uniform_(
                -1 / math.sqrt(self.encoder.in_features), 1 / math.sqrt(self.encoder.in_features),
                generator=generator,
            )
            weight.copy_(initial)
        self.requires_grad_(True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = CosineCubedLoss()
        rng = random.Random(seed)
        best_loss, best_weight, best_step = float("inf"), None, None
        history = []
        for step in range(steps):
            indices = rng.sample(range(len(train_embeddings)), min(batch_size, len(train_embeddings)))
            optimizer.zero_grad()
            loss = criterion(self(train_embeddings[indices].detach().to(weight)),
                             train_scores[indices].detach().to(weight.device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite projection loss; check feature and annotation variation.")
            loss.backward()
            optimizer.step()
            if step % validation_interval == 0 or step == steps - 1:
                with torch.no_grad():
                    val_loss = float(criterion(self(val_embeddings.to(weight)), val_scores.to(weight.device)))
                if not math.isfinite(val_loss):
                    raise ValueError("Nonfinite validation alignment; check feature variation.")
                history.append({"step": step, "train_loss": float(loss.detach()), "val_loss": val_loss})
                if val_loss < best_loss:
                    best_loss, best_step = val_loss, step
                    best_weight = weight.detach().clone()
                else:
                    break
        with torch.no_grad():
            weight.copy_(best_weight)
            alignment = cosine_cubed_similarity(self(val_embeddings.to(weight)), val_scores.to(weight.device))
        self.alignment_scores_ = alignment.cpu()
        self.history_, self.best_step_ = history, best_step
        return self.eval().requires_grad_(False)
