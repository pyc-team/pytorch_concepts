"""Graph generation and embedding aggregation."""

import torch
import torch.nn as nn


class GraphAggregator(nn.Module):
    """Aggregate source embeddings through a fixed or learned adjacency.

    Exactly one of ``generator`` and ``adjacency`` must be provided. With a
    generator, the first forward after :meth:`clear` calls ``generator()`` and
    stores the resulting adjacency in ``_last_adjacency``. Later forwards reuse
    that same adjacency until :meth:`clear` is called again. In CGM training
    this makes all endogenous CPDs evaluated in the same query use one graph,
    while still exposing the adjacency to graph regularization losses.

    Passing ``adjacency=...`` to :meth:`forward` is meant for fixed/evaluation
    contexts. During training with a learnable generator it is rejected because
    bypassing the generator would detach the graph parameters from the loss.
    """

    def __init__(self, generator=None, adjacency=None):
        super().__init__()
        if (generator is None) == (adjacency is None):
            raise ValueError("Pass exactly one of generator or adjacency.")
        self.generator = generator
        self.register_buffer(
            "fixed_adjacency",
            None if adjacency is None else adjacency.detach().clone(),
        )

    def graph(self):
        """Return and cache the learned or fixed adjacency for this query."""
        adjacency = (
            self.generator()
            if self.generator is not None
            else self.fixed_adjacency
        )
        self._last_adjacency = adjacency
        return adjacency

    def clear(self):
        """Forget the cached adjacency so the next forward materializes it."""
        self._last_adjacency = None

    @property
    def adjacency(self):
        if getattr(self, "_last_adjacency", None) is None:
            raise RuntimeError("The graph layer has not generated a graph yet.")
        return self._last_adjacency

    def forward(
        self, source_embeddings, *, adjacency=None,
        source_concepts=None, target_concept=None,
    ):
        """Aggregate ``source_embeddings`` over source-to-target edges.

        ``source_embeddings`` has trailing shape ``(source, embedding)`` and
        the adjacency has trailing shape ``(source, target)``. When
        ``source_concepts`` is provided, only those adjacency rows are used.
        When ``target_concept`` is provided, only that target column is
        returned; otherwise the full ``(..., target, embedding)`` tensor is
        returned.
        """
        if adjacency is not None and self.training and self.generator is not None:
            raise RuntimeError(
                "GraphAggregator received an explicit adjacency during training. "
                "Learned CGM training must call the graph generator so gradients "
                "flow through the graph parameters."
            )
        if adjacency is None:
            adjacency = (
                self.graph()
                if getattr(self, "_last_adjacency", None) is None
                else self._last_adjacency
            )
        # Evaluation graphs can be materialized after the parent model has
        # already been moved to an accelerator. Modules created at that point
        # retain the graph original (usually CPU) device, so align the graph
        # with the runtime embeddings before the batched matrix product.
        adjacency = adjacency.to(source_embeddings)
        if adjacency.shape[-2] != adjacency.shape[-1]:
            raise ValueError("adjacency must be square.")
        if source_concepts is not None:
            adjacency = adjacency[..., source_concepts, :]
        if source_embeddings.shape[-2] != adjacency.shape[-2]:
            raise ValueError("source embeddings and adjacency rows must match.")
        # Match the original CGM equation layer:
        #   source_embeddings:          (..., source, embedding)
        #   source_embeddings_by_dim:   (..., embedding, source)
        #   adjacency:                  (..., source, target)
        #   aggregated_by_dim:          (..., embedding, target)
        #   aggregated:                 (..., target, embedding)
        #
        # For each target j this computes:
        #   aggregated[..., j, :] = sum_i adjacency[..., i, j] * embedding_i
        source_embeddings_by_dim = source_embeddings.transpose(-2, -1)
        aggregated_by_dim = torch.matmul(source_embeddings_by_dim, adjacency)
        aggregated = aggregated_by_dim.transpose(-2, -1)

        if target_concept is not None:
            n_targets = adjacency.shape[-1]
            if not 0 <= target_concept < n_targets:
                raise IndexError(
                    f"target_concept must be in [0, {n_targets}), got {target_concept}."
                )
            return aggregated[..., target_concept, :]
        return aggregated


__all__ = ["GraphAggregator"]
