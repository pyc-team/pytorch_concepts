"""
Anchor-based predictors for Probabilistic Concept Bottleneck Models (ProbCBMs).

ProbCBM (Kim et al., ICML 2023, https://arxiv.org/abs/2306.01574) predicts
concepts and classes from distances in embedding space rather than from
linear logits. This means that each class has a learnable anchor in a
class-embedding space and class probabilities are a softmax over the scaled
negative distances to those.

Notice that we will use the same general class (:class:`EmbeddingAnchors`) to
hold and represent both the concept anchors and the class anchors (as they
can be seen as the same thing but in different spaces).
"""
import math

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Distribution, Normal

from ..base.layer import BaseConceptLayer


class EmbeddingAnchors(nn.Module):
    """
    A learnable table of anchor embeddings, and the distance metric over it.

    The table holds ``n_items`` items. These could be things like concepts,
    tasks, or anything else one wants to predict by proximity. Each of these
    items will be represented by one anchor per state it can take.
    Here, a binary item is treated as having two states (negative and positive),
    so the table always has shape ``(n_items, n_states, embedding_size)``, where
    ``n_states`` is always greater than or equal to 2.

    :meth:`logits` is the prediction rule: it measures how far a given
    embedding is from each of an item's anchors and turns those distances into
    logits, scaled by the learnable :attr:`scale`.

    :meth:`interpolate` runs the rule backwards, mixing an item's anchors by a
    set of activations (e.g., concept interventions).

    Args:
        n_items: Number of items (concepts, tasks, ...) with their own anchors.
        embedding_size: Dimensionality of each anchor embedding.
        cardinality: Number of states per item, using the PyC convention that a
            binary item has cardinality 1 (and so two anchors). Default 1. For
            now, for simplicity, we assume every item has the same cardinality.
        anchors: Optional initial anchors of shape
            ``(n_items, n_states, embedding_size)``. Sampled from
            ``init_distribution`` when omitted. Useful to warm-start from, say,
            class prototypes.
        init_distribution: Distribution the initial anchors are drawn from,
            which is how tables living in different spaces get initialised
            differently. Defaults to ``Normal(0, 1 / sqrt(embedding_size))``,
            and is ignored when ``anchors`` is given.
        init_scale: Initial value of the learnable distance scale. Default 5.
        per_item_scale: Whether each item gets its own distance scale. ``False``
            (the default) shares a single scale across the whole table, which
            is what ProbCBM does; ``True`` lets items calibrate independently,
            which helps when they are not equally easy to separate.
        normalize: Whether to L2-normalise the anchors at use, which puts them
            on the unit hypersphere alongside the embeddings they are compared
            against. Default True.
        distance_reduction: How to reduce the squared differences over the
            embedding axis, either ``"sum"`` (a plain Euclidean norm) or
            ``"mean"``. Notice that the two differ by a constant factor that the
            learnable :attr:`scale` could in theory absorb; the reference
            implementation happens to use ``"sum"`` for concepts and ``"mean"``
            for classes. Default is ``"sum"``.
        eps: Stabiliser added inside the square root. Default 1e-6.
    """

    def __init__(
        self,
        n_items: int,
        embedding_size: int,
        cardinality: int = 1,
        anchors: Optional[torch.Tensor] = None,
        init_distribution: Optional[Distribution] = None,
        init_scale: float = 5.0,
        per_item_scale: bool = False,
        normalize: bool = True,
        distance_reduction: str = "sum",
        eps: float = 1e-6,
    ):
        super().__init__()
        if distance_reduction not in ["sum", "mean"]:
            raise ValueError(
                f"distance_reduction must be 'sum' or 'mean', got "
                f"{distance_reduction!r}."
            )
        if not isinstance(cardinality, int):
            # TODO: support per-item cardinalities
            raise TypeError(
                f"cardinality must be an int, got {type(cardinality).__name__}."
                f" We do not currently support per-item cardinalities, so this "
                f"must be a single int."
            )
        if cardinality < 1:
            raise ValueError(
                f"cardinality must be >= 1, got {cardinality}."
            )
        if n_items < 1:
            raise ValueError(
                f"n_items must be >= 1, got {n_items}."
            )

        self.n_items = n_items
        self.embedding_size = embedding_size
        self.cardinality = cardinality
        # A binary item needs a (negative, positive) pair rather than a single
        # anchor, which is where the two conventions meet.
        self.n_states = cardinality if cardinality > 1 else 2
        self.per_item_scale = per_item_scale
        self.normalize = normalize
        self.distance_reduction = distance_reduction
        self.eps = eps

        anchor_shape = (n_items, self.n_states, embedding_size)
        if anchors is None:
            if init_distribution is None:
                init_distribution = Normal(
                    0.0,
                    1.0 / math.sqrt(embedding_size),
                )
            anchors = init_distribution.sample(anchor_shape)
        else:
            # We assume that the given anchors can be reshaped into
            # `anchor_shape`.
            anchors = torch.as_tensor(anchors, dtype=torch.float).reshape(
                anchor_shape
            )
        self.anchors = nn.Parameter(anchors)
        # A shared scale parameter for all items
        self.scale = nn.Parameter(torch.full(
            (n_items,) if per_item_scale else (1,),
            float(init_scale),
        ))

    @property
    def normalized(self) -> torch.Tensor:
        """
        The anchors as used, of shape ``(n_items, n_states, embedding_size)``,
        L2-normalised when :attr:`normalize` is set.
        """
        return (
            F.normalize(self.anchors, p=2, dim=-1)
            if self.normalize else self.anchors
        )

    def _anchors_for(self, items: Optional[List[int]]) -> torch.Tensor:
        """
        The anchors of ``items``, or of every item when ``items`` is None.
        """
        anchors = self.normalized
        return anchors if items is None else anchors[items]

    def _scale_for(self, items: Optional[List[int]]) -> torch.Tensor:
        """
        The distance scale of ``items``. Only a per-item scale needs slicing;
        a shared one broadcasts over every item as it is.
        """
        if items is None or not self.per_item_scale:
            return self.scale
        return self.scale[items]

    def distances(
        self,
        embeddings: torch.Tensor,
        items: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Distance from each embedding to each of its item's anchors.

        Args:
            embeddings: Embeddings of shape (batch, m * embedding_size) or
                (batch, m, embedding_size), where m is the number of items
                selected by ``items``.
            items: Optional indices selecting a subset of the table.

        Returns:
            torch.Tensor: Distances of shape (batch, m, n_states).
        """
        anchors = self._anchors_for(items)
        z = embeddings.reshape(embeddings.shape[0], -1, self.embedding_size)
        squared = (z.unsqueeze(2) - anchors.unsqueeze(0)).pow(2)
        reduced = (
            squared.sum(-1) if self.distance_reduction == "sum"
            else squared.mean(-1)
        )
        return torch.sqrt(reduced + self.eps)

    def logits(
        self,
        embeddings: torch.Tensor,
        items: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Predict each item's state from how close the embedding is to its
        anchors.

        Args:
            embeddings: Embeddings of shape (batch, m * embedding_size) or
                (batch, m, embedding_size).
            items: Optional indices selecting a subset of the table.

        Returns:
            torch.Tensor: Logits of shape (batch, m * cardinality).
        """
        distance = self.distances(embeddings, items)
        scale = self._scale_for(items)
        if self.cardinality == 1:
            # Binary item: one logit from the (negative, positive) pair.
            return scale * (distance[..., 0] - distance[..., 1])
        # ``distance`` carries a trailing state axis here, so the scale has to
        # be lined up against the item axis rather than broadcast into it.
        return (-scale.unsqueeze(-1) * distance).flatten(start_dim=1)

    def interpolate(
        self,
        values: torch.Tensor,
        items: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Mix each item's anchors by its activations, the inverse of
        :meth:`logits`.

        Args:
            values: Activations of shape (batch, m * cardinality). For a binary
                item these are the probabilities ``v``, weighting the anchors
                by ``[1 - v, v]``.
            items: Optional indices selecting a subset of the table.

        Returns:
            torch.Tensor: Mixed embeddings of shape
                (batch, m, embedding_size).
        """
        anchors = self._anchors_for(items)
        weights = values.reshape(values.shape[0], anchors.shape[0], -1)
        if self.cardinality == 1:
            weights = torch.cat([1.0 - weights, weights], dim=-1)
        return (weights.unsqueeze(-1) * anchors.unsqueeze(0)).sum(-2)


class AnchorPredictor(BaseConceptLayer):
    """
    Predicts a set of variables from how close an embedding is to their anchors.

    This is the prediction rule ProbCBM uses on both of its levels, so the model
    instantiates it twice with different arguments rather than having two
    layers. Going from the latent to the concepts, each concept owns a
    (negative, positive) anchor pair and the embedding is read against it
    directly. Going from the concepts to the tasks, a ``projection`` first maps
    the concept embeddings into a class-embedding space and each task owns one
    anchor per class.

    The layer builds and owns its own :class:`EmbeddingAnchors` table, exposed
    as :attr:`anchors`, so nothing has to be constructed on its behalf. That
    table carries the distance scale too, which means a model splitting one
    level across several predictors gets one scale per group rather than one
    per level; ``per_item_scale`` makes that split explicit instead.

    It also accepts either representation of its input, which is what lets the
    task head sit on top of either the concept embeddings or the concept
    activations. Given *embeddings* it uses them as they are; given
    *activations* it recovers embeddings by interpolating ``source_anchors``,
    the table those activations were decoded from. The two agree whenever an
    activation is hard (an intervention, or teacher forcing), since the
    interpolation *is* the ground-truth anchor, and differ only for a soft one.

    Args:
        n_items: Number of variables (concepts, tasks, ...) this layer predicts.
        embedding_size: Dimensionality of the space the anchors live in, i.e.
            of the projection's *output* when there is one.
        cardinality: Number of states per item, using the PyC convention that a
            binary item has cardinality 1 (and so two anchors). Default 1.
        projection: Optional map applied to the input before it is compared
            against the anchors, used by the concept-to-task level to reach the
            class-embedding space. ``None`` (the default) compares directly.
        source_anchors: The :class:`EmbeddingAnchors` the layer's *activation*
            input was decoded from, needed only when something upstream feeds
            it activations rather than embeddings. Several tables may be given,
            in which case their interpolations are concatenated in order, which
            is what a bottleneck split across several plates needs.
        **anchor_kwargs: Forwarded to :class:`EmbeddingAnchors`, e.g.
            ``per_item_scale``, ``normalize``, ``distance_reduction``, ``eps``,
            ``init_distribution``, ``init_scale`` or explicit ``anchors``.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import AnchorPredictor
        >>>
        >>> predictor = AnchorPredictor(n_items=4, embedding_size=16)
        >>> logits = predictor(embeddings=torch.randn(8, 4 * 16))
        >>> print(logits.shape)
        torch.Size([8, 4])

    References:
        Kim et al. "Probabilistic Concept Bottleneck Models", ICML 2023.
        https://arxiv.org/abs/2306.01574
    """

    def __init__(
        self,
        n_items: int,
        embedding_size: int,
        cardinality: int = 1,
        projection: Optional[nn.Module] = None,
        source_anchors: Optional[
            Union[EmbeddingAnchors, Sequence[EmbeddingAnchors]]
        ] = None,
        **anchor_kwargs,
    ):
        if source_anchors is None:
            source_anchors = []
        elif isinstance(source_anchors, EmbeddingAnchors):
            source_anchors = [source_anchors]

        super().__init__(
            out_concepts=n_items * cardinality,
            # What the layer reads is whatever its parents provide: embeddings
            # of the width the projection (or the anchors) expect, and/or the
            # activations of the tables it interpolates.
            in_embeddings=(
                projection.in_features if projection is not None
                else n_items * embedding_size
            ),
            in_concepts=(
                sum(t.n_items * t.cardinality for t in source_anchors) or None
            ),
        )
        self.anchors = EmbeddingAnchors(
            n_items=n_items,
            embedding_size=embedding_size,
            cardinality=cardinality,
            **anchor_kwargs,
        )
        self.projection = projection
        self.source_anchors = nn.ModuleList(source_anchors)

    def _embeddings_from(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Recover embeddings from activations by interpolating the anchors those
        activations were decoded from, one chunk per source table.
        """
        if not self.source_anchors:
            raise ValueError(
                f"{type(self).__name__} was given concept activations but no "
                f"`source_anchors` to interpolate them from; pass the table "
                f"the activations were decoded against."
            )
        chunks = []
        start = 0
        for table in self.source_anchors:
            width = table.n_items * table.cardinality
            chunks.append(table.interpolate(concepts[:, start:start + width]))
            start += width
        return torch.cat(chunks, dim=1)

    def forward(
        self,
        concepts: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict each item's state from the input, however it is represented.

        Args:
            concepts: Activations of the variables this layer predicts from, of
                shape (batch, n_source_items * source_cardinality).
            embeddings: Embeddings of shape (batch, n * size) or (batch, n,
                size), where the widths are those the projection (or the
                anchors) expect.

        Returns:
            torch.Tensor: Logits of shape (batch, n_items * cardinality).
        """
        if embeddings is None:
            embeddings = self._embeddings_from(concepts)
        if self.projection is not None:
            embeddings = self.projection(embeddings.flatten(start_dim=1))
        return self.anchors.logits(embeddings)
