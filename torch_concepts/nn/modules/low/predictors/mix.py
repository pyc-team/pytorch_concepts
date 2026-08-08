import torch

from torch_concepts import Annotations
from ..base.layer import BaseConceptLayer
from ....functional import grouped_concept_exogenous_mixture, replace_expand_cols
from typing import List, Union


class MixConceptEmbedding(BaseConceptLayer):
    """Shared machinery of the concept/embedding mixture layers.

    Holds the per-group mixture :math:`\\hat{c}_i w^+_i + (1 - \\hat{c}_i) w^-_i`
    of "Concept Embedding Models" (Espinosa Zarlenga et al., NeurIPS 2022) and
    nothing else: subclasses add whatever head consumes the mixed embeddings.

    A binary concept scores one Bernoulli probability but always mixes **two**
    state embeddings, :math:`w^+` and :math:`w^-`. ``expand_binary_embeddings``
    only decides where the second row comes from: ``True`` (default) takes one
    row per binary concept and fabricates the other with a learned
    ``Linear(m, 2m) + LeakyReLU``; ``False`` expects both rows from the caller —
    what
    :meth:`~torch_concepts.nn.modules.high.base.model.BaseModel.build_concept_embedding_variables`
    now supplies. The score axis is expanded to ``[c, 1-c]`` either way.

    Args:
        in_concepts: Annotations of the input concepts (their cardinalities and
            types drive the grouping).
        in_embeddings: Number of embedding features per state.
        out_concepts: Output width, interpreted by the subclass.
        expand_binary_embeddings: Fabricate a binary concept's second state
            embedding from its first. Default ``True`` (the historical contract).
    """

    def __init__(
        self,
        in_concepts: Annotations,
        in_embeddings: Union[int, Annotations],
        out_concepts: Union[int, Annotations],
        expand_binary_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_concepts=in_concepts,
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        self.expand_binary_embeddings = bool(expand_binary_embeddings)

        # `cardinalities_expanded`: embedding-row group sizes for
        # `grouped_concept_exogenous_mixture` (2 for binary, cardinality otherwise).
        # `binary_concept_columns`: those concepts' offsets in the unexpanded score
        # axis. Non-persistent buffers so they follow `.to(device)` without adding
        # state_dict keys older checkpoints would be missing.
        cardinalities = torch.tensor(in_concepts.cardinalities)
        binary_mask = torch.tensor([t == 'binary' for t in in_concepts.types], dtype=torch.bool)
        start_positions = torch.cumsum(cardinalities, dim=0) - cardinalities
        cardinalities_expanded = cardinalities.clone()
        cardinalities_expanded[binary_mask] = 2
        self.register_buffer("cardinalities_expanded", cardinalities_expanded, persistent=False)
        self.register_buffer("binary_concept_columns", start_positions[binary_mask], persistent=False)

        # Allocated only when used, so an all-categorical annotation (or
        # `expand_binary_embeddings=False`) carries no dead parameters.
        self.bernoulli_to_categorical_embedding_splitter = None
        if self.expand_binary_embeddings and len(self.binary_concept_columns) > 0:
            self.bernoulli_to_categorical_embedding_splitter = torch.nn.Sequential(
                torch.nn.Linear(self.in_embeddings_shape, self.in_embeddings_shape*2),
                torch.nn.LeakyReLU(),
                torch.nn.Unflatten(-1, (-1, self.in_embeddings_shape)),
            )

    def _mix(
        self,
        concepts: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess inputs and compute per-group mixed embeddings.

        Expands a binary concept's score to ``[c, 1-c]`` and — if
        :attr:`expand_binary_embeddings` — its embedding row to two. Returns
        ``c_mix`` of shape ``(batch, n_groups, in_embeddings)``; subclasses vary
        only the final aggregation.
        """
        idx = self.binary_concept_columns
        if len(idx) > 0:
            c = concepts[:, idx[:, None]]                       # (B, n_binary, 1)
            concepts_split = torch.cat([c, 1 - c], dim=-1)       # (B, n_binary, 2)
            if self.expand_binary_embeddings:
                embeddings_split = self.bernoulli_to_categorical_embedding_splitter(embeddings[:, idx])
                embeddings = replace_expand_cols(embeddings, idx, embeddings_split)
            concepts = replace_expand_cols(concepts, idx, concepts_split)
        return grouped_concept_exogenous_mixture(
            embeddings,
            concepts,
            groups=self.cardinalities_expanded.tolist(),
        )


class MixConceptEmbeddingToConcept(MixConceptEmbedding):
    """
    Concept predictor that mixes concept activations with embeddings.

    This predictor implements the Concept Embedding Model (CEM) task predictor that
    combines concept activations with learned embeddings using a mixture operation.

    Main reference: "Concept Embedding Models: Beyond the Accuracy-Explainability
    Trade-Off" (Espinosa Zarlenga et al., NeurIPS 2022).

    Attributes:
        in_concepts (int): Number of input concepts.
        in_embeddings (int): Number of embedding features.
        out_concepts (int): Number of output concepts.
        cardinalities (List[int]): Cardinalities for grouped concepts.
        predictor (nn.Module): Linear predictor module.

    Args:
        in_concepts: Number of input concepts.
        in_embeddings: Number of embedding features (must be even).
        out_concepts: Number of output concepts.
        cardinalities: List of concept group cardinalities. Required — must
            sum to ``in_concepts``.

    Example:
        >>> import torch
        >>> from torch_concepts.nn import MixConceptEmbeddingToConcept
        >>> from torch_concepts import Annotations
        >>>
        >>> # Create predictor: 3 concepts (cardinalities 3, 4, 3), 10 embedding dims, 2 outputs
        >>> in_ann = Annotations(labels=['color', 'shape', 'size'], cardinalities=[3, 4, 3])
        >>> predictor = MixConceptEmbeddingToConcept(
        ...     in_concepts=in_ann,
        ...     in_embeddings=10,
        ...     out_concepts=2,
        ... )
        >>>
        >>> # Generate random inputs
        >>> concepts = torch.randn(4, 10)  # batch_size=4, total logits (3+4+3=10)
        >>> embeddings = torch.randn(4, 10, 10)  # (batch, total_cardinality, emb_size)
        >>>
        >>> # Forward pass
        >>> output = predictor(concepts=concepts, embeddings=embeddings)
        >>> print(output.shape)
        torch.Size([4, 2])

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
        Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056
    """
    def __init__(
        self,
        in_concepts: Annotations,
        in_embeddings: Union[int, Annotations],
        out_concepts: Union[int, Annotations],
        expand_binary_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_concepts=in_concepts,
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
            expand_binary_embeddings=expand_binary_embeddings,
        )
        self.predictor = torch.nn.Linear(
            self.in_embeddings_shape * len(in_concepts.cardinalities),
            self.out_concepts_shape,
        )

    def forward(
        self,
        concepts: torch.Tensor,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the predictor.

        Args:
            concepts: Concept activations of shape ``(batch_size, in_concepts)``.
            embeddings: Concept embeddings of shape ``(batch_size, in_concepts, in_embeddings)``.

        Returns:
            torch.Tensor: Output concepts of shape (batch_size, out_concepts).
        """
        # For concepts with cardinality 1, split the Bernoulli probability into a categorical distribution
        c_mix = self._mix(concepts, embeddings)  # (batch, n_groups, in_embeddings)
        c_mix = c_mix.flatten(start_dim=1)      # (batch, n_groups * in_embeddings)
        return self.predictor(c_mix)



class MixConceptEmbeddingToEmbedding(MixConceptEmbedding):
    """Concept bottleneck layer of a Concept Bottleneck Generative Model.

    Mixes each concept's state embeddings by its predicted probability — the
    CEM mixture, :math:`w_i = \\hat{c}_i w^+_i + (1 - \\hat{c}_i) w^-_i`. Unlike
    :class:`MixConceptEmbeddingToConcept` there is no predictor head: the layer
    emits the mixed contexts :math:`[w_1, \\dots, w_k]` themselves. The caller
    concatenates the *unsupervised* context :math:`w_{k+1}` to obtain the full
    bottleneck :math:`w`, which the post-concept-bottleneck network (a
    generative model's decoder) consumes.

    The pre-defined concepts are not assumed to be complete in a generative
    setting, so the unsupervised context carries whatever they do not cover; an
    orthogonality penalty
    (:func:`~torch_concepts.nn.functional.concept_orthogonality`) keeps it from
    simply re-encoding them.

    Attributes:
        in_concepts (Annotations): Input concept annotations.
        in_embeddings (int): Number of embedding features per state.
        out_concepts (int): Width of the bottleneck the caller assembles,
            ``in_embeddings * (k + 1)`` — this layer contributes the first
            ``in_embeddings * k`` of it.

    Args:
        in_concepts: Annotations of the ``k`` supervised concepts.
        in_embeddings: Number of embedding features per state (``m``).

    Example:
        >>> import torch
        >>> from torch_concepts import Annotations
        >>> from torch_concepts.nn import MixConceptEmbeddingToEmbedding
        >>>
        >>> in_ann = Annotations(labels=['digit', 'color'], cardinalities=[10, 2])
        >>> layer = MixConceptEmbeddingToEmbedding(in_concepts=in_ann, in_embeddings=16)
        >>>
        >>> concepts = torch.rand(4, 12)          # (batch, 10 + 2 state scores)
        >>> embeddings = torch.randn(4, 12, 16)   # (batch, states, m)
        >>>
        >>> mixed = layer(concepts=concepts, embeddings=embeddings)
        >>> mixed.shape
        torch.Size([4, 2, 16])
        >>>
        >>> unknown = torch.randn(4, 1, 16)       # (batch, 1, m)
        >>> torch.cat([mixed, unknown], dim=-2).shape   # the full bottleneck
        torch.Size([4, 3, 16])

    Note:
        The paper defines the bottleneck as ``m(k + 1)`` wide — the concept
        contexts plus the unsupervised one. The authors' released code
        additionally prepends the concept probabilities themselves
        (``g_latent += sum(concept_bins)``); this layer follows the paper.

    References:
        Ismail et al. "Concept Bottleneck Generative Models", ICLR 2024.
        https://openreview.net/forum?id=L9U5MJJleF
    """

    def __init__(
        self,
        in_concepts: Annotations,
        in_embeddings: int,
        expand_binary_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_concepts=in_concepts,
            in_embeddings=in_embeddings,
            out_concepts=in_embeddings * (len(in_concepts.cardinalities) + 1),
            expand_binary_embeddings=expand_binary_embeddings,
        )

    def forward(
        self,
        concepts: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the concept bottleneck.

        Args:
            concepts: Concept probabilities of shape ``(batch_size, n_states)``.
            embeddings: State embeddings of shape ``(batch_size, n_states, in_embeddings)``.

        Returns:
            torch.Tensor: Mixed concept contexts of shape
            ``(batch_size, k, in_embeddings)``. Concatenate the unsupervised
            context along ``dim=-2`` for the full bottleneck.
        """
        return self._mix(concepts, embeddings)  # (batch, k, in_embeddings)


class MixSumConceptEmbeddingToConcept(MixConceptEmbeddingToConcept):
    """Like :class:`MixConceptEmbeddingToConcept` but aggregates group
    embeddings by **summing** across groups instead of flattening.

    The predictor therefore maps ``(batch, in_embeddings)`` → ``(batch, out_concepts)``
    rather than ``(batch, n_groups × in_embeddings)`` → ``(batch, out_concepts)``,
    which makes it group-count invariant and more parameter-efficient.
    """

    def __init__(
        self,
        in_concepts: int,
        in_embeddings: int,
        out_concepts: int,
        cardinalities: list[int] = None,
        bias: bool = True,
        **kwargs
    ):
        # FIXME: update to use Annotations for in_concepts and out_concepts, 
        # and remove cardinalities
        if cardinalities is None:
            cardinalities = [1] * in_concepts
        n_groups = len(cardinalities)
        types = ['binary' if c == 1 else 'categorical' for c in cardinalities]
        annotation = Annotations(
            labels=[f"c{i}" for i in range(n_groups)],
            cardinalities=cardinalities,
            types=types,
        )
        super().__init__(
            in_concepts=annotation,
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
            **kwargs,
        )
        self.predictor = torch.nn.Linear(in_embeddings, out_concepts, bias=bias)

    def forward(self, concepts: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        c_mix = self._mix(concepts, embeddings)  # same as CEM-layer (batch, n_groups, in_embeddings)
        c_mix = c_mix.sum(dim=1)                # (batch, in_embeddings)
        return self.predictor(c_mix)
