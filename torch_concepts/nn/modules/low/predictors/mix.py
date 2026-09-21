import torch
import torch.nn as nn
import numpy as np

from torch_concepts import Annotations
from ..base.layer import BaseConceptLayer
from ...utils import state_embedding_counts
from ....functional import grouped_concept_exogenous_mixture, replace_expand_cols
from typing import List, Union


class MixConceptEmbeddingToConceptEmbedding(torch.nn.Module):
    """Mix concept values with their state embeddings, concept by concept.

    ``concept_embeddings`` is a sequence of state-embedding banks and
    ``concept_values`` is the matching sequence of observed/predicted concept
    values. ``source_concepts`` maps each bank back to the original concept
    index, so callers may pass only a subset of concepts.

    With ``complete_output=False`` the output follows the supplied source order
    and has trailing shape ``(len(source_concepts), in_embeddings)``. With
    ``complete_output=True`` the output is scattered back to the full concept
    axis, with trailing shape ``(n_concepts, in_embeddings)`` and zeros for
    concepts that were not supplied. CGM uses the complete form before graph
    aggregation so the adjacency always indexes the original concept order.

    This module has trainable parameters only when binary embeddings are
    expanded. It does not cache outputs; if multiple CPDs call it with the same
    inputs, the same mixture is recomputed.
    """

    def __init__(
        self,
        in_embeddings,
        concept_types,
        cardinalities,
        expand_binary_embeddings=False,
        complete_output=False,
    ):
        super().__init__()

        self.in_embeddings = in_embeddings
        self.concept_types = list(concept_types)
        self.cardinalities = list(cardinalities)
        self.expand_binary_embeddings = expand_binary_embeddings
        self.complete_output = complete_output

        self.n_state_embeddings = state_embedding_counts(
            self.concept_types, self.cardinalities, expand_binary_embeddings
        )

        if expand_binary_embeddings:
            self.binary_embedding_expander = torch.nn.Sequential(
                torch.nn.Linear(
                    in_embeddings,
                    2 * in_embeddings,
                ),
                torch.nn.LeakyReLU(),
                torch.nn.Unflatten(
                    -1,
                    (2, in_embeddings),
                ),
            )
        else:
            self.binary_embedding_expander = None

    def forward(
        self,
        concept_embeddings,
        concept_values,
        source_concepts=None,
    ):
        """Return mixed embeddings for the supplied source concepts."""
        if not concept_embeddings:
            raise ValueError("At least one concept embedding is required.")

        if len(concept_embeddings) != len(concept_values):
            raise ValueError(
                "Each embedding bank needs one concept value."
            )

        if source_concepts is None:
            source_concepts = list(range(len(concept_embeddings)))
        else:
            source_concepts = list(source_concepts)

        if len(source_concepts) != len(concept_embeddings):
            raise ValueError(
                "source_concepts must contain one index per embedding bank."
            )
        if len(set(source_concepts)) != len(source_concepts):
            raise ValueError("source_concepts must not contain duplicates.")
        if any(
            not 0 <= source < len(self.concept_types)
            for source in source_concepts
        ):
            raise IndexError("source_concepts contains an out-of-range index.")

        # Mix each concept value with its corresponding embedding bank.
        mixed = torch.stack([
            self._mix(
                source_concept,
                embeddings,
                value,
            )
            for source_concept, embeddings, value in zip(
                source_concepts,
                concept_embeddings,
                concept_values,
            )
        ], dim=1)

        if not self.complete_output:
            return mixed

        complete = mixed.new_zeros(
            *mixed.shape[:-2],
            len(self.concept_types),
            self.in_embeddings,
        )
        complete[..., source_concepts, :] = mixed

        return complete

    def _mix(
        self,
        source_concept,
        embeddings,
        value,
    ):
        concept_type = self.concept_types[source_concept]
        cardinality = self.cardinalities[source_concept]
        if embeddings.shape[-1] != self.in_embeddings:
            raise ValueError(
                f"Embedding width must be {self.in_embeddings}, "
                f"got {embeddings.shape[-1]}."
            )
        value = value.to(dtype=embeddings.dtype, device=embeddings.device)

        if concept_type == "binary":
            if self.expand_binary_embeddings:
                if embeddings.shape[-2] != 1:
                    raise ValueError(
                        "Binary expansion expects one embedding."
                    )

                embeddings = self.binary_embedding_expander(
                    embeddings.squeeze(-2)
                )

            elif embeddings.shape[-2] != 2:
                raise ValueError(
                    "A binary concept expects two state embeddings."
                )

            value = torch.cat(
                [value, 1 - value],
                dim=-1,
            )

        elif (
            concept_type == "categorical"
            and value.shape[-1] == 1
        ):
            value = torch.nn.functional.one_hot(
                value.squeeze(-1).long(),
                cardinality,
            ).to(embeddings.dtype)

        elif embeddings.shape[-2] != cardinality:
            raise ValueError(
                f"Concept {source_concept} expects "
                f"{cardinality} state embeddings."
            )

        return grouped_concept_exogenous_mixture(
            embeddings,
            value,
            groups=[embeddings.shape[-2]],
        ).squeeze(-2)


class MixConceptEmbeddings(nn.Module):
    """Mix each concept's state embeddings by its predicted score.

    The per-group mixture :math:`\\hat{c}_i w^+_i + (1 - \\hat{c}_i) w^-_i` of
    "Concept Embedding Models" (Espinosa Zarlenga et al., NeurIPS 2022), returning
    the mixed contexts ``(batch, n_groups, in_embeddings)``.

    A binary concept supplies one state embedding; the second is derived from it
    by a learned ``Linear(m, 2m) + LeakyReLU``, with the score expanded to
    ``[c, 1-c]`` to match.

    Args:
        in_concepts: Annotations of the input concepts (their cardinalities and
            types drive the grouping).
        in_embeddings: Number of embedding features per state.

    Example:
        >>> import torch
        >>> from torch_concepts import Annotations
        >>> from torch_concepts.nn import MixConceptEmbeddings
        >>>
        >>> in_ann = Annotations(labels=['digit', 'color'], cardinalities=[10, 2])
        >>> layer = MixConceptEmbeddings(in_concepts=in_ann, in_embeddings=16)
        >>>
        >>> concepts = torch.rand(4, 12)          # (batch, 10 + 2 state scores)
        >>> embeddings = torch.randn(4, 12, 16)   # (batch, states, m)
        >>> layer(concepts=concepts, embeddings=embeddings).shape
        torch.Size([4, 2, 16])
    """

    def __init__(
        self,
        in_concepts: Annotations,
        in_embeddings: int,
    ):
        super().__init__()
        self.in_concepts = in_concepts
        self.in_embeddings = in_embeddings
        self.in_embeddings_shape = in_embeddings
        # find positions of concepts with cardinality 1 for Bernoulli to Categorical splitting
        self.cardinalities_expanded = torch.tensor(in_concepts.cardinalities)
        self.binary_mask = torch.from_numpy(np.array(in_concepts.types) != 'continuous')
        cumsum = torch.cumsum(self.cardinalities_expanded, dim=0)
        start_positions = cumsum - self.cardinalities_expanded
        bernoulli_mask = (self.cardinalities_expanded == 1) & self.binary_mask
        # This index is used directly against ``concepts`` and ``embeddings``
        # in ``forward``. Keep it as a buffer so ``module.to(device)`` moves it
        # together with those tensors when Lightning selects CUDA.
        self.register_buffer(
            "mask_cardinality_1", start_positions[bernoulli_mask]
        )
        self.cardinalities_expanded[bernoulli_mask] = 2

        self.bernoulli_to_categorical_embedding_splitter = torch.nn.Sequential(
            torch.nn.Linear(self.in_embeddings_shape, self.in_embeddings_shape*2),
            torch.nn.LeakyReLU(),
            torch.nn.Unflatten(-1, (-1, self.in_embeddings_shape)),
        )

    def forward(
        self,
        concepts: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess inputs and compute per-group mixed embeddings.

        Handles the Bernoulli→Categorical expansion for cardinality-1 concepts
        and returns ``c_mix`` of shape ``(batch, n_groups, in_embeddings)``.
        """
        if len(self.mask_cardinality_1) > 0:
            embeddings_split = self.bernoulli_to_categorical_embedding_splitter(embeddings[:, self.mask_cardinality_1])
            concepts_split = torch.cat([
                concepts[:, self.mask_cardinality_1[:, None]],
                1 - concepts[:, self.mask_cardinality_1[:, None]],
            ], dim=-1)
            embeddings = replace_expand_cols(embeddings, self.mask_cardinality_1, embeddings_split)
            concepts  = replace_expand_cols(concepts,  self.mask_cardinality_1, concepts_split)
        return grouped_concept_exogenous_mixture(
            embeddings,
            concepts,
            groups=list(self.cardinalities_expanded),
        )


class MixConceptEmbeddingToConcept(BaseConceptLayer):
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
        in_embeddings: int,
        out_concepts: Union[int, Annotations],
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_concepts=in_concepts,
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        self.mixture = MixConceptEmbeddings(
            in_concepts=in_concepts,
            in_embeddings=in_embeddings,
        )
        self.predictor = torch.nn.Linear(
            self.in_embeddings_shape * len(in_concepts.cardinalities),
            self.out_concepts_shape,
            bias=bias,
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
        c_mix = self.mixture(concepts, embeddings)  # (batch, n_groups, in_embeddings)
        c_mix = c_mix.flatten(start_dim=1)      # (batch, n_groups * in_embeddings)
        return self.predictor(c_mix)



class MixSumConceptEmbeddingToConcept(BaseConceptLayer):
    """Like :class:`MixConceptEmbeddingToConcept` but aggregates group
    embeddings by **summing** across groups instead of flattening.

    The predictor therefore maps ``(batch, in_embeddings)`` → ``(batch, out_concepts)``
    rather than ``(batch, n_groups × in_embeddings)`` → ``(batch, out_concepts)``,
    which makes it group-count invariant and more parameter-efficient.

    Args:
        in_concepts: Annotations of the input concepts (their cardinalities and
            types drive the grouping).
        in_embeddings: Number of embedding features per state.
        out_concepts: Output width.
        bias: Whether the predictor has a bias. ``False`` makes the layer exactly
            a sum of per-group linear maps, since a bias on the summed mixture is
            added once rather than once per group.
    """

    def __init__(
        self,
        in_concepts: Annotations,
        in_embeddings: int,
        out_concepts: Union[int, Annotations],
        bias: bool = True,
        **kwargs
    ):
        super().__init__(
            in_concepts=in_concepts,
            in_embeddings=in_embeddings,
            out_concepts=out_concepts,
        )
        self.mixture = MixConceptEmbeddings(
            in_concepts=in_concepts,
            in_embeddings=in_embeddings,
        )
        self.predictor = torch.nn.Linear(
            self.in_embeddings_shape, self.out_concepts_shape, bias=bias,
        )

    def forward(self, concepts: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        c_mix = self.mixture(concepts, embeddings)  # same as CEM-layer (batch, n_groups, in_embeddings)
        c_mix = c_mix.sum(dim=1)                # (batch, in_embeddings)
        return self.predictor(c_mix)
