import torch

from ..base.layer import BasePredictor
from ....functional import grouped_concept_exogenous_mixture, replace_expand_cols
from typing import List


class MixConceptExogegnousToConcept(BasePredictor):
    """
    Concept exogenous predictor with mixture of concept activations and exogenous features.

    This predictor implements the Concept Embedding Model (CEM) task predictor that
    combines concept activations with learned exogenous using a mixture operation.

    Main reference: "Concept Embedding Models: Beyond the Accuracy-Explainability
    Trade-Off" (Espinosa Zarlenga et al., NeurIPS 2022).

    Attributes:
        in_concepts (int): Number of input concepts.
        in_exogenous (int): Number of exogenous features.
        out_concepts (int): Number of output concepts.
        cardinalities (List[int]): Cardinalities for grouped concepts.
        predictor (nn.Module): Linear predictor module.

    Args:
        in_concepts: Number of input concepts.
        in_exogenous: Number of exogenous features (must be even).
        out_concepts: Number of output concepts.
        activation: Activation function for concept logits (default: sigmoid).
        cardinalities: List of concept group cardinalities (optional).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import MixConceptExogegnousToConcept
        >>>
        >>> # Create predictor with 10 concepts, 20 exogenous dims, 3 output concepts
        >>> predictor = MixConceptExogegnousToConcept(
        ...     in_concepts=10,
        ...     in_exogenous=10,  # Must be half of exogenous latent size when no cardinalities are provided
        ...     out_concepts=3,
        ...     activation=torch.sigmoid
        ... )
        >>>
        >>> # Generate random inputs
        >>> concepts = torch.randn(4, 10)  # batch_size=4, n_concepts=10
        >>> exogenous = torch.randn(4, 10, 20)  # (batch, n_concepts, emb_size)
        >>>
        >>> # Forward pass
        >>> output = predictor(concepts=concepts, exogenous=exogenous)
        >>> print(output.shape)  # torch.Size([4, 3])
        >>>
        >>> # With concept groups (e.g., color has 3 values, shape has 4, etc.)
        >>> predictor_grouped = MixConceptExogegnousToConcept(
        ...     in_concepts=10,
        ...     in_exogenous=20, # Must be equal to exogenous latent size when cardinalities are provided
        ...     out_concepts=3,
        ...     cardinalities=[3, 4, 3]  # 3 groups summing to 10
        ... )
        >>>
        >>> # Forward pass with grouped concepts
        >>> output = predictor_grouped(concepts=concepts, exogenous=exogenous)
        >>> print(output.shape)  # torch.Size([4, 3])

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
        Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056
    """
    def __init__(
        self,
        in_concepts: int,
        in_exogenous: int,
        out_concepts: int,
        cardinalities: List[int],
        **kwargs,
    ):
        super().__init__(
            in_concepts=in_concepts,
            in_exogenous=in_exogenous,
            out_concepts=out_concepts,
        )
        if cardinalities is None:
            raise ValueError("Cardinalities must be provided for MixConceptExogegnousToConcept.")
        else:
            self.cardinalities = cardinalities
            assert sum(self.cardinalities) == in_concepts

        # find positions of concepts with cardinality 1 for Bernoulli to Categorical splitting
        self.cardinalities_expanded = torch.tensor(self.cardinalities)
        cumsum = torch.cumsum(self.cardinalities_expanded, dim=0)
        start_positions = cumsum - self.cardinalities_expanded
        self.mask_cardinality_1 = start_positions[self.cardinalities_expanded == 1]
        self.cardinalities_expanded[self.cardinalities_expanded == 1] = 2

        self.bernoulli_to_categorical_exogenous_splitter = torch.nn.Sequential(
            torch.nn.Linear(in_exogenous, in_exogenous*2),
            torch.nn.LeakyReLU(),
            torch.nn.Unflatten(-1, (-1, in_exogenous)),
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(
                in_exogenous * len(self.cardinalities),
                out_concepts
            ),
            torch.nn.Unflatten(-1, (out_concepts,)),
        )

    def _mix(
        self,
        concepts: torch.Tensor,
        exogenous: torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess inputs and compute per-group mixed embeddings.

        Handles the Bernoulli→Categorical expansion for cardinality-1 concepts
        and returns ``c_mix`` of shape ``(batch, n_groups, exogenous_dim)``.
        Subclasses can call this and only vary the final aggregation step.
        """
        if len(self.mask_cardinality_1) > 0:
            exogenous_split = self.bernoulli_to_categorical_exogenous_splitter(exogenous[:, self.mask_cardinality_1])
            concepts_split = torch.cat([
                concepts[:, self.mask_cardinality_1[:, None]],
                1 - concepts[:, self.mask_cardinality_1[:, None]],
            ], dim=-1)
            exogenous = replace_expand_cols(exogenous, self.mask_cardinality_1, exogenous_split)
            concepts  = replace_expand_cols(concepts,  self.mask_cardinality_1, concepts_split)
        return grouped_concept_exogenous_mixture(
            exogenous,
            concepts,
            groups=list(self.cardinalities_expanded),
        )
    
    def forward(
        self,
        concepts: torch.Tensor,
        exogenous: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the predictor.

        Args:
            concepts: Concept of shape (batch_size, in_concepts).
            exogenous: Concept exogenous of shape (batch_size, in_concepts, exogenous_dim).

        Returns:
            torch.Tensor: Output concepts of shape (batch_size, out_concepts).
        """
        # For concepts with cardinality 1, split the Bernoulli probability into a categorical distribution
        c_mix = self._mix(concepts, exogenous)  # (batch, n_groups, exogenous_dim)
        c_mix = c_mix.flatten(start_dim=1)      # (batch, n_groups * exogenous_dim)
        return self.predictor(c_mix)



class MixSumConceptExogenousToConcept(MixConceptExogegnousToConcept):
    """Like :class:`MixConceptExogegnousToConcept` but aggregates group
    embeddings by **summing** across groups instead of flattening.

    The predictor therefore maps ``(batch, exogenous_dim)`` → ``(batch, out_concepts)``
    rather than ``(batch, n_groups × exogenous_dim)`` → ``(batch, out_concepts)``,
    which makes it group-count invariant and more parameter-efficient.
    """

    def __init__(
        self, 
        in_concepts: int, 
        in_exogenous: int, 
        out_concepts: int, 
        cardinalities: list[int] | None = None,
        bias: bool = True,
        **kwargs
    ):
        if cardinalities is None:
            cardinalities = [1] * in_concepts
        super().__init__(
            in_concepts=in_concepts,
            in_exogenous=in_exogenous,
            out_concepts=out_concepts,
            cardinalities=cardinalities,
            **kwargs,
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(in_exogenous, out_concepts, bias=bias),
            torch.nn.Unflatten(-1, (out_concepts,)),
        )

    def forward(self, concepts: torch.Tensor, exogenous: torch.Tensor) -> torch.Tensor:
        c_mix = self._mix(concepts, exogenous)  # same as CEM-layer (batch, n_groups, exogenous_dim)
        c_mix = c_mix.sum(dim=1)                # (batch, exogenous_dim)
        return self.predictor(c_mix)