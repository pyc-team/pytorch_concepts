import torch

from ..base.layer import BasePredictor
from ....functional import grouped_concept_exogenous_mixture
from typing import List, Callable


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
        activation: Callable = torch.sigmoid
    ):
        super().__init__(
            in_concepts=in_concepts,
            in_exogenous=in_exogenous,
            out_concepts=out_concepts,
            activation=activation,
        )
        assert in_exogenous % 2 == 0, "in_exogenous must be divisible by 2."
        if cardinalities is None:
            # assume all binary
            self.cardinalities = [1] * in_concepts
            predictor_in_features = in_exogenous * in_concepts
        else:
            self.cardinalities = cardinalities
            assert sum(self.cardinalities) == in_concepts
            predictor_in_features = (in_exogenous // 2) * len(self.cardinalities)

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(
                predictor_in_features,
                out_concepts
            ),
            torch.nn.Unflatten(-1, (out_concepts,)),
        )

    def forward(
        self,
        concepts: torch.Tensor,
        exogenous: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the predictor.

        Args:
            concepts: Concept logits of shape (batch_size, in_concepts).
            exogenous: Concept exogenous of shape (batch_size, in_concepts, exogenous_dim).

        Returns:
            torch.Tensor: Output concepts of shape (batch_size, out_concepts).
        """
        in_probs = self.activation(concepts)
        c_mix = grouped_concept_exogenous_mixture(exogenous, in_probs, groups=self.cardinalities)
        return self.predictor(c_mix.flatten(start_dim=1))
