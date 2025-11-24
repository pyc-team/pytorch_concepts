import torch

from ..base.layer import BasePredictor
from ....functional import grouped_concept_exogenous_mixture
from typing import List, Callable


class MixCUC(BasePredictor):
    """
    Concept exogenous predictor with mixture of concept activations and exogenous features.

    This predictor implements the Concept Embedding Model (CEM) task predictor that
    combines concept activations with learned exogenous using a mixture operation.

    Main reference: "Concept Embedding Models: Beyond the Accuracy-Explainability
    Trade-Off" (Espinosa Zarlenga et al., NeurIPS 2022).

    Attributes:
        in_features_endogenous (int): Number of input concept endogenous.
        in_features_exogenous (int): Number of exogenous features.
        out_features (int): Number of output features.
        cardinalities (List[int]): Cardinalities for grouped concepts.
        predictor (nn.Module): Linear predictor module.

    Args:
        in_features_endogenous: Number of input concept endogenous.
        in_features_exogenous: Number of exogenous features (must be even).
        out_features: Number of output task features.
        in_activation: Activation function for concept endogenous (default: sigmoid).
        cardinalities: List of concept group cardinalities (optional).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import MixCUC
        >>>
        >>> # Create predictor with 10 concepts, 20 exogenous dims, 3 tasks
        >>> predictor = MixCUC(
        ...     in_features_endogenous=10,
        ...     in_features_exogenous=10,  # Must be half of exogenous latent size when no cardinalities are provided
        ...     out_features=3,
        ...     in_activation=torch.sigmoid
        ... )
        >>>
        >>> # Generate random inputs
        >>> concept_endogenous = torch.randn(4, 10)  # batch_size=4, n_concepts=10
        >>> exogenous = torch.randn(4, 10, 20)  # (batch, n_concepts, emb_size)
        >>>
        >>> # Forward pass
        >>> task_endogenous = predictor(endogenous=concept_endogenous, exogenous=exogenous)
        >>> print(task_endogenous.shape)  # torch.Size([4, 3])
        >>>
        >>> # With concept groups (e.g., color has 3 values, shape has 4, etc.)
        >>> predictor_grouped = MixCUC(
        ...     in_features_endogenous=10,
        ...     in_features_exogenous=20, # Must be equal to exogenous latent size when cardinalities are provided
        ...     out_features=3,
        ...     cardinalities=[3, 4, 3]  # 3 groups summing to 10
        ... )
        >>>
        >>> # Forward pass with grouped concepts
        >>> task_endogenous = predictor_grouped(endogenous=concept_endogenous, exogenous=exogenous)
        >>> print(task_endogenous.shape)  # torch.Size([4, 3])

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
        Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056
    """
    def __init__(
        self,
        in_features_endogenous: int,
        in_features_exogenous: int,
        out_features: int,
        in_activation: Callable = torch.sigmoid,
        cardinalities: List[int] = None
    ):
        super().__init__(
            in_features_endogenous=in_features_endogenous,
            in_features_exogenous=in_features_exogenous,
            out_features=out_features,
            in_activation=in_activation,
        )
        assert in_features_exogenous % 2 == 0, "in_features_exogenous must be divisible by 2."
        if cardinalities is None:
            self.cardinalities = [1] * in_features_endogenous
            predictor_in_features = in_features_exogenous*in_features_endogenous
        else:
            self.cardinalities = cardinalities
            assert sum(self.cardinalities) == in_features_endogenous
            predictor_in_features = (in_features_exogenous//2)*len(self.cardinalities)

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(
                predictor_in_features,
                out_features
            ),
            torch.nn.Unflatten(-1, (out_features,)),
        )

    def forward(
        self,
        endogenous: torch.Tensor,
        exogenous: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the predictor.

        Args:
            endogenous: Concept endogenous of shape (batch_size, n_concepts).
            exogenous: Concept exogenous of shape (batch_size, n_concepts, emb_size).

        Returns:
            torch.Tensor: Task predictions of shape (batch_size, out_features).
        """
        in_probs = self.in_activation(endogenous)
        c_mix = grouped_concept_exogenous_mixture(exogenous, in_probs, groups=self.cardinalities)
        return self.predictor(c_mix.flatten(start_dim=1))
