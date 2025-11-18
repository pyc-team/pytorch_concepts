import torch

from ...base.layer import BasePredictor
from ...functional import grouped_concept_embedding_mixture
from typing import List, Callable, Union


class MixProbExogPredictor(BasePredictor):
    """
    Concept embedding predictor with mixture of concept activations and exogenous features.

    This predictor implements the Concept Embedding Model (CEM) task predictor that
    combines concept activations with learned embeddings using a mixture operation.

    Main reference: "Concept Embedding Models: Beyond the Accuracy-Explainability
    Trade-Off" (Espinosa Zarlenga et al., NeurIPS 2022).

    Attributes:
        in_features_logits (int): Number of input concept logits.
        in_features_exogenous (int): Number of exogenous embedding features.
        out_features (int): Number of output features.
        cardinalities (List[int]): Cardinalities for grouped concepts.
        predictor (nn.Module): Linear predictor module.

    Args:
        in_features_logits: Number of input concept logits.
        in_features_exogenous: Number of exogenous embedding features (must be even).
        out_features: Number of output task features.
        in_activation: Activation function for concept logits (default: sigmoid).
        cardinalities: List of concept group cardinalities (optional).

    Example:
        >>> import torch
        >>> from torch_concepts.nn import MixProbExogPredictor
        >>>
        >>> # Create predictor with 10 concepts, 20 embedding dims, 3 tasks
        >>> predictor = MixProbExogPredictor(
        ...     in_features_logits=10,
        ...     in_features_exogenous=10,  # Must be half of exogenous latent size when no cardinalities are provided
        ...     out_features=3,
        ...     in_activation=torch.sigmoid
        ... )
        >>>
        >>> # Generate random inputs
        >>> concept_logits = torch.randn(4, 10)  # batch_size=4, n_concepts=10
        >>> exogenous = torch.randn(4, 10, 20)  # (batch, n_concepts, emb_size)
        >>>
        >>> # Forward pass
        >>> task_logits = predictor(logits=concept_logits, exogenous=exogenous)
        >>> print(task_logits.shape)  # torch.Size([4, 3])
        >>>
        >>> # With concept groups (e.g., color has 3 values, shape has 4, etc.)
        >>> predictor_grouped = MixProbExogPredictor(
        ...     in_features_logits=10,
        ...     in_features_exogenous=20, # Must be equal to exogenous latent size when cardinalities are provided
        ...     out_features=3,
        ...     cardinalities=[3, 4, 3]  # 3 groups summing to 10
        ... )
        >>>
        >>> # Forward pass with grouped concepts
        >>> task_logits = predictor_grouped(logits=concept_logits, exogenous=exogenous)
        >>> print(task_logits.shape)  # torch.Size([4, 3])

    References:
        Espinosa Zarlenga et al. "Concept Embedding Models: Beyond the
        Accuracy-Explainability Trade-Off", NeurIPS 2022.
        https://arxiv.org/abs/2209.09056
    """
    def __init__(
        self,
        in_features_logits: int,
        in_features_exogenous: int,
        out_features: int,
        in_activation: Callable = torch.sigmoid,
        cardinalities: List[int] = None
    ):
        super().__init__(
            in_features_logits=in_features_logits,
            in_features_exogenous=in_features_exogenous,
            out_features=out_features,
            in_activation=in_activation,
        )
        assert in_features_exogenous % 2 == 0, "in_features_exogenous must be divisible by 2."
        if cardinalities is None:
            self.cardinalities = [1] * in_features_logits
            predictor_in_features = in_features_exogenous*in_features_logits
        else:
            self.cardinalities = cardinalities
            assert sum(self.cardinalities) == in_features_logits
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
        logits: torch.Tensor,
        exogenous: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the predictor.

        Args:
            logits: Concept logits of shape (batch_size, n_concepts).
            exogenous: Concept embeddings of shape (batch_size, n_concepts, emb_size).

        Returns:
            torch.Tensor: Task predictions of shape (batch_size, out_features).
        """
        in_probs = self.in_activation(logits)
        c_mix = grouped_concept_embedding_mixture(exogenous, in_probs, groups=self.cardinalities)
        return self.predictor(c_mix.flatten(start_dim=1))
