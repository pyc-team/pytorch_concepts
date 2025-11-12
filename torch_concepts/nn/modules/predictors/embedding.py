import torch

from ...base.layer import BasePredictor
from ...functional import grouped_concept_embedding_mixture
from typing import List, Callable, Union


class MixProbExogPredictor(BasePredictor):
    """
    ConceptEmbeddingLayer creates supervised concept embeddings.
    Main reference: `"Concept Embedding Models: Beyond the
    Accuracy-Explainability Trade-Off" <https://arxiv.org/abs/2209.09056>`_

    Attributes:
        in_features (int): Number of input features.
        annotations (Union[List[str], int]): Concept dimensions.
        activation (Callable): Activation function of concept scores.
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
        in_probs = self.in_activation(logits)
        c_mix = grouped_concept_embedding_mixture(exogenous, in_probs, groups=self.cardinalities)
        return self.predictor(c_mix.flatten(start_dim=1))
