import torch

from ...base.layer import BasePredictor
from typing import List, Callable, Union

from ...functional import prune_linear_layer


class ProbPredictor(BasePredictor):
    """
    ConceptLayer creates a bottleneck of supervised concepts.
    Main reference: `"Concept Layer
    Models" <https://arxiv.org/pdf/2007.04612>`_

    Attributes:
        in_features (int): Number of input features.
        annotations (Union[List[str], int]): Concept dimensions.
        activation (Callable): Activation function of concept scores.
    """

    def __init__(
        self,
        in_features_logits: int,
        out_features: int,
        in_activation: Callable = torch.sigmoid
    ):
        super().__init__(
            in_features_logits=in_features_logits,
            out_features=out_features,
            in_activation=in_activation,
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_logits,
                out_features
            ),
            torch.nn.Unflatten(-1, (out_features,)),
        )

    def forward(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        in_probs = self.in_activation(logits)
        probs = self.predictor(in_probs)
        return probs

    def prune(self, mask: torch.Tensor):
        self.in_features_logits = sum(mask.int())
        self.predictor[0] = prune_linear_layer(self.predictor[0], mask, dim=0)
