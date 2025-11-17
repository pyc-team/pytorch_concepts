import torch

from ....nn.base.layer import BaseConceptLayer
from typing import List, Union, Optional


class RandomPolicy(BaseConceptLayer):
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
        out_features: int,
        scale: float = 1.0,
    ):
        super().__init__(
            out_features=out_features,
        )
        self.scale = scale

    def forward(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        return torch.rand_like(logits).abs() * self.scale
