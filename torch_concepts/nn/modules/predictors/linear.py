import torch

from torch_concepts import Annotations, ConceptTensor
from ...base.layer import BasePredictor
from typing import List, Callable, Union, Dict, Tuple


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
        in_features_global: int,
        in_features_exogenous: int,
        in_features_logits: int,
        out_annotations: Annotations,
        in_activation: Callable = torch.sigmoid,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_features_global=in_features_global,
            in_features_exogenous=in_features_exogenous,
            in_features_logits=in_features_logits,
            out_annotations=out_annotations,
        )
        self.in_activation = in_activation # FIXME: this is the input activation, not the output!
        self.logit_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features_logits,
                self.out_annotations.shape[1],
                *args,
                **kwargs,
            ),
            torch.nn.Unflatten(-1, (self.out_annotations.shape[1],)),
        )

    def forward(
        self,
        logits: torch.Tensor,
        x: torch.Tensor=None,
        exogenous: torch.Tensor=None,
        *args,
        **kwargs,
    ) -> ConceptTensor:
        in_probs = self.in_activation(logits)
        probs = self.logit_layer(in_probs)
        return probs
